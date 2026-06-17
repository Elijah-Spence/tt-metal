// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief sfpi container type for an *integer* dst_reg[0] format (vInt→SMAG32, vSMag16→SMAG16,
 * vUInt16→UINT16).
 *
 * @note Floats (all widths share @c vFloat) and the 8-bit formats (no sfpi dst_reg conversion; raw
 *       SFPLOAD path) are handled in @ref zero_comp_traits, not here.
 */
template <DataFormat FMT>
struct dst_container {
    using type = sfpi::vInt;
};  // Int32
template <>
struct dst_container<DataFormat::Int16> {
    using type = sfpi::vSMag16;
};
template <>
struct dst_container<DataFormat::UInt16> {
    using type = sfpi::vUInt16;
};

/**
 * @brief Per-DataFormat load/store and 0/1 encoding for the comparison-to-zero result.
 *
 * @c load reads the element as raw @c vUInt bits for the shared predicate (see @ref _zero_comp_pred_);
 * @c store writes @c zero/@c one back in FMT's native encoding. The 16/32-bit and float formats use
 * the sfpi @c dst_reg[0] container; the 8-bit formats use a raw explicit-mode SFPLOAD/SFPSTORE.
 *
 * @tparam FMT: SFPU DataFormat (sfpu_math). Int32 / Int16 / Int8 signed, UInt16 / UInt8 unsigned, or
 *         any IEEE float width (Float32/Float16/Float16_b all share the float path).
 *
 * @todo Once sfpi adds an 8-bit dst_reg[0] conversion (vSMag8/vUInt8), add Int8→vSMag8 /
 *       UInt8→vUInt8 to dst_container and drop the is_8bit raw-SFPLOAD branches so 8-bit follows the
 *       same container path as the other formats.
 */
template <DataFormat FMT>
struct zero_comp_traits {
    static constexpr bool is_8bit = FMT == DataFormat::Int8 || FMT == DataFormat::UInt8;
    static constexpr bool is_float =
        FMT == DataFormat::Float32 || FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b;
    static constexpr std::uint32_t sfpmem_mode =
        FMT == DataFormat::Int8 ? ckernel::p_sfpu::sfpmem::INT8 : ckernel::p_sfpu::sfpmem::UINT8;

    // dst_reg[0] container carrying FMT's width/encoding (unused on the 8-bit raw path): vFloat for
    // every float width, else the per-format integer container.
    using container_t = std::conditional_t<is_float, sfpi::vFloat, typename dst_container<FMT>::type>;
    using result_t = std::conditional_t<is_float, sfpi::vFloat, sfpi::vInt>;

    static inline __attribute__((always_inline)) sfpi::vUInt load() {
        if constexpr (is_8bit) {
            return sfpi::vUInt(__builtin_rvtt_sfpload(0, sfpmem_mode, sfpi::SFPLOAD_ADDR_MODE_NOINC));
        } else {
            // Copy-init (not a functional cast) so the read routes through vReg's conversion
            // operator; the vNarrow container types (vUInt16/...) have only explicit ctors.
            container_t c = sfpi::dst_reg[0];
            return sfpi::reinterpret<sfpi::vUInt>(c);
        }
    }
    static inline __attribute__((always_inline)) result_t zero() { return result_t(0); }
    static inline __attribute__((always_inline)) result_t one() { return result_t(1); }
    static inline __attribute__((always_inline)) void store(result_t r) {
        if constexpr (is_8bit) {
            __builtin_rvtt_sfpstore(r.get(), 0, sfpmem_mode, ckernel::ADDR_MOD_6);
        } else {
            sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = sfpi::reinterpret<container_t>(r);
        }
    }
};

/**
 * @brief Lanes of @c u satisfying COMP_MODE against zero, read off the raw bit pattern.
 *
 * Each mode is built from two bit tests that read identically across every format (Quasar loads
 * signed integers as sign-magnitude and floats as IEEE, both with the sign in bit 31): the sign bit
 * @c u>>31 and the magnitude-zero test @c (u&0x7FFFFFFF)==0. Masking the sign bit makes the
 * magnitude test hold for both +0 (0x00000000) and -0 (0x80000000), so -0.0 / sign-magnitude -0
 * count as zero, matching IEEE (-0.0 == 0); "negative" is @c sign && magnitude-nonzero, excluding -0.0.
 *
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 */
template <SfpuType COMP_MODE>
inline __attribute__((always_inline)) sfpi::vBool _zero_comp_pred_(sfpi::vUInt u) {
    // Magnitude mask sourced from a const register (programmed once in _calculate_zero_comp_)
    // instead of two per-iteration sfploadi. Same (u & 0x7FFFFFFF)==0 logic -> ±0 still correct.
    const sfpi::vUInt mag = u & sfpi::reinterpret<sfpi::vUInt>(sfpi::vInt(sfpi::vConstIntPrgm0));
    if constexpr (COMP_MODE == SfpuType::equal_zero) {
        return mag == 0u;  // ±0
    } else if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
        return mag != 0u;
    } else if constexpr (COMP_MODE == SfpuType::less_than_zero) {
        return ((u >> 31) != 0u) && (mag != 0u);  // sign set and nonzero -> excludes -0.0
    } else if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
        return ((u >> 31) == 0u) && (mag != 0u);  // sign clear and nonzero
    } else if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
        return ((u >> 31) == 0u) || (mag == 0u);  // sign clear or zero (incl ±0)
    } else {                                      // less_than_equal_zero
        return ((u >> 31) != 0u) || (mag == 0u);  // sign set or zero (incl ±0)
    }
}

/**
 * @brief Program the shared comparison-to-zero state once: the magnitude-mask const register and
 * the dest-increment addr mod.
 *
 * @c vConstIntPrgm0 holds the 0x7FFFFFFF magnitude mask so @ref _zero_comp_pred_ clears the sign
 * bit with a single @c sfpand (no per-iteration immediate build). @c ADDR_MOD_6 (dest.incr=2) lets
 * the body's SFPSTORE advance the dest counter, so @ref _calculate_zero_comp_ needs no dst_reg++.
 * Both are HW state shared across every COMP_MODE/FMT instantiation, so programming them once here
 * (rather than per @ref _calculate_zero_comp_ call) keeps them out of the per-tile loop body.
 *
 * @note Call once after @ref _llk_math_eltwise_sfpu_init_ and before @ref _calculate_zero_comp_.
 */
inline void _init_zero_comp_() {
    sfpi::vConstIntPrgm0 = 0x7FFFFFFF;
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
}

/**
 * @brief Element-wise comparison-to-zero over a tile, written as 1/0 booleans.
 *
 * Defaults every result lane to 0 and writes 1 into the lanes satisfying COMP_MODE (see
 * @ref _zero_comp_pred_), in FMT's native encoding (see @ref zero_comp_traits). The body's SFPSTORE
 * rides @c ADDR_MOD_6 (dest.incr=2, programmed in @ref _init_zero_comp_) to advance the dest
 * counter, so the loop needs no dst_reg++.
 *
 * @tparam APPROXIMATION_MODE: Unused (no approx path); retained for dispatcher signature symmetry.
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32/Int16/Int8/UInt16/UInt8, or any IEEE float width
 *         (Float32/Float16/Float16_b share the width-agnostic float path — the SFPLOAD/SFPSTORE
 *         resolve the actual width from the dest format config). Anything else is a compile error.
 * @tparam COMP_MODE: Comparison-to-zero mode.
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32×16 face).
 * @note Requires @ref _init_zero_comp_ to have programmed @c vConstIntPrgm0 and @c ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, DataFormat FMT, SfpuType COMP_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_zero_comp_() {
    constexpr bool is_int_fmt = FMT == DataFormat::Int32 || FMT == DataFormat::Int16 || FMT == DataFormat::Int8 ||
                                FMT == DataFormat::UInt16 || FMT == DataFormat::UInt8;
    constexpr bool is_float_fmt =
        FMT == DataFormat::Float32 || FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b;
    static_assert(
        is_int_fmt || is_float_fmt,
        "_calculate_zero_comp_: unsupported FMT (expected an integer format or an IEEE float width)");

    using traits = zero_comp_traits<FMT>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vUInt bits = traits::load();

        typename traits::result_t result = traits::zero();
        v_if(_zero_comp_pred_<COMP_MODE>(bits)) { result = traits::one(); }
        v_endif;

        traits::store(result);
    }
}

}  // namespace sfpu
}  // namespace ckernel
