// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Per-DataFormat sfpi vector type and 0/1 encoding for the comparison-to-zero result.
 *
 * @c load reads the element and reinterprets the bits as a @c vUInt for the shared predicate (see
 * @ref _zero_comp_pred_); @c store writes @c zero/@c one back in FMT's native encoding. The 16/32-bit
 * and float formats ride the sfpi @c dst_reg[0] container that carries FMT's width (vInt→INT32,
 * vSMag16→INT16, vUInt16→UINT16, vFloat→implied float). The 8-bit formats have no such container
 * (sfpi exposes no 8-bit dst_reg conversion), so Int8/UInt8 issue an explicit-mode SFPLOAD/SFPSTORE
 * (sfpmem INT8 / UINT8) bridged through @c vUInt(sfpu_t) / @c vInt::get().
 *
 * @tparam FMT: SFPU DataFormat (sfpu_math). Int32 / Int16 / Int8 signed, UInt16 / UInt8 unsigned, or
 *         Float32 for every float width (the dispatcher routes Float16/Float16_b through the Float32
 *         path).
 */
template <DataFormat FMT>
struct zero_comp_traits;

template <>
struct zero_comp_traits<DataFormat::Int32> {
    using result_t = sfpi::vInt;
    static inline __attribute__((always_inline)) sfpi::vUInt load() { return sfpi::dst_reg[0]; }
    static inline __attribute__((always_inline)) result_t zero() { return 0; }
    static inline __attribute__((always_inline)) result_t one() { return 1; }
    static inline __attribute__((always_inline)) void store(result_t r) {
        sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = r;
    }
};

template <>
struct zero_comp_traits<DataFormat::Int16> {
    using result_t = sfpi::vInt;
    static inline __attribute__((always_inline)) sfpi::vUInt load() {
        sfpi::vSMag16 s = sfpi::dst_reg[0];
        return sfpi::reinterpret<sfpi::vUInt>(s);
    }
    static inline __attribute__((always_inline)) result_t zero() { return 0; }
    static inline __attribute__((always_inline)) result_t one() { return 1; }
    static inline __attribute__((always_inline)) void store(result_t r) {
        sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = sfpi::reinterpret<sfpi::vSMag16>(r);
    }
};

template <>
struct zero_comp_traits<DataFormat::UInt16> {
    using result_t = sfpi::vInt;
    static inline __attribute__((always_inline)) sfpi::vUInt load() {
        sfpi::vUInt16 u = sfpi::dst_reg[0];
        return sfpi::reinterpret<sfpi::vUInt>(u);
    }
    static inline __attribute__((always_inline)) result_t zero() { return 0; }
    static inline __attribute__((always_inline)) result_t one() { return 1; }
    static inline __attribute__((always_inline)) void store(result_t r) {
        sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = sfpi::reinterpret<sfpi::vUInt16>(r);
    }
};

// sfpi's dst_reg has no 8-bit conversion operator, so Int8/UInt8 can't use the dst_reg[0] container
// path; they issue raw SFPLOAD/SFPSTORE with an explicit sfpmem mode (INT8=SMAG8 / UInt8), bridged
// through @c sfpi::vUInt(sfpu_t) / @c vInt::get(). SMAG8 loads the sign in bit 31, so the shared
// predicate's @c u>>31 / @c u&0x7FFFFFFF read the same as the 16/32-bit signed paths.
template <>
struct zero_comp_traits<DataFormat::Int8> {
    using result_t = sfpi::vInt;
    static inline __attribute__((always_inline)) sfpi::vUInt load() {
        return sfpi::vUInt(__builtin_rvtt_sfpload(0, ckernel::p_sfpu::sfpmem::INT8, sfpi::SFPLOAD_ADDR_MODE_NOINC));
    }
    static inline __attribute__((always_inline)) result_t zero() { return 0; }
    static inline __attribute__((always_inline)) result_t one() { return 1; }
    static inline __attribute__((always_inline)) void store(result_t r) {
        __builtin_rvtt_sfpstore(r.get(), 0, ckernel::p_sfpu::sfpmem::INT8, ckernel::ADDR_MOD_6);
    }
};

template <>
struct zero_comp_traits<DataFormat::UInt8> {
    using result_t = sfpi::vInt;
    static inline __attribute__((always_inline)) sfpi::vUInt load() {
        return sfpi::vUInt(__builtin_rvtt_sfpload(0, ckernel::p_sfpu::sfpmem::UINT8, sfpi::SFPLOAD_ADDR_MODE_NOINC));
    }
    static inline __attribute__((always_inline)) result_t zero() { return 0; }
    static inline __attribute__((always_inline)) result_t one() { return 1; }
    static inline __attribute__((always_inline)) void store(result_t r) {
        __builtin_rvtt_sfpstore(r.get(), 0, ckernel::p_sfpu::sfpmem::UINT8, ckernel::ADDR_MOD_6);
    }
};

template <>
struct zero_comp_traits<DataFormat::Float32> {
    using result_t = sfpi::vFloat;
    static inline __attribute__((always_inline)) sfpi::vUInt load() {
        sfpi::vFloat f = sfpi::dst_reg[0];
        return sfpi::reinterpret<sfpi::vUInt>(f);
    }
    static inline __attribute__((always_inline)) result_t zero() { return 0.0f; }
    static inline __attribute__((always_inline)) result_t one() { return 1.0f; }
    static inline __attribute__((always_inline)) void store(result_t r) {
        sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = r;
    }
};

/**
 * @brief Lanes of @c u satisfying COMP_MODE against zero, read off the raw bit pattern.
 *
 * Each mode is built from two bit tests that read identically across every format (Quasar loads
 * signed integers as sign-magnitude and floats as IEEE, both with the sign in bit 31): the sign bit
 * @c u>>31 and the magnitude-zero test @c (u&0x7FFFFFFF)==0. Masking the sign bit makes the
 * magnitude test hold for both +0 (0x00000000) and -0 (0x80000000), so -0.0 / sign-magnitude -0
 * count as zero — matching IEEE, where -0.0 == 0. "Negative" is then @c sign && magnitude-nonzero,
 * which excludes -0.0. This matches the golden: eqz/gez/lez accept -0.0, while ltz/nez reject it.
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
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32/Int16/Int8/UInt16/UInt8 use their own traits; any
 *         IEEE float width (Float32/Float16/Float16_b) maps to the width-agnostic Float32 traits
 *         (the SFPLOAD/SFPSTORE resolve the actual width from the dest format config). Anything else
 *         is a compile error.
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

    // All float widths share the width-agnostic Float32 traits; integer formats use their own.
    constexpr DataFormat TRAITS_FMT = is_int_fmt ? FMT : DataFormat::Float32;
    using traits = zero_comp_traits<TRAITS_FMT>;

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
