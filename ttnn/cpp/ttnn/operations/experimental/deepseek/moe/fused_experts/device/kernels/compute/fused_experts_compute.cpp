// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {

// Compute for the fused routed-expert FFN.
//
// Compile-time args: { in_cb_index, out_cb_index, num_experts, intermediate_size, swiglu_limit_bits }
// Runtime args:      { num_tiles }
//
// For each selected expert e the reference (host) computation is:
//   gate_up = matmul(x, gate_up_w[e])          // [T, 2I]
//   act     = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)   // [T, I]
//   down    = matmul(act, down_w[e])           // [T, H]
//   acc    += down * routing_weights[:, e]
//
// TODO: implement the fused matmul + SwiGLU + weighted-accumulate pipeline,
// reusing CBs across the expert loop.
void MAIN {
    constexpr uint32_t in_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t num_experts = get_compile_time_arg_val(2);
    constexpr uint32_t intermediate_size = get_compile_time_arg_val(3);
    constexpr uint32_t swiglu_limit_bits = get_compile_time_arg_val(4);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    (void)in_cb_index;
    (void)out_cb_index;
    (void)num_experts;
    (void)intermediate_size;
    (void)swiglu_limit_bits;
    (void)num_tiles;

    // TODO: implement compute.
}

}  // namespace NAMESPACE
