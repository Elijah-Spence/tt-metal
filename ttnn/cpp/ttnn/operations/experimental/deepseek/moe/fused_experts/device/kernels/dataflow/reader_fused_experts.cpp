// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// Reader for the fused routed-expert FFN.
//
// Compile-time args: { in_cb_index, num_experts, <TensorAccessorArgs(input)>, <TensorAccessorArgs(routing_weights)> }
// Runtime args:      { input_addr, routing_weights_addr, num_tiles, start_tile_id,
//                      gate_up_addr[0..num_experts), down_addr[0..num_experts) }
//
// TODO: stream the activation tiles, the routing-weight column for each expert,
// and each expert's gate_up / down weight tiles into the compute kernel's CBs,
// looping over experts so that a fixed set of CBs is reused.
void kernel_main() {
    // Compile-time args.
    constexpr uint32_t in_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t num_experts = get_compile_time_arg_val(1);

    // Runtime args.
    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t routing_weights_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(arg_idx++);

    // Per-expert weight base addresses follow.
    // for (uint32_t e = 0; e < num_experts; ++e) gate_up_addr[e] = get_arg_val<uint32_t>(arg_idx++);
    // for (uint32_t e = 0; e < num_experts; ++e) down_addr[e]    = get_arg_val<uint32_t>(arg_idx++);

    (void)in_cb_index;
    (void)num_experts;
    (void)input_addr;
    (void)routing_weights_addr;
    (void)num_tiles;
    (void)start_tile_id;
    (void)arg_idx;

    // TODO: implement tile reads / CB pushes.
}
