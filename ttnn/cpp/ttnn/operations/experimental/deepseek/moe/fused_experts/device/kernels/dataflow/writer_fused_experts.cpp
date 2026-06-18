// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// Writer for the fused routed-expert FFN.
//
// Compile-time args: { out_cb_index, <TensorAccessorArgs(output)> }
// Runtime args:      { output_addr, num_tiles, start_tile_id }
//
// TODO: pop the accumulated output tiles from the compute CB and write them to the
// output buffer (interleaved or sharded per the output memory config).
void kernel_main() {
    constexpr uint32_t out_cb_index = get_compile_time_arg_val(0);

    uint32_t arg_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(arg_idx++);

    (void)out_cb_index;
    (void)output_addr;
    (void)num_tiles;
    (void)start_tile_id;

    // TODO: implement tile writes.
}
