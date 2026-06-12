// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side consumer for the single-DFB matrix
// sweep (DM → DFB → TRISC case).
//
// This kernel adds Tensix-as-consumer coverage that the DM-side consumer
// (dfb_consumer_2_0.cpp) doesn't reach. The DM producer fills the DFB L1 ring
// (NoC read from DRAM); this kernel drains it on the Tensix UNPACK/MATH path.
//
// IMPORTANT (SW/HW spec): a consumer must NOT issue wait_front (wait_tiles) then
// pop_front (pop_tiles) with NOTHING between them — the unpacker must actually
// READ the tile between the wait and the pop, or the buffer-descriptor / read
// pointer goes inconsistent and the unpacker traps. So this kernel does a real
// copy_tile (= llk_unpack_A unpack + math datacopy into the DST register) between
// wait_front and pop_front. It is drain-only (no dfb::out, no pack_tile): the
// copied tile is discarded — the point is to exercise the consumer UNPACK path
// HW-legally, not to produce output. unary_op_init_common runs the unpack/math
// HW configure (which programs the now-fixed buffer descriptor); we pass the
// input DFB id for both operands since there is no separate output DFB.
//
// Per-tile drain (wait_front(1)/copy/pop_front(1)) is correct for STRIDED, ALL,
// AND BLOCKED consumers: a Tensix consumer has no NoC burst to batch on
// block_size, so it simply unpacks one tile at a time over the same total count.
//
// Bindings (set by host KernelSpec):
//   dfb::in — CONSUMER (host binds the same DFB the DM producer pushes to).

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);

    DataflowBuffer dfb(dfb::in);

    // Configure the unpack/math pipeline against the input DFB. This issues the
    // unpack HW configure that programs the buffer descriptor (x/y/z dims) the
    // unpacker reads with — required before any copy_tile. No real output DFB, so
    // pass the input id for both operands (pack init is harmless; we never pack).
    unary_op_init_common(dfb.get_id(), dfb.get_id());

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; ++tile_id) {
        acquire_dst();
        dfb.wait_front(1);
        // The required "copy between wait and pop": unpack the front tile into DST.
        copy_tile(dfb.get_id(), 0, 0);
        dfb.pop_front(1);
        release_dst();
    }
    dfb.finish();
}
