# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the all-gather and reduce-scatter CCLs in the chunked-prefill MLA forward
(models/demos/deepseek_v3_d_p/tt/mla/mla.py).

The MLA forward runs four CCLs, all along the TP axis (tp_factor=4):

    | id          | op             | dim | per-device in -> out          | mla.py |
    |-------------|----------------|-----|-------------------------------|--------|
    | q_a_proj_rs | reduce_scatter |  3  | [1,1,S,1536] -> [1,1,S,384]   | :718   |
    | q_ag        | all_gather     |  3  | [1,1,S,384]  -> [1,1,S,1536]  | :729   |
    | kv_ag       | all_gather     |  1  | [1,1,S,576]  -> [1,4,S,576]   | :797   |
    | o_proj_rs   | reduce_scatter |  3  | [1,1,S,7168] -> [1,1,S,1792]  | :912   |

S is the PER-DEVICE sequence length (seq_len_local). On the 8x4 Galaxy with a 5120-token
chunk (sp=8), each chip sees chunk_size_global / sp = 5120 / 8 = 640 tokens. These tests fix
S=640 so that running on a 2x4 mesh reproduces exactly the per-device CCL shapes seen on 8x4
(the CCLs are on the TP axis, which is 4 on both meshes; only SP differs, and SP just adds
independent replicas of the same per-device op).

The op call signatures mirror mla.py exactly: DRAM interleaved, the model's semaphore layout
(all_gather -> [2 sems] + barrier; reduce_scatter -> [3 sems] + barrier, persistent_output_buffers
=None), and ccl_num_links = 2 on Blackhole else 1.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config as Cfg
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# MLA feature dims (kimi_k2_6 == deepseek_v3 for all of these; only num_heads differs and it
# does not enter the CCL widths).
Q_LORA_RANK = Cfg.Q_LORA_RANK  # 1536
KVPE_DIM = Cfg.KV_LORA_RANK + Cfg.QK_ROPE_HEAD_DIM  # 512 + 64 = 576
HIDDEN_SIZE = Cfg.EMB_SIZE  # 7168
TP_FACTOR = 4  # production TP; tp_axis size on both 8x4 and 2x4
SEQ_LOCAL = 640  # per-device seq: chunk_size_global(5120) / sp(8) on the 8x4 Galaxy

# (id, kind, dim, per-device-input feature size on the gathered/scattered dim)
#   rs: feat is the FULL width each device holds; output is feat // tp.
#   ag dim=3: feat is the PER-DEVICE width; output is feat * tp.
#   ag dim=1: feat is the channel width (576); the gather is over the head dim (dim1, 1 -> tp).
MLA_CCL_OPS = [
    ("q_a_proj_rs", "rs", 3, Q_LORA_RANK),  # mla.py:718  all-reduce part 1
    ("q_ag", "ag", 3, Q_LORA_RANK // TP_FACTOR),  # mla.py:729  all-reduce part 2
    ("kv_ag", "ag", 1, KVPE_DIM),  # mla.py:797  gather-then-reduce
    ("o_proj_rs", "rs", 3, HIDDEN_SIZE),  # mla.py:912  output reduce-scatter
]


def _make_global_semaphores(mesh_device, cores, n):
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]


def _run_mla_ccl(mesh_device, kind, dim, feat, topology, pcc_threshold=0.999):
    tp_axis = 1
    sp, tp = list(mesh_device.shape)
    # Production TP is 4; tp>TP_FACTOR (e.g. 1x8) is allowed for CCL connectivity/perf experiments.
    # The golden uses the live `tp`, so the op stays self-consistent as long as the scattered dim is
    # tile-aligned after the split.
    if kind == "rs":
        assert (feat // tp) % 32 == 0, f"rs scatter width {feat}//{tp} must be tile-aligned"
    num_links = 2 if is_blackhole() else 1

    # --- sub-device + semaphores (same scaffolding the model's TT_CCL uses) ---
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    barrier_sem = ttnn.create_global_semaphore(mesh_device, ccl_crs, 0)

    # --- inputs: each of the tp devices holds an independent [1,1,S,feat] tensor, laid out on
    #     dim1 so a tp-shard of dim1 hands each device its own slice. SP replicates (independent
    #     copies of the same per-device op). ---
    torch_in = torch.randn(1, tp, SEQ_LOCAL, feat, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, 1]),
    )

    try:
        if kind == "ag":
            mdgs = _make_global_semaphores(mesh_device, ccl_crs, 2)
            tt_out = ttnn.experimental.all_gather_async(
                tt_in,
                dim=dim,
                multi_device_global_semaphore=mdgs,
                barrier_semaphore=barrier_sem,
                num_links=num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                cluster_axis=tp_axis,
            )
        else:  # "rs"
            mdgs = _make_global_semaphores(mesh_device, ccl_crs, 3)
            tt_out = ttnn.experimental.reduce_scatter_minimal_async(
                tt_in,
                persistent_output_buffers=None,
                dim=dim,
                multi_device_global_semaphore=mdgs,
                barrier_semaphore=barrier_sem,
                num_links=num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                cluster_axis=tp_axis,
            )
        ttnn.synchronize_device(mesh_device)
        out_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, dim)),
        )[
            0:1
        ]  # SP replicas are identical; keep the first
    finally:
        mesh_device.reset_sub_device_stall_group()

    # --- golden ---
    if kind == "rs":
        # reduce over the tp devices (dim1), then the concat-over-tp readback on `dim` reassembles
        # the full scattered sum.
        golden = torch_in.to(torch.float32).sum(dim=1, keepdim=True)
        out_torch = out_torch.to(torch.float32)
    elif dim == 3:
        # all_gather along width: every device ends up with the full concat; tp copies are identical,
        # so take the first one back out of the concat readback.
        golden = torch.cat([torch_in[:, d : d + 1] for d in range(tp)], dim=3)
        out_torch = out_torch[:, :, :, : feat * tp]
    else:  # ag dim == 1 (head gather): output is the tp slices stacked on dim1
        golden = torch_in
        out_torch = out_torch[:, :tp]

    logger.info(f"{kind} dim={dim} feat={feat}: in {list(torch_in.shape)} -> out {list(out_torch.shape)}")
    passed, msg = comp_pcc(out_torch, golden, pcc_threshold)
    logger.info(f"PCC: {msg}")
    assert passed, f"{kind} dim={dim} feat={feat} FAILED: {msg}"


@pytest.mark.parametrize("ccl_id, kind, dim, feat", MLA_CCL_OPS, ids=[c[0] for c in MLA_CCL_OPS])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            },
            ttnn.Topology.Linear,
        ),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            },
            ttnn.Topology.Ring,
        ),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["line", "ring", "fabric2d"],
)
@pytest.mark.parametrize(
    "mesh_device", [(1, 4), (1, 8), (2, 4), (8, 4)], ids=["1x4", "1x8", "2x4", "8x4"], indirect=True
)
@pytest.mark.timeout(0)
def test_mla_ccl(mesh_device, device_params, topology, ccl_id, kind, dim, feat):
    """Each chunked-MLA all-gather / reduce-scatter at its per-device shape (seq_local=640),
    reproducing the 8x4 per-device load on a 2x4 mesh."""
    _run_mla_ccl(mesh_device, kind, dim, feat, topology)
