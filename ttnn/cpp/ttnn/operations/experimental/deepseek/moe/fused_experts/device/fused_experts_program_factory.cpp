// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include <bit>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels";
}  // namespace

ProgramDescriptor FusedExpertsDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& routing_weights = tensor_args.routing_weights;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* rw_buffer = routing_weights.buffer();
    auto* dst_buffer = output_tensor.buffer();

    const tt::DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t in_tile_size = tile_size(in_df);
    const uint32_t out_tile_size = tile_size(out_df);

    const uint32_t num_output_tiles = output_tensor.physical_volume() / constants::TILE_HW;

    IDevice* device = input_tensor.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_group_1, tiles_per_core_group_2] =
        split_work_to_cores(grid, num_output_tiles);

    ProgramDescriptor desc;

    // ------------------------------------------------------------------
    // Circular buffers.
    //
    // NOTE: Tensix has a hard limit of 32 circular buffers per core. The fused
    // kernel must reuse a small fixed set of CBs and loop over the expert
    // weights (whose addresses arrive as runtime args) rather than allocating
    // one CB per expert. See operation_attributes_t::expert_ids.
    // ------------------------------------------------------------------
    constexpr uint32_t in_cb_index = CBIndex::c_0;
    constexpr uint32_t num_in_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_in_tiles * in_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in_cb_index,
            .data_format = in_df,
            .page_size = in_tile_size,
        }}},
    });

    constexpr uint32_t out_cb_index = CBIndex::c_16;
    constexpr uint32_t num_out_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_out_tiles * out_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_df,
            .page_size = out_tile_size,
        }}},
    });

    // TODO: add CBs for gate_up matmul output (2I), the SwiGLU intermediate (I),
    // the down matmul output (H), and the per-expert weights. Reuse them across
    // experts to stay within the 32-CB / 341-runtime-arg budgets.

    // ------------------------------------------------------------------
    // Compile-time args.
    // ------------------------------------------------------------------
    const uint32_t num_experts = static_cast<uint32_t>(operation_attributes.expert_ids.size());

    std::vector<uint32_t> reader_ct_args = {in_cb_index, num_experts};
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*rw_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {out_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    std::vector<uint32_t> compute_ct_args = {
        in_cb_index,
        out_cb_index,
        num_experts,
        operation_attributes.intermediate_size,
        std::bit_cast<uint32_t>(operation_attributes.swiglu_limit),
    };

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kKernelDir) + "/dataflow/reader_fused_experts.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_ct_args;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kKernelDir) + "/dataflow/writer_fused_experts.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_ct_args;
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "/compute/fused_experts_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_ct_args;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };

    // ------------------------------------------------------------------
    // Runtime args.
    //
    // Each expert's gate_up / down weight buffer address is passed as a runtime
    // arg here. With E experts that is ~2*E address words plus per-tile bookkeeping,
    // which must stay under the Tensix limit of 341 unique+common runtime args
    // (tt::tt_metal::max_runtime_args). For large expert counts, pass the weight
    // addresses via an L1/DRAM address table instead.
    // ------------------------------------------------------------------
    for (uint32_t i = 0, tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            tiles_per_core = tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            tiles_per_core = tiles_per_core_group_2;
        } else {
            TT_THROW("fused_experts: core {} not in any work group", core.str());
        }

        KernelDescriptor::CoreRuntimeArgs reader_rt = {
            src_buffer->address(), rw_buffer->address(), tiles_per_core, tiles_written};
        for (const auto& w : tensor_args.gate_up_weights) {
            reader_rt.push_back(w.buffer()->address());
        }
        for (const auto& w : tensor_args.down_weights) {
            reader_rt.push_back(w.buffer()->address());
        }
        reader_desc.runtime_args.emplace_back(core, std::move(reader_rt));

        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{dst_buffer->address(), tiles_per_core, tiles_written});

        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{tiles_per_core});

        tiles_written += tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts
