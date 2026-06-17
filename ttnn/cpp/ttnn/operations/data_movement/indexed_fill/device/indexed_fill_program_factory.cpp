// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor IndexedFillProgramFactory::create_descriptor(
    const IndexedFillParams& /*operation_attributes*/, const IndexedFillInputs& tensor_args, Tensor& output) {
    const auto& batch_ids = tensor_args.batch_id;
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;

    IDevice* device = input_a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto set_of_core_ranges = num_cores_to_corerangeset(num_cores_x * num_cores_y, compute_with_storage_grid_size);
    CoreRangeSet all_cores(set_of_core_ranges);

    uint32_t B = input_a.padded_shape()[0];
    uint32_t b = input_b.padded_shape()[0];

    TT_ASSERT(batch_ids.padded_shape()[-1] == b);

    // parallelize across batch
    constexpr uint32_t cb_index = 0;
    constexpr uint32_t batch_cb_index = 1;

    // Reader kernel selects its data path via compile-time arg 3 (see indexed_fill_reader.cpp
    // header). This factory drives the interleaved one-batch-per-core layout, which maps to the
    // reader's generic path. The mode arg MUST be emitted explicitly: if omitted, the reader reads
    // input_a's TensorAccessorArgs config word (== IsDram for DRAM tensors) as the mode and wrongly
    // enters the shard-local path, reading runtime args this factory never sets -> device hang.
    constexpr uint32_t MODE_GENERIC = 0;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_a.dtype());

    uint32_t page_size = input_a.padded_shape()[-1] * input_a.element_size();
    uint32_t rounded_page_size = round_up_to_mul32(page_size);
    uint32_t batch_page_size = round_up_to_mul32(b * sizeof(uint32_t));

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * rounded_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = cb_data_format,
            .page_size = rounded_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * batch_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(batch_cb_index),
            .data_format = cb_data_format,
            .page_size = batch_page_size,
        }}},
    });

    Buffer* batch_ids_buffer = batch_ids.buffer();
    Buffer* input_a_buffer = input_a.buffer();
    Buffer* input_b_buffer = input_b.buffer();
    Buffer* output_buffer = output.buffer();

    // Reader compile-time args + tensor accessor args for batch_ids, input_a, input_b.
    // page_size is arg[2]; the mode selector is arg[3], directly before the appended accessor args
    // (the reader reads accessor args starting at index 4 via TensorAccessorArgs<4>).
    std::vector<uint32_t> reader_compile_time_args = {cb_index, batch_cb_index, page_size, MODE_GENERIC};
    TensorAccessorArgs(*input_a_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*input_b_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*batch_ids_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/kernels/dataflow/indexed_fill_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {cb_index, page_size};
    TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    auto cores = grid_to_cores(num_cores_x * num_cores_y, num_cores_x, num_cores_y, false);

    uint32_t batch_size_in_sticks = input_a.padded_shape()[1] * input_a.padded_shape()[2];

    // Generic-path geometry for dim=0, one batch (slice) per core:
    //   inner_count    = pages per batch                 (== batch_size_in_sticks)
    //   outer_count    = dims before dim 0               (== 1)
    //   outer_stride_a = B * inner_count                 (input_a page stride per outer step)
    //   outer_stride_b = b * inner_count                 (input_b page stride per outer step)
    // With outer_count == 1 the strides do not affect addressing, but are passed for completeness.
    constexpr uint32_t outer_count = 1;
    const uint32_t outer_stride_a = B * batch_size_in_sticks;
    const uint32_t outer_stride_b = b * batch_size_in_sticks;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        if (i < B) {
            // Active core: real work + Buffer* bindings for fast cache-hit patching.
            // Reader args: arg[4]=inner_count, arg[5]=slice_start, arg[6]=outer_count,
            //              arg[7]=outer_stride_a, arg[8]=outer_stride_b, arg[9]=num_slices.
            reader_desc.emplace_runtime_args(
                core,
                {batch_ids_buffer,
                 b,
                 input_a_buffer,
                 input_b_buffer,
                 batch_size_in_sticks,
                 i,
                 outer_count,
                 outer_stride_a,
                 outer_stride_b,
                 1u});
            writer_desc.emplace_runtime_args(
                core, {output_buffer, page_size, batch_size_in_sticks, i * batch_size_in_sticks});
        } else {
            // Idle core: short-circuits because num_slices (arg[9]) is 0.
            // Pass plain 0u for buffer slots so we don't register a BufferBinding here —
            // those bindings would force the framework to do a GetRuntimeArgs lookup on
            // every cache hit for cores that never read/write the buffer.  The kernel
            // ignores the address when its work count is 0.
            reader_desc.emplace_runtime_args(core, {0u, 0u, 0u, 0u, 0u, i, 0u, 0u, 0u, 0u});
            writer_desc.emplace_runtime_args(core, {0u, page_size, 0u, 0u});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
