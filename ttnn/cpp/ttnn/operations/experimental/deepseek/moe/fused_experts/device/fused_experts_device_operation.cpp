// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

FusedExpertsDeviceOperation::program_factory_t FusedExpertsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MultiCore{};
}

void FusedExpertsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    const auto& rw = tensor_args.routing_weights;

    TT_FATAL(x.storage_type() == StorageType::DEVICE, "fused_experts: input_tensor must be on device");
    TT_FATAL(rw.storage_type() == StorageType::DEVICE, "fused_experts: routing_weights must be on device");

    TT_FATAL(
        tensor_args.gate_up_weights.size() == tensor_args.down_weights.size(),
        "fused_experts: gate_up_weights ({}) and down_weights ({}) must have the same length",
        tensor_args.gate_up_weights.size(),
        tensor_args.down_weights.size());

    TT_FATAL(
        tensor_args.gate_up_weights.size() == attributes.expert_ids.size(),
        "fused_experts: weight count ({}) must match expert_ids count ({})",
        tensor_args.gate_up_weights.size(),
        attributes.expert_ids.size());

    // TODO: validate per-expert weight shapes against H / 2I / I, dtypes, and tile alignment.
}

void FusedExpertsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

FusedExpertsDeviceOperation::spec_return_value_t FusedExpertsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    // Output has the same shape/layout as the activations: [1, 1, T, H].
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), attributes.output_memory_config));
}

FusedExpertsDeviceOperation::tensor_return_value_t FusedExpertsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

std::tuple<FusedExpertsDeviceOperation::operation_attributes_t, FusedExpertsDeviceOperation::tensor_args_t>
FusedExpertsDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    const std::vector<uint32_t>& expert_ids,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config) {
    operation_attributes_t attributes{
        .intermediate_size = intermediate_size,
        .swiglu_limit = swiglu_limit,
        .expert_ids = expert_ids,
        .output_memory_config = memory_config.value_or(input_tensor.memory_config()),
    };
    tensor_args_t tensor_args{
        .input_tensor = input_tensor,
        .routing_weights = routing_weights,
        .gate_up_weights = gate_up_weights,
        .down_weights = down_weights,
    };
    return {std::move(attributes), std::move(tensor_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {
ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation::tensor_return_value_t
fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    const std::vector<uint32_t>& expert_ids,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation;
    auto [operation_attributes, tensor_args] = OperationType::invoke(
        input_tensor,
        routing_weights,
        gate_up_weights,
        down_weights,
        expert_ids,
        intermediate_size,
        swiglu_limit,
        memory_config);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
