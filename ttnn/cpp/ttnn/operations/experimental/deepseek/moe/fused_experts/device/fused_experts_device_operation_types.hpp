// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

// Non-tensor parameters of the fused routed-expert FFN.
struct operation_attributes_t {
    // SwiGLU intermediate size (I). gate_up weights are [H, 2I], down weights are [I, H].
    uint32_t intermediate_size{};

    // Clamp limit applied inside the SwiGLU activation: silu(clamp(gate, max=limit)) * clamp(up, -limit, limit).
    float swiglu_limit{};

    // Routing-weight column index for each weight pair (i.e. the original expert id of each "hit").
    // expert_ids[i] selects the column of `routing_weights` that scales the i-th expert's output.
    std::vector<uint32_t> expert_ids{};

    tt::tt_metal::MemoryConfig output_memory_config{};
};

// All tensors flowing in/out of the operation. This op is the concrete example of an op that takes
// an *array* of tensors: one gate_up / down weight tensor per selected ("hit") expert.
struct tensor_args_t {
    // Activations, [1, 1, T, H].
    const Tensor& input_tensor;

    // Per-token routing weights, [1, 1, T, E]. Columns indexed by operation_attributes_t::expert_ids.
    const Tensor& routing_weights;

    // One gate_up weight tensor per selected expert, each [H, 2I] (matmul-ready / transposed).
    std::vector<Tensor> gate_up_weights;

    // One down weight tensor per selected expert, each [I, H] (matmul-ready / transposed).
    std::vector<Tensor> down_weights;
};

using spec_return_value_t = TensorSpec;

using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts
