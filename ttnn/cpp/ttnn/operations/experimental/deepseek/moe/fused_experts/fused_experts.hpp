// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek::moe {

// Fused routed-expert FFN for DeepSeek V4-Flash decode.
//
// Replaces the per-expert host loop
//   for e in hit:
//       gate_up = matmul(x, gate_up_w[e]); act = swiglu(gate_up, intermediate, limit)
//       down    = matmul(act, down_w[e]);  acc += down * routing_weights[:, e]
// with a single device operation.
//
// Args:
//   input_tensor:     activations, [1, 1, T, H].
//   routing_weights:  per-token routing weights, [1, 1, T, E].
//   gate_up_weights:  one [H, 2I] weight tensor per selected expert.
//   down_weights:     one [I, H] weight tensor per selected expert.
//   expert_ids:       routing-weight column for each weight pair (the original expert id).
//   intermediate_size: SwiGLU intermediate size I.
//   swiglu_limit:     clamp limit used by the SwiGLU activation.
//   memory_config:    optional output memory config (defaults to the input's).
//
// Returns the accumulated output, [1, 1, T, H].
Tensor fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    const std::vector<uint32_t>& expert_ids,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek::moe
