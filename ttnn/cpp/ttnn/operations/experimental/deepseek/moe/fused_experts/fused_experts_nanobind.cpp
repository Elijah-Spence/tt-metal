// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/moe/fused_experts/fused_experts.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail {

void bind_fused_experts(nb::module_& mod) {
    ttnn::bind_function<"fused_experts", "ttnn.experimental.deepseek.moe.">(
        mod,
        R"doc(
        Experimental fused routed-expert FFN for DeepSeek V4-Flash.

        Fuses the per-expert matmul -> SwiGLU -> matmul -> weighted-accumulate loop
        into a single device operation.

        Args:
            input_tensor: Activations, [1, 1, T, H].
            routing_weights: Per-token routing weights, [1, 1, T, E].
            gate_up_weights: List of [H, 2I] weight tensors, one per selected expert.
            down_weights: List of [I, H] weight tensors, one per selected expert.
            expert_ids: Routing-weight column index for each weight pair.
            intermediate_size: SwiGLU intermediate size I.
            swiglu_limit: Clamp limit used by the SwiGLU activation.
            memory_config: Optional output memory config.
        )doc",
        &ttnn::experimental::deepseek::moe::fused_experts,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("routing_weights"),
        nb::arg("gate_up_weights"),
        nb::arg("down_weights"),
        nb::arg("expert_ids"),
        nb::arg("intermediate_size"),
        nb::arg("swiglu_limit"),
        nb::arg("memory_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_fused_experts(::nanobind::module_& mod) { fused_experts::detail::bind_fused_experts(mod); }

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
