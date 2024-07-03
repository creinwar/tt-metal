// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.hpp"
#include "ttnn/cpp/ttnn/validation.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {

using MatmulMultiCoreReuseProgramConfig = tt::operations::primary::MatmulMultiCoreReuseProgramConfig;
using MatmulMultiCoreReuseMultiCastProgramConfig = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig;
using MatmulMultiCoreReuseMultiCast1DProgramConfig =
    tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig;
// MatmulProgramConfig is the Union of the above types
using MatmulProgramConfig = tt::operations::primary::MatmulProgramConfig;

namespace operations {
namespace matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& shape) {
    auto is_batched = false;
    for (auto i = 0; i < shape.rank() - 2; ++i) {
        if (shape[i] > 1) {
            is_batched = true;
            break;
        }
    }
    return is_batched;
}

}  // namespace detail

const std::array<ttnn::TensorSchema, 3> input_tensor_schemas() {
    return {
        ttnn::TensorSchema{
            2, 4, {ttnn::float32, ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b}, {ttnn::TILE_LAYOUT}, true, false, true, false},
        ttnn::TensorSchema{
            2, 4, {ttnn::float32, ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b}, {ttnn::TILE_LAYOUT}, true, false, true, false},
        ttnn::TensorSchema{
            2, 4, {ttnn::float32, ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b}, {ttnn::TILE_LAYOUT}, true, false, true, true}};
}

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const std::string>& activation) {
    if (!activation.has_value()) {
	return std::nullopt;
    }
    return string_to_unary_with_param(activation.value());
}

ttnn::Tensor matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const MatmulProgramConfig> program_config,
    const ttnn::MemoryConfig& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid,
    const bool propagate_is_b_batched) {
    ttnn::validate_input_tensor("ttnn.matmul", input_tensor_a, input_tensor_schemas()[0]);
    ttnn::validate_input_tensor("ttnn.matmul", input_tensor_b, input_tensor_schemas()[1]);
    ttnn::validate_input_tensor("ttnn.matmul", bias, input_tensor_schemas()[2]);

    const auto input_tensor_a_shape = input_tensor_a.get_shape();
    const auto input_tensor_b_shape = input_tensor_b.get_shape();

    const auto width_a = input_tensor_a_shape[-1];
    const auto height_b = input_tensor_b_shape[-2];

    if (width_a != height_b) {
        TT_THROW("ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor");
    }

    auto input_b_is_batched = detail::is_input_batched(input_tensor_b_shape);
    bool batch_with_bias = input_b_is_batched && bias.has_value();
    TT_FATAL(!batch_with_bias, "Batched input not supported when bias exists (linear operation).");

    std::optional<CoreCoord> user_core_coord;
    const bool has_user_grid = core_grid.has_value();
    if (has_user_grid) {
	user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    const bool has_program_config = program_config.has_value();
    bool post_process_bias = false;
    if (bias.has_value()) {
        if (!has_program_config && !has_user_grid) {
	    post_process_bias = true;
	}
    }

    auto output_tensor = tt::operations::primary::matmul(
        input_tensor_a, input_tensor_b, post_process_bias ? std::nullopt : bias, program_config, memory_config, dtype, compute_kernel_config, false /*untilize_out*/, user_core_coord, get_fused_activation(activation), propagate_is_b_batched && input_b_is_batched);

    if (post_process_bias) {
        output_tensor = tt::operations::primary::bcast(
            output_tensor, bias.value(), tt::tt_metal::BcastOpMath::ADD, tt::tt_metal::BcastOpDim::H, memory_config);
    }

    if (activation.has_value() && !has_user_grid) {
        if (activation.value() == "relu") {
            output_tensor = ttnn::relu(output_tensor, memory_config);
        } else if (activation.value() == "gelu") {
            output_tensor = ttnn::gelu(output_tensor, false, memory_config);
        } else if (activation.value() == "silu") {
            output_tensor = ttnn::silu(output_tensor, memory_config);
        } else {
            TT_THROW("ttnn.matmul: Unsupported activation function");
        }
    }

    return output_tensor;
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
