// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <unordered_set>

#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_eager/tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_eager/tt_dnn/op_library/sliding_window_op_infra/halo_op.hpp"
#include "tt_eager/tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/cpp/ttnn/operations/matmul.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {
namespace conv2d {

struct Conv2dConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    DataType dtype = DataType::BFLOAT16;
    DataType weights_dtype = DataType::BFLOAT16;
    bool math_approx_mode_enabled = true;
    bool fp32_dest_acc_enabled = false;
    bool packer_l1_accum_enabled = false;
    string activation = "";
    uint32_t input_channels_alignment = 32;
    bool deallocate_activation = false;
    bool reallocate_halo_output = false;
    uint32_t act_block_h_override = 0;
    bool reshard_if_not_optimal = false;    // if true, override_sharding_config should not be set to true
    bool override_sharding_config = false;  // if true, reshard_if_not_optimal should not be set to true
    bool height_sharding = true;            // used only if override_sharding_config is true
    CoreRangeSet core_grid = {{}};          // used only if override_sharding_config is true
    bool transpose_shards = true;  // used only if override_sharding_config is true and if height sharding is false
    Layout output_layout = Layout::TILE;
    static constexpr auto attribute_names = std::forward_as_tuple(
        "math_fidelity",
        "dtype",
        "weights_dtype",
        "math_approx_mode_enabled",
        "fp32_dest_acc_enabled",
        "packer_l1_accum_enabled",
        "activation",
        "input_channels_alignment",
        "deallocate_activation",
        "reallocate_halo_output",
        "act_block_h_override",
        "reshard_if_not_optimal",
        "override_sharding_config",
        "height_sharding",
        "core_grid",
        "transpose_shards",
        "output_layout");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->math_fidelity,
            this->dtype,
            this->weights_dtype,
            this->math_approx_mode_enabled,
            this->fp32_dest_acc_enabled,
            this->packer_l1_accum_enabled,
            this->activation,
            this->input_channels_alignment,
            this->deallocate_activation,
            this->reallocate_halo_output,
            this->act_block_h_override,
            this->reshard_if_not_optimal,
            this->override_sharding_config,
            this->height_sharding,
            this->core_grid,
            this->transpose_shards,
            this->output_layout);
    }
};

extern const std::array<ttnn::TensorSchema, 3> input_schemas;

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor);

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num, uint32_t start_divisor);

uint32_t find_closest_common_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor);

ParallelConfig determine_parallel_config(
    bool height_sharding,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    Device& device,
    ShardOrientation block_shard_orientation,
    bool is_out_tiled = true);

uint32_t get_num_cores_nhw_from_parallel_config(const ParallelConfig& pconfig);

uint32_t get_num_cores_channels_from_parallel_config(const ParallelConfig& pconfig);

MemoryConfig create_sharded_memory_config_from_parallel_config(
    const Shape& tensor_shape, ParallelConfig& parallel_config, uint32_t tile_size);

tt::tt_metal::OptimizedConvParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config, uint32_t num_cores_nhw);

std::pair<uint32_t, uint32_t> determine_largest_subblock_size(
    uint32_t block_height, uint32_t block_width, bool fp32_accum);

tt::tt_metal::OptimizedConvBlockConfig determine_per_core_conv_block_config(
    const ParallelConfig& parallel_config,
    const tt::tt_metal::OptimizedConvParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t act_block_h_override,
    uint32_t window_w,
    bool fp32_accum,
    bool use_shallow_conv_variant);

std::tuple<ttnn::Tensor, ParallelConfig, bool> shard_or_reshard_tensor_if_required(
    Device& device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels);

void validate_weight_and_bias_tensors(
    const ttnn::Tensor& weight_tensor, std::optional<const ttnn::Tensor>& bias_tensor);

std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    Device& device);
std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    ttnn::Device& device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
    std::optional<const Conv2dConfig> conv_config_ = std::nullopt);

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn

template <>
struct fmt::formatter<ttnn::operations::conv2d::Conv2dConfig> : formatter<string_view> {
    // constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
    auto format(const ttnn::operations::conv2d::Conv2dConfig& t, fmt::format_context& ctx);
};
