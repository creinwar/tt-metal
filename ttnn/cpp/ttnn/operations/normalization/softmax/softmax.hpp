// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
// #include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "device/softmax_op.hpp"

namespace ttnn {
namespace operations::normalization {

struct ExecuteSoftmax {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16, ttnn::bfloat8_b}, {ttnn::TILE_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    // softmax
    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const int dim_arg,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto input_shape = input_tensor.get_shape();
        auto rank = input_shape.size();
        auto dim = dim_arg;
        if (dim < 0) {
            dim = rank + dim;
        }

        auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
        if (dim == rank - 1) {
            auto output_tensor =
                ttnn::operations::normalization::softmax(input_tensor_4D, memory_config.value_or(input_tensor.memory_config()), compute_kernel_config);
            return ttnn::reshape(output_tensor, input_shape);
        } else {
            auto dim_4D = dim + 4 - rank;
            auto output_tensor = tt::operations::primary::moreh_softmax(input_tensor_4D, dim_4D);
            return ttnn::reshape(output_tensor, input_shape);
        }
    }
};

struct ExecuteScaleMaskSoftmax {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16, ttnn::bfloat8_b}, {ttnn::TILE_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    // scale_mask_softmax
    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const bool is_causal_mask = false,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto input_shape = input_tensor.get_shape();

        auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
        auto output_tensor =
            ttnn::operations::normalization::scale_mask_softmax(input_tensor_4D, scale, mask, memory_config.value_or(input_tensor.memory_config()), is_causal_mask, compute_kernel_config);
        return ttnn::reshape(output_tensor, input_shape);
    }
};

struct ExecuteSoftmaxInPlace {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16, ttnn::bfloat8_b}, {ttnn::TILE_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    // softmax_in_place
    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto input_shape = input_tensor.get_shape();

        auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
        auto output_tensor =
            ttnn::operations::normalization::softmax_in_place(input_tensor_4D, program_config, compute_kernel_config);
        return ttnn::reshape(output_tensor, input_shape);
    }
};

struct ExecuteScaleMaskSoftmaxInPlace {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16, ttnn::bfloat8_b}, {ttnn::TILE_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    // scale_mask_softmax_in_place
    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const bool is_causal_mask = false,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto input_shape = input_tensor.get_shape();

        auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
        auto output_tensor =
            ttnn::operations::normalization::scale_mask_softmax_in_place(input_tensor_4D, scale, mask, program_config, is_causal_mask, compute_kernel_config);
        return ttnn::reshape(output_tensor, input_shape);
    }
};

struct ExecuteScaleCausalMaskHWSoftmaxInPlace {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16, ttnn::bfloat8_b}, {ttnn::TILE_LAYOUT}, true, false, false, false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    // scale_causal_mask_hw_dims_softmax_in_place
    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto input_shape = input_tensor.get_shape();

        auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
        auto output_tensor =
            ttnn::operations::normalization::scale_causal_mask_hw_dims_softmax_in_place(input_tensor_4D, scale, mask, program_config, compute_kernel_config);
        return ttnn::reshape(output_tensor, input_shape);
    }
};

}  // namespace operations::normalization

constexpr auto softmax = ttnn::register_operation<ttnn::operations::normalization::ExecuteSoftmax>("ttnn::softmax");
constexpr auto scale_mask_softmax = ttnn::register_operation<ttnn::operations::normalization::ExecuteScaleMaskSoftmax>("ttnn::scale_mask_softmax");
constexpr auto softmax_in_place = ttnn::register_operation<ttnn::operations::normalization::ExecuteSoftmaxInPlace>("ttnn::softmax_in_place");
constexpr auto scale_mask_softmax_in_place = ttnn::register_operation<ttnn::operations::normalization::ExecuteScaleMaskSoftmaxInPlace>("ttnn::scale_mask_softmax_in_place");
constexpr auto scale_causal_mask_hw_dims_softmax_in_place = ttnn::register_operation<ttnn::operations::normalization::ExecuteScaleCausalMaskHWSoftmaxInPlace>("ttnn::scale_causal_mask_hw_dims_softmax_in_place");

}  // namespace ttnn
