// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "ttnn/cpp/ttnn/multi_device.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteAllGather {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2,
            4,
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
            {ttnn::ROW_MAJOR_LAYOUT, ttnn::TILE_LAYOUT},
            true,
            false,
            false,
            false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_main_thread(
        const ttnn::Tensor& input_tensor,
        const uint32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
        return tt::operations::ccl::all_gather(input_tensor, dim, num_links, memory_config);
    }
};

struct ExecuteLineAllGather {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2,
            4,
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
            {ttnn::ROW_MAJOR_LAYOUT, ttnn::TILE_LAYOUT},
            true,
            false,
            false,
            false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_main_thread(
        const ttnn::Tensor& input_tensor,
        const uint32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
        return tt::operations::ccl::line_all_gather(input_tensor, dim, num_links, memory_config);
    }
};

}  // namespace ccl
}  // namespace operations

constexpr auto all_gather = ttnn::register_operation<ttnn::operations::ccl::ExecuteAllGather>("ttnn::all_gather");

constexpr auto line_all_gather = ttnn::register_operation<ttnn::operations::ccl::ExecuteLineAllGather>("ttnn::line_all_gather");

}  // namespace ttnn
