
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::ternary_backward {

template <TernaryBackwardOpType ternary_backward_op_type>
struct ExecuteTernaryBackward {

    static inline const std::array<TensorSchema, 4> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false}};
    }

    static inline std::vector<ttnn::Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }

    //Type 1: 3 inputs, 1 grad tensor, 1 float
    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, const Tensor &input_tensor_c, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c);
    }

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        float alpha,
        const MemoryConfig &memory_config) {
        auto op_type = utils::get_function_type(ternary_backward_op_type);
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, input_tensor_c_arg, alpha, memory_config);
        }

    //Q_ID, type1 args, optional output tensor for inputs based on are_required_outputs value
    template <typename... Args>
    static auto input_tensors_to_validate(uint8_t queue_id, const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, const Tensor &input_tensor_c, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c);
    }

    static std::vector<std::optional<ttnn::Tensor>> execute_on_main_thread(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt) {

        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        auto op_type = utils::get_function_type_opt(ternary_backward_op_type);
        return op_type(queue_id, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, input_tensor_c_arg, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    }

    //type1 args, optional output tensor for inputs based on are_required_outputs value
    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, const Tensor &input_tensor_c, std::vector<bool> are_required_outputs, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c);
    }

    static std::vector<std::optional<ttnn::Tensor>> execute_on_main_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt) {

        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        auto op_type = utils::get_function_type_opt_wo_qid(ternary_backward_op_type);
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, input_tensor_c_arg, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    }

};

}  // operations::ternary_backward

//type 1
constexpr auto addcmul_bw = ttnn::register_operation<operations::ternary_backward::ExecuteTernaryBackward<operations::ternary_backward::TernaryBackwardOpType::ADDCMUL_BW>>("ttnn::addcmul_bw");
constexpr auto addcdiv_bw = ttnn::register_operation<operations::ternary_backward::ExecuteTernaryBackward<operations::ternary_backward::TernaryBackwardOpType::ADDCDIV_BW>>("ttnn::addcdiv_bw");
constexpr auto where_bw = ttnn::register_operation<operations::ternary_backward::ExecuteTernaryBackward<operations::ternary_backward::TernaryBackwardOpType::WHERE_BW>>("ttnn::where_bw");

}  // namespace ttnn
