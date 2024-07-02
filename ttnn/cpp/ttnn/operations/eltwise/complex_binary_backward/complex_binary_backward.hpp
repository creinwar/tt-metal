
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_binary_backward_op.cpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::complex_binary_backward {

class ComplexTensor {
    private:
        std::array<Tensor,2> m_real_imag;

    public:

        ComplexTensor(std::array<Tensor,2> val): m_real_imag(val) {
            TT_ASSERT( m_real_imag[0].get_legacy_shape() == m_real_imag[1].get_legacy_shape() , "Tensor shapes of real and imag should be identical");
        }

        const Tensor& operator[](uint32_t index) const {
            return m_real_imag[index];
        }

        const Tensor& real() const {
            return m_real_imag[0];
        }

        const Tensor& imag() const {
            return m_real_imag[1];
        }

        void deallocate() {
            m_real_imag[0].deallocate();
            m_real_imag[1].deallocate();
        }
};

template <ComplexBinaryBackwardOpType complex_binary_backward_op_type>
struct ExecuteComplexBinaryBackward {

    static inline const std::array<TensorSchema, 3> input_tensor_schemas() {
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
                false}};
    }

    //Type 1: 2 inputs, 1 grad tensor
    template <typename... Args>
    static auto input_tensors_to_validate(const ComplexTensor &grad_tensor, const ComplexTensor &input_tensor_a, const ComplexTensor &input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    }

    static std::vector<ComplexTensor> execute_on_main_thread(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        float alpha,
        const MemoryConfig &memory_config) {

        auto op_type = utils::get_function_type1(complex_binary_backward_op_type);
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, alpha, memory_config);
        }

};

}
constexpr auto complex_add_bw = ttnn::register_operation<operations::complex_binary_backward::ExecuteComplexBinaryBackward<operations::complex_binary_backward::ComplexBinaryBackwardOpType::COMPLEX_ADD_BW>>("ttnn::complex_add_bw");

}
