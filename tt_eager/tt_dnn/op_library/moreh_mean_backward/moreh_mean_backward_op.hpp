/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <vector>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehMeanBackward {
    void validate(const std::vector<Tensor> &inputs) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &inputs) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &inputs) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }
};

operation::ProgramWithCallbacks moreh_mean_backward_program(const Tensor &output_grad, const Tensor &input_grad);

Tensor moreh_mean_backward(const Tensor &output_grad, const Tensor &input_grad);

}  // namespace primary

}  // namespace operations

}  // namespace tt
