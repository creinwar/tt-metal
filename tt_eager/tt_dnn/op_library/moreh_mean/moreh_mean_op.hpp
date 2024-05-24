// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehMean {
    int64_t dim;
    void validate(const std::vector<Tensor> &inputs) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &inputs) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &inputs) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;

    static constexpr auto attribute_names = std::forward_as_tuple("dim");
    const auto attribute_values() const { return std::forward_as_tuple(this->dim); }
};

operation::ProgramWithCallbacks moreh_mean_nc(const Tensor &input, const Tensor &output, int64_t dim);
// revised from reduce_op
operation::ProgramWithCallbacks moreh_mean_w(const Tensor &a, const Tensor &output);
operation::ProgramWithCallbacks moreh_mean_h(const Tensor &a, const Tensor &output);

Tensor moreh_mean_(
    const Tensor &input,
    std::optional<std::reference_wrapper<const Tensor>> output,
    const int64_t &dim,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor moreh_mean(
    const Tensor &input,
    const Tensor &output,
    std::vector<int64_t> &dims,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
