// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class ConcatOpParallelizationStrategy { MULTI_CORE, SHARDED_MULTI_CORE };

struct Concat {
    uint32_t dim;
    const MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    ConcatOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple("dim", "output_mem_config");
    const auto attribute_values() const { return std::forward_as_tuple(this->dim, this->output_mem_config); }
};

operation::ProgramWithCallbacks sharded_concat_multi_core(
    const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output);
operation::ProgramWithCallbacks concat_multi_core(
    const std::vector<Tensor> &input_tensors, const uint32_t dim, const Tensor &output);

// Ref: https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
// Notes: Non-empty tensors provided must have the same shape, except in the cat dimension.
Tensor concat(
    std::vector<Tensor> &input_tensors,
    const std::int64_t dim = 0,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
