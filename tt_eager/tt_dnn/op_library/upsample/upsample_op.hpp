// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {
namespace tt_metal {

enum class UpSampleParallelizationStrategy { MULTI_CORE, SINGLE_CORE };

struct UpSample {
    const int scale_factor_h_;
    const int scale_factor_w_;
    const MemoryConfig output_mem_config_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    UpSampleParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("scale_factor_h", "scale_factor_w", "output_mem_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(scale_factor_h_, scale_factor_w_, output_mem_config_);
    }
};

Tensor upsample(
    const Tensor &input,
    int scale_factor_h,
    int scale_factor_w,
    const MemoryConfig &out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

operation::ProgramWithCallbacks upsample_single_core(
    const Tensor &input, Tensor &output, uint32_t scale_factor_h, uint32_t scale_factor_w);
operation::ProgramWithCallbacks upsample_multi_core(
    const Tensor &input, Tensor &output, uint32_t scale_factor_h, uint32_t scale_factor_w);

}  // namespace tt_metal
}  // namespace tt
