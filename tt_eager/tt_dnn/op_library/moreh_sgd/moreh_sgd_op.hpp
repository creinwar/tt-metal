/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>

#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_sgd_(
    const Tensor &param_in,
    const Tensor &grad,
    std::optional<const Tensor> momentum_buffer_in,
    const Tensor &param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const CoreRange core_range,
    const DeviceComputeKernelConfig compute_kernel_config);

struct MorehSGD {
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    bool nesterov;
    bool momentum_initialized;
    const CoreRange core_range;  // unused for now
    MemoryConfig param_out_mem_config;
    MemoryConfig momentum_buffer_out_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::forward_as_tuple(
        "lr",
        "momentum",
        "dampening",
        "weight_decay",
        "nesterov",
        "momentum_initialized",
        "param_out_mem_config",
        "momentum_buffer_out_mem_config",
        "compute_kernel_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->lr,
            this->momentum,
            this->dampening,
            this->weight_decay,
            this->nesterov,
            this->momentum_initialized,
            this->param_out_mem_config,
            this->momentum_buffer_out_mem_config,
            this->compute_kernel_config);
    }
};

std::vector<std::optional<Tensor>> moreh_sgd(
    const Tensor &param_in,
    const Tensor &grad,
    std::optional<const Tensor> momentum_buffer_in,
    std::optional<const Tensor> param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const MemoryConfig &param_out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &momentum_buffer_out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
