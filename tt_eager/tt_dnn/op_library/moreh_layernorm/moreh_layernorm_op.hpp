// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/compute_kernel_config.hpp"
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

struct MorehLayerNorm {
    uint32_t normalized_dims;
    float eps;
    MemoryConfig memory_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    bool compute_mean;
    bool compute_rstd;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("normalized_dims", "eps", "memory_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->normalized_dims), std::cref(this->eps), std::cref(this->memory_config),
            std::cref(this->compute_kernel_config));
    }
};

operation::ProgramWithCallbacks moreh_layernorm_impl(
    const Tensor &input,
    uint32_t normalized_dims,
    float eps,
    Tensor &output,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    const std::optional<const Tensor> mean,
    const std::optional<const Tensor> rstd,
    const DeviceComputeKernelConfig compute_kernel_config);

std::vector<std::optional<Tensor>> moreh_layernorm(
    const Tensor &input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> beta = std::nullopt,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<const Tensor> mean = std::nullopt,
    const std::optional<const Tensor> rstd = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

namespace tt_metal {

std::vector<std::optional<Tensor>> moreh_layernorm(
    const Tensor &input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> beta = std::nullopt,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<const Tensor> mean = std::nullopt,
    const std::optional<const Tensor> rstd = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt
    );

}  // namespace tt_metal

}  // namespace tt
