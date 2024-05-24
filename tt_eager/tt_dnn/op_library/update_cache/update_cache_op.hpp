// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class UpdateCacheOpParallelizationStrategy { MULTI_CORE };

enum class UpdateCacheOpType { FILL, UPDATE };

operation::ProgramWithCallbacks update_cache_multi_core(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const uint32_t update_idx,
    const uint32_t batch_offset,
    DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks fill_cache_multi_core(
    const Tensor& cache_tensor, const Tensor& input_tensor, const uint32_t batch_idx, const uint32_t update_idx);

struct UpdateCache {
    const uint32_t batch_idx;
    const uint32_t update_idx;
    const uint32_t batch_offset;
    const UpdateCacheOpType op_type;
    const DeviceComputeKernelConfig compute_kernel_config;

    UpdateCacheOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("batch_idx", "update_idx", "batch_offset", "op_type", "compute_kernel_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->batch_idx, this->update_idx, this->batch_offset, this->op_type, this->compute_kernel_config);
    }

    const operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

inline Tensor fill_cache(const Tensor& cache_tensor, const Tensor& input_tensor, const uint32_t batch_idx) {
    std::vector<Tensor> dummy_output_tensors = {
        Tensor(operation::get_workers_for_op_output({cache_tensor, input_tensor}))};
    operation::launch_op(
        [batch_idx](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(UpdateCache{batch_idx, 0, 0, UpdateCacheOpType::FILL}, input_tensors);
        },
        {cache_tensor, input_tensor},
        dummy_output_tensors);
    return cache_tensor;
}

inline Tensor update_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const uint32_t update_idx,
    const uint32_t batch_offset,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    std::vector<Tensor> dummy_output_tensors = {
        Tensor(operation::get_workers_for_op_output({cache_tensor, input_tensor}))};
    operation::launch_op(
        [update_idx, batch_offset, compute_kernel_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& cache_tensor = input_tensors.at(0);
            auto& input_tensor = input_tensors.at(1);
            auto kernel_config_val =
                init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);
            return operation::run(
                UpdateCache{0, update_idx, batch_offset, UpdateCacheOpType::UPDATE, kernel_config_val},
                {cache_tensor, input_tensor});
        },
        {cache_tensor, input_tensor},
        dummy_output_tensors);
    return cache_tensor;
}

}  // namespace tt_metal

}  // namespace tt
