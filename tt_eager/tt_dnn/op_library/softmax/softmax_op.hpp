// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "common/base_types.hpp"
#include "common/core_coord.h"
#include "tensor/types.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

namespace transformers {
struct SoftmaxDefaultProgramConfig {
    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }
};
struct SoftmaxShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w;
    std::size_t block_h;
    std::size_t block_w;

    static constexpr auto attribute_names =
        std::forward_as_tuple("compute_with_storage_grid_size", "subblock_w", "block_h", "block_w");
    const auto attribute_values() const {
        return std::forward_as_tuple(compute_with_storage_grid_size, subblock_w, block_h, block_w);
    }
};

using SoftmaxProgramConfig = std::variant<SoftmaxDefaultProgramConfig, SoftmaxShardedMultiCoreProgramConfig>;

}  // namespace transformers

struct Softmax {
    const std::optional<float> scale;
    const bool inplace;
    const MemoryConfig output_mem_config;
    const tt::operations::primary::transformers::SoftmaxProgramConfig program_config;
    const bool is_causal_mask;
    const DeviceComputeKernelConfig compute_kernel_config;
    const bool is_scale_causal_mask_hw_dims_softmax;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    const operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "scale",
        "inplace",
        "output_mem_config",
        "program_config",
        "is_causal_mask",
        "compute_kernel_config",
        "is_scale_causal_mask_hw_dims_softmax");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            scale,
            inplace,
            output_mem_config,
            program_config,
            is_causal_mask,
            compute_kernel_config,
            is_scale_causal_mask_hw_dims_softmax);
    }
};

operation::ProgramWithCallbacks scale_mask_softmax_multi_core(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const std::optional<const Tensor> mask,
    std::optional<float> scale,
    bool causal_mask,
    DeviceComputeKernelConfig compute_kernel_config);

// hw_dims_only_causal_mask - represents if the causal mask is of shape [1, 1, h, w]
// valid only if causal_mask == true, and is interleaved
operation::ProgramWithCallbacks scale_mask_softmax_sharded_multi_core(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const std::optional<const Tensor> mask,
    std::optional<float> scale,
    bool causal_mask,
    bool hw_dims_only_causal_mask,
    CoreCoord grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config);

// const ref prevents in-place
Tensor softmax_in_place(
    Tensor& input_tensor,
    const transformers::SoftmaxProgramConfig& program_config = transformers::SoftmaxDefaultProgramConfig{},
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

namespace transformers {
// computes
// tmp1 = bcast_hw_mul(scale, x)  ; shape of scale is [1,1,32,32]
// tmp2 = bcast_add_w->h(tmp1, mask) ; shape of attn mask is [1,N,32,W]
// y = softmax(tmp2)              ; r=result
// If scale == 0.0f then just y = softmax(x) is computed
Tensor scale_mask_softmax_in_place(
    Tensor& input_tensor,
    std::optional<float> scale = std::nullopt,
    std::optional<const Tensor> mask = std::nullopt,
    const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
    const bool is_causal_mask = false,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

// Experimental feature. Does the same same as above, with the following assumptions:
// 1. Input must be sharded
// 2. Scale must exist
// 3. Attention mask must be interleaved and be of this shape [1, 1, H, W]
// 4. Causal mask argument is set to true.
Tensor scale_causal_mask_hw_dims_softmax_in_place(
    Tensor& input_tensor,
    std::optional<float> scale,
    std::optional<const Tensor> mask,
    const SoftmaxProgramConfig& program_config = SoftmaxShardedMultiCoreProgramConfig{},
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
}  // namespace transformers

}  // namespace primary
}  // namespace operations

namespace tt_metal {
Tensor softmax(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

namespace transformers {
Tensor scale_mask_softmax(
    const Tensor& input_tensor,
    std::optional<float> scale,
    std::optional<const Tensor> mask,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const bool is_causal_mask = false,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
}  // namespace transformers
}  // namespace tt_metal

}  // namespace tt
