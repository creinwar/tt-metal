// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>

// #include "tensor/tensor.hpp"
// #include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"

namespace ttnn::operations {
namespace halo {

struct Halo {
    SlidingWindowConfig config_;
    ParallelConfig parallel_config_;
    uint32_t pad_val_;
    bool remote_read_;
    bool transpose_mcast_;
    uint32_t reshard_num_cores_nhw_;
    uint32_t max_out_nsticks_per_core_;
    MemoryConfig output_memory_config_;
    Tensor pad_config_tensor_;
    Tensor local_config_tensor_;
    Tensor remote_config_tensor_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    // const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "config_",
        "parallel_config_",
        "pad_val_",
        "remote_read_",
        "transpose_mcast_",
        "reshard_num_cores_nhw_",
        "max_out_nsticks_per_core_",
        "output_memory_config_");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            config_,
            parallel_config_,
            pad_val_,
            remote_read_,
            transpose_mcast_,
            reshard_num_cores_nhw_,
            max_out_nsticks_per_core_,
            output_memory_config_);
    }
};

Tensor halo_op(
    const Tensor &input_tensor,
    const SlidingWindowConfig &config,
    uint32_t pad_val = 0x0,
    bool remote_read = false,
    bool transpose_mcast = true,
    uint32_t reshard_num_cores_nhw = 0,
    MemoryConfig output_memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace halo

}  // namespace ttnn::operations
