// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_dnn/op_library/ccl/ccl_common.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

namespace ttnn {

namespace utils {

namespace all_gather_op {
using tt::tt_metal::ccl::Topology;
}; // namespace all_gather_op

using tt::tt_metal::ccl::EriscDatamoverBuilder;


struct LineAllGather {
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<chip_id_t> receiver_device_id;
    const std::optional<chip_id_t> sender_device_id;
    const MemoryConfig output_mem_config;
    const all_gather_op::Topology topology;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim",
        "num_links",
        "ring_size",
        "ring_index",
        "receiver_device_id",
        "sender_device_id",
        "output_mem_config",
        "topology");

    const auto attribute_values() const {
        return std::forward_as_tuple(
            dim, num_links, ring_size, ring_index, receiver_device_id, sender_device_id, output_mem_config, topology);
    }
};

// All Gather Variants
std::vector<Tensor> line_all_gather_impl(
    const std::vector<Tensor>& input_tensors,
    const uint32_t dim,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const all_gather_op::Topology topology);
std::vector<Tensor> line_all_gather(
    const std::vector<Tensor> &input_tensors,
    const uint32_t dim,
    const uint32_t num_links = 1,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace utils

namespace operations {
namespace ccl {

Tensor line_all_gather(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

} // namespace ccl
} // namespace operations

}  // namespace ttnn
