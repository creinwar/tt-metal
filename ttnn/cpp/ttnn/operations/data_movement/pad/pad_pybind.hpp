// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "pad.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_pad(py::module& module) {
    auto doc =
        R"doc(
            pad(input_tensor: ttnn.Tensor, padding: Tuple[Tuple[int, int], ...], value: Union[int, float], *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            pad(input_tensor: ttnn.Tensor, output_tensor_shape::  ttnn.Shape, input_tensor_start:: ttnn.Shape, value: Union[int, float], *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
Pad tensor with constant value. Padded shape is accumulated if ttnn.pad is called on a tensor with padding. This version of the API only works with 4D tensors.

Args:
    * :attr:`input_tensor`: input tensor
    * :attr:`padding`: padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor.
    * :attr:`output_tensor_shape`: Final shape of padded tensor. This along with input_tensor_start can be used instead of padding.
    * :attr:`input_tensor_start`: Shape describing where to start padding. This along with output_tensor_shape can be used instead of padding.
    * :attr:`value`: value to pad with

Keyword Args:
    * :attr:`memory_config`: the memory configuration to use for the operation
    * :attr:`queue_id` (Optional[uint8]): command queue id
    )doc";

    using OperationType = decltype(ttnn::pad);
    ttnn::bind_registered_operation(
        module,
        ttnn::pad,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                std::vector<std::pair<uint32_t, uint32_t>> padding,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, padding, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("padding"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = true,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const Shape output_padded_shape,
                const Shape input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = true,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                }
        );
}
}  // namespace ttnn::operations::data_movement::detail
