// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/pool/pool.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace pool {

namespace detail {

void bind_global_avg_pool2d(py::module& module) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

        Applies {0} to :attr:`input_tensor` by performing a 2D adaptive average pooling over an input signal composed of several input planes. This operation computes the average of all elements in each channel across the entire spatial dimensions.

        .. math::
            {0}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): The input tensor to be pooled. Typically of shape (batch_size, channels, height, width).

        Keyword Args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            * :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor

        Returns:
            ttnn.Tensor: The tensor with the averaged values. The output tensor shape is (batch_size, channels, 1, 1).

        Example:

            >>> tensor = ttnn.from_torch(torch.randn((10, 3, 32, 32), dtype=ttnn.bfloat16), device=device)
            >>> output = {1}(tensor)
    )doc",
        ttnn::global_avg_pool2d.name(),
        ttnn::global_avg_pool2d.python_fully_qualified_name());

    bind_registered_operation(
        module,
        ttnn::global_avg_pool2d,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt});
}

}  // namespace detail

void py_module(py::module& module) { detail::bind_global_avg_pool2d(module); }

}  // namespace pool
}  // namespace operations
}  // namespace ttnn
