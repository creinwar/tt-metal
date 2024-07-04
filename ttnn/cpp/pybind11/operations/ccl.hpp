// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/ccl.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_ccl_operation(py::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t dim,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(input_tensor, dim, num_links, memory_config);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace detail


void py_module(py::module& module) {
    detail::bind_ccl_operation(
        module,
        ttnn::all_gather,
        R"doc(all_gather(input_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.all_gather(tensor, dim=0)

        )doc");

    detail::bind_ccl_operation(
        module,
        ttnn::line_all_gather,
        R"doc(line_all_gather(input_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.line_all_gather(tensor, dim=0)

        )doc");
}

}  // namespace ccl
}  // namespace operations
}  // namespace ttnn
