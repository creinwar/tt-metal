// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "permute.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_permute(py::module& module) {
    auto doc =
        R"doc(permute(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt, queue_id: int = 0) -> ttnn.Tensor

            Permutes the dimensions of the input tensor according to the specified permutation.

            Args:
                * :attr:`input_tensor`: Input Tensor for permute.
                * :attr:`dims`: the permutation of the dimensions of the input tensor.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor
                * :attr:`queue_id`: command queue id
                * :attr:`output_tensor`: optional output tensor

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
                >>> print(output.shape)
                [1, 1, 32, 64])doc";

    using OperationType = decltype(ttnn::permute);
    ttnn::bind_registered_operation(
        module,
        ttnn::permute,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const std::vector<int> &dims,
                std::optional<ttnn::Tensor> &optional_output_tensor,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, dims, memory_config, optional_output_tensor);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("dims"),
                py::kw_only(),
                py::arg("output_tensor").noconvert() = std::nullopt,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::data_movement::detail
