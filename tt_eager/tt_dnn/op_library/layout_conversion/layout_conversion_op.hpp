// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct LayoutConversionOnHost {
    Layout target_layout;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple("target_layout");
    const auto attribute_values() const { return std::forward_as_tuple(target_layout); }
};

Tensor layout_conversion_on_host(const Tensor &input_tensor, const Layout target_layout);

}  // namespace tt_metal

}  // namespace tt
