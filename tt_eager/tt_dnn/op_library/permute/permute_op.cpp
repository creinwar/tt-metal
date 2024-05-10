// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/permute/permute_op.hpp"

#include <functional>

#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace std::placeholders;

namespace tt {

namespace tt_metal {

Tensor permute_(const Tensor &a, std::vector<uint32_t> dims, const MemoryConfig& output_mem_config) {
    Device * device = a.device();

    TT_FATAL(dims.size() == 4, "Only 4D tensor are supported for permute.");
    uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];

    bool pad_n = H == 0 || W == 0;
    bool pad_c = H == 1 || W == 1;
    // Convert tensor back to original
    auto out_shape = a.get_legacy_shape();
    out_shape = {out_shape[N], out_shape[C], out_shape[H], out_shape[W]};

    auto output = a;
    static auto transpose_wh = std::bind(tt::tt_metal::transpose, _1, -2, -1, output_mem_config);
    static auto transpose_hc = std::bind(tt::tt_metal::transpose, _1, 1, -2, output_mem_config);
    static auto transpose_cn = std::bind(tt::tt_metal::transpose, _1, 0, 1, output_mem_config);
    if (N == 0 && C == 1 && H == 2 && W == 3) {
        output = a;
    } else if (N == 0 && C == 1 && H == 3 && W == 2) {
        output = transpose_wh(a);
    } else if (N == 0 && C == 2 && H == 1 && W == 3) {
        output = transpose_hc(a);
    } else if (N == 0 && C == 2 && H == 3 && W == 1) {
        output = transpose_wh(transpose_hc(a));
    } else if (N == 0 && C == 3 && H == 1 && W == 2) {
        output = transpose_hc(transpose_wh(a));
    } else if (N == 0 && C == 3 && H == 2 && W == 1) {
        output = transpose_wh(transpose_hc(transpose_wh(a)));
    } else if (N == 1 && C == 0 && H == 2 && W == 3) {
        output = transpose_cn(a);
    } else if (N == 1 && C == 0 && H == 3 && W == 2) {
        output = transpose_wh(transpose_cn(a));
    } else if (N == 1 && C == 2 && H == 0 && W == 3) {
        output = transpose_hc(transpose_cn(a));
    } else if (N == 1 && C == 2 && H == 3 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_cn(a)));
    } else if (N == 1 && C == 3 && H == 0 && W == 2) {
        output = transpose_hc(transpose_wh(transpose_cn(a)));
    } else if (N == 1 && C == 3 && H == 2 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(a))));
    } else if (N == 2 && C == 0 && H == 1 && W == 3) {
        output = transpose_cn(transpose_hc(a));
    } else if (N == 2 && C == 0 && H == 3 && W == 1) {
        output = transpose_wh(transpose_cn(transpose_hc(a)));
    } else if (N == 2 && C == 1 && H == 0 && W == 3) {
        output = transpose_cn(transpose_hc(transpose_cn(a)));
    } else if (N == 2 && C == 1 && H == 3 && W == 0) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_cn(a))));
    } else if (N == 2 && C == 3 && H == 0 && W == 1) {
        output = transpose_hc(transpose_wh(transpose_cn(transpose_hc(a))));
    } else if (N == 2 && C == 3 && H == 1 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(a)))));
    } else if (N == 3 && C == 0 && H == 1 && W == 2) {
        output = transpose_cn(transpose_hc(transpose_wh(a)));
    } else if (N == 3 && C == 0 && H == 2 && W == 1) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_wh(a))));
    } else if (N == 3 && C == 1 && H == 0 && W == 2) {
        output = transpose_cn(transpose_hc(transpose_cn(transpose_wh(a))));
    } else if (N == 3 && C == 1 && H == 2 && W == 0) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_cn(transpose_wh(a)))));
    } else if (N == 3 && C == 2 && H == 0 && W == 1) {
        output = transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(a)))));
    } else if (N == 3 && C == 2 && H == 1 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(a))))));
    } else {
        TT_ASSERT(false, "Illegal permute args");
    }
    return output;
    // return AutoFormat::format_output_tensor(output, out_shape, device, Layout::TILE);
}

Tensor permute(const Tensor &a, std::vector<std::int64_t> dims, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a}))};
    operation::launch_op(
        [dims, output_mem_config](
            std::vector<Tensor> input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors) mutable -> std::vector<Tensor> {
            auto& a = input_tensors.at(0);
            std::vector<uint32_t> normalized_dims(dims.size());
            std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [a](std::int64_t idx) {return a.get_legacy_shape().get_normalized_index(idx);});
            std::vector<uint32_t> seq_dims(dims.size());
            std::iota(seq_dims.begin(), seq_dims.end(), 0);
            if (normalized_dims == seq_dims) {
                return {a};
            }
            return {operation::decorate_as_composite(__func__, permute_)(a, normalized_dims, output_mem_config)};
        },
        {a},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace tt_metal

}  // namespace tt
