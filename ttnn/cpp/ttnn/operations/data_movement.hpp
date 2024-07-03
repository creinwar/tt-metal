// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_dnn/op_library/concat/concat_op.hpp"
#include "tt_eager/tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_eager/tt_dnn/op_library/repeat/repeat_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/upsample/upsample_op.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {

inline ttnn::Tensor concat(
    const std::vector<ttnn::Tensor>& input_tensors, int dim, const std::optional<MemoryConfig>& memory_config_arg) {
    TT_FATAL(input_tensors.size() > 0, "ttnn.concat: expected a non-empty list of Tensors!");

    const auto memory_config = memory_config_arg.value_or(ttnn::DRAM_MEMORY_CONFIG);

    if (input_tensors.size() == 1) {
        return ttnn::to_memory_config(input_tensors.at(0), memory_config, std::nullopt);
    }

    // TODO: Issue #8426: Add validation for ttnn.concat for sharded inputs
    // const bool all_tensors_are_tile_layout_without_padding = std::all_of(input_tensors.begin(), input_tensors.end(),
    // [dim](const ttnn::Tensor& input_tensor){
    //    return input_tensor.get_layout() == ttnn::TILE_LAYOUT and not has_tile_padding(input_tensor, dim);
    //});
    // TT_FATAL(all_tensors_are_tile_layout_without_padding, "Not Implemented");

    const ttnn::Tensor& first_tensor = input_tensors.front();
    const int rank = first_tensor.get_shape().rank();

    // Wrap dim
    dim = dim < 0 ? rank + dim : dim;
    TT_FATAL(
        dim >= 0 and dim < rank,
        "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}",
        dim,
        rank);

    const bool shapes_match =
        std::all_of(input_tensors.begin(), input_tensors.end(), [first_tensor, dim](const ttnn::Tensor& t) {
            const auto& ft_shape = first_tensor.get_shape();
            const auto& t_shape = t.get_shape();

            const bool ranks_match = ft_shape.rank() == t_shape.rank();
            bool non_concat_dims_match = true;
            for (int i = 0; i < ft_shape.rank(); i++) {
                non_concat_dims_match &= dim == i or t_shape[i] == ft_shape[i];
            }
            // bool non_concat_padded_dims_match = true;
            // for(int i = 0; i < ft_shape.rank(); i++) {
            //     non_concat_padded_dims_match &= dim == i or t_shape.with_tile_padding()[i] ==
            //     ft_shape.with_tile_padding()[i];
            // }
            return ranks_match and non_concat_dims_match;  // and non_concat_padded_dims_match;
        });

    TT_FATAL(
        shapes_match,
        "All dimensions must be the same size except for the dimension along which the contenation is taking place.");

    std::vector<ttnn::Tensor> itensor;
    std::transform(
        input_tensors.begin(),
        input_tensors.end(),
        std::back_inserter(itensor),
        [rank](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
            auto output = (rank < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
            return output;
        });
    // Convert dim after unsqueeze
    dim = dim + 4 - rank;
    auto output_tensor = tt::tt_metal::concat(itensor, dim, memory_config);
    while (output_tensor.get_shape().rank() > rank) {
        const auto shape = output_tensor.get_shape();
        const auto full_shape = output_tensor.get_shape().with_tile_padding();
        std::vector<uint32_t> shape_vec{};
        std::vector<uint32_t> full_shape_vec{};
        // int i = 0;
        // while(i < 3 and shape[i] == 1) i++;
        for (int i = 1; i < shape.rank(); i++) {
            shape_vec.push_back(shape[i]);
            full_shape_vec.push_back(full_shape[i]);
        }
        auto metal_shape = tt::tt_metal::Shape(shape_vec, full_shape_vec);
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(metal_shape));
    }

    return output_tensor;
}

struct UpSample {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2,
            4,
            {ttnn::bfloat16, ttnn::bfloat8_b},
            {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
            true,
            false,
            false,
            false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::make_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        std::variant<int, std::array<int, 2>, std::array<int, 3>, std::array<int, 4>> scale_factor,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

        int scale_h = 1;
        int scale_w = 1;
        std::visit(
            [&scale_h, &scale_w](auto&& sf) {
                using T = std::decay_t<decltype(sf)>;
                if constexpr (std::is_same_v<T, int>) {
                    scale_h = sf;
                    scale_w = sf;
                } else if constexpr (std::is_same_v<T, std::array<int, 2>>) {
                    scale_w = sf.at(0);
                    int scale_c = sf.at(1);
                    TT_FATAL(scale_c == 1);
                } else if constexpr (std::is_same_v<T, std::array<int, 3>>) {
                    scale_h = sf.at(0);
                    scale_w = sf.at(1);
                    int scale_c = sf.at(2);
                    TT_FATAL(scale_c == 1);
                } else if constexpr (std::is_same_v<T, std::array<int, 4>>) {
                    int scale_n = sf.at(0);
                    scale_h = sf.at(1);
                    scale_w = sf.at(2);
                    int scale_c = sf.at(3);
                    TT_FATAL(scale_n == 1);
                    TT_FATAL(scale_c == 1);
                } else {
                    // static_assert(false, "Unsupported scale factor");
                    static_assert(sizeof(T) != 0, "Type check failed.");
                }
            },
            scale_factor);

        // DEBUG
        // fmt::print("scale_h: {}, scale_w: {}\n", scale_h, scale_w);

        if (input_tensor.is_sharded()) {
            // TT_FATAL(not input_tensor.is_sharded());
            int shard_height = input_tensor.memory_config().shard_spec.value().shape[0];
            const auto batch_size = input_tensor.get_shape()[0];
            const auto input_h = input_tensor.get_shape()[1];
            const auto input_w = input_tensor.get_shape()[2];
            const auto num_channels = input_tensor.get_shape()[3];
            if (shard_height % input_w != 0) {
                TT_FATAL(shard_height % input_w != 0);
            }
        }

        return tt::tt_metal::upsample(input_tensor, scale_h, scale_w, mem_config);
    }
};

struct Repeat {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            4,  // min rank
            4,  // max rank
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::int32, ttnn::uint32},
            {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
            true,     // can_be_on_device
            false,    // can_be_on_cpu
            false,    // can_be_scalar
            false}};  // is_optional
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::make_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& shape,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat(input_tensor, shape.value(), mem_config);
        return output_tensor;
    }
};

struct RepeatInterleave {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            4,  // min rank
            4,  // max rank
            {ttnn::bfloat16},
            {ttnn::TILE_LAYOUT},
            true,     // can_be_on_device
            true,    // can_be_on_cpu
            false,    // can_be_scalar
            false}};  // is_optional
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::make_tuple(input_tensor);
    }

    // # This operation does not support the following cases:
    // #   - Shape([2[32], 2[32]]) -> repeats = 2, dim = 0
    // #   - Shape([2[32], 2[32]]) -> repeats = Tensor[1,2], dim = 1
    static ttnn::Tensor execute_on_worker_thread(const ttnn::Tensor& input_tensor,
                                                 uint32_t repeats,
                                                 int32_t dim,
                                                 std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat_interleave(input_tensor, repeats, dim, mem_config);
        return output_tensor;
    }
};



}  // namespace data_movement
}  // namespace operations

constexpr auto upsample = ttnn::register_operation<ttnn::operations::data_movement::UpSample>("ttnn::upsample");
constexpr auto repeat = ttnn::register_operation<ttnn::operations::data_movement::Repeat>("ttnn::repeat");
constexpr auto repeat_interleave = ttnn::register_operation<ttnn::operations::data_movement::RepeatInterleave>("ttnn::repeat_interleave");

}  // namespace ttnn
