#pragma once

#include <functional>
#include "third_party/magic_enum/magic_enum.hpp"

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class RotaryEmbeddingOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

operation::ProgramWithCallbacks rotary_embedding_single_core(const Tensor &input, const Tensor &cos, const Tensor &sin, Tensor &output);
operation::ProgramWithCallbacks rotary_embedding_multi_core(const Tensor &input, const Tensor &cos, const Tensor &sin, Tensor &output);

struct RotaryEmbedding {
    const uint32_t seq_len;
    const MemoryConfig output_mem_config;

    RotaryEmbeddingOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;


    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor rotary_embedding(const Tensor& input_tensor, const Tensor& cos, const Tensor& sin, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor.shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    uint32_t seq_len = input_tensor.shape()[-2];
    uint32_t B = input_tensor.shape()[0];
    uint32_t X = input_tensor.shape()[-1];
    TT_ASSERT(cos.shape()[0] == B && cos.shape()[1] == 1 && cos.shape()[-2] >= seq_len && cos.shape()[-1] == X, "Cos dims must match input dims");
    TT_ASSERT(sin.shape()[0] == B && sin.shape()[1] == 1 && sin.shape()[-2] >= seq_len && sin.shape()[-1] == X, "Sin dims must match input dims");
    TT_ASSERT(cos.shape() == sin.shape());
    Shape input_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape());
    FormatParams input_format_params = {.pad_shape=input_pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    Shape cos_pad_shape = AutoFormat::pad_to_tile_shape(cos.shape());
    FormatParams cos_format_params = {.pad_shape=cos_pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    Shape sin_pad_shape = AutoFormat::pad_to_tile_shape(sin.shape());
    FormatParams sin_format_params = {.pad_shape=sin_pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    return operation::run_with_autoformat(RotaryEmbedding{seq_len, output_mem_config}, {input_tensor, cos, sin}, {input_format_params, cos_format_params, sin_format_params}, {Layout::TILE}).at(0);
}


}  // namespace tt_metal

}  // namespace tt
