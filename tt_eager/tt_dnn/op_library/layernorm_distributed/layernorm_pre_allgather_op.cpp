// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_eager/tt_dnn/op_library/layernorm_distributed/layernorm_pre_allgather_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

void LayerNormPreAllGather::validate(const std::vector<Tensor> &input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Must have 1 input tensor");
    auto& tensor = input_tensors.at(0);

    TT_FATAL(tensor.get_layout() == Layout::TILE, "Only tilized inputs supported.");
    TT_FATAL(tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Only interleaved inputs supported.");
    TT_FATAL(tensor.get_dtype() == DataType::BFLOAT16 || tensor.get_dtype() == DataType::BFLOAT8_B || tensor.get_dtype() == DataType::FLOAT32, "Input data format not supported.");
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
}

std::vector<Shape> LayerNormPreAllGather::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    auto output_shape = input_tensor.get_legacy_shape();
    auto padding = output_shape.padding();
    uint32_t num_tiles_w = 1;
    if (this->norm_type == LayerNormType::LAYERNORM) {
        num_tiles_w = 2;
    }
    output_shape[3] = num_tiles_w * TILE_WIDTH;
    padding[3] = Padding::PadDimension{0, 31};

    return {Shape(output_shape, padding)};
}

std::vector<Tensor> LayerNormPreAllGather::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, input_tensor.memory_config());
}

operation::ProgramWithCallbacks LayerNormPreAllGather::create_program(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return layernorm_pre_allgather_multi_core(
        a, output_tensor, this->norm_type, this->compute_kernel_config
    );
}

tt::stl::reflection::Attributes LayerNormPreAllGather::attributes() const {
    return {
        {"norm_type", this->norm_type},
        {"compute_kernel_config", this->compute_kernel_config}
    };
}

}  // namespace tt_metal

}  // namespace tt
