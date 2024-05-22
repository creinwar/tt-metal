#pragma once

#include <optional>

#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "ttnn/operations/core.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

namespace transformers {

    Tensor llama_mlp_decode_forward(Tensor& input_tensor, const Tensor& w1, const Tensor& w2, const Tensor& w3);

    std::tuple<Tensor, Tensor, Tensor> llama_attn_qkv_decode_forward(const Tensor& input_tensor, const Tensor& rot_mat, const Tensor& wqkv, const MemoryConfig sharded_mem_config);

    Tensor llama_attn_mqa_decode_forward(Tensor& query_layer, Tensor& key_layer, Tensor& value_layer, const uint32_t start_pos, const Tensor& attn_masks, const uint32_t batch_offset, Tensor& K_cache, Tensor& V_cache, const float scale,
MemoryConfig kv_cache_mem_config, MemoryConfig dram_mem_config, MemoryConfig height_mem_config, MemoryConfig attn_output_memcfg, MemoryConfig scores_output_memcfg);

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
