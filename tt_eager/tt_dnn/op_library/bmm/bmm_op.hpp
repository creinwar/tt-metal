// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class MatmulParallelizationStrategy {
    MULTI_CORE,
    MULTI_CORE_REUSE,
    MULTI_CORE_REUSE_PADDING,
    MULTI_CORE_REUSE_OPTIMIZED,
    MULTI_CORE_REUSE_MCAST_2D_OPTIMIZED,
    MULTI_CORE_REUSE_MCAST_2D_TRANSPOSED_OPTIMIZED,
    MULTI_CORE_REUSE_MCAST_1D_IN0_OPTIMIZED,
    MULTI_CORE_REUSE_MCAST_1D_IN1_OPTIMIZED
};

/*
 * GENERAL MATMUL AND BMM
 */
operation::ProgramWithCallbacks matmul_multi_core(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_padding(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_padding(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, bool bcast_batch);

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const Tensor> bias,
    Tensor &output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    std::optional<UnaryWithParam> fused_activation,
    bool mcast_in0,
    bool untilize_out);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const Tensor> bias,
    Tensor &output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out);
operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    Tensor &output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    tt::tt_metal::DataType output_dtype,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool untilize_out);

/**
 * Falcon matmuls using operations::primary::matmul + program_config
 */
Tensor falcon_fused_qkv_matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt);
Tensor falcon_selfout_matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt);
Tensor falcon_dense_4h_to_h_matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    std::optional<bool> packer_l1_acc = std::nullopt);
Tensor falcon_dense_h_to_4h_matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias,
    std::optional<UnaryWithParam> fused_activation = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt);
Tensor falcon_lm_head_matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt);

/**
 * Resnet matmul for linear
 */
Tensor resnet_matmul(
    const Tensor &input_a,
    const Tensor &input_b,
    std::optional<const Tensor> bias,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    const MathFidelity math_fidelity = MathFidelity::LoFi);

/**
 * Generalized blocked matmul with support for tilize and untilize and mixed-prec
 */
struct BMMTilizeUntilize {
    const DataType out_dt_;
    const uint32_t in0_nblocks_h_, in0_nblocks_w_, in1_nblocks_w_;
    const uint32_t in0_block_ntiles_h_, in0_block_ntiles_w_, in1_block_ntiles_w_;
    const uint32_t out_subblock_ntiles_h_, out_subblock_ntiles_w_;
    const bool tilize_in0_, untilize_out_;
    const bool has_bias_;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "out_dt",
        "in0_nblocks_h",
        "in0_nblocks_w",
        "in1_nblocks_w",
        "in0_block_ntiles_h",
        "in0_block_ntiles_w",
        "in1_block_ntiles_w",
        "out_subblock_ntiles_h",
        "out_subblock_ntiles_w",
        "tilize_in0",
        "untilize_out",
        "has_bias");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->out_dt_,
            this->in0_nblocks_h_,
            this->in0_nblocks_w_,
            this->in1_nblocks_w_,
            this->in0_block_ntiles_h_,
            this->in0_block_ntiles_w_,
            this->in1_block_ntiles_w_,
            this->out_subblock_ntiles_h_,
            this->out_subblock_ntiles_w_,
            this->tilize_in0_,
            this->untilize_out_,
            this->has_bias_);
    }
};

/**
 * Blocked Matmul, with support for tilize a and untilize output.
 * NOTE: Takes blocks and subblock information as arguments.
 */
Tensor bmm_tilize_untilize(
    const Tensor &a,
    const Tensor &b,
    const Tensor &bias,
    DataType out_dt,
    uint32_t a_height_nblocks,
    uint32_t a_width_nblocks,
    uint32_t b_width_nblocks,
    uint32_t a_block_height_ntiles,
    uint32_t a_block_width_ntiles,
    uint32_t b_block_width_ntiles,
    uint32_t out_subblock_height_ntiles,
    uint32_t out_subblock_width_ntiles,
    bool tilize_in0,
    bool untilize_out,
    bool has_bias,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
operation::ProgramWithCallbacks bmm_single_core_tilize_untilize(
    const Tensor &in0,
    const Tensor &in1,
    Tensor &bias,
    DataType out_dt,
    uint32_t in0_height_nblocks,
    uint32_t in0_width_nblocks,
    uint32_t in1_width_nblocks,
    uint32_t in0_block_height_ntiles,
    uint32_t in0_block_width_ntiles,
    uint32_t in1_block_width_ntiles,
    uint32_t out_subblock_height_ntiles,
    uint32_t out_subblock_width_ntiles,
    bool tilize_in0,
    bool untilize_out,
    bool has_bias,
    Tensor &out,
    DeviceComputeKernelConfig compute_kernel_config);

}  // namespace tt_metal

namespace operations {

namespace primary {

using namespace tt_metal;

struct MatmulDefaultProgramConfig {
    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }
};

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1 for in1 iff B=1 for in0 (ie. single
// core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "per_core_M",
        "per_core_N");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->compute_with_storage_grid_size,
            this->in0_block_w,
            this->out_subblock_h,
            this->out_subblock_w,
            this->per_core_M,
            this->per_core_N);
    }
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool transpose_mcast;
    std::optional<UnaryWithParam> fused_activation;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "per_core_M",
        "per_core_N",
        "transpose_mcast",
        "fused_activation");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->compute_with_storage_grid_size,
            this->in0_block_w,
            this->out_subblock_h,
            this->out_subblock_w,
            this->per_core_M,
            this->per_core_N,
            this->transpose_mcast,
            this->fused_activation);
    }
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    std::optional<UnaryWithParam> fused_activation;
    bool mcast_in0;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "per_core_M",
        "per_core_N",
        "fuse_batch",
        "fused_activation",
        "mcast_in0");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->compute_with_storage_grid_size,
            this->in0_block_w,
            this->out_subblock_h,
            this->out_subblock_w,
            this->per_core_M,
            this->per_core_N,
            this->fuse_batch,
            this->fused_activation,
            this->mcast_in0);
    }
};

using MatmulProgramConfig = std::variant<
    MatmulDefaultProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig>;

struct Matmul {
    MatmulProgramConfig program_config;
    bool bcast_batch;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;
    const DeviceComputeKernelConfig compute_kernel_config;
    const bool untilize_out;

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    MatmulParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "program_config", "bcast_batch", "output_mem_config", "output_dtype", "compute_kernel_config", "untilize_out");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->program_config,
            this->bcast_batch,
            this->output_mem_config,
            this->output_dtype,
            this->compute_kernel_config,
            this->untilize_out);
    }
};

inline bool get_broadcast_batch(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MatmulProgramConfig &matmul_program_config) {
    bool broadcast_batch = input_tensor_b.get_legacy_shape()[0] * input_tensor_b.get_legacy_shape()[1] == 1;
    bool is_multi_core_reuse = std::visit(
        [](const auto &program_config) -> bool {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                return true;
            }
            return false;
        },
        matmul_program_config);
    if (is_multi_core_reuse) {
        broadcast_batch &= input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] > 1;
    }
    return broadcast_batch;
}

inline bool is_program_config_default(const MatmulProgramConfig &matmul_program_config) {
    using ProgramConfigType = std::decay_t<decltype(matmul_program_config)>;
    bool result = std::visit(
        [](const auto &program_config) -> bool {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (not std::is_same_v<ProgramConfigType, MatmulDefaultProgramConfig>) {
                return false;
            }
            return true;
        },
        matmul_program_config);
    return result;
}

MatmulProgramConfig create_matmul_1d_systolic_array_program_config(
    const ttnn::types::Shape &input_shape_a,
    const ttnn::types::Shape &input_shape_b,
    const CoreCoord &core_coord,
    const std::optional<const UnaryWithParam> fused_activation,
    const bool fp32_dest_acc_en);

MatmulProgramConfig create_matmul_program_config(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const CoreCoord> user_core_coord,
    std::optional<UnaryWithParam> fused_activation,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace primary

}  // namespace operations

}  // namespace tt

namespace bmm_op_utils {
using namespace tt::tt_metal;

// Ensure there are always symmetrical values. Different paths use different
// index ordering (0,1 vs 1,0) to meet test PCC requirements.
constexpr std::array<tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
}};

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(
    uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w);

CoreCoord get_core_range(
    uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

inline bool get_fp32_dest_acc_en(const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    bool fp32_dest_acc_en = false;
    if (compute_kernel_config) {
        std::visit(
            [&](auto &&compute_kernel_config) {
                using T = std::decay_t<decltype(compute_kernel_config)>;
                if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                    fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
                }
            },
            *compute_kernel_config);
    }
    return fp32_dest_acc_en;
}

// TODO: Remove get_mcast_1d_config and merge with general version?
tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig get_mcast_1d_config(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    bool fuse_batch = false,
    std::optional<UnaryWithParam> fused_activation = std::nullopt,
    bool mcast_in0 = true,
    bool out_sharded = false,
    std::optional<CoreCoord> compute_with_storage_grid_size = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

tuple<uint32_t, uint32_t> get_matmul_subblock_params(
    const uint32_t per_core_M,
    const uint32_t per_core_N,
    const bool per_core_M_equals_subblock_h_constraint,
    bool per_core_N_equals_subblock_w_constraint,
    bool fp32_dest_acc_en);

// TODO: Review usage of matmul bool; should probably infer this from batch
tt::operations::primary::MatmulProgramConfig get_matmul_program_config(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const MemoryConfig &output_mem_config,
    std::optional<UnaryWithParam> fused_activation = std::nullopt,
    const bool matmul = false,
    const std::optional<const CoreCoord> user_core_coord = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
}  // namespace bmm_op_utils

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

inline Tensor matmul(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias = std::nullopt,
    const MatmulProgramConfig &program_config = MatmulDefaultProgramConfig{},
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool untilize_out = false,
    std::optional<const CoreCoord> user_core_coord = std::nullopt,
    std::optional<UnaryWithParam> user_fused_activation = std::nullopt,
    std::optional<const bool> input_b_is_batched = std::nullopt,
    const bool needs_autoformat = false) {
    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<Tensor> output_tensors;
    if (bias) {
        optional_input_tensors.push_back(bias);
        output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}, {bias}))};
    } else {
        optional_input_tensors.push_back(std::nullopt);
        output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}))};
    }

    if (!needs_autoformat) {
        operation::launch_op(
            [program_config,
             mem_config,
             output_dtype,
             compute_kernel_config,
             untilize_out,
             user_core_coord,
             user_fused_activation,
             input_b_is_batched](
                const std::vector<Tensor> &input_tensors,
                const std::vector<std::optional<const Tensor>> &optional_input_tensors,
                const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto &input_tensor_a = input_tensors.at(0);
                const auto &input_tensor_b = input_tensors.at(1);
                auto arch = input_tensor_a.device()->arch();
                const auto increase_fidelity =
                    is_program_config_default(program_config) && !user_core_coord.has_value();
                auto math_fidelity = increase_fidelity ? MathFidelity::HiFi2 : MathFidelity::LoFi;
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, math_fidelity);
                bool broadcast_batch = get_broadcast_batch(input_tensor_a, input_tensor_b, program_config);
                auto matmul_program_config = program_config;
                if (user_core_coord.has_value()) {
                    TT_FATAL(
                        is_program_config_default(program_config),
                        "Cannot use both user core grid/coordinates and a program config");
                    matmul_program_config = create_matmul_program_config(
                        input_tensor_a, input_tensor_b, user_core_coord, user_fused_activation, compute_kernel_config);
                }
                if (is_program_config_default(matmul_program_config) && input_tensor_a.is_sharded()) {
                    bool bmm = input_b_is_batched.value_or(false);
                    matmul_program_config = bmm_op_utils::get_matmul_program_config(
                        input_tensor_a,
                        input_tensor_b,
                        mem_config,
                        std::nullopt,
                        !bmm,
                        user_core_coord,
                        compute_kernel_config);
                }
                return operation::run(
                    Matmul{
                        matmul_program_config,
                        broadcast_batch,
                        mem_config,
                        output_dtype.value_or(input_tensor_a.get_dtype()),
                        kernel_config_val,
                        untilize_out},
                    {input_tensor_a, input_tensor_b},
                    optional_input_tensors);
            },
            {input_tensor_a, input_tensor_b},
            output_tensors,
            optional_input_tensors);
    } else {
        operation::launch_with_autoformat(
            [program_config,
             mem_config,
             output_dtype,
             compute_kernel_config,
             untilize_out,
             user_core_coord,
             input_b_is_batched](
                const std::vector<Tensor> &input_tensors,
                const std::vector<std::optional<const Tensor>> &optional_input_tensors,
                const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto &input_tensor_a = input_tensors.at(0);
                const auto &input_tensor_b = input_tensors.at(1);
                auto arch = input_tensor_a.storage_type() == StorageType::DEVICE
                                ? input_tensor_a.device()->arch()
                                : AutoFormat::GetDefaultDevice()->arch();
                const auto program_config_default = is_program_config_default(program_config);
                auto math_fidelity = program_config_default ? MathFidelity::HiFi2 : MathFidelity::LoFi;
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, math_fidelity);
                bool broadcast_batch = get_broadcast_batch(input_tensor_a, input_tensor_b, program_config);
                return operation::run_with_autoformat(
                    Matmul{
                        program_config,
                        broadcast_batch,
                        mem_config,
                        output_dtype.value_or(input_tensor_a.get_dtype()),
                        kernel_config_val,
                        untilize_out},
                    {input_tensor_a, input_tensor_b},
                    optional_input_tensors);
            },
            {input_tensor_a, input_tensor_b},
            output_tensors,
            optional_input_tensors);
    }
    return output_tensors.at(0);
}

Tensor matmul_1d(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<const Tensor> bias,
    std::optional<MatmulMultiCoreReuseMultiCast1DProgramConfig> program_config = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool untilize_out = false);

}  // namespace primary

}  // namespace operations

}  // namespace tt
