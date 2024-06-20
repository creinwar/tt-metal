// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_stl/reflection.hpp"
#include "tt_eager/tt_dnn/op_library/conv/multi_core_optimized_conv_sharded/optimized_conv_op_width_sharded.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

tuple<CBHandle, CBHandle> create_CBs_for_sharded_input_v2_width(
    tt_metal::Program& program,
    const Tensor& input,
    CoreRange core,
    uint32_t num_cb0_tiles,
    uint32_t num_cb1_tiles,
    uint32_t num_cb0_tilized_tiles,
    uint32_t num_output_tiles,
    uint32_t num_reblock_cb_tiles,
    uint32_t num_writer_output_tiles,
    bool untilize_out,
    DataFormat act_df,
    DataFormat weight_df,
    DataFormat tilized_act_df,
    DataFormat out_df,
    DataFormat bias_df,
    bool weight_width_sliced,
    const Tensor& output,
    uint32_t bias_ntiles,
    bool with_bias,
    bool split_reader,
    bool fp32_dest_acc_en,
    bool packer_l1_acc_en) {
    std::cout << "CB creation stage" << std::endl;
    tt::DataFormat interm0_df =
        packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b) : out_df;

    uint32_t act_tile_size = tt_metal::detail::TileSize(act_df);
    uint32_t weight_tile_size = tt_metal::detail::TileSize(weight_df);
    uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);
    uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_df);

    CBHandle cb_sharded_act = 0;
    if (input.memory_config().is_sharded()) {
        uint32_t num_bytes_for_df = datum_size(act_df);
        auto shard_shape = input.shard_spec().value().shape;
        // 2D-sys-conv already has uint16_t indicies, TODO: do the same for 1D-sys-conv
        TT_ASSERT(
            shard_shape[0] <= (1 << 16), "Shard height must be less than 2^16, read pattern indicies are uint16_t");
        CircularBufferConfig cb_sharded_act_config =
            CircularBufferConfig(shard_shape[0] * shard_shape[1] * num_bytes_for_df, {{sharded_act_cb, act_df}})
                .set_page_size(sharded_act_cb, shard_shape[1] * num_bytes_for_df);
        // incoming data is the input cb instead of raw l1/dram addr
        cb_sharded_act_config.set_globally_allocated_address(*input.buffer());
        cb_sharded_act = tt_metal::CreateCircularBuffer(program, core, cb_sharded_act_config);

        if (weight_width_sliced) {
            // For 2D convs, each core creates and tilizes full input matrix then mcasts round robin style
            // Each core receives input into act_cb, so won't need a separate cb to receive
            // However, we need a separate cb to push ROW_MAJOR BFLOAT16 data for tilizing and configure act cb to be
            // output df

            // num_cb0_tiles is double buffered
            CircularBufferConfig cb_act_config =
                CircularBufferConfig(num_cb0_tiles * tilized_act_tile_size, {{act_cb, tilized_act_df}})
                    .set_page_size(act_cb, tilized_act_tile_size);
            auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);

            // num_cb0_tilized_tiles is single buffered
            CircularBufferConfig cb_act_row_major_bfloat16_config =
                CircularBufferConfig(num_cb0_tilized_tiles * act_tile_size, {{act_cb_row_major_bfloat16, act_df}})
                    .set_page_size(act_cb_row_major_bfloat16, act_tile_size);
            auto cb_act_row_major_bfloat16 =
                tt_metal::CreateCircularBuffer(program, core, cb_act_row_major_bfloat16_config);
        } else {
            // For 1D convs, locally create act matrix in act_cb, which is always ROW_MAJOR BFLOAT16
            // Then, tilize input in compute

            // Extra cb for second reader if we split act reads across two RISCs
            // In this case, the regular reader only does first half of reads along output block h
            if (split_reader) {
                num_cb0_tiles /= 2;

                CircularBufferConfig cb_act_config =
                    CircularBufferConfig(num_cb0_tiles * act_tile_size, {{act_cb_second_reader, act_df}})
                        .set_page_size(act_cb_second_reader, act_tile_size);
                auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);
            }

            CircularBufferConfig cb_act_config = CircularBufferConfig(num_cb0_tiles * act_tile_size, {{act_cb, act_df}})
                                                     .set_page_size(act_cb, act_tile_size);
            auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);
        }
    } else {
        TT_ASSERT(false, "Input must be sharded!");
    }

    CircularBufferConfig cb_weight_config =
        CircularBufferConfig(num_cb1_tiles * weight_tile_size, {{weight_cb, weight_df}})
            .set_page_size(weight_cb, weight_tile_size);
    auto cb_weight = tt_metal::CreateCircularBuffer(program, core, cb_weight_config);

    // Used for placing tilized activations
    CircularBufferConfig cb_src0_tilized_config =
        CircularBufferConfig(
            num_cb0_tilized_tiles * tilized_act_tile_size, {{tilize_mode_tilized_act_cb, tilized_act_df}})
            .set_page_size(tilize_mode_tilized_act_cb, tilized_act_tile_size);
    auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

    CBHandle cb_output = 0;
    if (untilize_out) {
        CircularBufferConfig cb_matmul_partials_config =
            CircularBufferConfig(num_output_tiles * interm0_single_tile_size, {{matmul_partials_cb, interm0_df}})
                .set_page_size(matmul_partials_cb, interm0_single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        CircularBufferConfig cb_reblock_config =
            CircularBufferConfig(num_reblock_cb_tiles * out_tile_size, {{untilize_mode_reblock_cb, out_df}})
                .set_page_size(untilize_mode_reblock_cb, out_tile_size);
        auto cb_reblock = tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_writer_output_tiles * out_tile_size, {{out0_cb, out_df}})
                .set_page_size(out0_cb, out_tile_size);
        if (output.is_sharded()) {
            cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
        }
        cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    } else {
        // Share buffer if same data format
        if (interm0_df == out_df) {
            CoreRangeSet cores(std::set<CoreRange>({core}));
            std::map<uint8_t, tt::DataFormat> cb_output_data_format_spec = {
                {out0_cb, out_df}, {matmul_partials_cb, out_df}};
            CircularBufferConfig cb_matmul_partials_config =
                CircularBufferConfig(num_output_tiles * out_tile_size, cb_output_data_format_spec)
                    .set_page_size(out0_cb, out_tile_size)
                    .set_page_size(matmul_partials_cb, out_tile_size);
            if (output.is_sharded()) {
                cb_matmul_partials_config = cb_matmul_partials_config.set_globally_allocated_address(*output.buffer());
            }
            cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_matmul_partials_config);
        } else {
            // Separate buffer if not same data format
            CircularBufferConfig cb_matmul_partials_config =
                CircularBufferConfig(num_output_tiles * interm0_single_tile_size, {{matmul_partials_cb, interm0_df}})
                    .set_page_size(matmul_partials_cb, interm0_single_tile_size);
            auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

            CircularBufferConfig cb_output_config =
                CircularBufferConfig(num_output_tiles * out_tile_size, {{out0_cb, out_df}})
                    .set_page_size(out0_cb, out_tile_size);
            if (output.is_sharded()) {
                cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
            }
            cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
        }
    }

    if (with_bias) {
        uint32_t bias_tile_size = tt_metal::detail::TileSize(bias_df);
        // bias input
        uint32_t bias_pagesize = bias_tile_size;
        CircularBufferConfig cb_bias_config = CircularBufferConfig(bias_ntiles * bias_pagesize, {{bias_cb, bias_df}})
                                                  .set_page_size(bias_cb, bias_pagesize);
        auto cb_bias = tt_metal::CreateCircularBuffer(program, core, cb_bias_config);

        log_debug(LogOp, "Bias CB: {}, npages: {}, pagesize: {}", bias_cb, bias_ntiles, bias_pagesize);
    }

    return {cb_sharded_act, cb_output};
}

operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_impl_width_sharded(
    tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const Shape& ashape,
    std::optional<const Tensor> bias,
    const std::optional<const Tensor> conv_reader_indices,
    vector<int> conv_params,
    uint32_t output_channels,
    bool untilize_out,
    bool has_bias,
    bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    uint32_t extra_padding_for_32B_alignment,
    bool use_shallow_conv_variant,
    bool transpose_mcast,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config){
         std::cout << "testing multi_core_optimized_conv_sharded_v2_impl 1" << std::endl;

    auto null_override_runtime_arguments_callback =
    [](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors) {
    };
    bool pass = true;
     tt_metal::Device* device = a.device();
    TT_ASSERT(a.get_layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_ASSERT(a.memory_config().is_sharded(), "Conv activation must be sharded.");
    TT_ASSERT(output_channels <= b.get_legacy_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    uint32_t weight_block_w_ntiles = parallelization_config.per_core_out_matrix_width_ntiles;
    uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntiles;
    uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;

    DataFormat act_df = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    DataFormat weight_df = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    DataFormat bias_df =
        has_bias ? tt_metal::datatype_to_dataformat_converter(bias.value().get_dtype()) : DataFormat::Float16_b;
    DataFormat tilized_act_df = out_df;

    uint32_t act_tile_size = tt_metal::detail::TileSize(act_df);
    uint32_t weight_tile_size = tt_metal::detail::TileSize(weight_df);
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);

    std::cout << "testing multi_core_optimized_conv_sharded_v2_impl 2" << std::endl;
    log_debug(LogOp, "act_df: {}", act_df);
    log_debug(LogOp, "weight_df: {}", weight_df);
    log_debug(LogOp, "out_df: {}", out_df);
    log_debug(LogOp, "bias_df: {}", bias_df);

    // compute kernel config
    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    bool packer_l1_acc;

    std::cout << "testing multi_core_optimized_conv_sharded_v2_impl 3" << std::endl;
    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = false;
                packer_l1_acc = false;
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
                packer_l1_acc = compute_kernel_config.packer_l1_acc;
            } else {
                TT_FATAL("arch not supported");
            }
        },
        compute_kernel_config);

    if (fp32_dest_acc_en and (out_subblock_h_ntiles * out_subblock_w_ntiles > 4)) {
        if (out_subblock_w_ntiles >= 4) {
            out_subblock_h_ntiles = 1;
            out_subblock_w_ntiles = find_max_block_size(out_subblock_w_ntiles, 4);
        } else {
            while (out_subblock_h_ntiles * out_subblock_w_ntiles > 4) {
                uint32_t div = find_max_divisor(out_subblock_h_ntiles, out_subblock_h_ntiles - 1);
                out_subblock_h_ntiles = find_max_block_size(out_subblock_h_ntiles, div);
            }
        }
    }
    // assert(out_block_h_ntiles == act_block_h_ntiles); // TODO: fix output block sizing
    TT_ASSERT(
        out_block_h_ntiles >= act_block_h_ntiles,
        "Output block height (in # of tiles) should be greater than or equal to activation block height (in # of "
        "tiles)");

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
    TT_ASSERT(b.get_layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_ASSERT(b.get_legacy_shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_ASSERT(b.get_legacy_shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.get_legacy_shape()[2];
    uint32_t weight_matrix_width = b.get_legacy_shape()[3];
    uint32_t weight_matrix_height_ntiles = weight_matrix_height / TILE_HEIGHT;
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;

    // Partitions conv inner dim into blocks to support sharding along this dim
    // TODO: Only 2D convs with sharded input use this, but we can uplift to support generically
    // TODO: Only updated variables which is affected, but there may be more that needs to account for this
    // TODO: Loop naming in reader, writer, and compute kernels could also be cleaned up
    // TODO: Can conv_act_c_blocks be same as num_blocks_act_w?
    auto shard_shape = a.shard_spec().value().shape;

    // parallelization config
    const auto& p_config = parallelization_config;
    uint32_t num_cores_x = p_config.grid_size.x;
    uint32_t num_cores_y = p_config.grid_size.y;
    uint32_t total_num_cores = num_cores_x * num_cores_y;
    assert(num_cores_x < 13);
    assert(num_cores_y < 10);
    uint32_t per_core_out_matrix_height_ntiles = p_config.per_core_out_matrix_height_ntiles;
    uint32_t per_core_out_matrix_width_ntiles = p_config.per_core_out_matrix_width_ntiles;

    // weight_width_sliced determines is 1d-sysarr-conv or 2d-sysarr-conv
    bool weight_width_sliced = per_core_out_matrix_width_ntiles < weight_matrix_width_ntiles;
    //weight_width_sliced = false;
    uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    uint32_t input_channels_padded = 0;
    if (weight_width_sliced && a.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED) {
        if (transpose_mcast) {
            TT_FATAL(conv_act_c_blocks == num_cores_y, "Expected conv_act_c_blocks to be equal to height of grid");
            input_channels_padded = shard_shape[1] * num_cores_y;
        } else {
            TT_FATAL(conv_act_c_blocks == num_cores_x, "Expected conv_act_c_blocks to be equal to width of grid");
            input_channels_padded = shard_shape[1] * num_cores_x;
        }
    } else {
        input_channels_padded = shard_shape[1] ;
    }
    if(a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED){
        input_channels_padded = shard_shape[1] * p_config.num_cores_c ;
    }
    TT_FATAL(input_channels_padded >= ashape[3], "Incorrect padding of input channels!");
    // check is for 16-byte alignment
    TT_FATAL(
        input_channels_padded % 16 == 0,
        "Expected input channels to be padded for 16 byte alignment in L1");  // TODO: For bfp16, check if its divisible
                                                                              // by 8 not 16.
    // Always use split reader for first conv in resnet which has input channels = 16
    // TODO: Expose option to split readers for 1D convs to python?
    bool split_reader = use_shallow_conv_variant;
    if (split_reader) {
        TT_FATAL(
            block_config.act_block_h_ntiles % block_config.out_subblock_h_ntiles == 0,
            "Out_block_h must be divisible by out_subblock_h!");
        TT_FATAL(
            (block_config.act_block_h_ntiles / block_config.out_subblock_h_ntiles) % 2 == 0,
            "Number of out_subblock_h must be divisible by 2 for split reader!");
    }
    Shape ashape_with_channels_padded = {ashape[0], ashape[1], ashape[2], input_channels_padded};
    uint32_t conv_act_size_h = ashape_with_channels_padded[1];
    uint32_t conv_act_size_w = ashape_with_channels_padded[2];
    uint32_t conv_act_size_c = ashape_with_channels_padded[3];
    uint32_t weight_size_h = (uint32_t)conv_params[0];  // filter_h
    uint32_t weight_size_w = (uint32_t)conv_params[1];  // filter_W
    uint32_t stride_h = (uint32_t)conv_params[2];
    uint32_t stride_w = (uint32_t)conv_params[3];
    uint32_t pad_h = (uint32_t)conv_params[4];
    uint32_t pad_w = (uint32_t)conv_params[5];

    // Compute the 2d matrix shape
    auto [act_matrix_shape, act_matrix_shape_unpadded] =
        optimized_conv_op_utils::compute_opt_conv_activation_as_mm_shape(
            ashape_with_channels_padded, conv_params, out_block_h_ntiles, extra_padding_for_32B_alignment);
    assert(act_matrix_shape.size() == 3);
    assert(act_matrix_shape[0] == 1);
    uint32_t act_matrix_height = (uint32_t)act_matrix_shape[1];
    uint32_t act_matrix_width = (uint32_t)act_matrix_shape[2];
    uint32_t act_matrix_height_unpadded = (uint32_t)act_matrix_shape_unpadded[1];
    uint32_t act_matrix_width_unpadded = (uint32_t)act_matrix_shape_unpadded[2];

    // TODO: Move all these asserts/checks to validate?

    if (has_bias) {
        // Tensor bias is of shape {output_channels}
        TT_ASSERT(bias.has_value());
        TT_ASSERT(bias.value().buffer() != nullptr);
        auto bias_shape_without_padding = bias.value().get_legacy_shape().without_padding();
        TT_ASSERT(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
    }

    // Normal matrix shape check
    std::cout << "act_matrix_width: " << act_matrix_width << " weight_matrix_height " << weight_matrix_height << std::endl;
    TT_ASSERT(act_matrix_width == weight_matrix_height, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(act_matrix_height % TILE_HEIGHT == 0, "Height of activation matrix needs to be divisible by 32");
    TT_ASSERT(act_matrix_width % TILE_WIDTH == 0, "Width of activation matrix needs to be divisible by 32");
    TT_ASSERT(weight_matrix_height % TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_ASSERT(weight_matrix_width % TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(
        a.storage_type() == StorageType::DEVICE && b.storage_type() == StorageType::DEVICE &&
        "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_ASSERT(
        a.buffer() != nullptr && b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
    if (has_bias) {
        TT_ASSERT(bias.value().storage_type() == StorageType::DEVICE, "Bias should be on device");
        TT_ASSERT(bias.value().device() == a.device(), "Bias should be on the same device as act tensor");
    }

    // Convert tensor dims to tile dims
    uint32_t act_matrix_height_ntiles = act_matrix_height / TILE_HEIGHT;
    uint32_t act_matrix_width_ntiles = act_matrix_width / TILE_WIDTH;

    assert(act_matrix_height_ntiles % act_block_h_ntiles == 0);
    assert(act_matrix_width_ntiles % act_block_w_ntiles == 0);
    assert(weight_matrix_width_ntiles % weight_block_w_ntiles == 0);
    assert(act_matrix_height_ntiles % out_block_h_ntiles == 0);

    uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t num_blocks_out_h = act_matrix_height_ntiles / out_block_h_ntiles;
    uint32_t num_blocks_act_w = act_matrix_width_ntiles / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_w_datums = act_matrix_width / num_blocks_act_w;
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;
    TT_ASSERT(
        (act_block_w_datums == round_up(conv_act_size_c * weight_size_w, TILE_WIDTH)) ||
        ((act_block_w_datums <= conv_act_size_c) && (conv_act_size_c % act_block_w_datums == 0)));

    // weight block info
    uint32_t weight_block_w_datums = weight_matrix_width / num_blocks_weight_w;
    assert(weight_block_w_ntiles % out_subblock_w_ntiles == 0);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;

    uint32_t num_groups = num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w;
    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width = round_up(output_channels, TILE_WIDTH);
    assert(output_channels_padded_to_tile_width <= weight_matrix_width);
    uint32_t output_width_num_tiles = output_channels_padded_to_tile_width / TILE_WIDTH;
    uint32_t num_blocks_output_w =
        (uint32_t)ceil((double)output_channels_padded_to_tile_width / (double)weight_block_w_datums);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0)
                                           ? weight_block_w_datums
                                           : (output_channels_padded_to_tile_width % weight_block_w_datums);
    assert(last_block_width_datums % TILE_WIDTH == 0);

    // sanity check
    assert(num_blocks_output_w == num_blocks_weight_w);

    uint32_t out_block_h_datums = out_block_h_ntiles * TILE_HEIGHT;

    tt_metal::Buffer* src0_dram_buffer = a.buffer();
    tt_metal::Buffer* src1_dram_buffer = b.buffer();

    tt_metal::Buffer* dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    // act
    uint32_t act_dram_addr = src0_dram_buffer->address();
    auto act_dram_noc_xy = src0_dram_buffer->noc_coordinates();
    uint32_t act_noc_x = act_dram_noc_xy.x;
    uint32_t act_noc_y = act_dram_noc_xy.y;

    assert(act_matrix_width_ntiles % act_block_w_ntiles == 0);
    assert(act_block_h_ntiles % out_subblock_h_ntiles == 0);
    assert(out_block_h_ntiles % out_subblock_h_ntiles == 0);
    uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    // weight
    uint32_t weight_dram_addr = src1_dram_buffer->address();

    // bias
    Buffer* bias_buffer = nullptr;
    uint32_t bias_dram_addr = 0;
    uint32_t bias_ntiles = 0;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_dram_addr = bias_buffer->address();
        bias_ntiles =
            bias.value().get_legacy_shape()[3] / constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
    }

    auto [conv_output_size_h, conv_output_size_w] = optimized_conv_op_utils::compute_opt_conv_output_face_shape(
        conv_act_size_h,
        conv_act_size_w,
        weight_size_h,
        weight_size_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        extra_padding_for_32B_alignment);

    std::map<string, string> reader_defines;

    if (act_matrix_height_unpadded < act_matrix_height) {
        reader_defines["ACT_BLOCK_HEIGHT_PADDING"] = "1";
    }

    if (conv_act_c_blocks > 1) {
        reader_defines["ACT_W_OUTER_BLOCKS"] = "1";
    }

    uint32_t output_height_padded_to_tile_height = round_up(act_matrix_height_unpadded, TILE_HEIGHT);
    uint32_t output_height_num_tiles = output_height_padded_to_tile_height / TILE_HEIGHT;
    assert(output_height_num_tiles <= act_matrix_height_ntiles);

    uint32_t src_dram_act_buffer_size_bytes = src0_dram_buffer->size();
    uint32_t src_dram_weight_buffer_size_bytes = src1_dram_buffer->size();
    uint32_t dst_l1_act_buffer_size_bytes =
        out_block_h_ntiles * act_block_w_ntiles * tt::tt_metal::detail::TileSize(act_df);
    uint32_t dst_l1_weight_buffer_size_bytes =
        weight_block_h_ntiles * weight_block_w_ntiles * tt::tt_metal::detail::TileSize(weight_df);

    // For debug
    {
        log_debug(LogOp, "multi_core_optimized_conv_sharded_v2_");
        log_debug(LogOp, "num_cores_x: {}", num_cores_x);
        log_debug(LogOp, "num_cores_y: {}", num_cores_y);
        log_debug(LogOp, "conv_act_size_h: {}", conv_act_size_h);
        log_debug(LogOp, "conv_act_size_w: {}", conv_act_size_w);
        log_debug(LogOp, "act_matrix_height: {}", act_matrix_height);
        log_debug(LogOp, "act_matrix_width: {}", act_matrix_width);
        log_debug(LogOp, "act_matrix_height_unpadded: {}", act_matrix_height_unpadded);
        log_debug(LogOp, "act_matrix_width_unpadded: {}", act_matrix_width_unpadded);
        log_debug(LogOp, "act_matrix_height_ntiles: {}", act_matrix_height_ntiles);
        log_debug(LogOp, "act_matrix_width_ntiles: {}", act_matrix_width_ntiles);
        log_debug(LogOp, "weight_matrix_width_ntiles: {}", weight_matrix_width_ntiles);
        log_debug(LogOp, "per_core_out_matrix_height_ntiles: {}", per_core_out_matrix_height_ntiles);
        log_debug(LogOp, "per_core_out_matrix_width_ntiles: {}", per_core_out_matrix_width_ntiles);
        log_debug(LogOp, "num_blocks_act_h: {}", num_blocks_act_h);
        log_debug(LogOp, "num_blocks_act_w: {}", num_blocks_act_w);
        log_debug(LogOp, "num_blocks_weight_w: {}", num_blocks_weight_w);
        log_debug(LogOp, "num_blocks_out_h: {}", num_blocks_out_h);
        log_debug(LogOp, "act_dram_addr: {}", act_dram_addr);
        log_debug(LogOp, "act_block_h_ntiles: {}", act_block_h_ntiles);
        log_debug(LogOp, "act_block_h_datums: {}", act_block_h_datums);
        log_debug(LogOp, "act_block_w_ntiles: {}", act_block_w_ntiles);
        log_debug(LogOp, "act_block_w_datums: {}", act_block_w_datums);
        log_debug(LogOp, "out_block_h_ntiles: {}", out_block_h_ntiles);
        log_debug(LogOp, "act_num_subblocks: {}", act_num_subblocks);
        log_debug(LogOp, "act_block_num_tiles: {}", act_block_num_tiles);
        log_debug(LogOp, "act_subblock_h_ntiles: {}", act_subblock_h_ntiles);
        log_debug(LogOp, "act_subblock_num_tiles: {}", act_subblock_num_tiles);
        log_debug(LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(LogOp, "weight_dram_addr: {}", weight_dram_addr);
        log_debug(LogOp, "weight_num_subblocks: {}", weight_num_subblocks);
        log_debug(LogOp, "weight_block_num_tiles: {}", weight_block_num_tiles);
        log_debug(LogOp, "weight_block_w_ntiles: {}", weight_block_w_ntiles);
        log_debug(LogOp, "weight_block_h_ntiles: {}", weight_block_h_ntiles);
        log_debug(LogOp, "has_bias: {}", has_bias);
        log_debug(LogOp, "bias_dram_addr: {}", bias_dram_addr);
        log_debug(LogOp, "bias_ntiles: {}", bias_ntiles);
        log_debug(LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(LogOp, "num_groups: {}", num_groups);
        log_debug(LogOp, "math_fidelity: {}", math_fidelity);
        log_debug(LogOp, "math_approx_mode: {}", math_approx_mode);
        log_debug(LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
        log_debug(LogOp, "packer_l1_acc: {}", packer_l1_acc);
        log_debug(LogOp, "total_active_cores {}", p_config.num_cores_c);

    }


    uint32_t total_active_num_cores = p_config.num_cores_c;

    //Assuming only a single row of cores.
    std::set<CoreRange> all_active_cores_set;
    all_active_cores_set.insert(
        CoreRange(CoreCoord(0, 0), CoreCoord(total_active_num_cores - 1, 0)));

    CoreRangeSet all_active_cores(all_active_cores_set);

    std::cout<<"Active Cores "<<all_active_cores.str()<<std::endl;

    std::vector<uint32_t> weights_reader_compile_args;

    weights_reader_compile_args={
        weight_cb,
        act_matrix_width_ntiles, //Height of Weights Matrix in Tiles
        per_core_out_matrix_width_ntiles,  //Width of Weights Matrix fetched by this core in Tiles
        weight_matrix_width_ntiles, //Width of the Full Weights Matrix
    };

    CircularBufferConfig cb_weight_config =
        CircularBufferConfig(act_matrix_width_ntiles * per_core_out_matrix_width_ntiles * weight_tile_size, {{weight_cb, weight_df}})
            .set_page_size(weight_cb, weight_tile_size);
    auto cb_weight = tt_metal::CreateCircularBuffer(program, all_active_cores, cb_weight_config);

    const string weights_reader_kernel_path = "tt_eager/tt_dnn/op_library/conv/kernels/width_sharded_weights_reader.cpp";
    auto weights_reader_kernel_id = CreateKernel(
        program,
        weights_reader_kernel_path,
        all_active_cores,
        DataMovementConfig{
            .processor=DataMovementProcessor::RISCV_1,
            .noc=NOC::NOC_1,
            .compile_args=weights_reader_compile_args
        }
    );

    for(uint32_t core_index = 0; core_index <total_active_num_cores; core_index++)
    {
        std::vector<uint32_t> weights_reader_runtime_args;
        uint32_t core_x = core_index%num_cores_x;
        uint32_t core_y = core_index/num_cores_x;
        weights_reader_runtime_args = {
            weight_dram_addr,
            core_x,
            core_y,
            core_index
        };
        SetRuntimeArgs(program, weights_reader_kernel_id, CoreCoord(core_x,core_y), weights_reader_runtime_args);

    }







    // auto mcast_sender_cores_vec = grid_to_cores(mcast_sender_cores.start, mcast_sender_cores.end, true);
    // auto mcast_receiver_cores_vec = corerange_to_cores(mcast_receiver_cores, std::nullopt, true);
    // auto override_runtime_arguments_callback =
    //     [reader_kernel_id = reader_id,
    //      mcast_sender_cores = mcast_sender_cores_vec,
    //      writer_mcast_sender_id = writer_mcast_sender_id,
    //      mcast_receiver_cores = mcast_receiver_cores_vec,
    //      writer_mcast_receiver_id = writer_mcast_receiver_id,
    //      cb_sharded_act = cb_sharded_act,
    //      cb_output = cb_output,
    //      total_active_num_cores = total_active_num_cores,
    //      num_cores_x = num_cores_x,
    //      num_cores_y = num_cores_y,
    //      has_bias = has_bias](
    //         const void* operation,
    //         Program& program,
    //         const std::vector<Tensor>& input_tensors,
    //         const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    //         const std::vector<Tensor>& output_tensors) {
    //         // Reader config indices is an optional static sharded tensor, so no need to update address
    //         TT_ASSERT(input_tensors.size() + optional_input_tensors.size() == 4);
    //         TT_ASSERT(output_tensors.size() == 1);

    //         auto src_buffer_a = input_tensors.at(0).buffer();
    //         auto src_buffer_b = input_tensors.at(1).buffer();
    //         bool src_a_is_sharded = input_tensors[0].is_sharded();

    //         std::optional<Buffer*> src_buffer_c = std::nullopt;
    //         if (has_bias) {
    //             src_buffer_c = optional_input_tensors.at(0).value().buffer();
    //             TT_ASSERT(src_buffer_c.value() != nullptr);
    //         }

    //         auto dst_buffer = output_tensors.at(0).buffer();
    //         bool out_sharded = output_tensors[0].is_sharded();

    //         auto& reader_kernel_args_by_core = GetRuntimeArgs(program, reader_kernel_id);

    //         auto& writer_sender_kernel_args_by_core = GetRuntimeArgs(program, writer_mcast_sender_id);
    //         for (const auto& core : mcast_sender_cores) {
    //             if (!src_a_is_sharded) {
    //                 auto& runtime_args = reader_kernel_args_by_core[core.x][core.y];
    //                 runtime_args[0] = src_buffer_a->address();
    //             }
    //             auto& runtime_args = writer_sender_kernel_args_by_core[core.x][core.y];
    //             runtime_args[0] = dst_buffer->address();
    //             runtime_args[1] = src_buffer_b->address();
    //             if (has_bias) {
    //                 runtime_args[2] = (*src_buffer_c)->address();
    //             }
    //         }

    //         if (mcast_receiver_cores.size() > 0) {
    //             auto& writer_receiver_kernel_args_by_core = GetRuntimeArgs(program, writer_mcast_receiver_id);
    //             for (const auto& core : mcast_receiver_cores) {
    //                 if (!src_a_is_sharded) {
    //                     auto& runtime_args = reader_kernel_args_by_core[core.x][core.y];
    //                     runtime_args[0] = src_buffer_a->address();
    //                 }
    //                 auto& runtime_args = writer_receiver_kernel_args_by_core[core.x][core.y];
    //                 runtime_args[0] = dst_buffer->address();
    //                 runtime_args[1] = src_buffer_b->address();
    //                 if (has_bias) {
    //                     runtime_args[2] = (*src_buffer_c)->address();
    //                 }
    //             }
    //         }

    //         if (src_a_is_sharded) {
    //             UpdateDynamicCircularBufferAddress(program, cb_sharded_act, *src_buffer_a);
    //         }

    //         if (out_sharded) {
    //             UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    //         }
    //     };


    return {.program = std::move(program), .override_runtime_arguments_callback = null_override_runtime_arguments_callback};
    }
}  // namespace tt_metal

}  // namespace tt
