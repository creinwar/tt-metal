// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/transformer_tms/transformer_tms.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace tt;

namespace tt {
namespace operations {
namespace primary {
namespace transformers {

operation::ProgramWithCallbacks multi_core_ssm_eltwise_mul(
    const Tensor& a,
    const Tensor& b,
    Tensor& output,
    const uint32_t hidden_size,
    MathFidelity math_fidelity,
    CoreCoord compute_with_storage_grid_size) {
    const auto &ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

    tt_metal::Device* device = a.device();

    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::Buffer* src1_buffer = b.buffer();

    tt_metal::Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat interm_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t interm_single_tile_size = tt_metal::detail::TileSize(interm_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    // Constants
    constexpr uint32_t ONE_TILE = 1;

    // Parallelize on bshape[-1]
    auto num_output_blocks_total = bshape[-1] / TILE_WIDTH;
    const bool row_major = false;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_output_blocks_total, row_major);

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();
    std::vector<CoreCoord> cores =
        grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, row_major);

    // Create circular buffers
    uint32_t src0_cb_index = CB::c_in0;
    uint32_t cb0_tiles = ONE_TILE * 4;  // double buffer
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(cb0_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1;
    uint32_t cb1_tiles = ONE_TILE * 4;  // double buffer
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(cb1_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = 16;
    uint32_t output_cb_tiles = ONE_TILE * 4;  // double buffer
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            output_cb_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    uint32_t interm_num_tiles = ONE_TILE * 4;  // double buffer
    uint32_t interm_cb_size = interm_num_tiles * interm_single_tile_size;
    uint32_t cb_intermed0_index = CB::c_intermed0;  // cb_in0_transposed
    tt_metal::CircularBufferConfig cb_intermed0_config =
        tt_metal::CircularBufferConfig(interm_cb_size, {{cb_intermed0_index, interm_data_format}})
            .set_page_size(cb_intermed0_index, interm_single_tile_size);
    auto cb_intermed0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    uint32_t cb_intermed1_index = CB::c_intermed1;  // cb_in1_transposed
    tt_metal::CircularBufferConfig cb_intermed1_config =
        tt_metal::CircularBufferConfig(interm_cb_size, {{cb_intermed1_index, interm_data_format}})
            .set_page_size(cb_intermed1_index, interm_single_tile_size);
    auto cb_intermed1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed1_config);

    uint32_t cb_intermed2_index = CB::c_intermed2;  // cb_in1_bcast_row
    tt_metal::CircularBufferConfig cb_intermed2_config =
        tt_metal::CircularBufferConfig(interm_cb_size, {{cb_intermed2_index, interm_data_format}})
            .set_page_size(cb_intermed2_index, interm_single_tile_size);
    auto cb_intermed2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed2_config);

    uint32_t cb_intermed3_index = CB::c_intermed3;  // cb_out_transposed
    tt_metal::CircularBufferConfig cb_intermed3_config =
        tt_metal::CircularBufferConfig(interm_cb_size, {{cb_intermed3_index, interm_data_format}})
            .set_page_size(cb_intermed3_index, interm_single_tile_size);
    auto cb_intermed3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed3_config);

    // Compile time args
    bool in0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)cb_intermed1_index,
        (std::uint32_t)cb_intermed2_index,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)in1_is_dram,
    };
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)out_is_dram,
    };
    std::vector<uint32_t> compute_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)cb_intermed0_index,
        (std::uint32_t)cb_intermed1_index,
        (std::uint32_t)cb_intermed2_index,
        (std::uint32_t)cb_intermed3_index,
    };

    std::map<string, string> ssm_eltwise_defines;
    if (ashape[-1] == TILE_WIDTH) {
        ssm_eltwise_defines["REPEAT_IN0"] = "1";
    }
    if (bshape[-1] == hidden_size) {
        ssm_eltwise_defines["REPEAT_INTERLEAVE_IN1"] = "1";
    }

    // Load kernels
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/transformer_tms/kernels/dataflow/reader_ssm_eltwise_mul.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, ssm_eltwise_defines));

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/transformer_tms/kernels/dataflow/writer_ssm_eltwise_mul.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/transformer_tms/kernels/compute/ssm_eltwise_mul.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_args,
            .defines = ssm_eltwise_defines});

    // Update runtime args function
    auto set_runtime_args = [reader_kernel_id,
                             writer_kernel_id,
                             compute_kernel_id,
                             compute_with_storage_grid_size,
                             all_cores,

                             // Args to iterate across cores
                             cores,
                             num_cores,
                             g1_numcores,
                             g2_numcores,
                             num_blocks_per_core_group_1,
                             num_blocks_per_core_group_2,
                             bshape,
                             hidden_size](Program& program, const Tensor& a, const Tensor& b, const Tensor& output) {
        tt_metal::Buffer* src0_buffer = a.buffer();
        tt_metal::Buffer* src1_buffer = b.buffer();
        tt_metal::Buffer* dst_buffer = output.buffer();

        // Default reader runtime args
        std::vector<uint32_t> reader_runtime_args = {
            0,
            0,
            0,
            0,
        };

        // Default writer runtime args
        std::vector<uint32_t> writer_runtime_args = {
            0,
            0,
            0,
            0,
        };

        // Default compute runtime args
        std::vector<uint32_t> compute_runtime_args = {
            0,
        };

        std::vector<std::vector<uint32_t>> all_reader_runtime_args = {cores.size(), reader_runtime_args};
        std::vector<std::vector<uint32_t>> all_writer_runtime_args = {cores.size(), writer_runtime_args};
        std::vector<std::vector<uint32_t>> all_compute_runtime_args = {cores.size(), compute_runtime_args};

        // Set runtime args
        uint32_t num_blocks_per_core;
        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
            const CoreCoord& core = cores.at(i);

            if (i < g1_numcores) {
                num_blocks_per_core = num_blocks_per_core_group_1;
            } else {
                num_blocks_per_core = num_blocks_per_core_group_2;
            }

            // Update core dependent runtime args
            all_reader_runtime_args[i][0] = src0_buffer->address();
            all_reader_runtime_args[i][1] = src1_buffer->address();
            all_reader_runtime_args[i][2] = num_blocks_per_core;
            all_reader_runtime_args[i][3] = num_blocks_written;

            all_writer_runtime_args[i][0] = dst_buffer->address();

            // update writer's num_tiles based on input_b already repeat_interleaved or not
            if (bshape[-1] == hidden_size) {
                all_writer_runtime_args[i][1] = num_blocks_per_core * TILE_WIDTH;
                all_writer_runtime_args[i][2] = num_blocks_written * TILE_WIDTH;
            } else {
                all_writer_runtime_args[i][1] = num_blocks_per_core;
                all_writer_runtime_args[i][2] = num_blocks_written;
            }

            all_compute_runtime_args[i][0] = num_blocks_per_core;

            num_blocks_written += num_blocks_per_core;
        }

        SetRuntimeArgs(program, reader_kernel_id, cores, all_reader_runtime_args);
        SetRuntimeArgs(program, writer_kernel_id, cores, all_writer_runtime_args);
        SetRuntimeArgs(program, compute_kernel_id, cores, all_compute_runtime_args);
    };

    set_runtime_args(program, a, b, output);

    auto override_runtime_arguments_callback = [set_runtime_args](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& output_tensor = output_tensors.at(0);

        set_runtime_args(program, input_tensors.at(0), input_tensors.at(1), output_tensor);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
