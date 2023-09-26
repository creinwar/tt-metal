// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/move/move_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

std::vector<CoreRange> get_multicast_regions(const Device *device, const CoreRangeSet &all_cores, const CoreCoord &logical_controller) {
    TT_ASSERT(0 < all_cores.ranges().size() <= 2);
    CoreCoord logical_zero = {.x = 0, .y = 0};
    TT_ASSERT(logical_controller == logical_zero);

    std::vector<CoreRange> logical_core_ranges;
    auto split_core_range_containing_controller = [&](const CoreRange &controller_core_range) {
        TT_ASSERT(controller_core_range.start == logical_controller);
        CoreRange right_block = CoreRange{.start = CoreCoord{.x = logical_controller.x + 1, .y = logical_controller.y}, .end = controller_core_range.end};
        CoreRange remaining_stick = CoreRange{
            .start = CoreCoord{.x = logical_controller.x, .y = logical_controller.y + 1},
            .end = CoreCoord{.x = logical_controller.x, .y = controller_core_range.end.y}
        };

        logical_core_ranges.push_back(right_block);
        logical_core_ranges.push_back(remaining_stick);
    };

    CoreRange range_0 = *all_cores.ranges().begin();
    if (all_cores.ranges().size() == 1) {
        split_core_range_containing_controller(range_0);
    } else {
        CoreRange range_1 = *all_cores.ranges().rbegin();
        if (range_0.start == logical_controller) {
            split_core_range_containing_controller(range_0);
            logical_core_ranges.push_back(range_1);
        } else if (range_1.start == logical_controller) {
            split_core_range_containing_controller(range_1);
            logical_core_ranges.push_back(range_0);
        } else {
            TT_THROW("Core {} is not included in set of core ranges!", logical_controller.str());
        }
    }

    TT_ASSERT(logical_core_ranges.size() == 2 or logical_core_ranges.size() == 3);
    return logical_core_ranges;
}

// This variant of move is invoked when the input buffer and output buffer overlap, which is possible because input buffer is deallocated before the op runs.
// In this case, data in each core needs to be moved to a temporary local location before being copied into the output buffer
operation::ProgramWithCallbacks move_multi_core_with_overlap(const Tensor &input, Tensor &output) {
    tt_metal::Program program{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t num_pages = output.volume() / TILE_HW;
    tt_metal::Device *device = output.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_pages);

    const auto num_dram_banks = device->num_banks(BufferType::DRAM);
    const auto num_l1_banks = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    uint32_t size_per_l1_bank = tt_metal::detail::SizeBytesPerBank(output.buffer()->size(), output.buffer()->page_size(), num_l1_banks);
    TT_ASSERT(size_per_l1_bank % single_tile_size == 0);

    // CB is being used as temp L1 buffer to copy src data into before writing to dst
    uint32_t cb_index = 0;
    tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(size_per_l1_bank, {{cb_index, cb_data_format}})
		.set_page_size(cb_index, single_tile_size);
    auto cb = tt_metal::CreateCircularBuffers(program, all_cores, cb_config);

    auto semaphore_addr = CreateSemaphore(program, all_cores, 0);

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> compile_time_args = {cb_index, (uint32_t)src_is_dram, (uint32_t)dst_is_dram};

    KernelID kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/move_interleaved_with_overlap.cpp",
        all_cores,
        DataMovementConfig{.compile_args = compile_time_args}
    );

    const CoreCoord logical_controller = CoreCoord{.x = 0, .y = 0};
    CoreCoord noc_controller = device->worker_core_from_logical_core(logical_controller);
    std::vector<CoreRange> logical_multicast_regions = get_multicast_regions(device, all_cores, logical_controller);

    std::vector<CoreRange> noc_multicast_regions;
    for (const auto &logical_cr : logical_multicast_regions) {
        CoreRange noc_cr = {.start = device->worker_core_from_logical_core(logical_cr.start), .end = device->worker_core_from_logical_core(logical_cr.end)};
        noc_multicast_regions.push_back(std::move(noc_cr));
    }

    CoreRange range_0_noc = noc_multicast_regions[0];
    CoreRange range_1_noc = noc_multicast_regions[1];
    // if third multicast is not needed range_2_noc will be ignored
    bool do_third_multicast = (noc_multicast_regions.size() == 3);

    uint32_t total_num_tiles = 0;
    for (uint32_t i = 0, tiles_handled_per_core = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        bool is_controller = (i == 0);
        std::vector<uint32_t> runtime_args = {
            src_buffer->address(),
            dst_buffer->address(),
            tiles_handled_per_core,
            num_tiles_per_core,
            semaphore_addr,
            (uint32_t)noc_controller.x,
            (uint32_t)noc_controller.y,
            /*control_value=*/(num_cores - 1),
            (uint32_t)is_controller,
            (uint32_t)range_0_noc.start.x,
            (uint32_t)range_0_noc.start.y,
            (uint32_t)range_0_noc.end.x,
            (uint32_t)range_0_noc.end.y,
            (uint32_t)logical_multicast_regions[0].size(),
            (uint32_t)range_1_noc.start.x,
            (uint32_t)range_1_noc.start.y,
            (uint32_t)range_1_noc.end.x,
            (uint32_t)range_1_noc.end.y,
            (uint32_t)logical_multicast_regions[1].size(),
            (uint32_t)noc_multicast_regions.back().start.x,
            (uint32_t)noc_multicast_regions.back().start.y,
            (uint32_t)noc_multicast_regions.back().end.x,
            (uint32_t)noc_multicast_regions.back().end.y,
            (uint32_t)logical_multicast_regions.back().size(),
            (uint32_t)do_third_multicast
        };
        SetRuntimeArgs(program, kernel_id, core, runtime_args);
        tiles_handled_per_core += num_tiles_per_core;
    }

    auto override_runtime_args_callback = [kernel_id, num_cores, num_cores_y](const Program &program, const std::vector<Buffer*>& input_buffers, const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto runtime_args = GetRuntimeArgs(program, kernel_id, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
                SetRuntimeArgs(program, kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
