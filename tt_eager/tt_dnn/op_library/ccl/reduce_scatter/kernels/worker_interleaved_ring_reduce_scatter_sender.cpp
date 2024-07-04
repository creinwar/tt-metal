// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/kernel_common/worker_edm_utils.hpp"

using tt::tt_metal::ccl::coord_t;

void kernel_main() {
    constexpr bool is_sharded = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    uint32_t arg_idx = 0;
    uint32_t const dst_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const eth_sender_l1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const eth_sender_l1_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const eth_sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const eth_sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const num_transfers = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const page_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const full_chunk_num_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const writer_send_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const half_cb_n_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const num_concurrent_workers = get_arg_val<uint32_t>(arg_idx++);

    coord_t const& output_tensor_shape = tt::tt_metal::ccl::coord_from_args(arg_idx);
    coord_t const& worker_slice_shape = tt::tt_metal::ccl::coord_from_args(arg_idx);
    coord_t worker_slice_base_offset = tt::tt_metal::ccl::coord_from_args(arg_idx);

    uint32_t total_eltwise_kernel_num_pages = get_arg_val<uint32_t>(arg_idx++);

    // Argument validation
    ASSERT(half_cb_n_pages >= full_chunk_num_pages);
    ASSERT(full_chunk_num_pages > 0);
    ASSERT(page_size > 0);
    ASSERT(half_cb_n_pages > 0);

    constexpr uint32_t cb_id_in0 = tt::CB::c_out0;
    constexpr uint32_t cb_id_in_short_circuit = tt::CB::c_out1;
    const DataFormat in0_df = get_dataformat(cb_id_in0);
#ifdef RM_INTERLEAVED
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = page_size};
#elif defined TILE_INTERLEAVED

    InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr, .page_size = page_size, .data_format = in0_df};
#endif

    // Used to wait until eth sender has space available
    volatile tt_l1_ptr uint32_t* writer_send_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_send_sem_addr);
    // This is different per writer core
    const uint64_t eth_l1_sender_base_noc_addr =
        get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_base_addr);
    // Used to signal eth sender that data is available. This is different per writer core
    const uint64_t eth_l1_sender_semaphore_addr =
        get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_sem_addr);

    uint32_t total_lifetime_cb_pages_popped_from_math = 0;
    while (worker_slice_base_offset.x < output_tensor_shape.x && worker_slice_base_offset.y < output_tensor_shape.y) {
        // First phase - we only forward messages to EDM
        coord_t valid_worker_slice_shape = coord_t(
            std::min(worker_slice_shape.x, output_tensor_shape.x - worker_slice_base_offset.x),
            std::min(worker_slice_shape.y, output_tensor_shape.y - worker_slice_base_offset.y));
        uint32_t const num_pages_to_write = valid_worker_slice_shape.x * valid_worker_slice_shape.y;

        ASSERT(total_lifetime_cb_pages_popped_from_math + num_pages_to_write <= total_eltwise_kernel_num_pages);
        for (uint32_t i = 0; i < num_transfers; ++i) {
            const uint32_t cb_in = i == 0 ? cb_id_in_short_circuit : cb_id_in0;
            for (uint32_t p = 0; p < num_pages_to_write; p += full_chunk_num_pages) {
                uint32_t n_pages = std::min(full_chunk_num_pages, num_pages_to_write - p);
                ASSERT(n_pages > 0);
                noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
                noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
                send_chunk(cb_in, n_pages, page_size, eth_l1_sender_base_noc_addr);
                noc_semaphore_inc(
                    eth_l1_sender_semaphore_addr,
                    tt::tt_metal::ccl::EriscDataMoverWorkerSignal::NEXT_MESSAGE_AVAILABLE);
                if (i != 0) {
                    total_lifetime_cb_pages_popped_from_math += n_pages;
                }
                if (n_pages < half_cb_n_pages) {
                    uint32_t num_filler_pages = half_cb_n_pages - n_pages;

                    ASSERT(p + n_pages == num_pages_to_write);
                    pop_filler_pages_from_cb(cb_in, num_filler_pages);
                    if (i != 0) {
                        total_lifetime_cb_pages_popped_from_math += num_filler_pages;
                    }
                }
            }
        }

        // write the final reduced chunk for this chip out to the output tensor
        // Second phase - Dump the local output to the output tensor
        uint32_t curr_ring_slice_start_page_offset = 0;
        const uint32_t worker_relative_start_offset_into_slice =
            worker_slice_base_offset.x + (worker_slice_base_offset.y * output_tensor_shape.x);
        auto current_worker_slice_offset = worker_slice_base_offset;
        const uint32_t starting_tile_id = curr_ring_slice_start_page_offset + worker_relative_start_offset_into_slice;
        uint32_t curr_tile_id = starting_tile_id;

        bool last_page_of_worker = false;
        for (uint32_t p = 0; p < num_pages_to_write; p += full_chunk_num_pages) {
            ASSERT(curr_tile_id < output_tensor_shape.x * output_tensor_shape.y);
            ASSERT(!last_page_of_worker);
            uint32_t n_pages = std::min(full_chunk_num_pages, num_pages_to_write - p);
            ASSERT(n_pages <= half_cb_n_pages);
            ASSERT(full_chunk_num_pages <= half_cb_n_pages);
            write_chunk_v2(
                curr_tile_id,
                current_worker_slice_offset,
                valid_worker_slice_shape,
                output_tensor_shape,  // In tiles for tile layout
                cb_id_in0,
                d,
                n_pages,
                page_size,
                last_page_of_worker);
            total_lifetime_cb_pages_popped_from_math += n_pages;
            if (n_pages < half_cb_n_pages) {
                uint32_t num_filler_pages = half_cb_n_pages - n_pages;
                ASSERT(p + n_pages == num_pages_to_write);
                pop_filler_pages_from_cb(cb_id_in0, num_filler_pages);
                total_lifetime_cb_pages_popped_from_math += num_filler_pages;
            }
        }

        worker_slice_base_offset = advance_slice_row_major(
            worker_slice_base_offset, worker_slice_shape, output_tensor_shape, num_concurrent_workers);
    }

    ASSERT(total_lifetime_cb_pages_popped_from_math <= total_eltwise_kernel_num_pages);
    for (; total_lifetime_cb_pages_popped_from_math < total_eltwise_kernel_num_pages;
         total_lifetime_cb_pages_popped_from_math++) {
        pop_filler_pages_from_cb(cb_id_in0, 1);
    }

    noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
    noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
    noc_semaphore_inc(
        eth_l1_sender_semaphore_addr, tt::tt_metal::ccl::EriscDataMoverWorkerSignal::TERMINATE_IMMEDIATELY);
}
