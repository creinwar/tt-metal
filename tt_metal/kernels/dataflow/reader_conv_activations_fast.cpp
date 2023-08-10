#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"

inline void pad_l1_buffer_with_zeroes(uint32_t l1_addr, uint32_t pad_size_bytes) {
    volatile std::uint32_t* dst = reinterpret_cast<volatile std::uint32_t*>(l1_addr);
    volatile std::uint32_t* end_dst = dst + (pad_size_bytes >> 2);  // Divide by 4 using right shift

    while (dst < end_dst) {
        *dst++ = 0;
    }

    uint32_t remainder = pad_size_bytes & 0x3;  // Get the remainder using bitwise AND
    if (remainder != 0) {
        volatile std::uint8_t* byte_dst = reinterpret_cast<volatile std::uint8_t*>(dst);
        for (uint32_t i = 0; i < remainder; ++i) {
            *byte_dst++ = 0;
        }
    }
}

void kernel_main() {
    uint32_t i = 0;
    uint32_t act_addr_dram_base  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_dram_noc_x = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_dram_noc_y = get_arg_val<uint32_t>(i); i+=1;

    uint32_t conv_act_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_c = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_h_ = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_w_ = get_arg_val<uint32_t>(i); i+=1;
    uint32_t pad_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t pad_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_weight_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_groups = get_arg_val<uint32_t>(i); i+=1;

    uint32_t act_matrix_height_unpadded = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width_unpadded = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_height = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_height_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_h_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_w_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_h_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_w_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t src_dram_act_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t dst_l1_act_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;

    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    //constexpr uint32_t act_block_width_padding_bytes = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_act = 0;
    constexpr uint32_t tile_size_pow2_exponent = 11;
    const DataFormat data_format = get_dataformat(cb_id_act);
    uint32_t channel_stick_size = conv_act_size_c;
    uint32_t channel_stick_size_bytes = channel_stick_size << 1;
    const InterleavedAddrGen<act_in_dram> s_act = {
        .bank_base_address = act_addr_dram_base,
        .page_size = channel_stick_size_bytes
    };

    // Assumptions. Must be true. Validate on host.
    // assert(act_block_w_datums == C * weight_size_w)
    // assert(num_blocks_act_w == weight_size_h)
    // assert(act_block_w_datums % C == 0)
    // assert(act_block_w_datums % 32 == 0)
    // assert(act_block_h_datums % 32 == 0)
    // assert(act_block_h_ntiles == act_block_h_datums/32)
    // assert(act_block_w_ntiles == act_block_w_datums/32)
    // assert(act_block_num_tiles == (act_block_h_datums * act_block_w_datums)/1024)

    uint32_t out_h = 0;
    uint32_t out_w = 0;
    uint32_t out_h_start = 0;
    uint32_t out_w_start = 0;
    DPRINT << "Running new conv reader" << ENDL();
    for(uint32_t nbh = 0; nbh < num_blocks_act_h; nbh++) {
        for(uint32_t nbr = 0; nbr < num_blocks_weight_w; nbr++) {
            uint32_t in_h_offset_within_kernel_window = 0;
            for (uint32_t nbw = 0; nbw < num_blocks_act_w; nbw++) {
                out_h = out_h_start;
                out_w = out_w_start;
                cb_reserve_back(cb_id_act, act_block_num_tiles);
                uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
                uint32_t l1_addr_offset = 0;
                for(uint32_t bh = 0; bh < act_block_h_datums; bh++) {
                    uint32_t in_h_offset = out_h * stride_h;
                    uint32_t in_w_offset = out_w * stride_w; // expect stride 1 or 2.. make this compile time args - also conv input width
                    uint32_t in_w_offset_within_kernel_window = 0;
                    for(uint32_t bw = 0; bw < weight_size_w; bw++) {
                        uint32_t read_size_bytes = channel_stick_size_bytes;
                        if (out_h < conv_output_size_h) {
                            uint32_t in_h = in_h_offset + in_h_offset_within_kernel_window;
                            uint32_t in_w = in_w_offset + in_w_offset_within_kernel_window;

                            if(in_h < pad_h || in_w < pad_w || in_h >= (conv_act_size_h + pad_h) || in_w >= (conv_act_size_w + pad_w)) {
                                // pad 0s in l1
                                uint32_t dst_addr = l1_write_addr_act + l1_addr_offset;
                                uint32_t pad_size_bytes = read_size_bytes;
                                pad_l1_buffer_with_zeroes(dst_addr, pad_size_bytes);
                            } else {
                                // read one channel from dram multi bank - row_id = channel_id
                                uint32_t in_h_raw = in_h - pad_h;
                                uint32_t in_w_raw = in_w - pad_w;
                                uint32_t channel_id = (in_h_raw * conv_act_size_w) + in_w_raw;
                                uint32_t dst_addr = l1_write_addr_act + l1_addr_offset;
                                uint64_t act_noc_addr = get_noc_addr(channel_id, s_act);
                                noc_async_read(act_noc_addr, dst_addr, read_size_bytes);
                            }
                        } // else { do nothing. let garbage rows be in l1 }
                        l1_addr_offset += read_size_bytes;
                        in_w_offset_within_kernel_window += 1;
                    } // for block width
                    // pad 0s for block padding on the right side of block.. only first conv since C%32 != 0.. ifdef with compile time define
                    #ifdef ACT_BLOCK_WIDTH_PADDING_BYTES
                        // pad 0s in l1
                        uint32_t dst_addr = l1_write_addr_act + l1_addr_offset;
                        pad_l1_buffer_with_zeroes(dst_addr, (uint32_t) ACT_BLOCK_WIDTH_PADDING_BYTES);
                        l1_addr_offset += (uint32_t) ACT_BLOCK_WIDTH_PADDING_BYTES;
                    #endif
                    if(out_w < conv_output_size_w - 1) {
                        out_w += 1;
                    } else {
                        out_h += 1;
                        out_w = 0;
                    }
                } // for block height
                in_h_offset_within_kernel_window += 1;
                noc_async_read_barrier();
                cb_push_back(cb_id_act, act_block_num_tiles);
            } // for num of act blocks in inner width dim
        } // for num of weight blocks in width dim
        out_h_start = out_h;
        out_w_start = out_w;
    } // for num of act blocks in height dim
}
