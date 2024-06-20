#include <stdint.h>
#include "dataflow_api.h"
// #include "debug/dprint.h"

void kernel_main()
{
    constexpr uint32_t cb_id_weight                 = get_compile_time_arg_val(0);
    constexpr uint32_t this_core_weights_height_nt  = get_compile_time_arg_val(1);
    constexpr uint32_t this_core_weights_width_nt   = get_compile_time_arg_val(2);
    constexpr uint32_t full_weights_width_nt        = get_compile_time_arg_val(3);

    uint32_t rt_arg_index = 0;

    const uint32_t weights_dram_address             = get_arg_val<uint32_t>(rt_arg_index++);
    const uint32_t core_x                           = get_arg_val<uint32_t>(rt_arg_index++);
    const uint32_t core_y                           = get_arg_val<uint32_t>(rt_arg_index++);
    const uint32_t core_index                       = get_arg_val<uint32_t>(rt_arg_index++);


    const uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    const DataFormat weight_df = get_dataformat(cb_id_weight);

    const InterleavedAddrGenFast<true> weights_ddr_addr_gen = {
        .bank_base_address = weights_dram_address,
        .page_size = weight_tile_nbytes,
        .data_format = weight_df
    };

    uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);
    uint32_t start_tile_index = core_index*this_core_weights_width_nt;

    //Bring in the weights so that the tiles are row major
    for(uint32_t fetch_weight_height_idx = 0; fetch_weight_height_idx < this_core_weights_height_nt; fetch_weight_height_idx++)
    {
        uint32_t current_tile_index = start_tile_index + (fetch_weight_height_idx*full_weights_width_nt);
        for(uint32_t fetch_weight_width_idx = 0; fetch_weight_width_idx < this_core_weights_width_nt; fetch_weight_width_idx++)
        {
            noc_async_read_tile(current_tile_index, weights_ddr_addr_gen, weight_write_l1_addr);
            weight_write_l1_addr += weight_tile_nbytes;
            current_tile_index++;
        }
    }
    noc_async_read_barrier();
}
