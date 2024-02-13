// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "n300_device_fixture.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;




/*
 * Simple test that does a round trip over the the eth links
 * and on the remote chip, does an eltwise unary increment
 * on the data
 *
 *                 eth              eth
 *                ┌──────┐         ┌──────┐
 *                │      │         │      │
 *                │      │         │      │
 *       ─────────┼─► ───┼─────────┼─► ───┼─┐
 *                │      │         │      │ │     Tensix
 *                │      │         │      │ │    ┌───────┐
 *                │      │         │      │ │    │       │
 *                └──────┘         └──────┘ └───►│       │
 *                                               │       │
 *                                               │  +1   │
 *                 eth               eth         │       │
 *                ┌──────┐         ┌──────┐   ┌──┤       │
 *                │      │         │      │   │  │       │
 *                │      │         │      │   │  └───────┘
 *       ◄────────┼─── ◄─┼─────────┼─── ◄─┼───┘
 *                │      │         │      │
 *                │      │         │      │
 *                │      │         │      │
 *                └──────┘         └──────┘
 *
 */

 // TODOL Add a that sends data over eth for different gather schemes
 // 1) Sharded L1 -> eth
 // 2) Interleaved L1 -> eth
 // 3) Sharded DRAM -> eth
 // 4) Interleaved DRAM -> eth
 // Performance testing of these should help understand performance differences
 // between different layout and synchronization schemes

bool RunWriteBWTest(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const size_t& byte_size,
    const size_t& src_eth_l1_byte_address,
    const size_t& dst_eth_l1_byte_address,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    uint32_t num_bytes_per_send) {
    bool pass = true;
    log_debug(
        tt::LogTest,
        "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} addr {}",
        byte_size,
        sender_device->id(),
        eth_sender_core.str(),
        src_eth_l1_byte_address,
        receiver_device->id(),
        eth_receiver_core.str(),
        dst_eth_l1_byte_address);
    // Generate inputs
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(
        sender_device->id(),
        sender_device->ethernet_core_from_logical_core(eth_sender_core),
        inputs,
        src_eth_l1_byte_address);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    llrt::write_hex_vec_to_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        all_zeros,
        dst_eth_l1_byte_address);

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send_looping.cpp",
        eth_sender_core,
        tt_metal::experimental::EthernetConfig{
            .eth_mode = tt_metal::Eth::SENDER,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {uint32_t(num_bytes_per_send), uint32_t(num_bytes_per_send >> 4)}});

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            (uint32_t)src_eth_l1_byte_address,
            (uint32_t)dst_eth_l1_byte_address,
            (uint32_t)byte_size,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive.cpp",
        eth_receiver_core,
        tt_metal::experimental::EthernetConfig{
            .eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});  // probably want to use NOC_1 here

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            (uint32_t)byte_size,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::detail::CompileProgram(sender_device, sender_program);
    tt::tt_metal::detail::CompileProgram(receiver_device, receiver_program);

    EnqueueProgram(sender_device->command_queue(), sender_program, false);
    EnqueueProgram(receiver_device->command_queue(), receiver_program, false);
    Finish(sender_device->command_queue());
    Finish(receiver_device->command_queue());

    auto readback_vec = llrt::read_hex_vec_from_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        dst_eth_l1_byte_address,
        byte_size);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_receiver_core.str() << std::endl;
        std::cout << readback_vec[0] << std::endl;
    }
    return pass;
}

TEST_F(N300DeviceFixture, N300WriteBandwidthTest) {
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    auto const& active_eth_cores = device_0->get_active_ethernet_cores();
    assert (active_eth_cores.size() > 0);
    const auto eth_sender_core_iter = active_eth_cores.begin();
    assert (eth_sender_core_iter != active_eth_cores.end());
    const auto& eth_sender_core = *eth_sender_core_iter;
    auto [device_id, eth_receiver_core] = device_0->get_connected_ethernet_core(eth_sender_core);

    for (uint32_t bytes_per_send : {16, 64}) {
        std::cout << "BW TEST: " << 64 << ", bytes_per_send: " << bytes_per_send << std::endl;
        RunWriteBWTest(
            device_0,
            device_1,
            64,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            eth_sender_core,
            eth_receiver_core,
            bytes_per_send);
    }

    for (uint32_t bytes_per_send : {16, 64, 128, 256, 520, 1040}) {
        std::cout << "BW TEST: " << 1040 << ", bytes_per_send: " << bytes_per_send << std::endl;
        RunWriteBWTest(
            device_0,
            device_1,
            1040,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            eth_sender_core,
            eth_receiver_core,
            bytes_per_send);
    }

    for (uint32_t bytes_per_send : {16, 64, 128, 256, 520, 1040, 2080}) {
        std::cout << "BW TEST: " << 2080 << ", bytes_per_send: " << bytes_per_send << std::endl;
        RunWriteBWTest(
            device_0,
            device_1,
            2080,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            eth_sender_core,
            eth_receiver_core,
            bytes_per_send);
    }


    for (uint32_t bytes_per_send : {16, 64, 128, 256, 520, 1040, 2080, 4160}) {
        std::cout << "BW TEST: " << 4160 << ", bytes_per_send: " << bytes_per_send << std::endl;
        RunWriteBWTest(
            device_0,
            device_1,
            4160,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            eth_sender_core,
            eth_receiver_core,
            bytes_per_send);
    }

    for (uint32_t bytes_per_send : {16, 64, 128, 256, 520, 1040, 2080, 4160, 8192, 16384}) {
        std::cout << "BW TEST: " << 16384 << ", bytes_per_send: " << bytes_per_send << std::endl;
        RunWriteBWTest(
            device_0,
            device_1,
            16384,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            eth_sender_core,
            eth_receiver_core,
            bytes_per_send);
    }

    for (uint32_t bytes_per_send : {16, 64, 128, 256, 520, 1040, 2080, 4160, 8192, 16384, 32768}) {
        std::cout << "BW TEST: " << 32768 << ", bytes_per_send: " << bytes_per_send << std::endl;
        RunWriteBWTest(
            device_0,
            device_1,
            32768,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            eth_sender_core,
            eth_receiver_core,
            bytes_per_send);
    }

    for (uint32_t bytes_per_send : {16, 64, 128, 256, 520, 1040, 2080, 4160, 8192, 16384, 32768}) {
        std::cout << "BW TEST: " << 65535 << ", bytes_per_send: " << bytes_per_send << std::endl;
        RunWriteBWTest(
            device_0,
            device_1,
            65535,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            eth_sender_core,
            eth_receiver_core,
            bytes_per_send);
    }

}


// TEST_F(N300DeviceFixture, N300Spiral1Times) {

// }

// TEST_F(N300DeviceFixture, N300Spiral700Times) {

// }
