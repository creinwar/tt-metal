// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "eth_l1_address_map.h"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/edm/erisc_async_datamover.hpp"

// Args Schema:
// 1) handshake addr
// 2) sender channels offset (indicates for the erisc channels, where the senders start
//    so sender and receivers don't clash when paired with sender/receiver on the other
//    end of the link.)
// 3) sender num channels (How many erisc channels to use. ALso how many buffers to instantiate)
//    Informs how many times to iterate through the next group of args
//    4) sender_buffer_address
//    5) sender_num_messages_to_send
//    6) sender_channel_size
//    7) sender_semaphores_base_address
//    8) worker_semaphore_address
//    9) sender_num_workers
//       Informs how many worker X/Y coords to accept in the next loop. Each X/Y pair is 2 uint16s
//       10) worker_coord(s)
// ...
// Repeat from step 2 for receiver side


using tt::tt_metal::ccl::WorkerXY;

template<uint8_t num_senders, uint8_t num_receivers>
struct sender_receiver_index_t {
    static constexpr bool ZERO_SENDERS = num_senders == 0;
    static constexpr bool ZERO_RECEIVERS = num_receivers == 0;
    static constexpr bool NUM_SENDERS_IS_POW_2 = !ZERO_SENDERS && (((num_senders - 1) & num_senders) == 0);
    static constexpr bool NUM_RECEIVERS_IS_POW_2 = !ZERO_RECEIVERS && (((num_receivers - 1) & num_receivers) == 0);
    static constexpr uint16_t SENDER_INCR_MASK = !ZERO_SENDERS ? num_senders - 1 : 0;
    static constexpr uint16_t RECEIVER_INCR_MASK = !ZERO_RECEIVERS ? num_receivers - 1 : 0;
    static constexpr uint16_t COMBINED_INCR_MASK = SENDER_INCR_MASK << 8 | RECEIVER_INCR_MASK;
    static constexpr uint16_t COMBINED_INCR = (1 << 8) | 1;
    union {
        struct {
            uint8_t sender;
            uint8_t receiver;
        };
        uint16_t combined;
    } index;
    union {
        struct {
            uint8_t sender;
            uint8_t receiver;
        };
        uint16_t combined;
    } real_index;
    union {
        struct {
            uint8_t sender;
            uint8_t receiver;
        };
        uint16_t combined;
    } start;

    sender_receiver_index_t(uint8_t send_start, uint8_t receive_start, uint8_t num_send, uint8_t num_receive) {
        start.sender = send_start;
        start.receiver = receive_start;
        index.sender = 0;
        index.receiver = 0;
        real_index.sender = send_start;
        real_index.receiver = receive_start;
    }

    FORCE_INLINE void increment() {
        if constexpr (NUM_SENDERS_IS_POW_2 and NUM_RECEIVERS_IS_POW_2) {
            index.combined = (index.combined + COMBINED_INCR) & COMBINED_INCR_MASK;
            real_index.combined = start.combined + index.combined;
        } else if constexpr (ZERO_RECEIVERS and NUM_SENDERS_IS_POW_2) {
            index.sender = (index.sender + 1) & SENDER_INCR_MASK;
            real_index.sender = start.sender + index.sender;
        } else if constexpr (ZERO_SENDERS and NUM_RECEIVERS_IS_POW_2) {
            index.receiver = (index.receiver + 1) & RECEIVER_INCR_MASK;
            real_index.receiver = start.receiver + index.receiver;
        } else {
            index.combined += COMBINED_INCR;
            index.sender = index.sender >= num_senders ? 0 : index.sender;
            index.receiver = index.receiver >= num_receivers ? 0 : index.receiver;
            real_index.combined = start.combined + index.combined;
        }
    }
};

// COMPILE TIME ARGS
// If true, will enable this erisc's sender functionality
static constexpr bool enable_sender_side = get_compile_time_arg_val(0) != 0;

// If true, will enable this erisc's receiver functionality
static constexpr bool enable_receiver_side = get_compile_time_arg_val(1) != 0;

static constexpr uint32_t num_senders = get_compile_time_arg_val(2);
static constexpr uint32_t num_receivers = get_compile_time_arg_val(3);

static constexpr tt::tt_metal::ccl::EriscDataMoverBufferSharingMode edm_buffer_sharing_mode =
    static_cast<tt::tt_metal::ccl::EriscDataMoverBufferSharingMode>(get_compile_time_arg_val(4));

static constexpr tt::tt_metal::ccl::EriscDataMoverTerminationMode terminate_on_worker_signal =
    static_cast<tt::tt_metal::ccl::EriscDataMoverTerminationMode>(get_compile_time_arg_val(5));

static constexpr bool use_compile_time_designated_handshake_sender = false;//get_compile_time_arg_val(6) != 0;
static constexpr bool is_handshake_sender = get_compile_time_arg_val(7) != 0;

static constexpr bool merge_channel_sync_and_payload = false;//get_compile_time_arg_val(8) != 0;

using EDM_CONFIG_T = erisc::datamover::EriscDatamoverConfig<edm_buffer_sharing_mode, terminate_on_worker_signal, merge_channel_sync_and_payload>;
using ChannelBufferT = erisc::datamover::ChannelBuffer<EDM_CONFIG_T>;

// Sender is advanceable
// eth_tx_q_full
// STATE: READY_FOR_ETH_TRANSFER, CHECK: sender_buffer_channel.eth_is_receiver_channel_send_done()
// NOT eth_tx_q_full
// STATE: WAITING_FOR_ETH, CHECK:  sender_buffer_channel.eth_is_receiver_channel_send_acked() || sender_buffer_channel.eth_is_receiver_channel_send_done();
// eth_tx_q_full
// STATE: CHECK: sender_channel_buffer.is_local_semaphore_full()

// Receiver is advanceable
// eth_tx_q_full (sends ack)
// STATE: WAITING_FOR_ETH, CHECK: buffer_channel.eth_bytes_are_available_on_channel()
// eth_tx_q_full (sends completion)
// STATE: WAITING_FOR_WORKER, CHECK: buffer_channel.is_local_semaphore_full()

FORCE_INLINE bool sender_channel_can_make_progress(ChannelBufferT const& sender_channel) {
    switch (sender_channel.get_state()) {
        case ChannelBufferT::STATE::SENDER_READY_FOR_ETH_TRANSFER:
            // checked condition
            return sender_channel.eth_is_receiver_channel_send_done();
        case ChannelBufferT::STATE::SENDER_WAITING_FOR_ETH:
            // checked condition
            return sender_channel.eth_is_receiver_channel_send_acked() || sender_channel.eth_is_receiver_channel_send_done();
        case ChannelBufferT::STATE::SENDER_WAITING_FOR_WORKER:
            // checked condition
            return sender_channel.is_local_semaphore_full();
        default:
            // DPRINT << "ERR S \n";// << (uint32_t)sender_channel.get_state() << "\n";
            return false;
    }
}

FORCE_INLINE bool receiver_channel_can_make_progress(ChannelBufferT const& receiver_channel) {
    switch (receiver_channel.get_state()) {
        case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_ETH:
            // checked condition
            return receiver_channel.eth_bytes_are_available_on_channel();
        case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_WORKER:
            // checked condition
            return receiver_channel.is_local_semaphore_full();
        case ChannelBufferT::STATE::RECEIVER_SIGNALING_WORKER:
            return true;
        default:
            // DPRINT << "ERR R \n";// << (uint32_t)receiver_channel.get_state() << "\n";
            return false;
    }
}


// {
//     bool any_senders_advanceable = false;
//     std::array<uint32_t, num_senders> advanceable_senders;
//     uint8_t advanceable_sender_index = 0;
//     for (uint32_t i = 0; i < num_senders; i++) {
//         advanceable_senders[i] = num_senders;
//     }
//     bool any_receivers_advanceable = false;
//     uint8_t advanceable_receiver_index = 0;
//     std::array<uint32_t, num_receivers> advanceable_receivers;
//     for (uint32_t i = 0; i < num_receivers; i++) {
//         advanceable_receivers[i] = num_receivers;
//     }

//     uint32_t idle_count = 0;

//     while (senders_in_progress || receivers_in_progress) {
//         while (any_senders_advanceable || any_receivers_advanceable) {
//             if (!eth_txq_is_busy()) {
//                 idle_count = 0;
//                 // TODO: Update states to predicate on channel type so we can just have one larger switch statement and
//                 // TODO: have proper load balancing
//                 if (any_receivers_advanceable) {
//                     uint8_t channel = advanceable_senders[advanceable_receiver_index] + receiver_channels_start;
//                     ChannelBufferT &current_receiver = buffer_channels[channel];

//                     switch (current_receiver.get_state()) {
//                         case ChannelBufferT::STATE::WAITING_FOR_ETH:
//                         erisc::datamover::receiver_eth_accept_payload_sequence_v2(current_receiver, num_receivers_complete, eth_transaction_ack_word_addr);
//                         // receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
//                         break;

//                         // case ChannelBufferT::STATE::SIGNALING_WORKER:
//                         // did_something_receiver =
//                         //     erisc::datamover::receiver_eth_notify_workers_payload_available_sequence(current_receiver);
//                         // break;

//                         case ChannelBufferT::STATE::WAITING_FOR_WORKER:
//                         erisc::datamover::receiver_noc_read_worker_completion_check_sequence_v2(
//                                             current_receiver, num_receivers_complete, eth_transaction_complete_addr);
//                         // receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
//                         break;

//                         default:
//                         break;
//                     };
//                     receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
//                     advanceable_receivers[advanceable_receiver_index] = num_receivers; // reset the entry
//                     advanceable_receiver_index = advanceable_receiver_index == num_receivers ? 0 : advanceable_receiver_index + 1; // goto next
//                     any_receivers_advanceable = advanceable_receivers[advanceable_receiver_index] < num_receivers;
//                 }
//                 if (senders_advanceable) {
//                     uint8_t channel = advanceable_senders[advanceable_sender_index] + sender_channels_start;
//                     ChannelBufferT &current_sender = buffer_channels[channel];
//                     switch (current_sender.get_state()) {
//                         case ChannelBufferT::STATE::WAITING_FOR_WORKER:
//                         erisc::datamover::sender_noc_receive_payload_ack_check_sequence_v2(current_sender, num_senders_complete);
//                         // senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
//                         break;

//                         case ChannelBufferT::STATE::READY_FOR_ETH_TRANSFER:
//                         erisc::datamover::sender_eth_send_data_sequence_v2(current_sender);
//                         break;

//                         case ChannelBufferT::STATE::WAITING_FOR_ETH: // Can remote due to short circuit
//                         erisc::datamover::sender_eth_check_receiver_ack_sequence_v2(current_sender, num_senders_complete);
//                         // senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
//                         break;

//                         default:
//                         break;
//                     };
//                     senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
//                     advanceable_senders[advanceable_sender_index] = num_senders; // reset the entry
//                     advanceable_sender_index = advanceable_sender_index == num_senders ? 0 : advanceable_sender_index + 1; // goto next
//                     any_senders_advanceable = advanceable_senders[advanceable_sender_index] < num_senders;
//                 }
//             }
//         }

//         idle_count++;

//         { // No more channels are advanceable so start checking through all of them to see if any are advanceable
//             {
//                 uint8_t index = advanceable_sender_index;
//                 for (uint32_t i = 0; i < num_senders; i++) {
//                     uint8_t channel = i + sender_channels_start;
//                     ChannelBufferT &sender = buffer_channels[channel];
//                     bool advanceable = sender_channel_can_make_progress(sender);
//                     if (advanceable) {
//                         if (sender.get_state() == ChannelBufferT::STATE::WAITING_FOR_ETH) {
//                             // Since we don't depend on tx q space being available, we can just complete this state quickly right now
//                             // If this ends up causing enough bubbles over eth, we can add them to a separate list that we check after
//                             // eth tx q is full or no other channels are advanceable
//                             erisc::datamover::sender_eth_check_receiver_ack_sequence_v2(sender, num_senders_complete)
//                             senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
//                         } else {
//                             advanceable_senders[index] = i;
//                             index = index == num_senders ? 0 : index + 1;
//                         }
//                     }
//                 }
//             }
//             {
//                 uint8_t index = 0;
//                 for (uint32_t i = 0; i < num_receivers; i++) {
//                     uint8_t channel = i + receiver_channels_start;
//                     ChannelBufferT &receiver = buffer_channels[channel];
//                     bool advanceable = receiver_channel_can_make_progress(channel);
//                     if (advanceable) {
//                         advanceable_receivers[index] = i;
//                         index = index == num_receivers ? 0 : index + 1;
//                     }
//                 }
//             }
//         }

//         if (idle_count > SWITCH_INTERVAL) {
//             idle_count = 0;
//             risc_context_switch();
//         }

//     }
// }

static constexpr bool use_optimized_edm_impl = true;

void kernel_main() {

    // DPRINT << "EDM (enable_sender | enable_receiver) " << uint32_t((enable_sender_side << 16) | enable_receiver_side) << ", on core " << (uint32_t)(my_y[0] << 16 | my_x[0]) << "\n";

    std::array<ChannelBufferT, eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS> buffer_channels;

    //
    std::array<uint32_t, eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS> printed_receiver_done;

    // SENDER ARGS
    uint32_t args_offset = 0;
    uint32_t handshake_addr = get_arg_val<uint32_t>(args_offset++);

    uint8_t const sender_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint32_t const sender_num_channels = num_senders;//get_arg_val<uint32_t>(args_offset++);
    uint8_t num_senders_with_no_work = 0;
    for (uint32_t channel = 0; channel < sender_num_channels; channel++) {
        uint32_t const sender_buffer_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const sender_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const sender_channel_size = get_arg_val<uint32_t>(args_offset++);
        // The erisc's local l1 copy of the semaphore workers remotely increment
        uint32_t const sender_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        // worker's semaphore L1 address
        const uint32_t worker_semaphore_address = get_arg_val<uint32_t>(args_offset++);
        const uint32_t sender_num_workers = get_arg_val<uint32_t>(args_offset++);
        const uint32_t workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += sender_num_workers;
        new (&buffer_channels[sender_channels_start + channel]) ChannelBufferT(
            sender_channels_start + channel,
            sender_buffer_address,
            sender_channel_size,
            worker_semaphore_address,
            sender_num_workers,
            sender_num_messages_to_send,
            (volatile tt_l1_ptr uint32_t *const)sender_semaphores_base_address,
            (const WorkerXY *)workers_xy_list_addr,
            true);
        if constexpr (terminate_on_worker_signal == EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) {
            if (sender_num_messages_to_send == 0) {
                num_senders_with_no_work++;
            }
        }
    }

    // Receiver args
    uint8_t const receiver_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint32_t const receiver_num_channels = num_receivers;//get_arg_val<uint32_t>(args_offset++);
    uint8_t num_receivers_with_no_work = 0;
    for (uint32_t channel = 0; channel < receiver_num_channels; channel++) {
        uint32_t const receiver_buffers_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const receiver_channel_size = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const worker_semaphore_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_num_workers = get_arg_val<uint32_t>(args_offset++);
        const uint32_t workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += receiver_num_workers;
        new (&buffer_channels[receiver_channels_start + channel]) ChannelBufferT(
            receiver_channels_start + channel,
            receiver_buffers_base_address,
            receiver_channel_size,
            worker_semaphore_address,
            receiver_num_workers,
            receiver_num_messages_to_send,
            (volatile tt_l1_ptr uint32_t *const)receiver_semaphores_base_address,
            (const WorkerXY *)workers_xy_list_addr,
            false);

        if constexpr (terminate_on_worker_signal == EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) {
            if (receiver_num_messages_to_send == 0) {
                num_receivers_with_no_work++;
            }
        }
    }

    // Handshake with other erisc to make sure it's safe to start sending/receiving
    // Chose an arbitrary ordering mechanism to guarantee one of the erisc's will always be "sender" and the other
    // will always be "receiver" (only for handshake purposes)
    {
        // DeviceZoneScopedN("EDM_HS");
        bool act_as_sender_in_handshake =
            (sender_channels_start < receiver_channels_start || receiver_num_channels == 0) && sender_num_channels > 0;
        erisc::datamover::eth_setup_handshake(handshake_addr, act_as_sender_in_handshake);
    }
    // DPRINT << "&chans="<< (uint32_t)(&erisc_info->channels[0]) << ", " << (uint32_t)((my_y[0] << 16)| my_x[0]) << "\n";
    uint32_t eth_transaction_ack_word_addr = handshake_addr + 16;
    uint32_t eth_transaction_complete_addr = handshake_addr + 32;

    // constexpr uint32_t SWITCH_INTERVAL = 2000000000;
    constexpr uint32_t SWITCH_INTERVAL = 100000;
    // uint32_t did_nothing_count = 0;

    uint32_t num_senders_complete = !enable_sender_side ? sender_num_channels : num_senders_with_no_work;
    uint32_t num_receivers_complete = !enable_receiver_side ? receiver_num_channels : num_receivers_with_no_work;
    bool senders_in_progress = num_senders_complete != sender_num_channels;
    bool receivers_in_progress = num_receivers_complete != receiver_num_channels;

    // if constexpr (true)
    {
        static constexpr uint8_t advanceable_oob_idx = num_receivers + num_senders;
        bool any_channels_advanceable = false;
        std::array<uint32_t, advanceable_oob_idx> advanceable_channels;
        uint8_t advanceable_index = 0;
        for (uint32_t i = 0; i < advanceable_oob_idx; i++) {
            advanceable_channels[i] = advanceable_oob_idx;
        }

        uint32_t idle_count = 0;

        bool printed = false;
        while (senders_in_progress || receivers_in_progress) {
            { // No more channels are advanceable so start checking through all of them to see if any are advanceable

                // No channels stuck in advanceable state
                // No message appear to be unacknowledged
                // ^^^ Before merging sender and receiver state checks

                // DeviceZoneScopedN("EDM_CHECK_ADVANCEABLE");
                {
                    // DeviceZoneScopedN("EDM_CHECK_ADVANCEABLE_SENDER");
                    uint8_t index = advanceable_index;
                    for (uint32_t i = 0; i < advanceable_oob_idx; i++) {
                        ChannelBufferT &edm_channel = buffer_channels[i];
                        if (edm_channel.is_done()) {
                            continue;
                        }
                        bool advanceable =
                            !edm_channel.is_done() &&
                            ((edm_channel.is_sender_side && sender_channel_can_make_progress(edm_channel)) ||
                             (!edm_channel.is_sender_side && receiver_channel_can_make_progress(edm_channel)));
                        if (advanceable) {
                            if (edm_channel.get_state() == ChannelBufferT::STATE::SENDER_WAITING_FOR_ETH) {
                                // Since we don't depend on tx q space being available, we can just complete this state quickly right now
                                // If this ends up causing enough bubbles over eth, we can add them to a separate list that we check after
                                // eth tx q is full or no other channels are advanceable
                                erisc::datamover::sender_eth_check_receiver_ack_sequence_v2(edm_channel, num_senders_complete);
                                // Technically only need to do this for message count termination mode
                                senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                                // DPRINT << "\t#msgs needed= " << (uint32_t)sender.total_num_messages_to_move << "\n";
                                // DPRINT << "\t#msgs fwded= " << (uint32_t)sender.get_messages_moved() << "\n";
                                // DPRINT << "\tsenders_in_progress= " << (uint32_t)senders_in_progress << "\n";
                            } else {
                                any_channels_advanceable = true;
                                advanceable_channels[index] = i;
                                index = index == advanceable_oob_idx - 1 ? 0 : index + 1;
                            }
                        }
                    }
                }
            }

            while (any_channels_advanceable) {
                idle_count = 0;
                if (!eth_txq_is_busy()) {
                    DeviceZoneScopedN("EDM_ADVANCE_RECEIVER");
                    uint8_t channel = advanceable_channels[advanceable_index];
                    ChannelBufferT &edm_channel = buffer_channels[channel];

                    switch (edm_channel.get_state()) {
                        case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_ETH:
                        // DPRINT << "RX WAITING_FOR_ETH\n";
                        erisc::datamover::receiver_eth_accept_payload_sequence_v2(edm_channel, num_receivers_complete, eth_transaction_ack_word_addr);
                        receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
                        break;

                        case ChannelBufferT::STATE::RECEIVER_SIGNALING_WORKER:
                        // Should be removable
                        erisc::datamover::receiver_eth_notify_workers_payload_available_sequence_v2(edm_channel);
                        break;

                        case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_WORKER:
                        // DPRINT << "RX WAITING_FOR_WORKER\n";
                        erisc::datamover::receiver_noc_read_worker_completion_check_sequence_v2(
                                            edm_channel, num_receivers_complete, eth_transaction_complete_addr);
                        receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
                        break;

                        ////////////////
                        case ChannelBufferT::STATE::SENDER_WAITING_FOR_WORKER:
                        // DPRINT << "SENDER SENDER_WAITING_FOR_WORKER\n";
                        erisc::datamover::sender_noc_receive_payload_ack_check_sequence_v2(edm_channel, num_senders_complete);
                        senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                        break;

                        case ChannelBufferT::STATE::SENDER_READY_FOR_ETH_TRANSFER:
                        // DPRINT << "SENDER READY_FOR_ETH_TRANSFER\n";
                        erisc::datamover::sender_eth_send_data_sequence_v2(edm_channel);
                        break;

                        case ChannelBufferT::STATE::SENDER_WAITING_FOR_ETH: // Can remote due to short circuit
                        // DPRINT << "SENDER WAITING_FOR_ETH\n";
                        erisc::datamover::sender_eth_check_receiver_ack_sequence_v2(edm_channel, num_senders_complete);
                        senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                        break;
                        /////////////////////////

                        default:
                        // DPRINT << "RX BUG in state: " << (uint32_t)current_receiver.get_state() << "\n";
                        DPRINT << "STATE BUG\n";
                        break;
                    };
                    advanceable_channels[advanceable_index] = advanceable_oob_idx;
                    advanceable_index = advanceable_index == advanceable_oob_idx - 1 ? 0 : advanceable_index + 1; // goto next
                    any_channels_advanceable = advanceable_channels[advanceable_index] < advanceable_oob_idx;

                }
            }

            idle_count++;

            if (idle_count > SWITCH_INTERVAL) {

                // for (uint32_t i = 0; i < num_senders; i++) {
                //     DPRINT << "A_S[] " << advanceable_senders[i] << "\n";
                // }
                // for (uint32_t i = 0; i < num_receivers; i++) {
                //     DPRINT << "A_R[] " << advanceable_receivers[i] << "\n";
                // }
                // DPRINT << "LONG IDLE\n";
                idle_count = 0;
                run_routing();
            }

        }
        // DPRINT << "DONE MAIN LOOP@@\n";

    }
    // else {
    //     uint32_t did_nothing_count = 0;
    //     auto send_recv_index = sender_receiver_index_t<num_senders,num_receivers>(sender_channels_start, receiver_channels_start, sender_num_channels, receiver_num_channels);
    //     while (senders_in_progress || receivers_in_progress) {
    //         DeviceZoneScopedN("EDM_LOOP_ITER");
    //         bool did_something_sender = false;
    //         bool did_something_receiver = false;

    //         uint32_t num_receivers_complete_old = num_receivers_complete;
    //         uint32_t num_senders_complete_old = num_senders_complete;
    //         //////////////////////////////////////
    //         // SENDER
    //         if constexpr (enable_sender_side) {
    //             ChannelBufferT &current_sender = buffer_channels[send_recv_index.real_index.sender];
    //             switch (current_sender.get_state()) {
    //                 case ChannelBufferT::STATE::SENDER_WAITING_FOR_WORKER:
    //                 did_something_sender =
    //                     erisc::datamover::sender_noc_receive_payload_ack_check_sequence(current_sender, num_senders_complete);
    //                 senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
    //                 break;

    //                 case ChannelBufferT::STATE::SENDER_READY_FOR_ETH_TRANSFER:
    //                 did_something_sender = erisc::datamover::sender_eth_send_data_sequence(current_sender);
    //                     break;

    //                 // case ChannelBufferT::STATE::SIGNALING_WORKER:
    //                 // did_something_sender = erisc::datamover::sender_notify_workers_if_buffer_available_sequence(
    //                 //                     current_sender, num_senders_complete);
    //                 // senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
    //                 // break;

    //                 case ChannelBufferT::STATE::SENDER_WAITING_FOR_ETH:
    //                 did_something_sender =
    //                     erisc::datamover::sender_eth_check_receiver_ack_sequence(current_sender, num_senders_complete);
    //                 senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
    //                 break;

    //                 default:
    //                 break;
    //             };
    //         }

    //         //////////////////////////////////////
    //         // RECEIVER
    //         if constexpr (enable_receiver_side) {
    //             ChannelBufferT &current_receiver = buffer_channels[send_recv_index.real_index.receiver];

    //             switch (current_receiver.get_state()) {
    //                 case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_ETH:
    //                 did_something_receiver = erisc::datamover::receiver_eth_accept_payload_sequence(current_receiver, num_receivers_complete, eth_transaction_ack_word_addr);
    //                 receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
    //                 break;

    //                 case ChannelBufferT::STATE::RECEIVER_SIGNALING_WORKER:
    //                 did_something_receiver =
    //                     erisc::datamover::receiver_eth_notify_workers_payload_available_sequence(current_receiver);
    //                 break;

    //                 case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_WORKER:
    //                 did_something_receiver = erisc::datamover::receiver_noc_read_worker_completion_check_sequence(
    //                                     current_receiver, num_receivers_complete, eth_transaction_complete_addr);
    //                 receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
    //                 break;

    //                 default:
    //                 break;
    //             };
    //         }
    //         send_recv_index.increment();
    //         //////////////////////////////////////

    //         // Enabling this block as is (with all the "did_something"s, seems to cause a loss of about
    //         // 0.5 GBps in throughput)
    //         if (did_something_sender || did_something_receiver) {
    //             did_nothing_count = 0;
    //         } else {
    //             if (did_nothing_count++ > SWITCH_INTERVAL) {
    //                 did_nothing_count = 0;
    //                 run_routing();
    //             }
    //         }
    //     }
    // }

    {
        // DeviceZoneScopedN("EDM_TEARDOWN");
    for (uint32_t s = 0; s < num_senders + num_receivers; s++ ) {
        auto const& channel = buffer_channels[s];
        // We need to explicitly check for channel send done because we may
        // advance sender channel state as soon as we receive an ack. Since we
        // may be the last active channel, and advance to done state just from ack
        // from the receiver ("I got a payload"), then we need to wait for done
        // at the very end here. Otherise if we invoke another erisc op back-to-back,
        // we may mess up transaction state because it's possible for receiver of this
        // op to send the completion done after that one has already started.
        uint32_t wait_count = 0;
        uint32_t wait_max = 50000;
        while(!channel.eth_is_receiver_channel_send_done()) {
            // wait_count++;
            // if (wait_count > wait_max) {

            //     DEBUG_STATUS("STK");
            //     run_routing();
            //     wait_count = 0;
            // }
        }
    }
    }


    DPRINT << "DONE TEARDOWN@@\n";
    DEBUG_STATUS("DONE");
}
