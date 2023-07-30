#include "tt_metal/impl/dispatch/device_command.hpp"

#include "tt_metal/common/logger.hpp"

DeviceCommand::DeviceCommand() {
    this->desc.at(this->wrap_idx) = 0;
    this->desc.at(this->finish_idx) = 0;
    this->desc.at(this->num_workers_idx) = 0;
    this->desc.at(this->num_multicast_messages_idx) = 0;
    this->desc.at(this->data_size_in_bytes_idx) = 0;
    this->desc.at(this->num_relay_buffer_reads_idx) = 0;
    this->desc.at(this->num_relay_buffer_writes_idx) = 0;
    this->desc.at(this->num_relay_program_writes_idx) = 0;
}

void DeviceCommand::finish() { this->desc.at(this->finish_idx) = 1; }

void DeviceCommand::set_num_workers(u32 num_workers) { this->desc.at(this->num_workers_idx) = num_workers; }

void DeviceCommand::set_num_multicast_messages(u32 num_multicast_messages) { this->desc.at(this->num_multicast_messages_idx) = num_multicast_messages; }

void DeviceCommand::set_multicast_message_noc_coord(u32 noc_coord, u32 num_messages) {
    this->desc.at(this->worker_launch_idx) = noc_coord;
    this->desc.at(this->worker_launch_idx + 1) = num_messages;
    this->worker_launch_idx += 2;
}

void DeviceCommand::add_buffer_instruction(
    u32 addr0,
    u32 addr0_noc,
    u32 addr1,

    u32 padded_buf_size,
    u32 burst_size,
    u32 page_size,
    u32 padded_page_size,
    u32 buf_type) {
    constexpr static u32 upper_bound_on_relay_buffer_entry_idx = CONTROL_SECTION_NUM_ENTRIES + NUM_DISPATCH_CORES + RELAY_BUFFER_NUM_ENTRIES;
    tt::log_assert(this->relay_buffer_entry_idx < upper_bound_on_relay_buffer_entry_idx, "relay_buffer_entry_idx ({}) out of bounds ({})", relay_buffer_entry_idx, upper_bound_on_relay_buffer_entry_idx);

    this->desc.at(this->relay_buffer_entry_idx) = addr0;
    this->desc.at(this->relay_buffer_entry_idx + 1) = addr0_noc;
    this->desc.at(this->relay_buffer_entry_idx + 2) = addr1;

    this->desc.at(this->relay_buffer_entry_idx + 3) = padded_buf_size;
    this->desc.at(this->relay_buffer_entry_idx + 4) = burst_size;
    this->desc.at(this->relay_buffer_entry_idx + 5) = page_size;
    this->desc.at(this->relay_buffer_entry_idx + 6) = padded_page_size;
    this->desc.at(this->relay_buffer_entry_idx + 7) = buf_type;
    this->relay_buffer_entry_idx += this->num_4B_words_in_relay_buffer_instruction;
}

void DeviceCommand::add_read_buffer_instruction(
    u32 dst,
    u32 dst_noc,
    u32 src,

    u32 padded_buf_size,
    u32 burst_size,
    u32 page_size,
    u32 padded_page_size,
    u32 buf_type) {
    this->desc.at(this->num_relay_buffer_reads_idx)++;
    tt::log_assert(this->desc.at(this->num_relay_buffer_reads_idx) <= NUM_DATA_MOVEMENT_INSTRUCTIONS, "There can be max {} read commands", NUM_DATA_MOVEMENT_INSTRUCTIONS);

    this->add_buffer_instruction(
        dst,
        dst_noc,
        src,

        padded_buf_size,
        burst_size,
        page_size,
        padded_page_size,
        buf_type);
}

void DeviceCommand::add_write_buffer_instruction(
    u32 src,
    u32 src_noc,
    u32 dst,

    u32 padded_buf_size,
    u32 burst_size,
    u32 page_size,
    u32 padded_page_size,
    u32 buf_type) {

    this->desc.at(this->num_relay_buffer_writes_idx)++;
    tt::log_assert(this->desc.at(this->num_relay_buffer_writes_idx) <= NUM_DATA_MOVEMENT_INSTRUCTIONS, "There can be max {} write commands", NUM_DATA_MOVEMENT_INSTRUCTIONS);

    this->add_buffer_instruction(
        src,
        src_noc,
        dst,

        padded_buf_size,
        burst_size,
        page_size,
        padded_page_size,
        buf_type);
}

void DeviceCommand::add_read_multi_write_instruction(
    u32 src, u32 src_noc, u32 transfer_size, vector<TrailingWriteCommand> write_commands) {
    this->desc.at(this->num_relay_program_writes_idx)++;

    this->desc.at(this->relay_program_entry_idx) = src;
    this->desc.at(this->relay_program_entry_idx + 1) = src_noc;
    this->desc.at(this->relay_program_entry_idx + 2) = transfer_size;
    this->desc.at(this->relay_program_entry_idx + 3) = write_commands.size();
    this->relay_program_entry_idx += 4;
    for (const TrailingWriteCommand& write_command : write_commands) {
        this->desc.at(this->relay_program_entry_idx) = write_command.src;
        this->desc.at(this->relay_program_entry_idx + 1) = write_command.dst;
        this->desc.at(this->relay_program_entry_idx + 2) = write_command.dst_noc;
        this->desc.at(this->relay_program_entry_idx + 3) = write_command.transfer_size;
        this->desc.at(this->relay_program_entry_idx + 4) = write_command.num_receivers;
        this->relay_program_entry_idx += 5;
    }
}

void DeviceCommand::set_data_size_in_bytes(u32 data_size_in_bytes) {
    this->desc.at(this->data_size_in_bytes_idx) = data_size_in_bytes;
}

u32 DeviceCommand::get_data_size_in_bytes() const { return this->desc.at(this->data_size_in_bytes_idx); }

const array<u32, DEVICE_COMMAND_NUM_ENTRIES>& DeviceCommand::get_desc() const { return this->desc; }
