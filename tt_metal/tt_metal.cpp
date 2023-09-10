// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <unordered_set>
#include <string>

#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tools/profiler/profiler.hpp"
#include "tools/cpuprof/cpuprof.h"

#include "common/executor.hpp"
#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/kernel_cache.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"


namespace tt {

namespace tt_metal {

namespace detail {
    inline void CompileBlankKernel(Device *device) {
        // Crude way to check if blank_op needs to be compiled or not
        // TODO(pgk):
        //  - fw is compiled every run
        //  - for unknown reasons, fw size can vary run to run
        //  - kernels from one run linked against fw from another run may clash
        //  - rebuid blank kernels once per run
        static bool compiled = false;
        if (compiled) {
            return;
        }

        build_kernel_for_riscv_options_t blank_build_options(device->id(), "blank_op");
        struct hlk_args_t {
            std::int32_t dummy;
        };
        void *hlk_args = new hlk_args_t{
            .dummy = 0,
        };
        blank_build_options.set_hlk_args_all_cores(hlk_args, sizeof(hlk_args_t));
        blank_build_options.set_hlk_file_name_all_cores("tt_metal/kernels/compute/blank.cpp");
        blank_build_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
        blank_build_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
        std::string arch_name = tt::get_string_lowercase(device->arch());

        generate_binaries_params_t default_params;
        detail::GenerateDeviceHeaders(device, &blank_build_options, blank_build_options.name);
        generate_binaries_all_riscs(&blank_build_options, blank_build_options.name, arch_name, default_params);

        compiled = true;
    }

    std::atomic<bool> enable_persistent_kernel_cache = false;

    void EnablePersistentKernelCache()
    {
        enable_persistent_kernel_cache = true;
    }

    void DisablePersistentKernelCache()
    {
        enable_persistent_kernel_cache = false;
    }
}

namespace {

void DownloadFirmware(Device *device, CoreCoord phys_core) {
    ZoneScoped;
    for (int riscv_id = 0; riscv_id < 5; riscv_id++) {
        string fname;
        switch (riscv_id) {
            case 0:
                fname = "brisc/brisc.hex";
                tt::llrt::program_brisc_startup_addr(device->cluster(), device->id(), phys_core);
                break;
            case 1: fname = "ncrisc/ncrisc.hex"; break;
            case 2: fname = "tensix_thread0/tensix_thread0.hex"; break;
            case 3: fname = "tensix_thread1/tensix_thread1.hex"; break;
            case 4: fname = "tensix_thread2/tensix_thread2.hex"; break;
        }
        tt::llrt::test_load_write_read_risc_binary(
            device->cluster(), fname, device->id(), phys_core, riscv_id, true);
    }
}

std::optional<uint32_t> get_semaphore_address(const Program &program, const CoreRange &core_range) {
    std::optional<uint32_t> address;
    auto start_core = core_range.start;
    auto end_core = core_range.end;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = CoreCoord{x, y};
            auto num_semaphores = program.num_semaphores(logical_core);
            if (num_semaphores == NUM_SEMAPHORES) {
                TT_THROW(
                    "Cannot add semaphore on core " + logical_core.str() + ". Max number of semaphores (" +
                    std::to_string(NUM_SEMAPHORES) + ") reached!");
            }
            uint32_t addr = num_semaphores == 0
                                ? SEMAPHORE_BASE
                                : program.semaphore_address(num_semaphores - 1) + ALIGNED_SIZE_PER_SEMAPHORE;
            if (!address.has_value()) {
                address = addr;
            } else if (addr != address) {
                TT_THROW(
                    "Expected semaphore on logical core " + logical_core.str() + " to be initialized at L1 address " +
                    std::to_string(address.value()) + " but it is at " + std::to_string(addr));
            }
        }
    }
    return address;
}
}  // namespace

Device *CreateDevice(int device_id) { return new Device(device_id); }

bool InitializeDevice(Device *device) {
    ZoneScoped;
    TT_ASSERT(not detail::GLOBAL_CQ, "GLOBAL_CQ should not be initialized prior to InitializeDevice!");

    bool init;
    if (device->initialize()) {
        static std::mutex build_mutex;
        static bool global_init_complete = false;

        {
            // Need a lock here to prevent the race of building mulitple times
            const std::lock_guard<std::mutex> lock(build_mutex);
            if (!global_init_complete) {
                build_kernel_for_riscv_options_t build_options(device->id());
                detail::GenerateDeviceHeaders(device, &build_options, "");
                std::string arch_name = tt::get_string_lowercase(device->arch());
                generate_binaries_params_t default_params;
                generate_binaries_all_riscs(&build_options,
                                            "",
                                            arch_name,
                                            default_params);

                global_init_complete = true;
            }
        }

        // Download to worker cores
        CoreCoord grid_size = device->logical_grid_size();
        int num_download_fws = 0;
        for (uint32_t y = 0; y < grid_size.y; y++) {
            for (uint32_t x = 0; x < grid_size.x; x++) {
                CoreCoord logical_core(x, y);
                CoreCoord phys_core = device->worker_core_from_logical_core(logical_core);
                DownloadFirmware(device, phys_core);
                num_download_fws++;

                // This is ugly
                // Enable ncrisc/trisc on worker cores since device dispatch
                // expects this.  Non device-dispatch will override and device
                // dispatch will set up the dispatch cores appropriately
                tt::llrt::enable_ncrisc(device->cluster(), device->id(), phys_core);
                tt::llrt::enable_triscs(device->cluster(), device->id(), phys_core);
            }
        }

        tt::llrt::watcher_attach(device, device->cluster(), device->id(),
                                 [&, device]() { return device->logical_grid_size(); },
                                 [&, device](CoreCoord core) { return device->worker_core_from_logical_core(core); },
                                 get_compile_outpath()
                                 );
        init = true;
    } else {
        init = false;
    }

    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        detail::GLOBAL_CQ = std::make_unique<CommandQueue>(device);
    }

    return init;
}

bool CloseDevice(Device *device) {
    tt::llrt::watcher_detach(device);

    // Needed to ensure that GLOBAL_CQ doesn't contain a closed device
    if (detail::GLOBAL_CQ) {
        detail::GLOBAL_CQ.reset(nullptr);
    }

    return device->close();
}

KernelID CreateDataMovementKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::optional<DataMovementConfig> &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    auto dm_config = detail::GetDataMovementConfig(program, file_name, core_ranges, config);
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        // This triggers the firmware to do logic only needed for slow dispatch
        dm_config.defines["TT_METAL_SLOW_DISPATCH_MODE"] = "1";
    }
    auto kernel = new DataMovementKernel(file_name, core_ranges, dm_config);
    detail::AddKernel(program, kernel);
    return kernel->id();
}

KernelID CreateComputeKernel(Program &program, const std::string &file_name, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const std::optional<ComputeConfig> &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    auto compute_config = config.has_value() ? config.value() : ComputeConfig{};
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        // This triggers the firmware to do logic only needed for fast dispatch
        compute_config.defines["TT_METAL_SLOW_DISPATCH_MODE"] = "1";
    }
    auto kernel = new ComputeKernel(file_name, core_ranges, compute_config);
    detail::AddKernel(program, kernel);
    return kernel->id();
}

const CircularBuffer &CreateCircularBuffer(
    Program &program,
    uint32_t buffer_index,
    const CoreCoord &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    CoreRange single_core_range = {.start = core, .end = core};
    return CreateCircularBuffers(
        program,
        std::set<u32>({buffer_index}),
        CoreRangeSet({single_core_range}),
        num_tiles,
        size_in_bytes,
        data_format,
        l1_address);
}

const CircularBuffer &CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return CreateCircularBuffers(
        program,
        std::set<u32>({buffer_index}),
        CoreRangeSet({core_range}),
        num_tiles,
        size_in_bytes,
        data_format,
        l1_address);
}

const CircularBuffer &CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return CreateCircularBuffers(
        program, std::set<u32>({buffer_index}), core_range_set, num_tiles, size_in_bytes, data_format, l1_address);
}

const CircularBuffer &CreateCircularBuffers(
    Program &program,
    const std::set<uint32_t> &buffer_indices,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address) {
    return program.add_circular_buffer(
        core_range_set, buffer_indices, num_tiles, size_in_bytes, data_format, l1_address);
}

uint32_t CreateSemaphore(Program &program, const CoreRange &core_range, uint32_t initial_value) {
    return CreateSemaphore(program, CoreRangeSet({core_range}), initial_value);
}

uint32_t CreateSemaphore(Program &program, const CoreRangeSet &core_range_set, uint32_t initial_value) {
    std::optional<uint32_t> address;
    TT_ASSERT(core_range_set.ranges().size() > 0, "Expecting a non-empty CoreRangeSet!");
    for (auto core_range : core_range_set.ranges()) {
        auto start_core = core_range.start;
        auto end_core = core_range.end;
        TT_ASSERT(start_core == end_core or start_core < end_core && "Invalid core range!");
        auto addr = get_semaphore_address(program, core_range);
        if (!address.has_value()) {
            address = addr;
        } else {
            TT_ASSERT(addr == address);
        }
    }
    TT_ASSERT(address.has_value(), "Expecting a valid Semaphore address!");

    program.add_semaphore(core_range_set, address.value(), initial_value);

    return address.value();
}

void WriteToDevice(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    ZoneScoped;
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("WriteToDevice");

    uint32_t host_buffer_size_bytes = host_buffer.size() * sizeof(uint32_t);
    TT_ASSERT(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer", host_buffer_size_bytes, buffer.size());

    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    TT_ASSERT(page_size % bytes_per_page_entry == 0);
    uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_type());
    uint32_t bank_index = 0;
    int data_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        page.insert(
            page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);
        switch (buffer.buffer_type()) {
            case BufferType::DRAM: {
                auto dram_channel = buffer.dram_channel_from_bank_id(bank_index);
                device->cluster()->write_dram_vec(
                    page, tt_target_dram{device->id(), dram_channel, 0}, absolute_address);
            } break;
            case BufferType::L1: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                llrt::write_hex_vec_to_core(
                    device->cluster(), device->id(), noc_coordinates, page, absolute_address);
            } break;
            default: TT_ASSERT(false && "Unsupported buffer type to write to device!");
        }

        bank_index = (bank_index + 1) % num_banks;
        data_index += num_entries_per_page;
    }
}

void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::L1: {
            WriteToDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_ASSERT(false && "Writing to host memory is unsupported!");
        } break;
        default: TT_ASSERT(false && "Unsupported buffer type!");
    }
}

void ReadFromDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    ZoneScoped;
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("ReadFromDevice");

    host_buffer.clear();  // overwrite the data
    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_type());

    uint32_t bank_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        switch (buffer.buffer_type()) {
            case BufferType::DRAM: {
                auto dram_channel = buffer.dram_channel_from_bank_id(bank_index);
                device->cluster()->read_dram_vec(
                    page, tt_target_dram{device->id(), dram_channel, 0}, absolute_address, page_size);
            } break;
            case BufferType::L1: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                page = llrt::read_hex_vec_from_core(
                    device->cluster(), device->id(), noc_coordinates, absolute_address, page_size);
            } break;
            default: TT_ASSERT(false && "Unsupported buffer type to write to device!");
        }

        // Copy page into host buffer
        for (uint32_t entry : page) {
            host_buffer.push_back(entry);
        }

        bank_index = (bank_index + 1) % num_banks;
    }

}

void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    Device *device = buffer.device();
    tt_cluster *cluster = buffer.device()->cluster();
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::L1: {
            if (buffer.buffer_type() == BufferType::DRAM) cluster->dram_barrier(device->pcie_slot()); else cluster->l1_barrier(device->pcie_slot());
            ReadFromDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_ASSERT(false && "Reading from host memory is unsupported!");
        } break;
        default: TT_ASSERT(false && "Unsupported buffer type!");
    }
}

void DeallocateBuffer(Buffer &buffer) { buffer.deallocate(); }

void GenerateBinaries(Device *device, build_kernel_for_riscv_options_t *build_options, const std::string &op_path_suffix, Kernel *kernel) {
    ZoneScoped;
    const std::string tracyPrefix = "GenerateBinaries_";
    ZoneName( (tracyPrefix + op_path_suffix).c_str(), op_path_suffix.length() + tracyPrefix.length());
    try {
        generate_descriptors(build_options, op_path_suffix);
        kernel->generate_binaries(device, build_options, op_path_suffix);
    } catch (std::runtime_error &ex) {
        log_fatal("Failed to generate binaries for {} {}", kernel->name(), ex.what());
    }
}

void SetCircularBufferDataFormat(
    Device *device, const Program &program, Kernel *kernel, build_kernel_for_riscv_options_t &build_options) {
    ZoneScoped;
    for (auto logical_core : kernel->logical_cores()) {
        auto cbs_on_core = program.circular_buffers_on_core(logical_core);
        for (auto circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer.buffer_indices()) {
                build_options.set_cb_dataformat_all_cores(static_cast<CB>(buffer_index), circular_buffer.data_format());
            }
        }
    }
}

#ifdef GENERATE_HASH_LOG
#include <fstream>
#endif

size_t KernelCompileHash(
    Kernel *kernel, build_kernel_for_riscv_options_t &build_options, const int &device_id) {
    string compile_hash_str = std::to_string(std::hash<tt_hlk_desc>{}(build_options.hlk_desc));
    compile_hash_str += kernel->compute_hash();
    size_t compile_hash = std::hash<std::string>{}(compile_hash_str);

#ifdef GENERATE_HASH_LOG
    static std::ofstream f("/tmp/hashlog.txt");
    static std::mutex mutex_;
    {
        unique_lock<mutex> lock;
        f << kernel->name() << " :: "
          << std::hash<tt_hlk_desc>{}(build_options.hlk_desc) << " :: "
          << kernel->compute_hash() << " :: "
          << compile_hash_str << " "
          << compile_hash << std::endl << std::flush;
    }
#endif
    return compile_hash;
}

void CompileKernel(Device *device, Program &program, Kernel *kernel) {
    build_kernel_for_riscv_options_t build_options(device->id(), kernel->name());
    ZoneScoped;

    kernel->set_build_options(build_options);
    SetCircularBufferDataFormat(device, program, kernel, build_options);

    auto kernel_hash = KernelCompileHash(kernel, build_options, device->id());
    std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash);

    bool cache_hit = true;
    bool path_exists = std::filesystem::exists(build_options.outpath + kernel_path_suffix);
    if ( detail::enable_persistent_kernel_cache && path_exists ) {
        if ( not detail::HashLookup::inst().exists(kernel_hash) ) detail::HashLookup::inst().add(kernel_hash);
    } else if ( detail::HashLookup::inst().add(kernel_hash) ) {
        cache_hit = false;
        GenerateBinaries(device, &build_options, kernel_path_suffix, kernel);
    }

    if (detail::CompilationReporter::enabled()) {
        detail::CompilationReporter::inst().add_kernel_compile_stats(program, kernel, cache_hit, kernel_hash);
    }

    kernel->set_binary_path(kernel_path_suffix);
}

// Just adds blank kernels, doesn't compile them or read binaries
void AddBlankKernels(Device *device, Program &program) {
    ZoneScoped;

    // This can be smarter by combining core ranges into maximal rectangles but this code can be removed once we load BRISC FW separately from the kernel binary
    std::set<CoreRange> unique_core_ranges_without_brisc_kernel,
                        unique_core_ranges_without_ncrisc_kernel,
                        unique_core_ranges_without_compute_kernel ;
    vector<KernelID> blank_ids;
    for (auto &[logical_core, kernel_group] : program.core_to_kernel_group()) {
        CoreRange core_range = {.start = logical_core, .end = logical_core};
        if (not kernel_group.riscv0_id.has_value())
            unique_core_ranges_without_brisc_kernel.insert(core_range);
        if (not kernel_group.riscv1_id.has_value())
            unique_core_ranges_without_ncrisc_kernel.insert(core_range);
        if (not kernel_group.compute_id.has_value())
            unique_core_ranges_without_compute_kernel.insert(core_range);
    }

    if (not unique_core_ranges_without_brisc_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_brisc_kernel);
        KernelID blank_kernel_id = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/blank.cpp", core_range_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        blank_ids.push_back(blank_kernel_id);
    }

    if (not unique_core_ranges_without_ncrisc_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_ncrisc_kernel);
        KernelID blank_kernel_id = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/blank.cpp", core_range_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        blank_ids.push_back(blank_kernel_id);
    }

    if (not unique_core_ranges_without_compute_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_compute_kernel);
        KernelID blank_kernel_id = CreateComputeKernel(program, "tt_metal/kernels/compute/blank.cpp", core_range_set);
        blank_ids.push_back(blank_kernel_id);
    }
}

bool CompileProgram(Device *device, Program &program) {
    ZoneScoped;
    TT_ASSERT(
        device->is_initialized(),
        "Device needs to be initialized before program {} compilation! Generating headers for banking information is "
        "dependent on information that is set during device initialization.",
        program.get_id());

    tt::tt_metal::detail::CompileBlankKernel(device);

    bool pass = true;
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("CompileProgram");
    bool profile_kernel = getDeviceProfilerState();
    std::vector<std::future<void>> events;
    log_assert(
        !(profile_kernel && tt_is_print_server_running()), "Debug print server is running, profiling is not allowed");
    tt_set_profiler_state_for_debug_print(profile_kernel);

    // add blank kernels to program, we do this serially before we start compiling all kernels in parallel
    AddBlankKernels(device, program);

    // compile all kernels in parallel, including the blanks
    for (auto kernel_id : program.kernel_ids()) {
        auto kernel = detail::GetKernel(program, kernel_id);
        events.emplace_back ( detail::async ( [kernel, device, &program] {
            CompileKernel(device, program, kernel);
        } ) );
    }

    for (auto & f : events)
        f.wait();

    for (auto kernel_id : program.kernel_ids()) {
        auto kernel = detail::GetKernel(program, kernel_id);
        events.emplace_back ( detail::async ( [kernel, device] { kernel->read_binaries(device->id()); }));
    }

    for (auto & f : events)
        f.wait();

    program.construct_core_range_set_for_worker_cores();

    if (detail::CompilationReporter::enabled()) {
        detail::CompilationReporter::inst().flush_program_entry(program, detail::enable_persistent_kernel_cache);
    }
    if (detail::MemoryReporter::enabled()) {
        detail::MemoryReporter::inst().flush_program_memory_usage(program, device);
    }
    return pass;
}

void ConfigureKernelGroup(const Program &program, const KernelGroup &kernel_group, Device *device, const CoreCoord &logical_core) {
    if (kernel_group.compute_id.has_value()) {
        detail::GetKernel(program, kernel_group.compute_id.value())->configure(device, logical_core);
    }
    if (kernel_group.riscv1_id.has_value()) {
        detail::GetKernel(program, kernel_group.riscv1_id.value())->configure(device, logical_core);
    }
    detail::GetKernel(program, kernel_group.riscv0_id.value())->configure(device, logical_core);
}

bool ConfigureDeviceWithProgram(Device *device, const Program &program) {
    ZoneScoped;
    bool pass = true;
    detail::DispatchStateCheck( false );
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("ConfigureDeviceWithProgram");

    std::unordered_set<CoreCoord> worker_cores;
    auto cluster = device->cluster();
    auto device_id = device->id();

    for (auto &[logical_core, kernel_group] : program.core_to_kernel_group()) {
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        worker_cores.insert(worker_core);

        if (program.circular_buffers_on_core(logical_core).size()) {
            detail::ValidateCircularBufferRegion(program, device, logical_core);
        }
        // CircularBufferConfigVec -- common across all kernels, so written once to the core
        llrt::CircularBufferConfigVec circular_buffer_config_vec = llrt::create_circular_buffer_config_vector();

        // Load firmware into L1 of worker core
        llrt::disable_ncrisc(cluster, device_id, worker_core);     // PROF_BEGIN("CONF_DISABLE_NCTR")
        llrt::disable_triscs(cluster, device_id, worker_core);     // PROF_END("CONF_DISABLE_NCTR")

        ConfigureKernelGroup(program, kernel_group, device, logical_core); // PROF_BEGIN("CONF_KERN") PROF_END("CONF_KERN")

        // Initialize registers to INVALID
        constexpr static uint32_t INVALID = 0x4321;  // PROF_BEGIN("WRITE_HEX")
        uint32_t stream_register_address = STREAM_REG_ADDR(0, 24);
        llrt::write_hex_vec_to_core(
            cluster, device_id, worker_core, {INVALID}, stream_register_address);  // PROF_END("WRITE_HEX")

        auto cbs_on_core = program.circular_buffers_on_core(logical_core);         // PROF_BEGIN("CBS")
        for (auto circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer.buffer_indices()) {
                llrt::set_config_for_circular_buffer(
                    circular_buffer_config_vec,
                    buffer_index,
                    circular_buffer.address(),
                    circular_buffer.size(),
                    circular_buffer.num_tiles());
            }
        }  // PROF_END("CBS")

        if (cbs_on_core.size()) {
            llrt::write_circular_buffer_config_vector_to_core(
                cluster,
                device_id,
                worker_core,
                circular_buffer_config_vec);  // PROF_BEGIN("WRITE_CBS") PROF_END("WRITE_CBS")
        }

        program.init_semaphores(*device, logical_core);
    }

    // Skip loading of blank kernels to storage cores
    for (const auto &core : device->cluster()->get_soc_desc(device->id()).storage_cores) {
        const auto logical_coord = get_core_coord_from_relative(core, device->logical_grid_size());
        worker_cores.insert(device->worker_core_from_logical_core(logical_coord));
    }

    // Load blank kernel to all riscs of all cores excluding those in worker_cores
    const llrt::TensixRiscsOptions riscs_options = llrt::TensixRiscsOptions::ALL_RISCS;  // PROF_BEGIN("LOAD_BLANK")
    llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(
        cluster, device_id, riscs_options, worker_cores);                                // PROF_END("LOAD_BLANK")

    return pass;
}

void SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    detail::GetKernel(program, kernel_id)->set_runtime_args(logical_core, runtime_args);
}

void SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
            CoreCoord logical_core(x, y);
            SetRuntimeArgs(program, kernel_id, logical_core, runtime_args);
        }
    }
}

void SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    for (auto core_range : core_range_set.ranges()) {
        SetRuntimeArgs(program, kernel_id, core_range, runtime_args);
    }
}

std::vector<uint32_t> GetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreCoord &logical_core) {
    return detail::GetKernel(program, kernel_id)->runtime_args(logical_core);
}

void WriteRuntimeArgsToDevice(Device *device, const Program &program) {
    ZoneScoped;
    auto cluster = device->cluster();
    auto device_id = device->id();
    detail::DispatchStateCheck( false );

    auto get_l1_arg_base_addr = [](const RISCV &riscv) {
        uint32_t l1_arg_base = 0;
        switch (riscv) {
            case RISCV::BRISC: {
                l1_arg_base = BRISC_L1_ARG_BASE;
            } break;
            case RISCV::NCRISC: {
                l1_arg_base = NCRISC_L1_ARG_BASE;
            } break;
            default: log_assert(false, "Unsupported {} processor does not support runtime args", riscv);
        }
        return l1_arg_base;
    };

    for (auto kernel_id : program.kernel_ids()) {
        const auto kernel = detail::GetKernel(program, kernel_id);
        auto processor = kernel->processor();
        for (const auto &[logical_core, rt_args] : kernel->runtime_args()) {
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            tt::llrt::write_hex_vec_to_core(cluster, device_id, worker_core, rt_args, get_l1_arg_base_addr(processor));
        }
    }
}

bool core_runs_ncrisc(const Program &program, const CoreCoord &logical_core) {
    auto kernel_group = program.kernels_on_core(logical_core);
    return kernel_group.riscv1_id.has_value();
}

bool core_runs_triscs(const Program &program, const CoreCoord &logical_core) {
    auto kernel_group = program.kernels_on_core(logical_core);
    return kernel_group.compute_id.has_value();
}

llrt::TensixRiscsOptions GetRiscOptionFromCoreConfig(bool core_runs_ncrisc, bool core_runs_triscs) {
    auto risc_option = llrt::TensixRiscsOptions::BRISC_ONLY;
    if (core_runs_ncrisc and not core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::BRISC_NCRISC;
    } else if (not core_runs_ncrisc and core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::BRISC_TRISCS;
    } else if (core_runs_ncrisc and core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::ALL_RISCS;
    }
    return risc_option;
}

bool LaunchKernels(Device *device, const Program &program, bool stagger_start) {
    bool pass = true;
    {//Profiler scope start
    ZoneScoped;
    detail::DispatchStateCheck( false );
    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("LaunchKernels");

    auto cluster = device->cluster();
    auto device_id = device->id();

    cluster->dram_barrier(pcie_slot);
    cluster->l1_barrier(pcie_slot);

    // Deassert cores used in the program
    TensixSoftResetOptions soft_reset_option = stagger_start ? TENSIX_DEASSERT_SOFT_RESET : TENSIX_DEASSERT_SOFT_RESET_NO_STAGGER;
    auto logical_cores_used_in_program = program.logical_cores();
    for (const CoreCoord &logical_core : logical_cores_used_in_program) {
        CoreCoord worker_core = device->worker_core_from_logical_core(logical_core);
        cluster->set_tensix_risc_reset_on_core(tt_cxy_pair(device_id, worker_core), soft_reset_option);
    }

    bool riscs_are_done = false;
    while (not riscs_are_done) {
        riscs_are_done = true;
        for (const auto &logical_core : logical_cores_used_in_program) {
            // Check if all the riscs on the core are done
            bool ncrisc_runs = core_runs_ncrisc(program, logical_core);
            bool triscs_run = core_runs_triscs(program, logical_core);
            auto risc_option = GetRiscOptionFromCoreConfig(ncrisc_runs, triscs_run);
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            riscs_are_done &=
                llrt::internal_::check_if_riscs_on_specified_core_done(cluster, device_id, risc_option, worker_core);
        }
    }

    // Reset the mailboxes on each core to enable multiple launches of the same program
    // without needing to re-configure the device
    for (const auto &logical_core : logical_cores_used_in_program) {
        bool ncrisc_runs = core_runs_ncrisc(program, logical_core);
        bool triscs_run = core_runs_triscs(program, logical_core);
        auto risc_option = GetRiscOptionFromCoreConfig(ncrisc_runs, triscs_run);
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        llrt::internal_::setup_riscs_on_specified_core(cluster, device_id, risc_option, worker_core);
        // Reset the cores that were running
        cluster->set_tensix_risc_reset_on_core(tt_cxy_pair(device_id, worker_core), TENSIX_ASSERT_SOFT_RESET);
    }

    }//Profiler scope end
    detail::DumpDeviceProfileResults(device,program);
    return pass;
}


}  // namespace tt_metal

}  // namespace tt
