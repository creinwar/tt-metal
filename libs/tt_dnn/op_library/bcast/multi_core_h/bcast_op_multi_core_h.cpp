#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "constants.hpp"


using namespace tt::tt_metal;
using namespace tt::constants;


namespace tt {

namespace tt_metal {

static const string op_name = "bcast";
static const string perf_folder = "/tmp/tt_perf/ops/";

Tensor bcast_multi_core_h(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim, uint32_t call_count) {
    TT_ASSERT(bcast_dim == BcastOpDim::H);
    tt_metal::SetProfilerDir(perf_folder + op_name + "/" + to_string(call_count));

    const auto ashape = a.shape();
    const auto bshape = b.shape();
    uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

    tt_metal::Program program = tt_metal::Program();

    tt_metal::Device *device = a.device();

    auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;
    auto num_cores = std::min(Ht, num_cores_x * num_cores_y);
    std::vector<uint32_t> Ht_per_core(num_cores, Ht / num_cores);
    for(uint32_t i = 0; i < Ht % num_cores; i++){
        Ht_per_core[i]++;
    }

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr and b.device() != nullptr, "Operands to bcast need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to bcast need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to bcast need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * 1024;

    // This should allocate a DRAM buffer on the device
    tt_metal::Tensor output = Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::TILE, device);

    const char* reader_name = bcast_op_utils::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::MULTI_CORE_H);
    const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);

	std::vector<tt_metal::DataMovementKernel *> binary_reader_kernels;
    std::vector<tt_metal::DataMovementKernel *> unary_writer_kernels;
    for (uint32_t i = 0; i < num_cores; i++){
		CoreCoord core = {i / num_cores_y, i % num_cores_y};

		uint32_t src0_cb_index = 0;
		uint32_t num_input_tiles = 2;
		auto cb_src0 = tt_metal::CreateCircularBuffer(
			program,
			device,
			src0_cb_index,
			core,
			num_input_tiles,
			num_input_tiles * single_tile_size,
			DataFormat::Float16_b
		);

		uint32_t src1_cb_index = 1;
		auto cb_src1 = tt_metal::CreateCircularBuffer(
			program,
			device,
			src1_cb_index,
			core,
			num_input_tiles,
			num_input_tiles * single_tile_size,
			DataFormat::Float16_b
		);

		uint32_t ouput_cb_index = 16; // output operands start at index 16
		uint32_t num_output_tiles = 2;
		auto cb_output = tt_metal::CreateCircularBuffer(
			program,
			device,
			ouput_cb_index,
			core,
			num_output_tiles,
			num_output_tiles * single_tile_size,
			DataFormat::Float16_b
		);

		tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
			program,
			reader_name,
			core,
			tt_metal::DataMovementProcessor::RISCV_1,
			tt_metal::NOC::RISCV_1_default);
			binary_reader_kernels.push_back(binary_reader_kernel);

		tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
			program,
			"tt_metal/kernels/dataflow/writer_unary_8bank_input_cols_batched.cpp",
			core,
			tt_metal::DataMovementProcessor::RISCV_0,
			tt_metal::NOC::RISCV_0_default);
			unary_writer_kernels.push_back(unary_writer_kernel);

		// TODO(AP): add dimensions and op params
		vector<uint32_t> compute_kernel_args = {
			NC, // B
			Ht_per_core[i], // Ht
			Wt  // Wt
		};

		bool fp32_dest_acc_en = false;
		bool math_approx_mode = false;
		auto bcast_kernel = tt_metal::CreateComputeKernel(
			program,
			compute_name,
			core,
			compute_kernel_args,
			MathFidelity::HiFi4,
			fp32_dest_acc_en,
			math_approx_mode
		);
		bcast_op_utils::add_defines(bcast_kernel, bcast_dim, bcast_math);
	}

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////

    constexpr bool profile_device = true;
    tt_metal::CompileProgram(device, program, profile_device);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
	for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores; num_Wtiles_read+=Ht_per_core[i]*Wt, i++){
		CoreCoord core = {i / num_cores_y, i % num_cores_y};
		uint32_t num_tensor_tiles_per_core = NC*Ht_per_core[i]*Wt;

		uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;
		tt_metal::WriteRuntimeArgsToDevice(
			device,
			binary_reader_kernels[i],
			core,
			{
				a.buffer()->address(), // 0
				0, // 1
				0, // 2
				num_tensor_tiles_per_core, // 3
				b.buffer()->address(), // 4
				0, // 5
				0, // 6
				num_btensor_tiles, // 7
				num_tensor_tiles_per_core, // 8
				NC, // 9
				Ht_per_core[i], // 10
				Wt, // 11
				bnc1, // 12
				num_Wtiles_read, // 13
				Ht*Wt, // 14
			}
		);

		tt_metal::WriteRuntimeArgsToDevice(
			device, unary_writer_kernels[i], core,
			{
				output.buffer()->address(),
				0,
				0,
				Ht_per_core[i],
				Wt,
				num_Wtiles_read,
				0,
				NC,
				Ht*Wt,
			}
		);
	}

    tt_metal::ConfigureDeviceWithProgram(device, program);

    tt_metal::LaunchKernels(device, program);
    tt_metal::FreshProfilerDeviceLog();
    tt_metal::DumpDeviceProfileResults(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
