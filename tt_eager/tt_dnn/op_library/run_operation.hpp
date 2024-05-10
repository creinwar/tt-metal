// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt_eager/tensor/tensor.hpp>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/operation_history.hpp"
#include "tt_stl/concepts.hpp"

namespace tt::tt_metal {

namespace operation {

template <typename ConcreteOperation>
auto generic_create_output_tensors(
    const ConcreteOperation& operation,
    const Tensors& input_tensors,
    const std::optional<DataType> output_dtype,
    const Layout output_layout,
    const std::optional<MemoryConfig>& output_mem_config) -> ProgramOutputTensors<ConcreteOperation> {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_shapes = operation.compute_output_shapes(input_tensors);

    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    OutputTensors output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(create_device_tensor(
            output_shape,
            output_dtype.value_or(input_tensors.at(0).get_dtype()),
            output_layout,
            input_tensor.device(),
            output_mem_config.value_or(input_tensors.at(0).memory_config())));
    }
    return output_tensors;
}

namespace run_operation_state {
namespace detail {
struct RunOperationState {

    RunOperationState() {}

    void push_composite_parent_name(const char* parent_name) {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        this->composite_parent_names.push_back(parent_name);
    }

    void pop_composite_parent_name() {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        this->composite_parent_names.pop_back();
    }

    bool is_composite_operation() const {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        return not composite_parent_names.empty();
    }

    const auto& get_composite_parent_names() const {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        return this->composite_parent_names;
    }

  private:
    mutable std::mutex parent_name_mutex;
    std::vector<const char*> composite_parent_names{};
};

inline RunOperationState OPERATION_STATE{};

}  // namespace detail

inline void push_composite_parent_name(const char* parent_name) {
    detail::OPERATION_STATE.push_composite_parent_name(parent_name);
}

inline void pop_composite_parent_name() {
    detail::OPERATION_STATE.pop_composite_parent_name();
}

inline bool is_composite_operation() {
    return detail::OPERATION_STATE.is_composite_operation();
}

inline const auto& get_composite_parent_names() {
    return detail::OPERATION_STATE.get_composite_parent_names();
}

}  // namespace run_operation_state


namespace detail {
template<typename ReturnType, typename... Args>
struct CompositeOperation {

    const char* name;
    std::function<ReturnType(Args...)> function;

    constexpr ReturnType operator()(Args... args) const {
        run_operation_state::push_composite_parent_name(this->name);
        ReturnType output = this->function(args...);
        run_operation_state::pop_composite_parent_name();
        return output;
    }
};

}  // namespace detail

template<typename ReturnType, typename... Args>
constexpr auto decorate_as_composite(const char* name, std::function<ReturnType(Args...)>&& function) {
  return detail::CompositeOperation<ReturnType, Args...>{.name=name, .function=function};
}

template<typename FunctionType>
constexpr auto decorate_as_composite(const char* name, FunctionType function) {
  return decorate_as_composite(name, std::function(function));
}

#ifdef DEBUG
namespace detail {

template <typename OperationType>
std::string operation_type_to_string() {
    if constexpr (std::is_same_v<OperationType, HostOperation<Tensors>>) {
        return "host<Tensors>";
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation<Tensors>>) {
        return "device<Tensors>";
    } else if constexpr (std::is_same_v<OperationType, HostOperation<OptionalTensors>>) {
        return "host<OptionalTensors>";
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation<OptionalTensors>>) {
        return "device<OptionalTensors>";
    } else if constexpr (std::is_same_v<OperationType, ExternalOperation>) {
        return "external";
    } else {
        static_assert(tt::stl::concepts::always_false_v<OperationType>, "OperationType is not supported!");
    }
}

static operation_history::TensorRecord create_tensor_record(const Tensor& tensor) {
    return std::visit(
        [&](const auto& storage) -> operation_history::TensorRecord {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout(), std::nullopt
                };
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout(), tensor.memory_config()};
            }
            else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            }
            else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            }
            else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            } else {
                raise_unsupported_storage<T>();
            }
        },
        tensor.get_storage());
}

template <typename OperationType>
static void append_operation_to_operation_history(
    const std::size_t ttnn_operation_id,
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors) {

    std::vector<operation_history::TensorRecord> input_tensor_records;
    input_tensor_records.reserve(input_tensors.size() + optional_input_tensors.size());

    for (const auto& tensor : input_tensors) {
        input_tensor_records.emplace_back(create_tensor_record(tensor));
    }
    for (const auto& tensor : optional_input_tensors) {
        if (tensor.has_value()) {
            input_tensor_records.emplace_back(create_tensor_record(tensor.value()));
        }
    }

    std::optional<bool> program_cache_hit = std::nullopt;
    std::optional<tt::stl::hash::hash_t> program_hash = std::nullopt;
    if constexpr (std::is_same_v<OperationType, DeviceOperation<typename OperationType::OutputTensors>>) {
        auto& program_cache = input_tensors[0].device()->program_cache;
        if (program_cache.is_enabled()) {
            program_hash = operation.compute_program_hash(input_tensors, optional_input_tensors);
            auto program_pointer = program_cache.find(program_hash.value());
            program_cache_hit = program_pointer.has_value();
        }
    }

    operation_history::append(operation_history::OperationRecord{
        ttnn_operation_id,
        boost::core::demangle(typeid(OperationType).name()),
        operation.get_type_name(),
        operation.attributes(),
        input_tensor_records,
        run_operation_state::get_composite_parent_names(),
        program_cache_hit,
        program_hash,
    });
}

}  // namespace detail

template<typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {}) {
    tt::log_debug(
        tt::LogOp,
        "Launching Operation: \"{}\" ({})",
        operation.get_type_name(),
        detail::operation_type_to_string<OperationType>());

    if (run_operation_state::is_composite_operation()) {
        tt::log_debug(tt::LogOp, "Composite Parents: {}", run_operation_state::get_composite_parent_names());
    }

    if (not operation.attributes().empty()) {
        tt::log_debug(tt::LogOp, "Attributes:");
        for (auto&& [name, value] : operation.attributes()) {
            tt::log_debug(tt::LogOp, "\t{} = {}", name, value);
        }
    }

    tt::log_debug(tt::LogOp, "Input Tensors:");
    for (auto index = 0; index < input_tensors.size(); index++) {
        const auto& tensor = input_tensors[index];
        tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
    }

    if (not optional_input_tensors.empty()) {
        tt::log_debug(tt::LogOp, "Optional Input Tensors:");
        for (auto index = 0; index < optional_input_tensors.size(); index++) {
            const auto& tensor = optional_input_tensors[index];
            tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
        }
    }

    tt::log_debug(tt::LogOp, "");

    if (operation_history::enabled()) {
        detail::append_operation_to_operation_history(
            ttnn::OPERATION_ID, operation, input_tensors, optional_input_tensors);
    }
}
#else

template <typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {}) {}
#endif

inline uint32_t assign_id()
{
    static std::atomic<uint32_t> atomic_count{0};
    return atomic_count.fetch_add(1);
}

template<class OutputTensors=Tensors>
OutputTensors run(
    const HostOperation<OutputTensors>& operation,
    const Tensors& input_tensors
);

template<class OutputTensors=Tensors>
OutputTensors run(
    CommandQueue& queue,
    const DeviceOperation<OutputTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {});

template<class OutputTensors=Tensors>
OutputTensors run(
    const DeviceOperation<OutputTensors>& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {});

template<typename ConcreteOperation>
inline auto run(
    ConcreteOperation&& concrete_op,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors={},
    const OptionalTensors& optional_output_tensors={}
) -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    if constexpr (detail::is_host_operation<ConcreteOperation>()) {
        TT_ASSERT(optional_input_tensors.empty());
        const auto operation = HostOperation(concrete_op);
        return run<OutputTensors>(operation, input_tensors);
    } else if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto operation = DeviceOperation(concrete_op);
        return run<OutputTensors>(operation, input_tensors, optional_input_tensors, optional_output_tensors);
    } else {
        static_assert(tt::stl::concepts::always_false_v<ConcreteOperation>, "Unsupported Operation");
    }
}

void launch_op(
    std::function<std::vector<Tensor>(const Tensors&, const OptionalConstTensors&)>&& op_func,
    const std::vector<Tensor> input_tensors,
    std::vector<Tensor>& output_tensors,
    const std::vector<std::optional<const Tensor>> optional_input_tensors = {}
);

std::vector<Device*> get_workers_for_op_output(const std::vector<Tensor>&& inputs, const std::vector<std::optional<const Tensor>>&& optional_inputs = {});

} //namespace operation

} //namespace tt::tt_metal
