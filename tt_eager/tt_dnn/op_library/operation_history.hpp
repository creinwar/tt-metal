// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_eager/tensor/tensor.hpp>

#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

#ifdef DEBUG

namespace operation_history {

struct TensorRecord {
    const StorageType storage_type;
    const Shape shape;
    const DataType data_type;
    const Layout layout;
    const std::optional<MemoryConfig> memory_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("storage_type", "shape", "data_type", "layout", "memory_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->storage_type, this->shape, this->data_type, this->layout, this->memory_config);
    }
};

struct OperationRecord {
    const std::size_t ttnn_operation_id;
    const std::string operation_type;
    const std::string operation_name;
    const tt::stl::reflection::Attributes attributes;
    const std::vector<TensorRecord> input_tensor_records;
    const std::vector<const char*> composite_parent_names{};
    std::optional<bool> program_cache_hit;
    const std::optional<tt::stl::hash::hash_t> program_hash;
};

namespace detail {

struct OperationHistory {
    ~OperationHistory();

    void append(OperationRecord&& record);
    void dump_to_csv();
    void clear();

   private:
    std::mutex op_history_mutex;
    std::vector<OperationRecord> records;
};

inline OperationHistory OPERATION_HISTORY{};

}  // namespace detail

template <typename... Args>
inline void append(Args&&... args) {
    detail::OPERATION_HISTORY.append(std::forward<Args>(args)...);
}

const char* csv_file_name();

bool enabled();

void dump_to_csv();
void clear();

}  // namespace operation_history

#endif

}  // namespace tt_metal

}  // namespace tt
