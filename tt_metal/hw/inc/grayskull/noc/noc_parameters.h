// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "third_party/umd/src/firmware/riscv/grayskull/noc/noc_parameters.h"

#ifdef _NOC_PARAMETERS_H_

#define PCIE_NOC_X 0
#define PCIE_NOC_Y 4

#define PCIE_NOC1_X
#define PCIE_NOC1_Y

// Address formats
#define NOC_XY_ENCODING(x, y) ((((uint32_t)(y)) << (NOC_ADDR_NODE_ID_BITS)) | (((uint32_t)(x))))

#define NOC_XY_PCIE_ENCODING(x, y, noc_index)
    NOC_XY_ENCODING(x, y) | \
    ((noc_index ? (x == PCIE_NOC1_X and y == PCIE_NOC1_Y) : (x == PCIE_NOC_X and y == PCIE_NOC_Y)) * NOC_PCIE_ADDR_BASE) \

#define NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end)                                          \
    ((x_start) << (2 * NOC_ADDR_NODE_ID_BITS)) | ((y_start) << (3 * NOC_ADDR_NODE_ID_BITS)) | (x_end) | \
        ((y_end) << (NOC_ADDR_NODE_ID_BITS))

// Alignment restrictions
#define NOC_L1_READ_ALIGNMENT_BYTES       16
#define NOC_L1_WRITE_ALIGNMENT_BYTES      16
#define NOC_PCIE_READ_ALIGNMENT_BYTES     32
#define NOC_PCIE_WRITE_ALIGNMENT_BYTES    16
#define NOC_DRAM_READ_ALIGNMENT_BYTES     32
#define NOC_DRAM_WRITE_ALIGNMENT_BYTES    16

#endif
