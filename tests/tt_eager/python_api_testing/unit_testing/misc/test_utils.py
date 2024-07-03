# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
from models.utility_functions import is_wormhole_b0
import copy

compute_kernel_options = [
    False,  # for grayskull
]
compute_kernel_ids = ["fp32_dest_acc_en=False"]
if is_wormhole_b0:
    compute_kernel_options.append(True)
    compute_kernel_ids.append("fp32_dest_acc_en=True")


def get_compute_kernel_options(compute_kernel_options):
    if compute_kernel_options is None:
        return None
    if is_wormhole_b0():
        fp32_dest_acc_en = compute_kernel_options
        packer_l1_acc = False
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )
    else:
        # Grayskull doesn't support fp32 but test passing a GS config is ok
        compute_kernel_config = ttl.tensor.GrayskullComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
        )
    return compute_kernel_config


def to_cpu(npu_tensor, shape, *, cpu_layout=ttl.tensor.Layout.ROW_MAJOR):
    if npu_tensor is None:
        return None

    shape = list(shape)

    unpad_shape = copy.copy(shape)

    if shape == []:
        unpad_shape = [1, 1]

    if len(shape) == 1:
        unpad_shape = [1] + shape

    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(unpad_shape).to_torch().reshape(shape)

    return cpu_tensor


def to_npu(
    cpu_tensor,
    device,
    *,
    npu_layout=ttl.tensor.Layout.TILE,
    npu_dtype=ttl.tensor.DataType.BFLOAT16,
    shape=None,
):
    if cpu_tensor is None:
        return None

    if shape is not None:
        cpu_tensor = cpu_tensor.view(shape)

    if len(cpu_tensor.shape) == 1:
        cpu_tensor = cpu_tensor.reshape([1, len(cpu_tensor)])

    if len(cpu_tensor.shape) == 0:
        cpu_tensor = cpu_tensor.reshape([1, 1])

    npu_tensor = ttl.tensor.Tensor(cpu_tensor, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return npu_tensor
