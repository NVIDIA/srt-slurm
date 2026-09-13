# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accelerator-specific runtime environment helpers."""

from typing import Literal

AcceleratorVendor = Literal["nvidia", "amd"]


def visible_device_environment(vendor: AcceleratorVendor, device_ids: str) -> dict[str, str]:
    """Return the vendor-native environment used to restrict visible GPUs.

    ROCm recommends ``ROCR_VISIBLE_DEVICES`` for GPU isolation on Linux.  Do
    not also set ``HIP_VISIBLE_DEVICES`` or ``CUDA_VISIBLE_DEVICES`` here:
    those variables are interpreted after ROCr device masking by some stacks,
    so repeating physical indices can accidentally hide the selected devices.
    """
    if vendor == "nvidia":
        return {"CUDA_VISIBLE_DEVICES": device_ids}
    if vendor == "amd":
        return {"ROCR_VISIBLE_DEVICES": device_ids}
    raise ValueError(f"Unsupported accelerator vendor: {vendor}")
