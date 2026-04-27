#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pip install msgpack

if [ -n "${DYNAMO_VERSION:-}" ] || [ -n "${DYNAMO_WHEEL_NAME:-}" ]; then
    if [ -f /configs/install-ai-dynamo.sh ]; then
        bash /configs/install-ai-dynamo.sh
    else
        echo "ERROR: /configs/install-ai-dynamo.sh not found for ai-dynamo wheel install" >&2
        exit 1
    fi
fi

# Upgrade FlashInfer for the NVLink one-sided all-to-all bf16 dispatch patch.
# flashinfer-python / flashinfer-cubin publish on PyPI; flashinfer-jit-cache is
# CUDA-specific and only on the cu130 index. --index-url replaces PyPI entirely,
# so split into two calls.
pip install --upgrade flashinfer-python==0.6.9 flashinfer-cubin==0.6.9
pip install --upgrade flashinfer-jit-cache==0.6.9 --index-url https://flashinfer.ai/whl/cu130

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

if [ -f /configs/patches/vllm_nvlink_one_sided_bf16_fix.py ]; then
    python3 /configs/patches/vllm_nvlink_one_sided_bf16_fix.py
fi
