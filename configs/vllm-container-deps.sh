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

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi
