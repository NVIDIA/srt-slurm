#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pip install msgpack

if [ -f /configs/install-ai-dynamo.sh ]; then
    bash /configs/install-ai-dynamo.sh
else
    python3 -m pip install --pre --no-deps --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com "ai-dynamo-runtime==1.2.0.dev20260426" "ai-dynamo==1.2.0.dev20260426"
fi

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi
