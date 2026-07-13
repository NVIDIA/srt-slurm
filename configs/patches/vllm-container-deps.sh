#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if ! command -v numactl >/dev/null 2>&1; then
    apt-get -y update
    apt-get install -y --no-install-recommends --allow-change-held-packages numactl
fi

if ! python3 -c 'import msgpack' >/dev/null 2>&1; then
    python3 -m pip install --no-deps msgpack
fi

# Some Mooncake wheels are linked against the CUDA 12 runtime soname even when
# vLLM and Triton use CUDA 13.
if ! python3 -c 'import ctypes; ctypes.CDLL("libcudart.so.12")' >/dev/null 2>&1; then
    python3 -m pip install --no-deps "nvidia-cuda-runtime-cu12>=12.8"
    cuda12_runtime_lib="$(python3 - <<'PY'
from pathlib import Path
import site

for base in site.getsitepackages() + [site.getusersitepackages()]:
    candidate = Path(base) / "nvidia" / "cuda_runtime" / "lib"
    if (candidate / "libcudart.so.12").exists():
        print(candidate)
        break
else:
    raise SystemExit("nvidia-cuda-runtime-cu12 installed but libcudart.so.12 was not found")
PY
)"
    echo "${cuda12_runtime_lib}" >/etc/ld.so.conf.d/nvidia-cuda-runtime-cu12.conf
    ldconfig
    python3 -c 'import ctypes; ctypes.CDLL("libcudart.so.12")'
fi

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi
