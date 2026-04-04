#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eux

echo "=== Rebuilding DeepEP with kNumMaxTopK=16 for Qwen3.5 (topk=10) ==="

DEEPEP_SRC="/sgl-workspace/DeepEP"

if [ ! -d "$DEEPEP_SRC" ]; then
    echo "ERROR: DeepEP source not found at $DEEPEP_SRC (mount via extra_mount)"
    exit 1
fi

cd "$DEEPEP_SRC"

# Find NVSHMEM
NVSHMEM_DIR=$(find /usr/local -name "nvshmem" -type d -not -path "*/flashinfer*" 2>/dev/null | head -1)
if [ -z "${NVSHMEM_DIR:-}" ]; then
    echo "ERROR: NVSHMEM installation not found under /usr/local" >&2
    exit 1
fi
echo "NVSHMEM_DIR=$NVSHMEM_DIR"

# Fix missing nvshmem symlinks (container has .so.3 but not .so)
NVSHMEM_LIB="$NVSHMEM_DIR/lib"
if [ ! -f "$NVSHMEM_LIB/libnvshmem_host.so" ] && [ -f "$NVSHMEM_LIB/libnvshmem_host.so.3" ]; then
    echo "Creating missing nvshmem symlinks..."
    ln -sf libnvshmem_host.so.3 "$NVSHMEM_LIB/libnvshmem_host.so"
fi

# Apply kNumMaxTopK=16 patch (Qwen3.5 uses topk=10, default kNumMaxTopK=8 is insufficient)
# Note: source has both kNumMaxTopK (uppercase) and kNumMaxTopk (lowercase) as separate variables
sed -i 's/kNumMaxTopK[[:space:]]*=[[:space:]]*[0-9][0-9]*/kNumMaxTopK = 16/g' csrc/kernels/internode_ll.cu
sed -i 's/kNumMaxTopk[[:space:]]*=[[:space:]]*[0-9][0-9]*/kNumMaxTopk = 16/g' csrc/kernels/internode_ll.cu

# Verify the patch was applied
grep -q "kNumMaxTop. = 16" csrc/kernels/internode_ll.cu && echo "Patch verified: kNumMaxTopK/k=16" || {
    echo "ERROR: kNumMaxTopK patch failed to apply!"; exit 1;
}

# Build with full output so we can debug failures
# set -e will auto-exit on failure
TORCH_CUDA_ARCH_LIST="10.0" \
NVSHMEM_DIR="$NVSHMEM_DIR" \
pip install -e . --no-build-isolation 2>&1

echo "=== DeepEP rebuild complete ==="
python3 -c "import deep_ep; print('deep_ep imported successfully')"
