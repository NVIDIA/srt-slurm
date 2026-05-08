#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install sglang-router 0.3.2 and rebuild DeepEP with kNumMaxTopK=16.
# Used by tasks/bench-sgl-router/*.yaml via setup_script.
set -eux

echo "=== [1/3] Installing sglang-router==0.3.2 ==="
pip install --break-system-packages sglang-router==0.3.2

python3 -c "import sglang_router; print('sglang_router version:', sglang_router.__version__)"

echo "=== [2/3] Patching flashinfer ensure_symlink to be idempotent (race between DP workers) ==="
# flashinfer/jit/cubin_loader.py:ensure_symlink races when multiple DP workers
# start in parallel — both try to create the same friendly-name symlink and the
# loser raises FileExistsError. Clean stale baked-in symlinks AND patch the
# function to tolerate existing symlinks.
rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins/flashinfer 2>/dev/null || true
python3 - <<'PYEOF'
import pathlib, re
p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cubin_loader.py')
s = p.read_text()
marker = '# SRT-SLURM-PATCH: idempotent symlink'
if marker not in s:
    # Replace the `link.symlink_to(target)` call with a try/except that tolerates
    # races between DP workers creating the same symlink concurrently.
    new = re.sub(
        r'(\n(\s*)link\.symlink_to\(target\)\n)',
        r'\n\2try:\n\2    link.symlink_to(target)  ' + marker +
        r'\n\2except FileExistsError:\n\2    pass\n',
        s, count=1,
    )
    assert new != s, 'failed to patch ensure_symlink'
    p.write_text(new)
    print('Patched flashinfer cubin_loader.py:ensure_symlink for DP race')
else:
    print('flashinfer cubin_loader.py already patched')
PYEOF

echo "=== [3/4] Rebuilding DeepEP ==="
bash /configs/rebuild-deepep.sh

echo "=== [4/4] Seeding node-local DeepGEMM cache from shared NFS ==="
# Multi-node decode (DEP16/DEP32) races on a shared NFS DeepGEMM cache: each
# node's first-rank JIT-compiles the same kernels and cuobjdump hits a
# partial/racing cubin, crashing at kernel_runtime.hpp:45. Fix: each node uses
# a node-local writable cache seeded from the shared read-mostly cache.
if [ -d /configs/deepgemm-cache ]; then
    mkdir -p /tmp/deepgemm-cache
    # -n: no-clobber; -r: recursive. Only copies files that don't exist locally.
    cp -rn /configs/deepgemm-cache/. /tmp/deepgemm-cache/ 2>/dev/null || true
    ncu=$(find /tmp/deepgemm-cache -name 'kernel.cubin' 2>/dev/null | wc -l)
    echo "Seeded /tmp/deepgemm-cache with ${ncu} cached kernels (node-local, writable)"
fi

echo "=== setup-router-and-deepep.sh complete ==="
