#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bash /configs/patches/vllm-container-deps-nixl-cu13.sh

python3 - <<'PY'
from pathlib import Path
import glob
import site
import sys

rel = Path("vllm/third_party/deep_gemm/include/deep_gemm/comm/barrier.cuh")
roots = []

for getter in (site.getsitepackages,):
    try:
        roots.extend(Path(p) for p in getter())
    except Exception:
        pass

try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass

roots.extend(Path(p) for p in sys.path if p)
roots.extend(Path(p) for p in glob.glob("/usr/local/lib/python*/dist-packages"))
roots.extend(Path(p) for p in glob.glob("/usr/local/lib/python*/site-packages"))

targets = []
seen = set()
for root in roots:
    target = (root / rel).resolve()
    if target in seen:
        continue
    seen.add(target)
    if target.exists():
        targets.append(target)

if not targets:
    raise RuntimeError(f"DeepGEMM barrier header not found under Python paths: {rel}")

old_cycles = "constexpr int64_t kNumTimeoutCycles = 30ll * 2000000000ll;"
new_cycles = "constexpr int64_t kNumTimeoutCycles = 300ll * 2000000000ll;"
old_msg = "DeepGEMM NVLink barrier timeout (30s)"
new_msg = "DeepGEMM NVLink barrier timeout (300s)"

patched = 0
for target in targets:
    text = target.read_text()
    if new_cycles in text:
        print(f"[deepgemm-timeout-hotfix] {target}: already patched")
        continue
    if old_cycles not in text:
        raise RuntimeError(
            f"{target}: expected DeepGEMM 30s timeout marker not found; refusing to patch"
        )
    text = text.replace(old_cycles, new_cycles, 1)
    text = text.replace(old_msg, new_msg, 1)
    target.write_text(text)
    patched += 1
    print(f"[deepgemm-timeout-hotfix] patched {target}: 30s -> 300s")

print(f"[deepgemm-timeout-hotfix] complete; patched={patched}, checked={len(targets)}")
PY
