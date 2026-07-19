# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Increase vLLM's vendored DeepGEMM NVLink barrier timeout to 300 seconds.

This adapts sgl-project/DeepGEMM#57 to the DeepGEMM source layout vendored by
the vLLM image used by midcurve.yaml. DeepGEMM hashes included headers for its
JIT cache, so changing this file before importing vLLM causes affected kernels
to be recompiled automatically on first use.
"""

import importlib.util
import sys
from pathlib import Path

RELATIVE_TARGET = Path("third_party/deep_gemm/include/deep_gemm/comm/barrier.cuh")

OLD_TIMEOUT = (
    "        // Update status and wait arrival (with 30s timeout, at 2 GHz)\n"
    "        constexpr int64_t kNumTimeoutCycles = 30ll * 2000000000ll;"
)
NEW_TIMEOUT = (
    "        // Update status and wait arrival (with 300s timeout, at 2 GHz)\n"
    "        constexpr int64_t kNumTimeoutCycles = 300ll * 2000000000ll;"
)
OLD_DIAGNOSTIC = "DeepGEMM NVLink barrier timeout (30s):"
NEW_DIAGNOSTIC = "DeepGEMM NVLink barrier timeout (300s):"


def find_target() -> Path:
    """Locate barrier.cuh without importing vLLM or DeepGEMM."""
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        for package_dir in spec.submodule_search_locations:
            candidate = Path(package_dir) / RELATIVE_TARGET
            if candidate.exists():
                return candidate

    candidates = (
        Path("/usr/local/lib/python3.12/dist-packages/vllm") / RELATIVE_TARGET,
        Path("/usr/local/lib/python3.12/site-packages/vllm") / RELATIVE_TARGET,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def patch_target(target: Path) -> bool:
    """Patch target and return True, or return False if already patched."""
    if not target.exists():
        raise RuntimeError(f"Target not found: {target}")

    content = target.read_text()
    old_counts = (content.count(OLD_TIMEOUT), content.count(OLD_DIAGNOSTIC))
    new_counts = (content.count(NEW_TIMEOUT), content.count(NEW_DIAGNOSTIC))

    if new_counts == (1, 1) and old_counts == (0, 0):
        return False
    if old_counts != (1, 1) or new_counts != (0, 0):
        raise RuntimeError(
            "Expected exactly one unpatched timeout and diagnostic anchor; "
            f"found old={old_counts}, new={new_counts}. The vendored DeepGEMM source may have drifted."
        )

    patched = content.replace(OLD_TIMEOUT, NEW_TIMEOUT, 1).replace(OLD_DIAGNOSTIC, NEW_DIAGNOSTIC, 1)
    target.write_text(patched)
    return True


def main() -> None:
    if len(sys.argv) > 2:
        print(f"Usage: {Path(sys.argv[0]).name} [barrier.cuh]", file=sys.stderr)
        raise SystemExit(2)

    target = Path(sys.argv[1]) if len(sys.argv) == 2 else find_target()
    try:
        changed = patch_target(target)
    except (OSError, RuntimeError) as exc:
        print(f"[vllm-deepgemm-timeout-fix] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if changed:
        print(f"[vllm-deepgemm-timeout-fix] Increased NVLink barrier timeout to 300s in {target}", file=sys.stderr)
    else:
        print("[vllm-deepgemm-timeout-fix] Already patched, skipping.", file=sys.stderr)


if __name__ == "__main__":
    main()
