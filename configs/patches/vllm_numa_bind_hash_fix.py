# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Patch vLLM's ParallelConfig.compute_hash to exclude NUMA-bind fields
(numa_bind / numa_bind_nodes / numa_bind_cpus) from the DP consistency hash.

Symptom (seen on GB300, 1 worker, DP=4, numa-bind=True):
    RuntimeError: Configuration mismatch detected for engine 3.
    All DP workers must have identical configurations for parameters that
    affect collective communication ...

Root cause: when numa-bind is enabled, each DP rank auto-detects and stores
its own per-rank NUMA node in ParallelConfig.numa_bind_nodes. These per-rank
values enter compute_hash(), so ranks on different NUMA nodes produce
different hashes and fail the DP startup check. NUMA binding affects only
host-side memory locality, not collective-communication semantics, so it is
safe to exclude from the DP hash.

Reference: vllm/config/parallel.py, ParallelConfig.compute_hash(),
ignored_factors set.
"""

import os
import sys
from pathlib import Path


def _find_target() -> Path | None:
    override = os.environ.get("VLLM_PARALLEL_CONFIG_PATH")
    if override:
        path = Path(override)
        return path if path.is_file() else None

    candidates = (
        Path("/usr/local/lib/python3.12/dist-packages/vllm/config/parallel.py"),
        Path("/usr/local/lib/python3.12/site-packages/vllm/config/parallel.py"),
    )
    for path in candidates:
        if path.is_file():
            return path

    for root in (Path("/usr/local/lib"), Path("/usr/lib")):
        for path in root.glob("python*/site-packages/vllm/config/parallel.py"):
            if path.is_file():
                return path
        for path in root.glob("python*/dist-packages/vllm/config/parallel.py"):
            if path.is_file():
                return path
    return None

# Idempotency: if any of our additions is already present, skip.
MARKER = '"numa_bind",'

# Anchor: the last entry of the existing ignored_factors set in the
# upstream compute_hash method. We insert the three numa fields just
# before the closing brace.
OLD = '            "_api_process_rank",\n        }'

NEW = (
    '            "_api_process_rank",\n'
    "            # srt-slurm-sa hotfix: numa-bind fields are per-rank runtime\n"
    "            # topology, not collective-communication semantics.\n"
    '            "numa_bind",\n'
    '            "numa_bind_nodes",\n'
    '            "numa_bind_cpus",\n'
    "        }"
)


def main():
    target = _find_target()
    if target is None:
        print(
            "[vllm-numa-bind-hash-fix] ParallelConfig source not found; "
            "skipping version-specific hotfix.",
            file=sys.stderr,
        )
        return

    content = target.read_text()

    if MARKER in content:
        print("[vllm-numa-bind-hash-fix] Already patched, skipping.", file=sys.stderr)
        return

    count = content.count(OLD)
    if count == 0:
        print(
            "[vllm-numa-bind-hash-fix] Could not find ignored_factors anchor. "
            "vLLM version may have drifted; inspect ParallelConfig.compute_hash().",
            file=sys.stderr,
        )
        sys.exit(1)
    if count > 1:
        print(
            f"[vllm-numa-bind-hash-fix] Anchor is ambiguous ({count} occurrences); "
            "refusing to patch.",
            file=sys.stderr,
        )
        sys.exit(1)

    content = content.replace(OLD, NEW)
    target.write_text(content)
    print(
        "[vllm-numa-bind-hash-fix] Added numa_bind/numa_bind_nodes/numa_bind_cpus "
        "to ParallelConfig.compute_hash ignored_factors.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
