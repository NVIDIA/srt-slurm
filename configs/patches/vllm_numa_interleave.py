# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Change vLLM native NUMA binding from strict membind to interleave 0,1.

The B300 TP4 SimpleCPU offload control allocates 1 TiB of pinned host KV
memory.  With GPUs 0-3, vLLM's native NUMA binding puts every rank on NUMA 0
and uses ``--membind=0`` for the worker subprocesses.  That exhausts the
socket-local memory even though the other socket has ample capacity.

This experiment preserves vLLM's CPU affinity selection while changing only
the memory policy of worker and EngineCore subprocesses to
``--interleave=0,1``.
"""

import importlib.util
import sys
from pathlib import Path

MARKER = "# srt-slurm-sa: interleave NUMA memory across nodes 0 and 1"

REPLACEMENTS = (
    (
        '        return f"--physcpubind={cpu_binding} --membind={numa_node}"',
        f'        {MARKER}\n        return f"--physcpubind={{cpu_binding}} --interleave=0,1"',
    ),
    (
        '    return f"--cpunodebind={numa_node} --membind={numa_node}"',
        f'    {MARKER}\n    return f"--cpunodebind={{numa_node}} --interleave=0,1"',
    ),
    (
        '        return f"--physcpubind={cpu_binding} --membind={membind_arg}"',
        f'        {MARKER}\n        return f"--physcpubind={{cpu_binding}} --interleave=0,1"',
    ),
    (
        '    return f"--cpunodebind={membind_arg} --membind={membind_arg}"',
        f'    {MARKER}\n    return f"--cpunodebind={{membind_arg}} --interleave=0,1"',
    ),
    (
        '        t for t in numactl_args.split() if not t.startswith("--membind=")',
        """        t
        for t in numactl_args.split()
        if not t.startswith(("--membind=", "--interleave="))""",
    ),
)


def find_target() -> Path:
    """Locate vLLM's NUMA utility without importing the package."""
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        for package_dir in spec.submodule_search_locations:
            candidate = Path(package_dir) / "utils" / "numa_utils.py"
            if candidate.exists():
                return candidate

    candidates = (
        Path("/usr/local/lib/python3.12/dist-packages/vllm/utils/numa_utils.py"),
        Path("/usr/local/lib/python3.12/site-packages/vllm/utils/numa_utils.py"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def patch_target(target: Path) -> bool:
    """Patch target and return True, or False when already patched."""
    if not target.exists():
        raise RuntimeError(f"Target not found: {target}")

    content = target.read_text()
    if content.count(MARKER) == 4:
        if any(old in content for old, _new in REPLACEMENTS):
            raise RuntimeError("Found a partially patched NUMA utility")
        return False

    old_counts = tuple(content.count(old) for old, _new in REPLACEMENTS)
    new_counts = tuple(content.count(new) for _old, new in REPLACEMENTS)
    if old_counts != (1, 1, 1, 1, 1) or new_counts != (0, 0, 0, 0, 0):
        raise RuntimeError(
            "Expected one copy of every vLLM NUMA anchor; "
            f"found old={old_counts}, new={new_counts}. The image source may have drifted."
        )

    patched = content
    for old, new in REPLACEMENTS:
        patched = patched.replace(old, new, 1)
    target.write_text(patched)
    return True


def main() -> None:
    if len(sys.argv) > 2:
        print(f"Usage: {Path(sys.argv[0]).name} [numa_utils.py]", file=sys.stderr)
        raise SystemExit(2)

    target = Path(sys.argv[1]) if len(sys.argv) == 2 else find_target()
    try:
        changed = patch_target(target)
    except (OSError, RuntimeError) as exc:
        print(f"[vllm-numa-interleave] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if changed:
        print(
            "[vllm-numa-interleave] Preserved CPU binding and changed worker/EngineCore "
            f"memory policy to --interleave=0,1 in {target}",
            file=sys.stderr,
        )
    else:
        print("[vllm-numa-interleave] Already patched, skipping.", file=sys.stderr)


if __name__ == "__main__":
    main()
