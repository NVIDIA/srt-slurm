# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate explicit physical NUMA maps and pin API processes before exec'ing Python.

Self-contained copy of InferenceMAX ``infx/bench/launch_affinity.py`` (identical at PR 1859
commits 4455c46e and 12c6b381), so the MiniMax-M3.1 Flash R37 recipes run without an
InferenceMAX checkout. ``python3`` in this directory wraps it; put the directory first on PATH
with the srtslurm.yaml line

    default_bash_preamble: export PATH=/configs/m31-launch-bin:$PATH

Behaviour (unchanged from the CI copy):
- ``INFMAX_NUMA_PHYSICAL=1`` on ``python3 -m dynamo.vllm``: ``--numa-bind-nodes`` and
  ``--numa-bind-cpus`` list one entry per *physical* GPU; keep only the entries for the GPUs in
  ``CUDA_VISIBLE_DEVICES``, in that order.
- ``INFMAX_API_CPUS`` on ``python3 -m dynamo.frontend`` / ``-m dynamo.vllm``: pin the API process
  to that CPU list before exec.
Every other invocation is passed through untouched.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping


def remap_numa(argv: list[str], environment: Mapping[str, str]) -> list[str]:
    """Select the physical-GPU entries for a process with a restricted device mask."""
    result = list(argv)
    if result[:2] != ["-m", "dynamo.vllm"] or environment.get("INFMAX_NUMA_PHYSICAL") != "1":
        return result
    mask = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not mask:
        raise ValueError("INFMAX_NUMA_PHYSICAL requires CUDA_VISIBLE_DEVICES")
    devices = [int(value) for value in mask.split(",")]
    if min(devices) < 0 or len(set(devices)) != len(devices):
        raise ValueError(f"invalid CUDA_VISIBLE_DEVICES: {mask}")
    for flag in ("--numa-bind-nodes", "--numa-bind-cpus"):
        start = result.index(flag) + 1
        stop = start
        while stop < len(result) and not result[stop].startswith("--"):
            stop += 1
        values = result[start:stop]
        if max(devices) >= len(values):
            raise ValueError(f"{flag} does not cover physical GPUs {mask}")
        result[start:stop] = [values[device] for device in devices]
    return result


def main(argv: list[str]) -> int:
    """Execute the unmodified Python interpreter with the translated launch arguments."""
    args = remap_numa(argv, os.environ)
    cpus = os.environ.get("INFMAX_API_CPUS")
    if args[:2] in (["-m", "dynamo.frontend"], ["-m", "dynamo.vllm"]) and cpus:
        selected: set[int] = set()
        for field in cpus.split(","):
            bounds = [int(value) for value in field.split("-")]
            if len(bounds) == 1:
                selected.add(bounds[0])
            elif len(bounds) == 2 and bounds[0] <= bounds[1]:
                selected.update(range(bounds[0], bounds[1] + 1))
            else:
                raise ValueError(f"invalid INFMAX_API_CPUS: {cpus}")
        os.sched_setaffinity(0, selected)
    if args != argv or cpus:
        print(f"[launch-affinity] api_cpus={cpus!r} argv={args!r}", file=sys.stderr, flush=True)
    os.execv(sys.executable, [sys.executable, *args])  # noqa: S606 - preserve argv without a shell.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
