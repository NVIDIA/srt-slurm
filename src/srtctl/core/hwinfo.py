# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVLink / MNNVL health snapshots taken around a run.

An uncorrectable NVLink fault kills a job with a CUDA error that says nothing
about which link failed, and by the time anyone looks the allocation is gone.
This module captures the fabric state on every worker node before the run and
again when it ends, into ``logs/hwinfo/before.out`` and ``logs/hwinfo/after.out``.
The interesting part is the difference between the two: NVLink error counters
that moved point at the link that degraded.

Collection is best effort by construction — it must never be the reason a
benchmark fails.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext

logger = logging.getLogger(__name__)

HWINFO_DIRNAME = "hwinfo"

COLLECT_SCRIPT = Path(__file__).resolve().parent.parent / "runtime_scripts" / "collect_hwinfo.sh"

# Generous enough for a slow driver call on every node, short enough that a
# wedged node cannot hold up job startup or cleanup for long.
COLLECT_TIMEOUT_S = 180


def record_hwinfo_snapshot(runtime: RuntimeContext, phase: str) -> Path | None:
    """Capture NVLink/MNNVL state on all worker nodes without failing a job.

    Returns the merged snapshot path, or None when nothing could be collected.
    """
    try:
        return _collect(runtime, phase)
    except Exception as error:  # noqa: BLE001
        logger.warning("Failed to capture %s hwinfo snapshot: %s", phase, error)
        return None


def _collect(runtime: RuntimeContext, phase: str) -> Path | None:
    nodes = list(dict.fromkeys(runtime.nodes.worker))
    if not nodes:
        logger.debug("No worker nodes to snapshot for %s hwinfo", phase)
        return None

    hwinfo_dir = runtime.log_dir / HWINFO_DIRNAME
    parts_dir = hwinfo_dir / f".{phase}.parts"
    # A retried phase must not merge stale parts from the previous attempt.
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    for het_group, chunk in _node_chunks(runtime, nodes):
        proc = start_srun_process(
            command=["bash", str(COLLECT_SCRIPT), str(parts_dir), phase],
            nodes=len(chunk),
            ntasks=len(chunk),
            nodelist=chunk,
            # Runs on the host: the IMEX config and MNNVL channel devices are
            # not visible inside the job container.
            container_image=None,
            srun_options=runtime.srun_options,
            het_group=het_group,
        )
        try:
            proc.wait(timeout=COLLECT_TIMEOUT_S)
        except Exception as error:  # noqa: BLE001
            logger.warning("hwinfo collection on %s did not finish: %s", ",".join(chunk), error)
            proc.kill()

    snapshot_path = hwinfo_dir / f"{phase}.out"
    _merge_parts(nodes, parts_dir, snapshot_path)
    shutil.rmtree(parts_dir, ignore_errors=True)

    logger.info("Wrote %s hardware snapshot: %s", phase, snapshot_path)
    return snapshot_path


def _node_chunks(runtime: RuntimeContext, nodes: list[str]) -> list[tuple[int | None, list[str]]]:
    """Group nodes so each srun stays inside one heterogeneous-job component."""
    if not runtime.nodes.het:
        return [(None, nodes)]

    groups: dict[int, list[str]] = {}
    for node in nodes:
        group = runtime.nodes.het_group_for(node)
        if group is None:
            logger.warning("Skipping hwinfo for %s: node is not in any het component", node)
            continue
        groups.setdefault(group, []).append(node)
    return [(group, members) for group, members in sorted(groups.items())]


def _merge_parts(nodes: list[str], parts_dir: Path, snapshot_path: Path) -> None:
    """Concatenate per-node parts in node order into one readable file.

    Each task writes its own part so that output from twelve nodes cannot
    interleave into an unreadable mess.
    """
    sections = []
    merged: set[Path] = set()
    for node in nodes:
        part = parts_dir / f"{node}.part"
        if part.is_file():
            sections.append(part.read_text(errors="replace"))
            merged.add(part)
        else:
            sections.append(f"===== {node} =====\n\n[no hwinfo collected from this node]\n\n")

    # A node whose SLURM name differs from what the task reported still has its
    # snapshot appended rather than silently dropped.
    for part in sorted(parts_dir.glob("*.part")):
        if part not in merged:
            sections.append(part.read_text(errors="replace"))

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text("".join(sections))
