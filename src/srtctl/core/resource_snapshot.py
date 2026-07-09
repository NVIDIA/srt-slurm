# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the CPU and GPU allocation visible to a running SLURM job."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import platform
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

RESOURCE_SNAPSHOT_FILENAME = "resource_snapshot.json"

_SLURM_RESOURCE_ENV_KEYS = (
    "SLURM_CPUS_ON_NODE",
    "SLURM_CPUS_PER_GPU",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_NTASKS",
    "SLURM_TRES_PER_TASK",
)


def parse_slurm_cpus_per_node(value: str | None) -> tuple[int, ...]:
    """Expand SLURM's compressed CPU counts, such as ``72(x2),36``."""
    if not value:
        return ()

    counts: list[int] = []
    for item in value.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\(x(\d+)\))?\s*", item)
        if match is None:
            return ()
        cpu_count = int(match.group(1))
        repetitions = int(match.group(2) or "1")
        counts.extend([cpu_count] * repetitions)
    return tuple(counts)


def _allocated_cpus_per_node(environ: Mapping[str, str]) -> tuple[int, ...]:
    """Return per-node CPU allocation, preferring het-component variables."""
    het_values = sorted(
        (
            (int(match.group(1)), value)
            for key, value in environ.items()
            if (match := re.fullmatch(r"SLURM_JOB_CPUS_PER_NODE_HET_GROUP_(\d+)", key))
        ),
        key=lambda item: item[0],
    )
    if het_values:
        counts: list[int] = []
        for _, value in het_values:
            parsed = parse_slurm_cpus_per_node(value)
            if not parsed:
                return ()
            counts.extend(parsed)
        return tuple(counts)

    return parse_slurm_cpus_per_node(environ.get("SLURM_JOB_CPUS_PER_NODE"))


def _compress_cpu_ids(cpu_ids: Sequence[int]) -> str:
    """Format CPU IDs as a Linux cpuset string (for example, ``0-3,8``)."""
    if not cpu_ids:
        return ""

    values = sorted(set(cpu_ids))
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _process_affinity() -> tuple[int | None, str | None]:
    try:
        cpu_ids = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None, None
    return len(cpu_ids), _compress_cpu_ids(cpu_ids)


def _cpu_model() -> str | None:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "cpu model", "hardware"}:
                return value.strip() or None
    return None


def _worker_gpu_count(config: SrtConfig) -> int:
    resources = config.resources
    return resources.prefill_gpus + resources.decode_gpus + resources.num_agg * resources.gpus_per_agg


def collect_resource_snapshot(
    config: SrtConfig,
    runtime: RuntimeContext,
    *,
    environ: Mapping[str, str] | None = None,
    affinity: tuple[int | None, str | None] | None = None,
    minimum_cpus_per_gpu: float | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable snapshot of the job's effective resources."""
    env = os.environ if environ is None else environ
    node_names = tuple(dict.fromkeys((runtime.nodes.infra, *runtime.nodes.worker)))
    node_count = len(node_names)
    configured_gpu_count = node_count * config.resources.gpus_per_node
    worker_gpu_count = _worker_gpu_count(config)
    gpu_count_basis = worker_gpu_count or configured_gpu_count

    cpus_per_node = _allocated_cpus_per_node(env)
    allocated_cpu_count = sum(cpus_per_node) if cpus_per_node else None
    current_node_cpu_count: int | None = None
    with contextlib.suppress(KeyError, TypeError, ValueError):
        current_node_cpu_count = int(env["SLURM_CPUS_ON_NODE"])

    affinity_cpu_count, affinity_list = affinity if affinity is not None else _process_affinity()

    if allocated_cpu_count is None and node_count == 1:
        allocated_cpu_count = current_node_cpu_count or affinity_cpu_count

    if minimum_cpus_per_gpu is None:
        from srtctl.core.config import get_srtslurm_setting

        configured_minimum = get_srtslurm_setting("minimum_cpus_per_gpu", 1.0)
        minimum_cpus_per_gpu = float(configured_minimum if configured_minimum is not None else 1.0)
    minimum_cpu_count = math.ceil(gpu_count_basis * minimum_cpus_per_gpu)

    if minimum_cpus_per_gpu <= 0:
        check_status = "disabled"
        check_message = "CPU allocation warning is disabled (minimum_cpus_per_gpu <= 0)."
    elif allocated_cpu_count is None:
        check_status = "unknown"
        check_message = "Could not determine the job's allocated CPU count from the SLURM environment."
    elif allocated_cpu_count < minimum_cpu_count:
        check_status = "warning"
        check_message = (
            f"CPU allocation may be too small: {allocated_cpu_count} CPUs for {gpu_count_basis} backend GPUs; "
            f"the configured baseline of {minimum_cpus_per_gpu:g} CPU/GPU requires at least {minimum_cpu_count}. "
            "Increase the SLURM CPU request (for example, cpus-per-task/cpus-per-gpu) or use an exclusive node."
        )
    else:
        check_status = "ok"
        check_message = (
            f"CPU allocation meets the configured baseline: {allocated_cpu_count} CPUs for "
            f"{gpu_count_basis} backend GPUs (minimum {minimum_cpu_count})."
        )

    slurm_environment = {key: env[key] for key in _SLURM_RESOURCE_ENV_KEYS if key in env}
    slurm_environment.update(
        {key: value for key, value in sorted(env.items()) if key.startswith("SLURM_JOB_CPUS_PER_NODE_HET_GROUP_")}
    )

    return {
        "version": 1,
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job_id": runtime.job_id,
        "hardware": {
            "architecture": platform.machine(),
            "cpu_model": _cpu_model(),
            "host_logical_cpus": os.cpu_count(),
        },
        "nodes": {
            "count": node_count,
            "names": list(node_names),
        },
        "gpus": {
            "type": config.resources.gpu_type,
            "configured_per_node": config.resources.gpus_per_node,
            "configured_total": configured_gpu_count,
            "backend_total": worker_gpu_count,
        },
        "cpus": {
            "allocated_total": allocated_cpu_count,
            "allocated_per_node": list(cpus_per_node),
            "current_node_allocated": current_node_cpu_count,
            "process_affinity_count": affinity_cpu_count,
            "process_affinity_list": affinity_list,
        },
        "slurm_environment": slurm_environment,
        "cpu_check": {
            "status": check_status,
            "minimum_cpus_per_gpu": minimum_cpus_per_gpu,
            "gpu_count_basis": gpu_count_basis,
            "minimum_cpu_count": minimum_cpu_count,
            "available_cpu_count": allocated_cpu_count,
            "message": check_message,
        },
    }


def load_resource_snapshot(log_dir: Path) -> dict[str, Any] | None:
    """Load a resource snapshot, returning ``None`` when it is unavailable."""
    try:
        data = json.loads((log_dir / RESOURCE_SNAPSHOT_FILENAME).read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def format_resource_summary(snapshot: Mapping[str, Any], snapshot_path: Path) -> str:
    """Render a compact startup summary intended for humans and log-reading agents."""
    nodes = snapshot["nodes"]
    hardware = snapshot["hardware"]
    gpus = snapshot["gpus"]
    cpus = snapshot["cpus"]
    cpu_check = snapshot["cpu_check"]

    allocated_cpus = cpus["allocated_total"]
    gpu_count_basis = cpu_check["gpu_count_basis"]
    if allocated_cpus is not None and gpu_count_basis:
        cpu_gpu_ratio = f"{allocated_cpus / gpu_count_basis:.2f} allocated"
    else:
        cpu_gpu_ratio = "unknown"

    per_node = ", ".join(str(value) for value in cpus["allocated_per_node"]) or "unknown"
    affinity = cpus["process_affinity_count"] if cpus["process_affinity_count"] is not None else "unknown"
    affinity_list = f" [{cpus['process_affinity_list']}]" if cpus["process_affinity_list"] else ""
    node_names = ", ".join(nodes["names"])

    return "\n".join(
        (
            "=" * 60,
            "Runtime Resource Summary",
            "=" * 60,
            f"  Nodes: {nodes['count']} ({node_names})",
            (
                f"  CPU hardware: {hardware['cpu_model'] or 'unknown'} "
                f"({hardware['host_logical_cpus'] or 'unknown'} logical, {hardware['architecture']})"
            ),
            (
                f"  GPUs: {gpus['backend_total']} backend / {gpus['configured_total']} configured "
                f"({gpus['configured_per_node']}/node, {gpus['type']})"
            ),
            f"  CPUs: {allocated_cpus if allocated_cpus is not None else 'unknown'} total; per node: {per_node}",
            f"  CPU affinity: {affinity}{affinity_list}",
            (
                f"  CPU/GPU: {cpu_gpu_ratio} vs {cpu_check['minimum_cpus_per_gpu']:g} minimum "
                f"— {str(cpu_check['status']).upper()}"
            ),
            f"  Snapshot: {snapshot_path}",
            "=" * 60,
        )
    )


def _update_job_metadata(output_dir: Path, job_id: str, snapshot: dict[str, Any]) -> None:
    metadata_path = output_dir / f"{job_id}.json"
    if not metadata_path.exists():
        return

    try:
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            return
        resources = metadata.setdefault("resources", {})
        if not isinstance(resources, dict):
            return
        resources["cpu_allocation"] = snapshot["cpus"]
        resources["cpu_check"] = snapshot["cpu_check"]
        temp_path = metadata_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(metadata, indent=2) + "\n")
        temp_path.replace(metadata_path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        logger.warning("Failed to add CPU allocation to job metadata %s: %s", metadata_path, error)


def record_resource_snapshot(config: SrtConfig, runtime: RuntimeContext) -> dict[str, Any] | None:
    """Capture, persist, and log the effective allocation without failing a job."""
    try:
        snapshot = collect_resource_snapshot(config, runtime)
        snapshot_path = runtime.log_dir / RESOURCE_SNAPSHOT_FILENAME
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
        _update_job_metadata(runtime.log_dir.parent, runtime.job_id, snapshot)

        logger.info("\n%s", format_resource_summary(snapshot, snapshot_path))
        cpu_check = snapshot["cpu_check"]
        if cpu_check["status"] == "warning":
            logger.warning("CPU ALLOCATION WARNING: %s", cpu_check["message"])
        elif cpu_check["status"] == "unknown":
            logger.warning("CPU allocation check unavailable: %s", cpu_check["message"])
        else:
            logger.info("CPU allocation check: %s", cpu_check["message"])
        return snapshot
    except Exception as error:
        logger.warning("Failed to capture resource snapshot: %s", error)
        return None
