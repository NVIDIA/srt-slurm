# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the CPU and GPU allocation visible to a running SLURM job."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.core.fingerprint import cpu_model_from_cpuinfo

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
        return cpu_model_from_cpuinfo(Path("/proc/cpuinfo").read_text(errors="replace"))
    return None


def _worker_gpu_count(config: SrtConfig) -> int:
    resources = config.resources
    return resources.prefill_gpus + resources.decode_gpus + resources.num_agg * resources.gpus_per_agg


def _backend_gpus_by_node(config: SrtConfig, runtime: RuntimeContext) -> dict[str, int]:
    """Return backend GPU demand by node, using the same endpoint placement as workers."""
    resources = config.resources
    try:
        if runtime.nodes.het:
            from srtctl.core.topology import allocate_endpoints_het

            endpoints = allocate_endpoints_het(
                num_prefill=resources.num_prefill,
                gpus_per_prefill=resources.gpus_per_prefill,
                prefill_nodes=runtime.nodes.prefill_group,
                num_decode=resources.num_decode,
                gpus_per_decode=resources.gpus_per_decode,
                decode_nodes=runtime.nodes.decode_group,
                gpus_per_node=resources.gpus_per_node,
            )
        else:
            endpoints = config.backend.allocate_endpoints(
                num_prefill=resources.num_prefill,
                num_decode=resources.num_decode,
                num_agg=resources.num_agg,
                gpus_per_prefill=resources.gpus_per_prefill,
                gpus_per_decode=resources.gpus_per_decode,
                gpus_per_agg=resources.gpus_per_agg,
                gpus_per_node=resources.gpus_per_node,
                available_nodes=runtime.nodes.worker,
                spread_workers=resources.spread_workers,
            )
    except Exception:
        logger.debug("Failed to derive backend node GPU allocation", exc_info=True)
        return {}

    gpu_counts: dict[str, int] = {}
    for endpoint in endpoints:
        for node in endpoint.nodes:
            gpu_counts[node] = gpu_counts.get(node, 0) + len(endpoint.gpu_indices)
    return gpu_counts


def _backend_node_allocations(
    node_names: Sequence[str],
    cpus_per_node: Sequence[int],
    backend_gpu_counts: Mapping[str, int],
) -> list[dict[str, int | str]]:
    """Pair SLURM per-node CPU counts with backend GPU demand when ordering is known."""
    if len(cpus_per_node) != len(node_names):
        return []

    cpus_by_node = dict(zip(node_names, cpus_per_node, strict=True))
    return [
        {
            "node": node,
            "allocated_cpus": cpus_by_node[node],
            "backend_gpus": backend_gpu_counts[node],
        }
        for node in node_names
        if backend_gpu_counts.get(node, 0) > 0
    ]


def collect_resource_snapshot(
    config: SrtConfig,
    runtime: RuntimeContext,
    *,
    environ: Mapping[str, str] | None = None,
    affinity: tuple[int | None, str | None] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable snapshot of the job's effective resources."""
    env = os.environ if environ is None else environ
    node_names = tuple(dict.fromkeys((runtime.nodes.infra, *runtime.nodes.worker)))
    node_count = len(node_names)
    configured_gpu_count = node_count * config.resources.gpus_per_node
    worker_gpu_count = _worker_gpu_count(config)
    gpu_count_basis = worker_gpu_count or configured_gpu_count
    cpus_per_node = _allocated_cpus_per_node(env)
    backend_node_allocations = _backend_node_allocations(
        node_names,
        cpus_per_node,
        _backend_gpus_by_node(config, runtime),
    )

    allocated_cpu_count = sum(cpus_per_node) if cpus_per_node else None
    current_node_cpu_count: int | None = None
    with contextlib.suppress(KeyError, TypeError, ValueError):
        current_node_cpu_count = int(env["SLURM_CPUS_ON_NODE"])

    affinity_cpu_count, affinity_list = affinity if affinity is not None else _process_affinity()

    effective_cpu_count: int | None = allocated_cpu_count
    effective_cpu_source = "slurm_allocation" if allocated_cpu_count is not None else "unknown"
    effective_node: str | None = None
    if node_count == 1:
        local_cpu_counts = [
            (source, count)
            for source, count in (
                ("slurm_allocation", allocated_cpu_count),
                ("slurm_current_node", current_node_cpu_count),
                ("process_affinity", affinity_cpu_count),
            )
            if count is not None
        ]
        if local_cpu_counts:
            effective_cpu_count = min(count for _, count in local_cpu_counts)
            effective_cpu_source = (
                local_cpu_counts[0][0] if len(local_cpu_counts) == 1 else "minimum_of_available_local_signals"
            )
    elif backend_node_allocations:
        most_constrained = min(
            backend_node_allocations,
            key=lambda item: (int(item["allocated_cpus"]) / int(item["backend_gpus"]), int(item["allocated_cpus"])),
        )
        effective_cpu_count = int(most_constrained["allocated_cpus"])
        gpu_count_basis = int(most_constrained["backend_gpus"])
        effective_cpu_source = "minimum_backend_node_allocation"
        effective_node = str(most_constrained["node"])

    minimum_cpu_count = gpu_count_basis

    if effective_cpu_count is None:
        check_status = "unknown"
        check_message = "Could not determine effective CPU capacity from SLURM or process affinity."
    elif effective_cpu_count < minimum_cpu_count:
        check_status = "warning"
        node_context = f" on backend node {effective_node}" if effective_node else ""
        node_suffix = " on that node" if effective_node else ""
        check_message = (
            f"Effective CPU capacity may be too small{node_context}: "
            f"{effective_cpu_count} CPUs for {gpu_count_basis} backend GPUs{node_suffix}; "
            f"the baseline of 1 CPU/GPU requires at least {minimum_cpu_count}. "
            "Increase the SLURM CPU request (for example, cpus-per-task/cpus-per-gpu) or use an exclusive node."
        )
    else:
        check_status = "ok"
        if effective_node:
            check_message = (
                f"Effective CPU capacity meets the baseline on backend node {effective_node}: "
                f"{effective_cpu_count} CPUs for {gpu_count_basis} backend GPUs "
                f"(minimum {minimum_cpu_count})."
            )
        else:
            check_message = (
                f"Effective CPU capacity meets the baseline: {effective_cpu_count} CPUs for "
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
            "effective_for_check": effective_cpu_count,
            "backend_node_allocations": backend_node_allocations,
        },
        "slurm_environment": slurm_environment,
        "cpu_check": {
            "status": check_status,
            "gpu_count_basis": gpu_count_basis,
            "minimum_cpu_count": minimum_cpu_count,
            "available_cpu_count": effective_cpu_count,
            "available_cpu_source": effective_cpu_source,
            "node": effective_node,
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
    effective_cpus = cpu_check["available_cpu_count"]
    gpu_count_basis = cpu_check["gpu_count_basis"]
    if effective_cpus is not None and gpu_count_basis:
        cpu_gpu_ratio = f"{effective_cpus / gpu_count_basis:.2f} effective"
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
            f"  CPU/GPU: {cpu_gpu_ratio} vs 1 minimum — {str(cpu_check['status']).upper()}",
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
