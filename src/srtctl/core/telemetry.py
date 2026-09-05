# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tachometer configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from srtctl.core.slurm import get_hostname_ip
from srtctl.ports import FRONTEND_PUBLIC_PORT

if TYPE_CHECKING:
    from srtctl.cli.mixins.frontend_stage import FrontendTopology
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import TachometerConfig, TelemetryExporterConfig
    from srtctl.core.topology import Process


# Tachometer owns its final storage directory and rejects a pre-existing leaf
# (see tachometer-scraper `parse_storage`). srtctl must therefore create only the
# parent and hand the scraper a not-yet-existing leaf. This mirrors what the
# direct-host path already does in templates/local_lifecycle.sh.j2.
TACHOMETER_STORAGE_PARENT = "raw"
TACHOMETER_STORAGE_LEAF = "scrape"


@dataclass(frozen=True)
class TelemetryEndpoint:
    """One telemetry endpoint entry in the scraper config."""

    name: str
    url: str
    frequency: float
    filter: str | None = None
    node_metadata: dict[str, str] = field(default_factory=dict)
    gpu_metadata: dict[str, dict[str, str]] = field(default_factory=dict)


def generate_tachometer_config(
    *,
    processes: list[Process],
    frontend_topology: FrontendTopology,
    runtime: RuntimeContext,
    tachometer: TachometerConfig,
    dcgm_exporter: TelemetryExporterConfig | None = None,
    frontend_type: str = "dynamo",
    exclude_urls: frozenset[str] | set[str] = frozenset(),
) -> str:
    """Generate Tachometer TOML from backend and frontend topology.

    ``exclude_urls`` is the set of ``/metrics`` URLs the benchmark client
    already polls (``AIPERF_SERVER_METRICS_URLS``). Tachometer scrapes the
    complement so a worker endpoint is never double-polled — the extra scrape
    load has previously made a submission irreproducible. Frontend, DCGM and
    node-exporter endpoints are never excluded: the frontend scrape is cheap
    and Tachometer is the only whole-window, per-replica capture of it.
    """
    dcgm_exporter = dcgm_exporter or tachometer.resolved_dcgm_exporter
    node_exporter = tachometer.resolved_node_exporter
    endpoints: list[TelemetryEndpoint] = []
    physical_nodes: dict[str, list[Process]] = {}
    for process in processes:
        physical_nodes.setdefault(process.node, []).append(process)

    for node in sorted(physical_nodes):
        node_processes = physical_nodes[node]
        node_metadata = {"hostname": node, "job_id": runtime.job_id, "run_name": runtime.run_name}
        node_metadata.update(tachometer.extra_metadata)

        gpu_metadata: dict[str, dict[str, str]] = {}
        for process in node_processes:
            for gpu_idx in sorted(process.gpu_indices):
                gpu_metadata[str(gpu_idx)] = {
                    "worker_index": str(process.endpoint_index),
                    "worker_process": str(process.node_rank),
                    "worker_role": process.endpoint_mode,
                }

        if dcgm_exporter is not None:
            endpoints.append(
                TelemetryEndpoint(
                    name=f"dcgm_{node}",
                    url=f"http://{node}:{dcgm_exporter.port}/metrics",
                    frequency=tachometer.default_frequency,
                    filter="dcgm",
                    node_metadata=node_metadata,
                    gpu_metadata=gpu_metadata,
                )
            )
        if node_exporter is not None:
            endpoints.append(
                TelemetryEndpoint(
                    name=f"node_exporter_{node}",
                    url=f"http://{node}:{node_exporter.port}/metrics",
                    frequency=tachometer.default_frequency,
                    filter="node_exporter",
                    node_metadata=node_metadata,
                )
            )

    for process in sorted(processes, key=lambda p: (p.endpoint_mode, p.endpoint_index, p.node_rank, p.node)):
        # Every rank is a target (vLLM agg followers excepted below): follower
        # metadata columns keep rows distinguishable, and rank coverage is
        # exactly what the physical-process client list provides for vLLM DP.
        if frontend_type == "vllm" and process.endpoint_mode == "agg" and not process.is_leader:
            continue
        if frontend_type == "vllm-router" and process.http_port <= 0:
            continue
        node_ip = get_hostname_ip(process.node, runtime.network_interface)
        if frontend_type == "vllm" and process.endpoint_mode == "agg":
            port = FRONTEND_PUBLIC_PORT
        elif frontend_type == "vllm-router":
            port = process.http_port
        else:
            port = process.sys_port
        url = f"http://{node_ip}:{port}/metrics"
        if url in exclude_urls:
            # The benchmark client already polls this endpoint on its own
            # cadence; scrape the complement instead of double-polling.
            continue
        node_metadata = {
            "hostname": process.node,
            "worker_index": str(process.endpoint_index),
            "worker_process": str(process.node_rank),
            "worker_role": process.endpoint_mode,
        }
        node_metadata.update(tachometer.extra_metadata)
        endpoints.append(
            TelemetryEndpoint(
                name=f"backend_{process.endpoint_mode}{process.endpoint_index}_rank{process.node_rank}",
                url=url,
                frequency=tachometer.default_frequency,
                filter="backend",
                node_metadata=node_metadata,
            )
        )

    frontend_nodes = frontend_topology.frontend_nodes
    if frontend_type == "vllm":
        # Direct vLLM has no separate frontend process. Its public endpoint is
        # the aggregate leader, which may differ from the Slurm/orchestrator
        # head recorded in FrontendTopology.
        agg_leader_nodes = [
            process.node
            for process in sorted(processes, key=lambda p: (p.endpoint_index, p.node_rank, p.node))
            if process.endpoint_mode == "agg" and process.is_leader
        ]
        if agg_leader_nodes:
            frontend_nodes = list(dict.fromkeys(agg_leader_nodes))

    for frontend_index, node in enumerate(frontend_nodes):
        node_ip = get_hostname_ip(node, runtime.network_interface)
        node_metadata = {
            "frontend_index": str(frontend_index),
            "hostname": node,
        }
        node_metadata.update(tachometer.extra_metadata)
        endpoints.append(
            TelemetryEndpoint(
                name=f"frontend{frontend_index}",
                url=f"http://{node_ip}:{frontend_topology.frontend_port}/metrics",
                frequency=tachometer.default_frequency,
                filter="frontend",
                node_metadata=node_metadata,
            )
        )

    return _dump_toml(
        endpoints=endpoints,
        storage=str(runtime.log_dir / tachometer.storage_subdir / TACHOMETER_STORAGE_PARENT / TACHOMETER_STORAGE_LEAF),
    )


def _dump_toml(*, endpoints: list[TelemetryEndpoint], storage: str) -> str:
    """Render a compact TOML document without extra dependencies."""
    lines = [f"storage = {json.dumps(storage)}", ""]
    for endpoint in endpoints:
        lines.append("[[endpoints]]")
        lines.append(f"name = {json.dumps(endpoint.name)}")
        lines.append(f"url = {json.dumps(endpoint.url)}")
        lines.append(f"frequency = {endpoint.frequency}")
        if endpoint.filter is not None:
            lines.append(f"filter = {json.dumps(endpoint.filter)}")
        if endpoint.node_metadata:
            lines.append("[endpoints.node_metadata]")
            for key, value in sorted(endpoint.node_metadata.items()):
                lines.append(f"{json.dumps(key)} = {json.dumps(value)}")
        if endpoint.gpu_metadata:
            lines.append("[endpoints.gpu_metadata]")
            for gpu_idx, metadata in sorted(endpoint.gpu_metadata.items(), key=lambda item: int(item[0])):
                fields = ", ".join(f"{json.dumps(k)} = {json.dumps(v)}" for k, v in sorted(metadata.items()))
                lines.append(f"{json.dumps(gpu_idx)} = {{ {fields} }}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
