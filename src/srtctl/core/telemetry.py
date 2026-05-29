# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from srtctl.core.slurm import get_hostname_ip

if TYPE_CHECKING:
    from srtctl.cli.mixins.frontend_stage import FrontendTopology
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import TelemetryConfig
    from srtctl.core.topology import Process


@dataclass(frozen=True)
class TelemetryEndpoint:
    """One telemetry endpoint entry in the scraper config."""

    name: str
    url: str
    frequency: float
    filter: str | None = None
    node_metadata: dict[str, str] = field(default_factory=dict)
    gpu_metadata: dict[str, dict[str, str]] = field(default_factory=dict)


def generate_telemetry_config(
    *,
    processes: list[Process],
    frontend_topology: FrontendTopology,
    runtime: RuntimeContext,
    telemetry: TelemetryConfig,
) -> str:
    """Generate telemetry TOML from backend and frontend topology."""
    dcgm_exporter = telemetry.dcgm_exporter
    node_exporter = telemetry.node_exporter
    if dcgm_exporter is None or node_exporter is None:
        raise ValueError("Telemetry exporters must be configured before generating telemetry config")

    endpoints: list[TelemetryEndpoint] = []
    physical_nodes: dict[str, list[Process]] = {}
    for process in processes:
        physical_nodes.setdefault(process.node, []).append(process)

    for node in sorted(physical_nodes):
        node_processes = physical_nodes[node]
        node_metadata = {"hostname": node, "job_id": runtime.job_id, "run_name": runtime.run_name}
        node_metadata.update(telemetry.extra_metadata)

        gpu_metadata: dict[str, dict[str, str]] = {}
        for process in node_processes:
            for gpu_idx in sorted(process.gpu_indices):
                gpu_metadata[str(gpu_idx)] = {
                    "worker_index": str(process.endpoint_index),
                    "worker_process": str(process.node_rank),
                    "worker_role": process.endpoint_mode,
                }

        endpoints.append(
            TelemetryEndpoint(
                name=f"dcgm_{node}",
                url=f"http://{node}:{dcgm_exporter.port}/metrics",
                frequency=telemetry.default_frequency,
                filter="dcgm",
                node_metadata=node_metadata,
                gpu_metadata=gpu_metadata,
            )
        )
        endpoints.append(
            TelemetryEndpoint(
                name=f"node_exporter_{node}",
                url=f"http://{node}:{node_exporter.port}/metrics",
                frequency=telemetry.default_frequency,
                filter="node_exporter",
                node_metadata=node_metadata,
            )
        )

    for process in sorted(processes, key=lambda p: (p.endpoint_mode, p.endpoint_index, p.node_rank, p.node)):
        node_ip = get_hostname_ip(process.node, runtime.network_interface)
        node_metadata = {
            "hostname": process.node,
            "worker_index": str(process.endpoint_index),
            "worker_process": str(process.node_rank),
            "worker_role": process.endpoint_mode,
        }
        node_metadata.update(telemetry.extra_metadata)
        endpoints.append(
            TelemetryEndpoint(
                name=f"backend_{process.endpoint_mode}{process.endpoint_index}_rank{process.node_rank}",
                url=f"http://{node_ip}:{process.sys_port}/metrics",
                frequency=telemetry.default_frequency,
                filter="backend",
                node_metadata=node_metadata,
            )
        )

    for frontend_index, node in enumerate(frontend_topology.frontend_nodes):
        node_ip = get_hostname_ip(node, runtime.network_interface)
        node_metadata = {
            "frontend_index": str(frontend_index),
            "hostname": node,
        }
        node_metadata.update(telemetry.extra_metadata)
        endpoints.append(
            TelemetryEndpoint(
                name=f"frontend{frontend_index}",
                url=f"http://{node_ip}:{frontend_topology.frontend_port}/metrics",
                frequency=telemetry.default_frequency,
                filter="frontend",
                node_metadata=node_metadata,
            )
        )

    return _dump_toml(
        endpoints=endpoints,
        storage=f"/logs/{telemetry.storage_subdir}",
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
