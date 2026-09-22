# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tachometer configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from srtctl.core.ip_utils import url_host
from srtctl.core.slurm import get_hostname_ip

if TYPE_CHECKING:
    from collections.abc import Sequence

    from srtctl.cli.mixins.frontend_stage import FrontendTopology
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import TachometerConfig
    from srtctl.core.topology import Process


# Tachometer owns its final storage directory and rejects a pre-existing leaf
# (see tachometer-scraper `parse_storage`). srtctl must therefore create only the
# parent and hand the scraper a not-yet-existing leaf.
TACHOMETER_STORAGE_PARENT = "raw"
TACHOMETER_STORAGE_LEAF = "scrape"


@dataclass(frozen=True)
class ServiceMetricsTarget:
    """One node of a ``services[]`` entry that serves Prometheus metrics.

    ``endpoint`` is the scraper endpoint name prefix (the service name unless the
    kind sets one); ``gpu_metadata`` asks for the per-GPU worker labels DCGM rows
    need. Built by ``TelemetryStageMixin._service_metrics_targets``.
    """

    service: str
    node: str
    url: str
    filter: str = "passthrough"
    endpoint: str | None = None
    gpu_metadata: bool = False

    @property
    def endpoint_name(self) -> str:
        return f"{self.endpoint or self.service}_{self.node}"


@dataclass(frozen=True)
class TelemetryEndpoint:
    """One telemetry endpoint entry in the scraper config."""

    name: str
    url: str
    collect_interval_ms: int
    filter: str | None = None
    node_metadata: dict[str, str] = field(default_factory=dict)
    gpu_metadata: dict[str, dict[str, str]] = field(default_factory=dict)


def generate_tachometer_config(
    *,
    processes: list[Process],
    frontend_topology: FrontendTopology,
    runtime: RuntimeContext,
    tachometer: TachometerConfig,
    frontend_type: str = "dynamo",
    frontend_metrics_port: int | None = None,
    service_targets: Sequence[ServiceMetricsTarget] = (),
) -> str:
    """Generate Tachometer TOML from the worker and frontend topology plus the services' metrics.

    Workers and the frontend are described by ``processes`` and
    ``frontend_topology``; everything else that serves metrics is a
    ``services[]`` entry with a ``metrics`` annotation (its own or its kind's),
    resolved to one ``ServiceMetricsTarget`` per node by the telemetry stage.
    The DCGM, node and process exporters arrive that way too, so a job with no
    workers (``frontend.type: none``) is scraped wherever its services run.

    Every endpoint is scraped even when the benchmark client polls the same
    URL (``AIPERF_SERVER_METRICS_URLS``): double-polling has been validated
    as harmless, and unconditional coverage keeps Tachometer the one
    whole-window, per-replica capture regardless of what the client does.

    ``frontend_type: trtllm_serve`` targets a different surface than Dynamo:
    workers bind only their OpenAI ``http_port`` (leaders; the DYN_SYSTEM_PORT
    sys-ports are never created in this mode) and both the workers and the
    disaggregated orchestrator serve Prometheus at ``/prometheus/metrics`` —
    the worker's ``/metrics`` route is JSON iteration stats and the
    orchestrator registers no ``/metrics`` route at all. Endpoint names keep
    the Dynamo pattern (``backend_{mode}{index}_rank{rank}``, ``frontend{i}``)
    so downstream grouping is identical across the two frontends. Aggregate
    trtllm-serve is out of scope (disagg-only coverage).

    ``frontend_type: sglang`` (the Model Gateway over native ``sglang.launch_server``
    workers) has no Dynamo system ports either: worker leaders serve ``/metrics``
    on their OpenAI ``http_port`` (srtctl passes ``--enable-metrics``; followers
    of a multi-node worker serve nothing), and the gateway serves Prometheus on
    its own listener (``frontend_metrics_port``, ``--prometheus-port``), not on
    the routing port.
    """
    from srtctl.frontends import FRONTEND_NONE, get_frontend

    # The frontend says which rank serves metrics on which port and at what path;
    # a services-only job has no frontend and no worker processes.
    frontend = None if frontend_type == FRONTEND_NONE else get_frontend(frontend_type)
    metrics_path = frontend.metrics_path if frontend is not None else "/metrics"
    endpoints: list[TelemetryEndpoint] = []
    # Per-GPU worker labels, attached to DCGM targets so GPU rows carry the rank that owns the GPU.
    gpu_metadata_by_node: dict[str, dict[str, dict[str, str]]] = {}
    for process in processes:
        for gpu_idx in sorted(process.gpu_indices):
            gpu_metadata_by_node.setdefault(process.node, {})[str(gpu_idx)] = {
                "worker_index": str(process.endpoint_index),
                "worker_process": str(process.node_rank),
                "worker_role": process.endpoint_mode,
            }

    for process in sorted(processes, key=lambda p: (p.endpoint_mode, p.endpoint_index, p.node_rank, p.node)):
        # Every rank that serves metrics is a target: follower metadata columns
        # keep rows distinguishable, and rank coverage is exactly what the
        # physical-process client list provides for vLLM DP. Which ranks serve
        # (Dynamo: every rank on its system port; native servers: the leader or
        # each routable pool on its HTTP port) is the frontend's call.
        port = frontend.worker_metrics_port(process, runtime) if frontend is not None else None
        if port is None:
            continue
        node_ip = get_hostname_ip(process.node, runtime.network_interface)
        url = f"http://{url_host(node_ip)}:{port}{metrics_path}"
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
                collect_interval_ms=tachometer.collect_interval_ms,
                filter="backend",
                node_metadata=node_metadata,
            )
        )

    # A services-only job has no frontend process, so nothing listens on the
    # frontend port. A frontend whose worker is itself the endpoint (direct vLLM
    # or SGLang) has no separate process either: the endpoint is the aggregate
    # leader's node, which may differ from the orchestrator head in FrontendTopology.
    frontend_nodes: list[str] = []
    if frontend is not None:
        frontend_nodes = frontend.direct_endpoint_nodes(processes) or list(frontend_topology.frontend_nodes)

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
                url=f"http://{url_host(node_ip)}:{frontend_metrics_port or frontend_topology.frontend_port}{metrics_path}",
                collect_interval_ms=tachometer.collect_interval_ms,
                filter="frontend",
                node_metadata=node_metadata,
            )
        )

    for target in service_targets:
        node_metadata = {
            "hostname": target.node,
            "job_id": runtime.job_id,
            "run_name": runtime.run_name,
            "service": target.service,
        }
        node_metadata.update(tachometer.extra_metadata)
        endpoints.append(
            TelemetryEndpoint(
                name=target.endpoint_name,
                url=target.url,
                collect_interval_ms=tachometer.collect_interval_ms,
                filter=target.filter,
                node_metadata=node_metadata,
                gpu_metadata=gpu_metadata_by_node.get(target.node, {}) if target.gpu_metadata else {},
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
        lines.append(f"collect_interval_ms = {endpoint.collect_interval_ms}")
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
