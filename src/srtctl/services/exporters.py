# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: dcgm-exporter`` and ``type: node-exporter``: the metrics exporters tachometer scrapes.

Implied on every worker node whenever tachometer runs with its default exporters
(``observability.tachometer.default_exporters``, on by default); declaring one by
name changes its container (a ``srtslurm.yaml`` alias for air-gapped clusters),
its command, or drops it with ``enabled: false``. Both images are distroless, so
they launch without the bash wrapper. Non-critical: a dead exporter costs its
metrics, never the run.

The power-telemetry path (``telemetry.enabled``) owns its own DCGM exporter and
is untouched; the implicit ``dcgm-exporter`` service steps aside when it is on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.services.config import ServiceConfig

DCGM_EXPORTER_IMAGE = "nvcr.io#nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04"
NODE_EXPORTER_IMAGE = "quay.io#prometheus/node-exporter:v1.8.2"
# Deliberately off the conventional 9400/9100: managed clusters may run host exporters there.
DCGM_EXPORTER_PORT = 9401
NODE_EXPORTER_PORT = 9101


class _ExporterKind(ServiceKind):
    builds_command = True
    default_start = "after_frontend"
    default_critical = False
    default_placement = "workers"
    use_bash_wrapper = False
    option_keys = ("port", "collect_interval_ms")


@register_service("dcgm-exporter")
class DcgmExporterService(_ExporterKind):
    """NVIDIA DCGM exporter; samples NVML as often as tachometer scrapes (``options.collect_interval_ms``)."""

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        port = int(service.options.get("port", DCGM_EXPORTER_PORT))
        interval = int(service.options.get("collect_interval_ms", 1000))
        return ["dcgm-exporter", f"--collect-interval={interval}", "--address", f":{port}", *service.args]

    def container_fallback(self, config) -> str | None:
        return DCGM_EXPORTER_IMAGE


@register_service("node-exporter")
class NodeExporterService(_ExporterKind):
    """Prometheus node exporter with the CPU, InfiniBand, and meminfo collectors."""

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        port = int(service.options.get("port", NODE_EXPORTER_PORT))
        return [
            "/bin/node_exporter",
            f"--web.listen-address=:{port}",
            "--collector.disable-defaults",
            "--collector.cpu",
            "--collector.infiniband",
            "--collector.meminfo",
            *service.args,
        ]

    def container_fallback(self, config) -> str | None:
        return NODE_EXPORTER_IMAGE
