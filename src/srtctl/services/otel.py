# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-local OTLP capture for Dynamo's native request-lifecycle spans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import yaml
from marshmallow import ValidationError

from srtctl.ports import OTEL_GRPC_PORT
from srtctl.services.config import ServiceReadinessConfig, TcpProbe
from srtctl.services.exporters import resolve_host_binary
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.services.config import ServiceConfig

OTEL_SERVICE_NAME = "otel-collector"
DEFAULT_OTEL_BINARY = "configs/otelcol-contrib"
OTEL_CONFIG_NAME = "otel-collector.yaml"


def lifecycle_otel_env(services: Sequence[ServiceConfig], head_ip: str, component: str) -> dict[str, str]:
    """Defaults for Dynamo launches; explicit recipe environment wins at the caller."""
    service = next((s for s in services if s.name == OTEL_SERVICE_NAME), None)
    if service is not None and (not service.enabled or service.type != "otel-collector"):
        return {}
    port = _port(service) if service is not None else OTEL_GRPC_PORT
    return {
        "DYN_LIFECYCLE_TRACE_ENABLED": "1",
        "OTEL_EXPORT_ENABLED": "1",
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://{head_ip}:{port}",
        "OTEL_SERVICE_NAME": f"dynamo-{component}",
    }


def _port(service: ServiceConfig) -> int:
    return int(service.options.get("port", OTEL_GRPC_PORT))


@register_service("otel-collector")
class OtelCollectorService(ServiceKind):
    """Host-native collector, ready before Dynamo starts and stopped after it flushes."""

    builds_command = True
    default_start = "before_workers"
    default_critical = True
    option_keys = ("binary", "port")

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        if service.effective_placement != "head" or service.effective_pool is not None or service.nodes is not None:
            raise ValidationError("otel-collector runs on the head node, where Dynamo's default OTLP endpoint points")
        if service.effective_start != "before_workers":
            raise ValidationError("otel-collector must start before_workers to capture the full lifecycle")
        if service.container is not None:
            raise ValidationError("otel-collector runs host-native; set options.binary instead of container")
        try:
            port = _port(service)
        except (TypeError, ValueError):
            raise ValidationError("otel-collector options.port must be an integer between 1 and 65535") from None
        if not 1 <= port <= 65535:
            raise ValidationError("otel-collector options.port must be between 1 and 65535")

    def host_native(self, service: ServiceConfig) -> bool:
        return True

    def prepare(self, service: ServiceConfig, runtime: RuntimeContext, config: SrtConfig) -> None:
        binary = str(service.options.get("binary", DEFAULT_OTEL_BINARY))
        if resolve_host_binary(binary) is None:
            raise RuntimeError(f"OTel collector binary {binary!r} not found; run make setup ARCH=<compute_arch>")
        output = runtime.log_dir / "otel"
        output.mkdir(parents=True, exist_ok=True)
        signals = ("traces", "logs")
        collector_config = {
            "receivers": {"otlp": {"protocols": {"grpc": {"endpoint": f"0.0.0.0:{_port(service)}"}}}},
            "exporters": {
                f"file/{signal}": {
                    "path": str(output / f"{signal}.jsonl"),
                    "format": "json",
                    "append": True,
                    "flush_interval": "1s",
                }
                for signal in signals
            },
            "service": {
                "telemetry": {"logs": {"level": "info"}, "metrics": {"level": "none"}},
                "pipelines": {
                    signal: {"receivers": ["otlp"], "processors": [], "exporters": [f"file/{signal}"]}
                    for signal in signals
                },
            },
        }
        (runtime.log_dir / OTEL_CONFIG_NAME).write_text(yaml.safe_dump(collector_config, sort_keys=False))

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        binary = str(service.options.get("binary", DEFAULT_OTEL_BINARY))
        resolved = resolve_host_binary(binary)
        return [str(resolved or binary), "--config", str(ctx.runtime.log_dir / OTEL_CONFIG_NAME)]

    def readiness(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> ServiceReadinessConfig:
        return ServiceReadinessConfig(tcp=TcpProbe(port=_port(service)))
