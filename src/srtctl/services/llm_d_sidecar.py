# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: llm-d-sidecar``: llm-d's P/D sidecar in front of every routable decode worker.

Implied by ``frontend.type: llm-d`` when the job has prefill and decode workers.
The Endpoint Picker routes a request to a decode endpoint and names the prefill
worker in the ``x-prefiller-host-port`` header; the sidecar on that decode
endpoint sends the prefill request first, then hands the returned
``kv_transfer_params`` to its own vLLM, whose KV connector pulls the cache from
the prefill worker. The sidecar is what the router addresses for a decode
worker, so it binds the worker's ``Process.proxy_port`` and proxies everything
else (``/metrics`` included) to the worker's HTTP port.

Upstream: llm-d-router ``cmd/pd-sidecar`` and ``pkg/sidecar/proxy`` at v0.11.0
(https://github.com/llm-d/llm-d-router/tree/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/sidecar/proxy).
Its ``GET /health`` answers 200 on its own, so it starts before the workers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from marshmallow import ValidationError

from srtctl.services.config import HttpProbe, ServiceReadinessConfig
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Process
    from srtctl.services.config import ServiceConfig

LLM_D_SIDECAR_TYPE = "llm-d-sidecar"

# The sidecar's --kv-connector protocol for the vLLM connector class the decode workers run.
# Upstream's table of protocols: pkg/sidecar/proxy/options.go (supportedKVConnectors).
SIDECAR_KV_CONNECTORS: dict[str, str] = {
    "NixlConnector": "nixlv2",
}

# Argv shown by dry-run, where there is no worker to read the ports from.
_PREVIEW_PORTS = ["--port=<worker_proxy_port>", "--model-server-port=<worker_http_port>"]


def sidecar_kv_connector(backend: Any) -> str:
    """The ``--kv-connector`` protocol for the decode workers' KV connector; raises ``ValueError`` when unmapped."""
    from srtctl.backends.vllm import VLLMBackend

    # The sidecar drives vLLM's KV transfer protocol; another engine's decode has no row.
    connector = backend.kv_connector_class("decode") if isinstance(backend, VLLMBackend) else None
    try:
        return SIDECAR_KV_CONNECTORS[str(connector)]
    except KeyError:
        supported = ", ".join(f"{name} ({protocol})" for name, protocol in SIDECAR_KV_CONNECTORS.items())
        raise ValueError(
            f"the llm-d P/D sidecar has no protocol for the {backend.type} decode KV connector {connector!r}; "
            f"supported (vLLM): {supported}"
        ) from None


@register_service(LLM_D_SIDECAR_TYPE)
class LLMDSidecarService(ServiceKind):
    """llm-d P/D sidecar, one per routable decode worker; implied by the llm-d frontend."""

    builds_command = True
    default_command = ("pd-sidecar",)
    default_start = "before_workers"
    default_critical = True
    default_placement = "decode"
    default_per = "worker"

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        from srtctl.frontends import FRONTEND_NONE, get_frontend

        proxied = (
            frozenset()
            if config.frontend.type == FRONTEND_NONE
            else get_frontend(config.frontend.type).proxied_worker_modes(config)
        )
        if service.effective_per != "worker" or service.effective_placement not in proxied:
            raise ValidationError(
                f"services[{service.name}] (type {LLM_D_SIDECAR_TYPE}) fronts the workers the frontend proxies "
                f"({', '.join(sorted(proxied)) or 'none for this recipe'}); set placement.node to one of them "
                "and placement.per: worker"
            )

    def container_fallback(self, config: SrtConfig) -> str | None:
        """The router image (``frontend.container_image``), which ships the sidecar next to the EPP."""
        return config.frontend.container_image

    def attaches_to(self, process: Process) -> bool:
        return process.proxy_port is not None

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        command = list(service.command) if service.command is not None else list(self.default_command)
        if ctx.process is None or ctx.config is None or ctx.process.proxy_port is None:
            managed = [*_PREVIEW_PORTS, "--kv-connector=<decode connector>"]
        else:
            connector = sidecar_kv_connector(ctx.config.backend_for_role(ctx.process.endpoint_mode))
            managed = [
                f"--port={ctx.process.proxy_port}",
                f"--model-server-port={ctx.process.http_port}",
                f"--kv-connector={connector}",
            ]
        # The router reaches the sidecar over plain HTTP inside the job.
        return [*command, *managed, "--secure-proxy=false", *service.args]

    def readiness(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> ServiceReadinessConfig | None:
        if ctx.process is None or ctx.process.proxy_port is None:
            return None
        return ServiceReadinessConfig(
            http=HttpProbe(port=ctx.process.proxy_port, path="/health"),
            timeout_seconds=self.default_readiness_timeout,
        )
