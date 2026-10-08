# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: llm-d-sidecar``: llm-d's P/D sidecar in front of every routable decode worker.

The llm-d frontend implies this service for P/D jobs. The sidecar calls the
prefill worker named in ``x-prefiller-host-port``, then passes the returned
``kv_transfer_params`` to its local vLLM, which pulls the prefill KV cache.
It listens on ``Process.proxy_port`` and forwards other routes, including
``/metrics``, to the worker's HTTP port.

Upstream: llm-d-router ``cmd/pd-sidecar`` and ``pkg/sidecar/proxy`` at v0.11.0
(https://github.com/llm-d/llm-d-router/tree/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/sidecar/proxy).
Its ``GET /health`` is independent of vLLM, allowing it to start before workers.
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
_MANAGED_FLAGS = frozenset({"--port", "--model-server-port", "--kv-connector", "--secure-proxy"})


def sidecar_kv_connector(backend: Any) -> str:
    """Select the decode connector's sidecar protocol, or raise ``ValueError``.

    For ``MultiConnector``, use the first supported wrapped connector.
    """
    from srtctl.backends.vllm import VLLMBackend

    # Only vLLM's KV-transfer protocol is supported.
    connectors = backend.kv_connector_classes("decode") if isinstance(backend, VLLMBackend) else ()
    protocol = next((SIDECAR_KV_CONNECTORS[name] for name in connectors if name in SIDECAR_KV_CONNECTORS), None)
    if protocol is None:
        supported = ", ".join(f"{name} ({protocol})" for name, protocol in SIDECAR_KV_CONNECTORS.items())
        raise ValueError(
            f"the llm-d P/D sidecar has no protocol for the {backend.type} decode KV connector "
            f"{' > '.join(connectors) or None!r}; supported (vLLM, alone or in a MultiConnector): {supported}"
        )
    return protocol


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

        conflicts = {
            arg.split("=", 1)[0]
            for arg in [*(service.command or []), *service.args]
            if arg.startswith("--") and arg.split("=", 1)[0] in _MANAGED_FLAGS
        }
        if conflicts:
            raise ValidationError(
                f"services[{service.name}].command/args sets {', '.join(sorted(conflicts))}, "
                "which srtctl manages for the worker's ports and KV-transfer protocol"
            )
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
        """Use the frontend image, which also contains ``pd-sidecar``."""
        return config.frontend.container_image

    def attaches_to(self, process: Process) -> bool:
        return process.proxy_port is not None

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        command = list(service.command) if service.command is not None else list(self.default_command or ())
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
