# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: lmcache-server``: the LMCache multiprocess server, one per worker node.

The vLLM connector (``connector: lmcache-mp``) only talks to the server on its
own node, so the kind runs an instance on every worker node, before workers,
and gates on the server's HTTP healthcheck. ``args`` are appended to the fixed
command, which sets only the ports srtctl owns. LMCache must be installed in
the job container: the connector is imported inside the vLLM worker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.ports import LMCACHE_HTTP_PORT, LMCACHE_SERVER_PORT
from srtctl.services.config import HttpProbe, ServiceReadinessConfig
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.services.config import ServiceConfig


@register_service("lmcache-server")
class LMCacheServerService(ServiceKind):
    """LMCache MP server on each worker node; ``args`` are appended to the fixed command."""

    builds_command = True
    default_start = "before_workers"
    default_critical = True
    default_placement = "workers"
    # L1 preallocation pins host memory; large pools outlast the base class's 120 s.
    default_readiness_timeout = 300

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        return [
            "lmcache",
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            str(LMCACHE_SERVER_PORT),
            "--http-host",
            "0.0.0.0",
            "--http-port",
            str(LMCACHE_HTTP_PORT),
            *service.args,
        ]

    def readiness(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> ServiceReadinessConfig | None:
        return ServiceReadinessConfig(
            http=HttpProbe(port=LMCACHE_HTTP_PORT, path="/healthcheck"),
            timeout_seconds=self.default_readiness_timeout,
        )
