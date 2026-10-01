# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shepherd Model Gateway router frontend (`frontend.type: smg`).

SMG (https://github.com/smg-project/smg) takes its workers as static URLs on the
command line (``smg launch --worker-urls ...`` or ``--pd-disaggregation --prefill
URL [BOOTSTRAP_PORT] --decode URL``), lists them at ``GET /workers`` with
``stats.{prefill,decode,regular}_count``, and detects each HTTP worker's engine
itself, so the static-router base covers launch, readiness, and worker shape for
every backend.
"""

from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend
from srtctl.ports import SMG_METRICS_PORT

if TYPE_CHECKING:
    from srtctl.core.topology import Process


@register_frontend("smg")
class SMGFrontend(StaticRouterFrontend):
    """Shepherd Model Gateway in front of any engine's direct HTTP workers."""

    type: ClassVar[str] = "smg"
    required_backend: ClassVar[str | None] = None
    executable: ClassVar[tuple[str, ...]] = ("smg", "launch")
    pd_flag: ClassVar[str] = "--pd-disaggregation"
    process_name: ClassVar[str] = "smg"
    # SMG waits at most --worker-startup-timeout-secs (1800 s by default) for a startup
    # worker to register; a large model can load for longer, so every worker is probed first.
    wait_for_workers_before_start: ClassVar[bool] = True

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        """SMG serves Prometheus on its own listener, not on the routing port."""
        return SMG_METRICS_PORT

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: "list[Process]",
    ) -> list[str]:
        """Bind SMG's Prometheus listener on srtctl's port instead of upstream's default."""
        normalized = {str(key).replace("_", "-") for key in (config.frontend.args or {})}
        if "prometheus-port" in normalized:
            raise ValueError("frontend.args.prometheus-port is managed by srtctl for smg")
        return ["--prometheus-port", str(SMG_METRICS_PORT)]
