# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang Model Gateway router frontend (`frontend.type: sglang-router`).

The router-free single-replica mode is `frontend.type: sglang`; see sglang_direct.py.
"""

from typing import Any, ClassVar

from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.core.topology import Process
from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend
from srtctl.ports import SGLANG_ROUTER_METRICS_PORT


def router_metrics_port(frontend_args: dict[str, Any] | None) -> int:
    """The port the Model Gateway serves Prometheus metrics on: the recipe's ``prometheus-port`` or srtctl's default."""
    args = frontend_args or {}
    for key in ("prometheus-port", "prometheus_port"):
        if args.get(key) is not None:
            return int(args[key])
    return SGLANG_ROUTER_METRICS_PORT


@register_frontend("sglang-router")
class SGLangRouterFrontend(StaticRouterFrontend):
    """SGLang Model Gateway static router (`frontend.type: sglang-router`)."""

    type: ClassVar[str] = "sglang-router"
    required_backend: ClassVar[str | None] = "sglang"
    executable: ClassVar[tuple[str, ...]] = ("python", "-m", "sglang_router.launch_router")
    pd_flag: ClassVar[str] = "--pd-disaggregation"
    process_name: ClassVar[str] = "sglang_router"
    log_label: ClassVar[str] = "router"
    # Preserve the historical launch shape used by dry-run/topology callers
    # that construct the frontend before populating worker processes.
    allow_empty_workers: ClassVar[bool] = True
    # The gateway drops a static worker that does not answer within its startup window, and a
    # large model can load for longer than that, so every worker is probed before the router starts.
    wait_for_workers_before_start: ClassVar[bool] = True

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        """The gateway serves Prometheus on its own listener, not on the routing port."""
        return router_metrics_port(frontend_args)

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
    ) -> list[str]:
        """Expose the gateway's Prometheus listener unless the recipe configured it itself.

        The Model Gateway only starts its metrics server when ``--prometheus-port``
        is given (``PrometheusConfig`` is ``None`` otherwise), and tachometer is on
        by default, so srtctl always asks for it on every interface.
        """
        frontend_args = config.frontend.args or {}
        normalized = {str(key).replace("_", "-") for key in frontend_args}
        managed: list[str] = []
        if "prometheus-port" not in normalized:
            managed.extend(["--prometheus-port", str(SGLANG_ROUTER_METRICS_PORT)])
        if "prometheus-host" not in normalized:
            managed.extend(["--prometheus-host", "0.0.0.0"])
        return managed

    def worker_scheme(self, backend: Any, mode: str) -> str:
        return "grpc" if backend.is_grpc_mode(mode) else "http"

    def resolve_worker_host(self, node: str, network_interface: str | None) -> str:
        return get_hostname_ip(node, network_interface)

    def start_process(self, **kwargs: Any) -> Any:
        return start_srun_process(**kwargs)
