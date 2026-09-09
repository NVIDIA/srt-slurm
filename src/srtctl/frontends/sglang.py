# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang Model Gateway router frontend."""

from typing import Any, ClassVar

from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.core.topology import Process
from srtctl.frontends.static_router import StaticRouterFrontend
from srtctl.ports import SGLANG_ROUTER_METRICS_PORT


def router_metrics_port(frontend_args: dict[str, Any] | None) -> int:
    """The port the Model Gateway serves Prometheus metrics on: the recipe's ``prometheus-port`` or srtctl's default."""
    args = frontend_args or {}
    for key in ("prometheus-port", "prometheus_port"):
        if args.get(key) is not None:
            return int(args[key])
    return SGLANG_ROUTER_METRICS_PORT


class SGLangFrontend(StaticRouterFrontend):
    """SGLang Model Gateway static router."""

    type: ClassVar[str] = "sglang"
    backend_type: ClassVar[str] = "sglang"
    executable: ClassVar[tuple[str, ...]] = ("python", "-m", "sglang_router.launch_router")
    pd_flag: ClassVar[str] = "--pd-disaggregation"
    process_name: ClassVar[str] = "sglang_router"
    log_label: ClassVar[str] = "router"
    # Preserve the historical launch shape used by dry-run/topology callers
    # that construct the frontend before populating worker processes.
    allow_empty_workers: ClassVar[bool] = True

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
        del backend, backend_processes
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
        del network_interface
        return get_hostname_ip(node)

    def start_process(self, **kwargs: Any) -> Any:
        return start_srun_process(**kwargs)
