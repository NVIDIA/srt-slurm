# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official AToMesh router frontend for native ATOM workers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.frontends.static_router import StaticRouterFrontend

if TYPE_CHECKING:
    from srtctl.core.topology import Process


class AtomeshFrontend(StaticRouterFrontend):
    """Route aggregate or disaggregated traffic to native ATOM servers."""

    type: ClassVar[str] = "atomesh"
    backend_type: ClassVar[str] = "atom"
    executable: ClassVar[tuple[str, ...]] = ("atomesh", "launch")
    pd_flag: ClassVar[str] = "--pd-disaggregation"
    process_name: ClassVar[str] = "atomesh"

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """ATOM exposes transfer topology through ``/kv_transfer_info``."""
        del backend, process

    def get_pre_start_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        return [
            f"{worker.url.rstrip('/')}/health"
            for worker in self.collect_workers(backend, backend_processes, network_interface)
        ]

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
    ) -> list[str]:
        del backend, backend_processes
        normalized = {str(key).replace("_", "-") for key in (config.frontend.args or {})}
        if "backend" in normalized:
            raise ValueError("frontend.args.backend is managed by srtctl for atomesh")
        return ["--backend", "atom"]
