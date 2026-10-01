# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official AToMesh router frontend for native ATOM workers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process


@register_frontend("atomesh")
class AtomeshFrontend(StaticRouterFrontend):
    """Route aggregate or disaggregated traffic to native ATOM servers."""

    type: ClassVar[str] = "atomesh"
    required_backend: ClassVar[str] = "atom"
    executable: ClassVar[tuple[str, ...]] = ("atomesh", "launch")
    pd_flag: ClassVar[str] = "--pd-disaggregation"
    process_name: ClassVar[str] = "atomesh"
    wait_for_workers_before_start: ClassVar[bool] = True

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> None:
        """Native ATOM workers do not expose the Prometheus metrics endpoint."""

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """ATOM exposes transfer topology through ``/kv_transfer_info``."""
        return None

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
    ) -> list[str]:
        normalized = {str(key).replace("_", "-") for key in (config.frontend.args or {})}
        if "backend" in normalized:
            raise ValueError("frontend.args.backend is managed by srtctl for atomesh")
        return ["--backend", "atom"]
