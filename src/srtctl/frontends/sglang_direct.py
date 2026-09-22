# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct SGLang frontend (`frontend.type: sglang`).

For a single aggregate SGLang worker the OpenAI-compatible HTTP server is the
worker process itself (`sglang.launch_server` bound to the public port). No
router process starts. Use `frontend.type: sglang-router` for several replicas
or a prefill/decode layout, and `frontend.type: dynamo` for KV-aware routing.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from srtctl.core.health import WorkerHealthResult, probe_direct_server
from srtctl.frontends.base import agg_leader_nodes, logical_health_expectations, register_frontend

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process
    from srtctl.services.implicit import EffectiveService

logger = logging.getLogger(__name__)


@register_frontend("sglang")
class SGLangFrontend:
    """Direct SGLang OpenAI server frontend.

    Intentionally narrow: exactly one aggregate worker, which binds the public
    port itself. Readiness is the worker's ``/health`` plus ``/v1/models``.
    """

    required_backend: ClassVar[str | None] = "sglang"
    worker_launch: ClassVar[Literal["dynamo", "direct"]] = "direct"
    expands_node_local_dp: ClassVar[bool] = False

    @property
    def type(self) -> str:
        return "sglang"

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """The one ``sglang.launch_server`` is the endpoint, so it binds the public port."""
        return "public"

    metrics_path: ClassVar[str] = "/metrics"

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        """The aggregate leader binds the public port; its followers serve nothing."""
        if process.endpoint_mode == "agg" and process.is_leader:
            return runtime.frontend_port
        return None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        return runtime.frontend_port if process.is_leader else None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        """The leader's server on the public port carries the control routes; followers have none."""
        return runtime.frontend_port if process.is_leader else None

    def profiling_control_is_leader_only(self, config: Any) -> bool:
        return False

    def direct_endpoint_nodes(self, processes: list[Process]) -> list[str]:
        return agg_leader_nodes(processes)

    def worker_ready_port(self, process: Process) -> int:
        return process.sys_port

    def probe_ready(self, host: str, port: int, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        """The worker's own /health, then /v1/models must list the model."""
        return probe_direct_server(host, port)

    def health_expectations(self, config: Any, processes: list[Process] | None) -> tuple[int, int, str]:
        return logical_health_expectations(config)

    def validate(self, config: Any) -> None:
        """One aggregate ``sglang.launch_server`` owns the public port.

        Several replicas or a prefill/decode layout need ``sglang-router`` (or
        ``dynamo``); a schema 2 recipe that still says ``sglang`` for those is an
        old router recipe and is rejected rather than silently run unbalanced.
        """
        if config.frontend.enable_multiple_frontends:
            raise ValueError(
                "frontend.type: sglang binds sglang.launch_server directly; set frontend.enable_multiple_frontends: false"
            )
        if config.resources.is_disaggregated:
            raise ValueError(
                "frontend.type: sglang supports one aggregate worker only, not a prefill/decode layout. "
                "The SGLang router is frontend.type: sglang-router (renamed in 2.0; `srtctl migrate` rewrites "
                "schema 1 recipes)."
            )
        if config.resources.num_agg != 1:
            raise ValueError(
                f"frontend.type: sglang supports exactly one aggregate worker, got {config.resources.num_agg}. "
                "sglang.launch_server owns the public port directly and there is no router to balance "
                "replicas. Use frontend.type: sglang-router (the SGLang Model Gateway, renamed in 2.0) or dynamo."
            )
        if config.dynamo.sidecar:
            raise ValueError("frontend.type: sglang does not support dynamo.sidecar; use frontend.type: dynamo")

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        return []

    def implied_services(self, config: Any) -> list[EffectiveService]:
        return []

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        return None

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,
    ) -> list[ManagedProcess]:
        if config.backend.type != "sglang":
            raise ValueError(f"frontend.type: sglang requires engine sglang (got {config.backend.type!r})")
        if topology.uses_nginx or len(topology.frontend_nodes) != 1:
            raise ValueError(
                "frontend.type: sglang binds sglang.launch_server directly to the public port; "
                "set frontend.enable_multiple_frontends: false"
            )
        if config.resources.is_disaggregated:
            raise ValueError("frontend.type: sglang supports one aggregate worker only; use sglang-router or dynamo")
        if config.resources.num_agg != 1:
            raise ValueError(
                f"frontend.type: sglang supports exactly one aggregate worker, got {config.resources.num_agg}; "
                "use frontend.type: sglang-router or dynamo to balance between replicas"
            )

        logger.info(
            "frontend.type=sglang: no separate frontend process; sglang.launch_server owns port %d",
            topology.public_port,
        )
        return []
