# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Direct vLLM frontend implementation.

For aggregate vLLM jobs the OpenAI-compatible HTTP server is the worker
process itself (`vllm serve`). There is no separate router/frontend process.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from srtctl.core.health import WorkerHealthResult
from srtctl.frontends.base import agg_leader_nodes, register_frontend

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


@register_frontend("vllm")
class VLLMFrontend:
    """Direct vLLM OpenAI server frontend.

    This frontend is intentionally narrow: a single aggregate vLLM worker, with
    that worker binding the public OpenAI port directly. Disaggregated layouts
    and multiple aggregate replicas still need a real router/orchestrator such
    as Dynamo, since nothing here load-balances between endpoints.
    """

    required_backend: ClassVar[str | None] = "vllm"
    worker_launch: ClassVar[Literal["dynamo", "direct"]] = "direct"
    expands_node_local_dp: ClassVar[bool] = False

    @property
    def type(self) -> str:
        return "vllm"

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """The one ``vllm serve`` is the endpoint, so it binds the public port in every mode it runs."""
        del mode
        return "public"

    metrics_path: ClassVar[str] = "/metrics"

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        """The aggregate leader binds the public port; its followers serve nothing."""
        if process.endpoint_mode == "agg" and process.is_leader:
            return runtime.frontend_port
        return None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        del config
        return runtime.frontend_port if process.is_leader else None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        """One control server for the whole worker, on the public port."""
        del process, config
        return runtime.frontend_port

    def profiling_control_is_leader_only(self, config: Any) -> bool:
        del config
        return True

    def direct_endpoint_nodes(self, processes: list[Process]) -> list[str]:
        return agg_leader_nodes(processes)

    def worker_ready_port(self, process: Process) -> int:
        return process.sys_port

    def validate(self, config: Any) -> None:
        """The one aggregate ``vllm serve`` owns the public port: no nginx fan-out, no P/D, one worker."""
        if config.frontend.enable_multiple_frontends:
            raise ValueError(
                "frontend.type: vllm binds vllm serve directly; set frontend.enable_multiple_frontends: false"
            )
        if config.resources.is_disaggregated:
            raise ValueError("frontend.type: vllm supports aggregate jobs only, not disaggregated layouts")
        if config.resources.num_agg != 1:
            raise ValueError(
                f"frontend.type: vllm supports exactly one aggregate worker, got {config.resources.num_agg}. "
                "vllm serve owns the public port directly and there is no router to load-balance "
                "replicas, so extra workers would either idle or collide on the port. "
                "Use frontend.type: dynamo to run multiple aggregate workers, or scale a single "
                "worker across nodes with resources.agg_nodes."
            )

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        return WorkerHealthResult(
            ready=True,
            message="vLLM OpenAI server healthy",
            prefill_ready=expected_prefill,
            prefill_expected=expected_prefill,
            decode_ready=expected_decode,
            decode_expected=expected_decode,
        )

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        del backend, backend_processes, network_interface
        return []

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        if not args:
            return []
        result = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key}")
            elif value is not False and value is not None:
                result.extend([f"--{key}", str(value)])
        return result

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,
    ) -> list[ManagedProcess]:
        if config.backend.type != "vllm":
            raise ValueError(f"frontend.type: vllm requires backend.type: vllm (got {config.backend.type!r})")
        if topology.uses_nginx or len(topology.frontend_nodes) != 1:
            raise ValueError(
                "frontend.type: vllm binds vllm serve directly to the public port; "
                "set frontend.enable_multiple_frontends: false"
            )
        if config.resources.is_disaggregated:
            raise ValueError("frontend.type: vllm supports aggregate vLLM jobs only")
        if config.resources.num_agg != 1:
            raise ValueError(
                f"frontend.type: vllm supports exactly one aggregate worker, got {config.resources.num_agg}; "
                "use frontend.type: dynamo to route between multiple workers"
            )

        logger.info("frontend.type=vllm: no separate frontend process; vllm serve owns port %d", topology.public_port)
        return []
