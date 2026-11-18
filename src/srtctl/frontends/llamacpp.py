# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct llama.cpp server: one aggregate worker owns the public endpoint."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from srtctl.core.health import WorkerHealthResult, probe_direct_server
from srtctl.frontends.base import Frontend, agg_leader_nodes, register_frontend
from srtctl.ports import FRONTEND_PUBLIC_PORT

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process


@register_frontend("llamacpp")
class LlamaCppFrontend(Frontend):
    """``llama-server`` supplies the OpenAI endpoint, health checks and optional metrics."""

    required_backend: ClassVar[str | None] = "llamacpp"
    # The server is already scraped as a worker; there is no router metrics target.
    metrics_path: ClassVar[str | None] = None

    @property
    def type(self) -> str:
        return "llamacpp"

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        return "public"

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        return runtime.frontend_port if process.endpoint_mode == "agg" and process.is_leader else None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        return runtime.frontend_port if process.is_leader else None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        # llama.cpp has no SGLang/vLLM iteration-triggered profiling API.
        return None

    def direct_endpoint_nodes(self, processes: list[Process]) -> list[str]:
        return agg_leader_nodes(processes)

    def worker_ready_port(self, process: Process) -> int:
        return FRONTEND_PUBLIC_PORT

    def probe_ready(
        self, host: str, port: int, expected_prefill: int, expected_decode: int, config: Any
    ) -> WorkerHealthResult:
        return probe_direct_server(host, port)

    def validate(self, config: Any) -> None:
        if config.frontend.enable_multiple_frontends:
            raise ValueError("frontend.type: llamacpp requires frontend.enable_multiple_frontends: false")
        if config.topology.is_disaggregated or config.topology.num_agg != 1:
            raise ValueError(
                "frontend.type: llamacpp requires exactly one aggregate worker; prefill/decode is unsupported"
            )
        if config.topology.agg_nodes != 1 or config.topology.gpus_per_agg > config.resources.gpus_per_node:
            raise ValueError("frontend.type: llamacpp requires roles.agg.nodes: 1 and GPUs that fit on one node")
        if config.dynamo.sidecar:
            raise ValueError("frontend.type: llamacpp does not support dynamo.sidecar")
        if config.model.path.startswith("hf:"):
            raise ValueError("llamacpp requires a local model.path directory and engine.model_file; hf: is unsupported")
        if config.profiling.enabled and not config.profiling.is_nsys_time:
            raise ValueError("llamacpp supports only time-based profiling; use profiling.type: nsys-time")

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,
    ) -> list[ManagedProcess]:
        # Validation already guarantees one worker owns the public port; no router is launched.
        self.validate(config)
        return []
