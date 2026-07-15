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
from typing import TYPE_CHECKING, Any

from srtctl.core.health import WorkerHealthResult

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class VLLMFrontend:
    """Direct vLLM OpenAI server frontend.

    This frontend is intentionally narrow: aggregate vLLM jobs only, with the
    backend worker binding the public OpenAI port directly. Disaggregated vLLM
    still needs a real router/orchestrator such as Dynamo.
    """

    @property
    def type(self) -> str:
        return "vllm"

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
        if config.resources.is_disaggregated or config.resources.num_agg != 1:
            raise ValueError("frontend.type: vllm supports aggregate vLLM jobs only")

        leaders = [p for p in backend_processes if p.endpoint_mode == "agg" and p.is_leader]
        if len(leaders) != 1:
            raise ValueError("frontend.type: vllm requires exactly one aggregate server process")
        leader = leaders[0]
        if leader.node != topology.frontend_nodes[0] or leader.http_port != topology.public_port:
            raise ValueError(
                "direct vLLM topology does not match its aggregate server "
                f"({leader.node}:{leader.http_port} != {topology.frontend_nodes[0]}:{topology.public_port})"
            )

        logger.info("frontend.type=vllm: no separate frontend process; vllm serve owns port %d", topology.public_port)
        return []
