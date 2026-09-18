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
from typing import TYPE_CHECKING, Any

from srtctl.core.health import WorkerHealthResult

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class SGLangFrontend:
    """Direct SGLang OpenAI server frontend.

    Intentionally narrow: exactly one aggregate worker, which binds the public
    port itself. Readiness is the worker's ``/health`` plus ``/v1/models``.
    """

    @property
    def type(self) -> str:
        return "sglang"

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
            message="SGLang OpenAI server healthy",
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
        del runtime, backend, backend_processes, stop_event
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
