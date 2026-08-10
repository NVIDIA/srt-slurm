# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM Router frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend

if TYPE_CHECKING:
    from srtctl.core.topology import Process


@register_frontend("vllm-router")
class VLLMRouterFrontend(StaticRouterFrontend):
    """Route requests to direct vLLM OpenAI-compatible worker endpoints."""

    type: ClassVar[str] = "vllm-router"
    backend_type: ClassVar[str] = "vllm"
    executable: ClassVar[tuple[str, ...]] = ("vllm-router",)
    pd_flag: ClassVar[str] = "--vllm-pd-disaggregation"
    process_name: ClassVar[str] = "vllm_router"

    def get_managed_frontend_args(self, config: Any) -> list[str]:
        """Keep Router's worker wait alive for srtctl's model-readiness window."""
        frontend_args = config.frontend.args or {}
        if "worker-startup-timeout-secs" in frontend_args:
            return []

        health_check = config.health_check
        timeout_seconds = health_check.max_attempts * health_check.interval_seconds
        return ["--worker-startup-timeout-secs", str(timeout_seconds)]

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """Advertise vLLM's NIXL side-channel port to the P/D router."""
        return process.nixl_port
