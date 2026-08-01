# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base types and protocols for frontend configurations.

Frontend types handle:
- Starting router/frontend processes
- Health checking with appropriate endpoints
- Building CLI arguments from config
"""

import threading
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from srtctl.core.health import WorkerHealthResult
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

# Supported frontend types - extensible by adding new literals
FrontendType = Literal["dynamo", "sglang", "trtllm_serve", "vllm"]


class FrontendProtocol(Protocol):
    """Protocol that all frontend implementations must implement.

    Each frontend is responsible for:
    1. Starting router/frontend processes on designated nodes
    2. Providing health check endpoint and response parsing
    3. Building CLI arguments from config
    """

    @property
    def type(self) -> str:
        """Frontend type identifier (e.g., 'dynamo', 'sglang')."""
        ...

    @property
    def health_endpoint(self) -> str:
        """HTTP endpoint for health checks (e.g., '/health', '/workers')."""
        ...

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> "WorkerHealthResult":
        """Parse health check response and return worker status."""
        ...

    def start_frontends(
        self,
        topology: Any,  # FrontendTopology
        runtime: "RuntimeContext",
        config: Any,  # SrtConfig
        backend: Any,  # BackendProtocol
        backend_processes: list["Process"],
        stop_event: "threading.Event | None" = None,
    ) -> list["ManagedProcess"]:
        """Start frontend processes on designated nodes.

        Args:
            topology: FrontendTopology describing where to run frontends
            runtime: Runtime context with paths and settings
            config: Full SrtConfig
            backend: Backend protocol for mode-specific info
            backend_processes: List of backend worker processes
            stop_event: Optional event to abort any readiness waits a frontend
                performs while starting (frontends that return immediately ignore it)

        Returns:
            List of ManagedProcess instances for started frontends
        """
        ...

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert frontend args dict to CLI argument list."""
        ...


def get_frontend(frontend_type: str) -> FrontendProtocol:
    """Get frontend implementation by type.

    Args:
        frontend_type: Frontend type string (e.g., 'dynamo', 'sglang')

    Returns:
        Instantiated frontend implementation

    Raises:
        ValueError: If frontend type is unknown
    """
    # Import here to avoid circular imports
    from srtctl.frontends.dynamo import DynamoFrontend
    from srtctl.frontends.sglang import SGLangFrontend
    from srtctl.frontends.trtllm_serve import TRTLLMServeFrontend
    from srtctl.frontends.vllm import VLLMFrontend

    if frontend_type == "dynamo":
        return DynamoFrontend()
    elif frontend_type == "sglang":
        return SGLangFrontend()
    elif frontend_type == "trtllm_serve":
        return TRTLLMServeFrontend()
    elif frontend_type == "vllm":
        return VLLMFrontend()
    else:
        raise ValueError(f"Unknown frontend type: {frontend_type!r}. Supported: dynamo, sglang, trtllm_serve, vllm")
