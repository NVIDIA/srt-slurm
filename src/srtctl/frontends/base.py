# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base types and protocols for frontend configurations.

Frontend types handle:
- Starting router/frontend processes
- Health checking with appropriate endpoints
- Building CLI arguments from config
"""

import shlex
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

if TYPE_CHECKING:
    from srtctl.core.health import WorkerHealthResult
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

# Supported frontend types - extensible by adding new literals
FrontendType = Literal["dynamo", "sglang", "trtllm_serve", "vllm", "vllm-router"]

FrontendFactory = Callable[[], "FrontendProtocol"]
_FRONTEND_REGISTRY: dict[str, FrontendFactory] = {}
_FrontendClass = TypeVar("_FrontendClass", bound=type)


def register_frontend(*names: str) -> Callable[[_FrontendClass], _FrontendClass]:
    """Register a frontend implementation under one or more config names."""

    def decorator(frontend_class: _FrontendClass) -> _FrontendClass:
        for name in names:
            if name in _FRONTEND_REGISTRY:
                raise ValueError(f"Frontend type {name!r} is already registered")
            _FRONTEND_REGISTRY[name] = frontend_class
        return frontend_class

    return decorator


def _load_builtin_frontends() -> None:
    """Import built-ins once so their registration decorators run."""
    from srtctl.frontends import dynamo, sglang, trtllm_serve, vllm, vllm_router  # noqa: F401


def build_setup_script_preamble(setup_script: str | None) -> str | None:
    """Build the standard in-container recipe setup-script invocation."""
    if not setup_script:
        return None
    script_name = shlex.quote(setup_script)
    return (
        f"setup_script={script_name} && "
        'script_path="/configs/${setup_script}" && '
        'patch_script_path="/configs/patches/${setup_script}" && '
        'echo "Running setup script: ${script_path} (fallback ${patch_script_path})" && '
        'if [ -f "${script_path}" ]; then bash "${script_path}"; '
        'elif [ -f "${patch_script_path}" ]; then bash "${patch_script_path}"; '
        'else echo "WARNING: ${script_path} or ${patch_script_path} not found"; fi'
    )


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

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list["Process"],
    ) -> list[str]:
        """Return backend URLs that must be directly healthy before benchmarking.

        Frontends that discover or gate their own backends return an empty list.
        Static adapters may use this hook to require readiness at the exact URLs
        they advertise to their router.
        """
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
    _load_builtin_frontends()
    try:
        factory = _FRONTEND_REGISTRY[frontend_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_FRONTEND_REGISTRY))
        raise ValueError(f"Unknown frontend type: {frontend_type!r}. Supported: {supported}") from exc
    return factory()
