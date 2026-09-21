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
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:
    from srtctl.core.health import WorkerHealthResult
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

# ``frontend.type: none`` is a services-only job: no router process, no OpenAI
# endpoint, no worker-count health gate (see SrtConfig._validate_services_only).
# It has no implementation and every stage short-circuits on it before calling
# get_frontend().
FRONTEND_NONE = "none"


class FrontendProtocol(Protocol):
    """Protocol that all frontend implementations must implement.

    Each frontend is responsible for:
    1. Starting router/frontend processes on designated nodes
    2. Providing health check endpoint and response parsing
    3. Building CLI arguments from config
    4. Its own recipe-level rules (``required_backend``, ``validate``)

    An implementation registers with ``@register_frontend("<type>")``; the
    recipe's ``frontend.type`` is resolved through that registry and nowhere
    else, so adding a frontend is one module under ``srtctl/frontends/``
    imported from the package ``__init__``.
    """

    #: Backend type this frontend requires, or ``None`` for any backend.
    #: ``SrtConfig._validate_frontend`` enforces it at config load.
    required_backend: ClassVar[str | None]

    @property
    def type(self) -> str:
        """Frontend type identifier (e.g., 'dynamo', 'sglang')."""
        ...

    def validate(self, config: Any) -> None:
        """Recipe-level rules for this frontend.

        Raise ``ValueError`` with the user-facing message; the schema reports it
        as a load-time ValidationError so ``srtctl dry-run`` catches it before an
        allocation is spent. The backend pairing is checked before this runs.
        """
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
        network_interface: str | None = None,
    ) -> list[str]:
        """Return backend URLs that must be directly healthy before traffic."""
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


_FRONTENDS: dict[str, type] = {}


def register_frontend(name: str):
    """Class decorator registering a frontend implementation under ``frontend.type: <name>``."""

    def decorator(cls):
        _FRONTENDS[name] = cls
        return cls

    return decorator


def _load_registry() -> None:
    # The package __init__ imports every implementation module, which registers
    # it. Imported lazily: implementations import core modules that import this
    # one, and get_frontend() is only ever called at run time.
    import srtctl.frontends  # noqa: F401


def list_frontend_types() -> list[str]:
    """Every accepted ``frontend.type``, including ``none``."""
    _load_registry()
    return sorted([*_FRONTENDS, FRONTEND_NONE])


def get_frontend(frontend_type: str) -> FrontendProtocol:
    """Instantiate the registered frontend implementation for ``frontend_type``.

    Raises:
        ValueError: for ``none`` (which has no implementation) and for unknown types
    """
    _load_registry()
    if frontend_type == FRONTEND_NONE:
        raise ValueError(
            "frontend.type 'none' has no frontend implementation: services-only jobs skip the frontend layer "
            "and the health gate, so nothing should ask for one"
        )
    try:
        implementation = _FRONTENDS[frontend_type]
    except KeyError:
        raise ValueError(
            f"Unknown frontend type: {frontend_type!r}. Supported: {', '.join(list_frontend_types())}"
        ) from None
    return implementation()
