# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared implementation for frontends whose workers register themselves.

The counterpart of :class:`~srtctl.frontends.static_router.StaticRouterFrontend`.
A static router is launched with its worker URLs on the command line; a
dynamic frontend is launched with no worker list, workers announce
themselves over a discovery plane (Dynamo: etcd and NATS), and readiness is
the frontend's own registration count checked against the allocated
topology. Consequences shared by every such frontend live here: it fronts
any engine, it needs no per-worker URL gate before traffic, and no worker is
itself the public endpoint.

A router binary that can also take static URLs (vLLM Router's ZMQ discovery
mode) is a mode of a static router, not a dynamic frontend.

Dynamo is the only implementation today. A subclass sets ``type`` and
``worker_launch``, registers with ``@register_frontend``, parses its own
registration count in ``parse_health``, decides which rank serves which port,
and launches the process in ``start_frontends``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from srtctl.core.topology import Process


class DynamicFrontend:
    """Base class for frontends that discover their workers through registration."""

    type: ClassVar[str]
    # Registration does not care which engine registers.
    required_backend: ClassVar[str | None] = None
    expands_node_local_dp: ClassVar[bool] = False
    metrics_path: ClassVar[str] = "/metrics"

    @property
    def health_endpoint(self) -> str:
        """The frontend reports its registered workers here; ``parse_health`` counts them."""
        return "/health"

    def validate(self, config: Any) -> None:
        """Recipe-level rules beyond the backend pairing; none by default."""
        del config

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """A registered worker never binds the public port; the frontend owns it."""
        del mode
        return "allocated"

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        """Registration is the readiness gate; there is no per-worker URL to poll first."""
        del backend, backend_processes, network_interface
        return []

    def direct_endpoint_nodes(self, processes: list[Process]) -> list[str]:
        """The frontend process is the endpoint, never a worker."""
        del processes
        return []

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert ``frontend.args`` to CLI flags, keys verbatim."""
        if not args:
            return []
        result: list[str] = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key}")
            elif value is not False and value is not None:
                result.extend([f"--{key}", str(value)])
        return result
