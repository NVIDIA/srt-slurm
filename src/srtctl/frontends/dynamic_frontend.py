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

from srtctl.core.health import WorkerHealthResult, probe_json_health
from srtctl.frontends.base import frontend_args_to_cli, logical_health_expectations

if TYPE_CHECKING:
    from srtctl.core.topology import Process
    from srtctl.services.implicit import EffectiveService


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

    def parse_health(self, response_json: dict, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        """Count the registered workers in the frontend's health body; each implementation knows its format."""
        raise NotImplementedError(f"{type(self).__name__} must parse its own registration count")

    def probe_ready(self, host: str, port: int, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        """One GET of the registration endpoint, parsed against the expected counts."""
        return probe_json_health(host, port, self.health_endpoint, self.parse_health, expected_prefill, expected_decode)

    def health_expectations(self, config: Any, processes: list[Process] | None) -> tuple[int, int, str]:
        """One registration per logical worker unless the implementation knows better."""
        return logical_health_expectations(config)

    def validate(self, config: Any) -> None:
        """Recipe-level rules beyond the backend pairing; none by default."""

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """A registered worker never binds the public port; the frontend owns it."""
        return "allocated"

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        """Registration is the readiness gate; there is no per-worker URL to poll first."""
        return []

    def direct_endpoint_nodes(self, processes: list[Process]) -> list[str]:
        """The frontend process is the endpoint, never a worker."""
        return []

    def implied_services(self, config: Any) -> list[EffectiveService]:
        """The discovery plane the frontend needs; none unless the implementation brings one."""
        return []

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        """Metrics share the routing port unless the implementation runs a separate listener."""
        return None

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert ``frontend.args`` to CLI flags, keys verbatim."""
        return frontend_args_to_cli(args)
