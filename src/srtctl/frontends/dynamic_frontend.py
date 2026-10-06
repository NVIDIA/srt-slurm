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

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from srtctl.core.health import WorkerHealthResult, probe_json_health
from srtctl.frontends.base import Frontend, frontend_args_to_cli

if TYPE_CHECKING:
    from srtctl.backends.base import WorkerMode


class DynamicFrontend(Frontend):
    """Base class for frontends that discover their workers through registration."""

    def worker_metrics_path(self, backend: Any, mode: WorkerMode) -> str | None:
        """Every rank's Dynamo system server answers ``/metrics``, whatever the engine."""
        return self.metrics_path

    @property
    def health_endpoint(self) -> str:
        """The frontend reports its registered workers here; ``parse_health`` counts them."""
        return "/health"

    @abstractmethod
    def parse_health(self, response_json: dict, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        """Count the registered workers in the frontend's health body; each implementation knows its format."""
        raise NotImplementedError(f"{type(self).__name__} must parse its own registration count")

    def probe_ready(
        self, host: str, port: int, expected_prefill: int, expected_decode: int, config: Any
    ) -> WorkerHealthResult:
        """One GET of the registration endpoint, parsed against the expected counts."""
        return probe_json_health(host, port, self.health_endpoint, self.parse_health, expected_prefill, expected_decode)

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """A registered worker never binds the public port; the frontend owns it."""
        return "allocated"

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert ``frontend.args`` to CLI flags, keys verbatim."""
        return frontend_args_to_cli(args)
