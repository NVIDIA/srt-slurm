# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The frontend protocol and registry.

A frontend owns everything the rest of srtctl needs to know about the router
(or the direct server that stands in for one): which backend it pairs with,
its recipe rules, how its workers are launched and which ports they bind,
which rank serves metrics or an endpoint, how readiness is probed and counted,
the services it implies, and how its process starts.
"""

import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

if TYPE_CHECKING:
    from srtctl.core.health import WorkerHealthResult
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process
    from srtctl.services.implicit import EffectiveService

# ``frontend.type: none`` is a services-only job: no router process, no OpenAI
# endpoint, no worker-count health gate (see SrtConfig._validate_services_only).
# It has no implementation and every stage short-circuits on it before calling
# get_frontend().
FRONTEND_NONE = "none"


class FrontendProtocol(Protocol):
    """Protocol that all frontend implementations must implement.

    Each frontend answers, for the rest of srtctl:
    1. Which backend it pairs with and what its recipe rules are
       (``required_backend``, ``validate``)
    2. How its workers are launched and which port each binds
       (``worker_launch``, ``worker_api_port``, ``expands_node_local_dp``)
    3. Which rank serves metrics, an endpoint, or profiler control, and where
       (``worker_metrics_port``, ``worker_endpoint_port``, ``profiling_control_port``,
       ``direct_endpoint_nodes``, ``worker_ready_port``, ``metrics_path``)
    4. How readiness is probed and counted (``probe_ready``, ``health_expectations``,
       ``get_backend_health_urls``)
    5. What it brings along (``implied_services``, ``frontend_metrics_port``)
    6. How its process starts (``start_frontends``)

    An implementation registers with ``@register_frontend("<type>")``; the
    recipe's ``frontend.type`` is resolved through that registry and nowhere
    else, so adding a frontend is one module under ``srtctl/frontends/``
    imported from the package ``__init__``.
    """

    #: Backend type this frontend requires, or ``None`` for any backend.
    #: ``SrtConfig._validate_frontend`` enforces it at config load.
    required_backend: ClassVar[str | None]

    #: How this frontend's workers are launched. ``dynamo`` workers are
    #: ``dynamo.<engine>`` processes that register with the Dynamo runtime;
    #: ``direct`` workers are the engine's own OpenAI server (``vllm serve``,
    #: ``sglang.launch_server``, ``trtllm-serve``). Backends read this instead
    #: of comparing frontend names.
    worker_launch: ClassVar[Literal["dynamo", "direct"]]

    #: The router expands each advertised URL into its node-local hybrid-LB DP
    #: ranks (vLLM Router). The vLLM backend launches one hybrid-LB API per node
    #: for such a frontend and refuses the deprecated per_gpu layout.
    expands_node_local_dp: ClassVar[bool]

    @property
    def type(self) -> str:
        """Frontend type identifier (e.g., 'dynamo', 'sglang')."""
        ...

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """Which port a direct worker of ``mode`` binds.

        ``public``: the worker is the endpoint itself and binds
        ``runtime.frontend_port``. ``allocated``: a router fronts it and it binds
        its own ``Process.http_port``. Dynamo workers serve no HTTP API and never
        consult this.
        """
        ...

    #: Path where this frontend's workers and router serve Prometheus metrics.
    metrics_path: ClassVar[str]

    def worker_metrics_port(self, process: "Process", runtime: "RuntimeContext") -> int | None:
        """Port on ``process.node`` serving Prometheus metrics at ``metrics_path`` for this rank.

        ``None`` when the rank serves none: a follower of a native multi-node
        server, or a layout the frontend does not scrape. Every rank of a Dynamo
        worker serves its own system port; this is the telemetry view.
        """
        ...

    def worker_endpoint_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        """Port a benchmark addresses this worker's HTTP endpoint on, one per logical worker.

        ``None`` for a rank that is not addressable on its own (followers behind
        a leader). Feeds the ``PREFILL_IPS``-style benchmark env and custom
        benchmarks' metrics URLs.
        """
        ...

    def profiling_control_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        """Port carrying this rank's profiler control routes for iteration-triggered captures, or ``None``."""
        ...

    def profiling_control_is_leader_only(self, config: Any) -> bool:
        """Whether one control server per logical endpoint, on its leader, fronts every rank."""
        ...

    def direct_endpoint_nodes(self, processes: list["Process"]) -> list[str]:
        """Nodes whose worker is itself the public endpoint, in topology order.

        Empty when a router process owns the public port; then the frontend
        topology's nodes are the endpoint.
        """
        ...

    def worker_ready_port(self, process: "Process") -> int:
        """Port polled for a worker's own ``/health`` during sequential endpoint start."""
        ...

    def health_expectations(self, config: Any, processes: list["Process"] | None) -> tuple[int, int, str]:
        """Expected ``(prefill, decode)`` counts in the units this frontend's readiness reports, plus a description.

        Aggregate workers count as decode. Dynamo counts registered generate
        instances (one per vLLM DP rank or node-local process), vLLM Router
        counts the ranks it expands each advertised URL into, every other
        frontend counts logical workers. ``processes`` is ``None`` before the
        topology is known; implementations fall back to logical counts then.
        """
        ...

    def probe_ready(
        self, host: str, port: int, expected_prefill: int, expected_decode: int, config: Any
    ) -> "WorkerHealthResult":
        """One readiness probe against the public endpoint at ``host:port``.

        Raise ``requests.RequestException`` while the endpoint is unreachable;
        ``wait_for_model`` retries until its timeout. Anything else that is not
        ready comes back as a result whose message explains why. ``config`` is
        the recipe, for frontends whose readiness contract depends on it (the
        vLLM Router in discovery mode).
        """
        ...

    def validate(self, config: Any) -> None:
        """Recipe-level rules for this frontend.

        Raise ``ValueError`` with the user-facing message; the schema reports it
        as a load-time ValidationError so ``srtctl dry-run`` catches it before an
        allocation is spent. The backend pairing is checked before this runs.
        """
        ...

    def implied_services(self, config: Any) -> list["EffectiveService"]:
        """Services this frontend needs that the recipe did not name (Dynamo: its discovery plane)."""
        ...

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        """Port of a Prometheus listener separate from the routing port, or ``None`` when metrics share it."""
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


def frontend_args_to_cli(args: dict[str, Any] | None) -> list[str]:
    """``frontend.args`` as CLI flags with keys verbatim: ``True`` is a bare flag, ``False``/``None`` are dropped."""
    if not args:
        return []
    result: list[str] = []
    for key, value in args.items():
        if value is True:
            result.append(f"--{key}")
        elif value is not False and value is not None:
            result.extend([f"--{key}", str(value)])
    return result


def logical_health_expectations(config: Any) -> tuple[int, int, str]:
    """Expected counts in logical workers: aggregate workers count as decode."""
    r = config.resources
    if r.num_agg > 0:
        return 0, r.num_agg, f"{r.num_agg} agg"
    return r.num_prefill, r.num_decode, f"{r.num_prefill}P + {r.num_decode}D"


def agg_leader_nodes(processes: list["Process"]) -> list[str]:
    """Nodes of the aggregate workers' leader ranks, in topology order, without repeats.

    For a frontend whose one aggregate worker is the public endpoint, these are
    the endpoint nodes.
    """
    ordered = sorted(processes, key=lambda p: (p.endpoint_index, p.node_rank, p.node))
    return list(dict.fromkeys(p.node for p in ordered if p.endpoint_mode == "agg" and p.is_leader))


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
