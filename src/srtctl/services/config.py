# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe dataclasses for the top-level ``services:`` block.

A service is any long-running process srtctl launches next to the workers,
frontend, and benchmark client: an experimental router built from a PR, a
standalone Mooncake Store per worker node, a debugging HTTP server. One list,
one shape; ``type`` selects a :class:`~srtctl.services.registry.ServiceKind`
that supplies defaults and injects the environment that kind needs. See
``docs/services.md``.
"""

import builtins
import logging
from dataclasses import field
from typing import Any, ClassVar

from marshmallow import Schema, ValidationError, pre_load
from marshmallow_dataclass import dataclass

from srtctl.core.source import SourceConfig

logger = logging.getLogger(__name__)

# Where a service runs. head / infra are one node; prefill / decode / agg are the
# distinct physical nodes the role's workers land on; workers is every worker node.
SERVICE_PLACEMENTS: tuple[str, ...] = (
    "head",
    "infra",
    "dedicated",
    "prefill",
    "decode",
    "agg",
    "workers",
    "compute",
    "all",
)
SINGLE_NODE_PLACEMENTS: frozenset[str] = frozenset({"head", "infra", "dedicated"})
# Placements whose nodes carry engine workers, the only ones ``placement.per: worker`` can attach to.
PER_WORKER_PLACEMENTS: tuple[str, ...] = ("prefill", "decode", "agg", "workers")
# How many instances a placed node gets: one, or one per engine worker on it.
SERVICE_PERS: tuple[str, ...] = ("node", "worker")

# When a service starts relative to the rest of the job. ``infra`` is the discovery
# plane (etcd, NATS) that everything else may depend on; ``before_workers`` runs after
# it and before any worker; ``after_frontend`` once workers and the frontend are healthy.
SERVICE_STARTS: tuple[str, ...] = ("infra", "before_workers", "after_frontend")

# services[].source is the shared git-at-an-immutable-ref shape.
ServiceSourceConfig = SourceConfig


@dataclass(frozen=True)
class ServicePlacementConfig:
    """Where a service runs.

    Attributes:
        node: ``head`` or ``infra`` (one instance), ``dedicated`` (reserve the
            infra node exclusively; infra-class kinds only), ``prefill`` /
            ``decode`` / ``agg`` (one instance per distinct physical node that
            role's workers use), ``workers`` (one instance per engine worker
            node; on a service that owns nodes, its own pool), ``compute``
            (engine worker nodes plus every pool), or ``all`` (every node of
            the allocation).
        pool: Run on the nodes another service owns (``services[].nodes``), one
            instance per node of that pool. Replaces ``node``.
        per: ``node`` (default): one instance per placed node. ``worker``: one
            instance per engine worker on each placed node, attached to that
            worker: it runs with the worker's ``CUDA_VISIBLE_DEVICES`` and sees
            ``{worker_role}``, ``{worker_index}``, ``{worker_node_rank}``,
            ``{worker_gpus}``, ``{worker_gpu_count}``. A sidecar in the
            Kubernetes sense (the GPU Memory Service next to each vLLM worker).
            Only with ``node`` in ``prefill``, ``decode``, ``agg``, ``workers``.
    """

    node: str = "head"
    pool: str | None = None
    per: str = "node"

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.node not in SERVICE_PLACEMENTS:
            raise ValidationError(
                f"services[].placement.node must be one of {', '.join(SERVICE_PLACEMENTS)}; got {self.node!r}"
            )
        if self.per not in SERVICE_PERS:
            raise ValidationError(
                f"services[].placement.per must be one of {', '.join(SERVICE_PERS)}; got {self.per!r}"
            )
        if self.per == "worker":
            if self.pool is not None:
                raise ValidationError("services[].placement.per: worker attaches to engine workers, not to a pool")
            if self.node not in PER_WORKER_PLACEMENTS:
                raise ValidationError(
                    f"services[].placement.per: worker needs placement.node in {', '.join(PER_WORKER_PLACEMENTS)}; "
                    f"got {self.node!r}"
                )
        if self.pool is not None:
            if not str(self.pool).strip():
                raise ValidationError("services[].placement.pool must name a service that declares nodes")
            if self.node != "head":
                raise ValidationError("services[].placement: give either node or pool, not both")


@dataclass(frozen=True)
class TcpProbe:
    """Ready when ``port`` accepts a TCP connection on the service node."""

    port: int

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValidationError("readiness.tcp.port must be between 1 and 65535")


@dataclass(frozen=True)
class HttpProbe:
    """Ready when ``GET http://<node>:<port><path>`` returns ``status``."""

    port: int
    path: str = "/health"
    status: int = 200

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValidationError("readiness.http.port must be between 1 and 65535")
        if not self.path.startswith("/"):
            raise ValidationError("readiness.http.path must start with '/'")
        if not 100 <= self.status <= 599:
            raise ValidationError("readiness.http.status must be an HTTP status code")


@dataclass(frozen=True)
class ServiceMetricsConfig:
    """One Prometheus endpoint a service serves: ``port``, ``path`` (default ``/metrics``), ``nodes``, ``name``.

    The scrape annotation. Tachometer builds its target list from these: one
    target per node the service runs on, or only its first node when ``nodes``
    is ``first`` (a cluster whose head serves the metrics: a trainer's
    collector on the Ray head). ``name`` is the endpoint's name in the parquet,
    ``<name>_<node>``; it defaults to the service name and is required when a
    service declares more than one endpoint. Kinds that always publish metrics
    (the exporters) supply theirs; a recipe writes the block for anything else.
    """

    port: int
    path: str = "/metrics"
    nodes: str = "all"
    name: str | None = None

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValidationError("metrics.port must be between 1 and 65535")
        if not self.path.startswith("/"):
            raise ValidationError("metrics.path must start with '/'")
        if self.nodes not in ("all", "first"):
            raise ValidationError(f"metrics.nodes must be all or first; got {self.nodes!r}")
        if self.name is not None and not self.name.strip():
            raise ValidationError("metrics.name must not be empty")


@dataclass(frozen=True)
class LogProbe:
    """Ready when the service's log file contains a line matching the regular expression ``pattern``."""

    pattern: str

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        import re

        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValidationError(f"readiness.log.pattern is not a valid regular expression: {exc}") from None


@dataclass(frozen=True)
class ServiceReadinessConfig:
    """Readiness gate: the launch blocks until the probe passes on every service node.

    Exactly one probe: ``tcp`` (a port accepts connections), ``http`` (a URL
    returns a status), or ``log`` (the service log matches a pattern). ``port``
    alone is shorthand for ``tcp``.

    Attributes:
        port: Shorthand for ``tcp: {port: <port>}``.
        tcp: TCP connect probe.
        http: HTTP GET probe.
        log: Log-pattern probe against ``service_<name>.out``.
        timeout_seconds: How long to wait per node before failing the job.
        interval_seconds: Seconds between probe attempts.
    """

    port: int | None = None
    tcp: TcpProbe | None = None
    http: HttpProbe | None = None
    log: LogProbe | None = None
    timeout_seconds: int = 120
    interval_seconds: int = 2

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.port is not None:
            if self.tcp is not None:
                raise ValidationError("services[].readiness: give either port (shorthand) or tcp, not both")
            object.__setattr__(self, "tcp", TcpProbe(port=self.port))
        probes = [name for name in ("tcp", "http", "log") if getattr(self, name) is not None]
        if len(probes) != 1:
            raise ValidationError(
                f"services[].readiness needs exactly one probe: port, tcp, http, or log; got {', '.join(probes) or 'none'}"
            )
        if self.timeout_seconds <= 0:
            raise ValidationError("services[].readiness.timeout_seconds must be positive")
        if self.interval_seconds <= 0:
            raise ValidationError("services[].readiness.interval_seconds must be positive")

    @property
    def probe(self) -> TcpProbe | HttpProbe | LogProbe:
        for name in ("tcp", "http", "log"):
            value = getattr(self, name)
            if value is not None:
                return value
        raise AssertionError("unreachable: __post_init__ guarantees one probe")

    @property
    def probe_port(self) -> int | None:
        """The port the service is expected to own, when the probe implies one."""
        probe = self.probe
        return probe.port if isinstance(probe, TcpProbe | HttpProbe) else None

    def describe(self) -> str:
        probe = self.probe
        if isinstance(probe, TcpProbe):
            what = f"tcp/{probe.port}"
        elif isinstance(probe, HttpProbe):
            what = f"http://<node>:{probe.port}{probe.path} -> {probe.status}"
        else:
            what = f"log matches /{probe.pattern}/"
        return f"{what}, timeout={self.timeout_seconds}s"


class _ServiceSchema(Schema):
    """``metrics: {port: ..}`` is sugar for a one-endpoint list; a service may serve several."""

    @pre_load
    def _metrics_as_list(self, data, **kwargs):
        if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
            data = {**data, "metrics": [data["metrics"]]}
        return data


@dataclass(frozen=True, base_schema=_ServiceSchema)
class ServiceConfig:
    """One entry of the top-level ``services:`` list.

    Attributes:
        name: Unique label; names the log file (``service_<name>.out``) and the
            tracked process.
        type: Service kind. ``generic`` (default) launches exactly what you
            wrote; ``mooncake-store`` runs a standalone Mooncake Store wired to
            the managed master. See ``docs/services.md`` for the kinds.
        command: Argv to launch (not shell-interpreted). Required for
            ``generic``; typed kinds supply a default.
        args: Extra argv appended to ``command``.
        container: Container image or ``srtslurm.yaml`` alias. Defaults to the
            kind's fallback image, then the job container.
        env: Environment for the service process, on top of what the kind injects.
        source: Optional git source to clone before ``build_command`` and
            ``command`` run. Single-node placements only.
        build_command: Argv run once inside the service container, from the
            clone, before ``command`` starts. Only meaningful with ``source``.
        nodes: Whole nodes this service owns: its pool. Pools add to the
            allocation next to the engine roles' nodes and are carved after them
            in declaration order, so a Ray cluster, a sandbox fleet and an engine
            role can each have their own nodes in one recipe. An owner is placed
            on its own pool (``placement.node: workers``); other services join it
            with ``placement.pool: <name>``.
        placement: Where the service runs. Defaults to the kind's placement
            (``head`` for generic services, ``infra`` for etcd/nats/mooncake-master,
            ``workers`` for the exporters).
        start: ``after_frontend`` (default for ``generic``) or
            ``before_workers`` (default for ``mooncake-store``).
        readiness: Optional TCP port gate; the job waits for it on every
            service node before continuing.
        inherit_discovery_env: Inject ``ETCD_ENDPOINTS`` / ``NATS_SERVER`` so
            the service can register with the job's Dynamo discovery plane.
        critical: When true a crash fails the run, like a worker dying. Default
            false for ``generic`` (a dead sidecar costs its own log, not the
            run) and true for ``mooncake-store``. Set true for anything in
            the live request path.
        terminal: This service is the job's run: the job ends when every
            instance of every terminal service has exited, and the worst exit
            code becomes the job's. A recipe with a terminal service has no
            benchmark step (``benchmark.type`` stays ``manual``); a torchrun
            pool that trains to completion is the shape.
        preamble: Shell run inside the container before ``command``
            (``ulimit`` and friends).
        cpus_per_task: Optional ``srun --cpus-per-task``.
        cpu_bind: Optional ``srun --cpu-bind``.
        srun_options: Extra srun options for this service only.
        build_timeout_seconds: Kill ``build_command`` after this many seconds.
        enabled: ``false`` drops the service, including an implicit one
            (``etcd`` / ``nats`` under the Dynamo frontend, the default
            exporters) declared here by name.
        external: For discovery-plane kinds (``etcd``, ``nats``,
            ``mooncake-master``): use this already-running endpoint and launch
            nothing; the URL is what the job's processes are pointed at.
        options: Kind-specific settings (``nats``: ``max_payload_mb``;
            ``mooncake-master``: ``store_config`` and ``device_names_by_gpu``
            for vLLM). Unknown keys are rejected by the kind.
        metrics: Prometheus endpoints this service serves: one mapping or a
            list of ``{port, path, nodes, name}`` (``path`` defaults to
            ``/metrics``, ``nodes`` to ``all``). Tachometer scrapes each on every
            node the service runs on, or on its first node with ``nodes: first``,
            as endpoint ``<name>_<node>`` where ``name`` defaults to the service
            name. The exporter kinds declare theirs; write it for a generic
            service that publishes metrics, or on a ``ray`` service whose head
            serves a trainer's collector and router.
    """

    name: str
    type: str = "generic"
    command: list[str] | None = None
    args: list[str] = field(default_factory=list)
    container: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    source: ServiceSourceConfig | None = None
    build_command: list[str] | None = None
    placement: ServicePlacementConfig | None = None
    nodes: int | None = None
    start: str | None = None
    readiness: ServiceReadinessConfig | None = None
    inherit_discovery_env: bool = True
    critical: bool | None = None
    terminal: bool = False
    preamble: str | None = None
    cpus_per_task: int | None = None
    cpu_bind: str | None = None
    srun_options: dict[str, str] = field(default_factory=dict)
    # Wall-clock budget for build_command; the build srun is killed when it runs out
    # so a hung build cannot hold the allocation until walltime.
    build_timeout_seconds: int = 1800
    enabled: bool = True
    external: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    metrics: list[ServiceMetricsConfig] = field(default_factory=list)

    # builtins.type: the ``type`` field above shadows the builtin inside the class body.
    Schema: ClassVar[builtins.type[Schema]] = Schema

    def __post_init__(self) -> None:
        from srtctl.services.registry import get_service_kind, list_service_types

        if not self.name.strip():
            raise ValidationError("services[].name must be a non-empty string")
        label = f"services[{self.name}]"
        if self.type not in list_service_types():
            raise ValidationError(
                f"{label}.type {self.type!r} is not a known service type (known: {', '.join(list_service_types())})"
            )
        kind = get_service_kind(self.type)
        if self.command is not None and not self.command:
            raise ValidationError(f"{label}.command, if set, must be non-empty (omit it to use the type's default)")
        if self.command is None and kind.default_command is None and not kind.builds_command:
            raise ValidationError(f"{label}.command is required for type {self.type!r}")
        if any(not str(part).strip() for part in [*(self.command or []), *self.args]):
            raise ValidationError(f"{label}.command/args must not contain empty arguments")
        if self.build_command is not None and not self.build_command:
            raise ValidationError(f"{label}.build_command, if set, must be non-empty (omit it entirely instead)")
        if self.source is not None and self.effective_placement not in SINGLE_NODE_PLACEMENTS:
            raise ValidationError(
                f"{label}.source requires a single-node placement (head or infra); got placement.node="
                f"{self.effective_placement!r}"
            )
        if self.source is not None and not self.build_command:
            logger.warning(
                "%s sets 'source' without 'build_command'; the source is cloned but nothing builds it "
                "before 'command' runs. This is almost always a mistake.",
                label,
            )
        if self.start is not None and self.start not in SERVICE_STARTS:
            raise ValidationError(f"{label}.start must be one of {', '.join(SERVICE_STARTS)}; got {self.start!r}")
        if self.cpus_per_task is not None and self.cpus_per_task <= 0:
            raise ValidationError(f"{label}.cpus_per_task must be positive")
        if self.build_timeout_seconds <= 0:
            raise ValidationError(f"{label}.build_timeout_seconds must be positive")
        if self.external is not None and not str(self.external).strip():
            raise ValidationError(f"{label}.external must be a non-empty endpoint")
        if self.external is not None and not kind.supports_external:
            raise ValidationError(f"{label}.external is only valid for discovery-plane kinds, not {self.type!r}")
        if self.effective_placement == "dedicated" and not kind.supports_dedicated:
            raise ValidationError(
                f"{label}.placement.node: dedicated is only supported for infra-class kinds (etcd, nats, "
                f"mooncake-master); use head, infra, or workers for type {self.type!r}"
            )
        if self.nodes is not None:
            if self.nodes < 1:
                raise ValidationError(f"{label}.nodes must be at least 1; got {self.nodes}")
            if self.placement is not None and self.placement.pool is not None:
                raise ValidationError(f"{label}.nodes makes the service its own pool; drop placement.pool")
            if self.effective_placement != "workers":
                raise ValidationError(
                    f"{label}.nodes owns a pool, so placement.node must be workers, meaning that pool "
                    f"(got {self.effective_placement!r})"
                )
        if self.effective_per == "worker":
            if self.nodes is not None:
                raise ValidationError(f"{label}.placement.per: worker attaches to engine workers; it cannot own a pool")
            if self.effective_placement not in PER_WORKER_PLACEMENTS:
                raise ValidationError(
                    f"{label}.placement.per: worker needs placement.node in {', '.join(PER_WORKER_PLACEMENTS)}; "
                    f"got {self.effective_placement!r}"
                )
            if self.readiness is not None and self.readiness.log is None:
                raise ValidationError(
                    f"{label}.placement.per: worker instances share their node's ports, so readiness must be a log probe"
                )
        if len(self.metrics) > 1:
            names = [endpoint.name for endpoint in self.metrics]
            if any(name is None for name in names) or len(set(names)) != len(names):
                raise ValidationError(
                    f"{label}.metrics declares {len(self.metrics)} endpoints; give each a distinct name "
                    "(it becomes the endpoint's name in the parquet)"
                )
        unknown_options = set(self.options) - set(kind.option_keys)
        if unknown_options:
            raise ValidationError(
                f"{label}.options has keys type {self.type!r} does not understand: {', '.join(sorted(unknown_options))}"
            )

    # -- effective values (type defaults applied) ------------------------------

    @property
    def effective_command(self) -> list[str]:
        """``command`` plus ``args``, with the kind's default command when none is written."""
        from srtctl.services.registry import get_service_kind

        base = self.command if self.command is not None else list(get_service_kind(self.type).default_command or ())
        return [*base, *self.args]

    def preview_command(self) -> list[str]:
        """The command as the kind would launch it, with placeholders for runtime values (dry-run)."""
        from srtctl.services.registry import ServiceLaunchContext, get_service_kind

        return get_service_kind(self.type).build_command(self, ServiceLaunchContext.preview())

    @property
    def effective_start(self) -> str:
        from srtctl.services.registry import get_service_kind

        return self.start if self.start is not None else get_service_kind(self.type).default_start

    @property
    def effective_placement(self) -> str:
        """``placement.node`` as written, else the kind's default (``head`` for generic services).

        A service that owns nodes, or rides on a pool, is placed on ``workers``: that pool.
        """
        from srtctl.services.registry import get_service_kind

        if self.placement is not None:
            return "workers" if self.placement.pool is not None else self.placement.node
        if self.nodes is not None:
            return "workers"
        return get_service_kind(self.type).default_placement

    @property
    def effective_pool(self) -> str | None:
        """The pool this service rides on: its own when it owns nodes, else ``placement.pool``."""
        if self.nodes is not None:
            return self.name
        return self.placement.pool if self.placement is not None else None

    @property
    def effective_per(self) -> str:
        """``placement.per`` as written, else the kind's default (``node`` for every kind but ``gms``)."""
        from srtctl.services.registry import get_service_kind

        if self.placement is not None:
            return self.placement.per
        return get_service_kind(self.type).default_per

    @property
    def effective_critical(self) -> bool:
        from srtctl.services.registry import get_service_kind

        return self.critical if self.critical is not None else get_service_kind(self.type).default_critical
