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
from typing import ClassVar

from marshmallow import Schema, ValidationError
from marshmallow_dataclass import dataclass

logger = logging.getLogger(__name__)

# Where a service runs. head / infra are one node; prefill / decode / agg are the
# distinct physical nodes the role's workers land on; workers is every worker node.
SERVICE_PLACEMENTS: tuple[str, ...] = ("head", "infra", "prefill", "decode", "agg", "workers")
SINGLE_NODE_PLACEMENTS: frozenset[str] = frozenset({"head", "infra"})

# When a service starts relative to the rest of the job.
SERVICE_STARTS: tuple[str, ...] = ("before_workers", "after_frontend")

# Immutable-ref guard for services[].source.rev.
_MOVING_REFS: frozenset[str] = frozenset({"main", "master", "HEAD"})


@dataclass(frozen=True)
class ServiceSourceConfig:
    """Git source to build a service from before launching it.

    Attributes:
        git: Repository URL to clone.
        rev: Immutable ref to check out: a commit SHA, a tag, or
            ``refs/pull/<n>/head`` for an unmerged PR. Branch names are
            rejected because they move out from under a build.
        path: Optional subdirectory of the clone that ``build_command`` and
            ``command`` run from. Defaults to the repository root.
    """

    git: str
    rev: str
    path: str | None = None

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if not self.git.strip():
            raise ValidationError("services[].source.git must be a non-empty repository URL")
        if not self.rev.strip():
            raise ValidationError("services[].source.rev must be a non-empty immutable ref")
        if self.rev.strip() in _MOVING_REFS:
            raise ValidationError(
                "services[].source.rev must be an immutable ref (commit SHA, tag, or refs/pull/<n>/head), "
                f"not a moving branch name: {self.rev!r}"
            )
        if self.path is not None and not self.path.strip():
            raise ValidationError("services[].source.path must not be blank when set")


@dataclass(frozen=True)
class ServicePlacementConfig:
    """Where a service runs.

    Attributes:
        node: ``head`` or ``infra`` (one instance), ``prefill`` / ``decode`` /
            ``agg`` (one instance per distinct physical node that role's
            workers use), or ``workers`` (one instance per worker node).
    """

    node: str = "head"

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.node not in SERVICE_PLACEMENTS:
            raise ValidationError(
                f"services[].placement.node must be one of {', '.join(SERVICE_PLACEMENTS)}; got {self.node!r}"
            )


@dataclass(frozen=True)
class ServiceReadinessConfig:
    """TCP readiness gate: the launch blocks until ``port`` accepts connections on every service node.

    Attributes:
        port: TCP port the service listens on.
        timeout_seconds: How long to wait per node before failing the job.
    """

    port: int
    timeout_seconds: int = 120

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValidationError("services[].readiness.port must be between 1 and 65535")
        if self.timeout_seconds <= 0:
            raise ValidationError("services[].readiness.timeout_seconds must be positive")


@dataclass(frozen=True)
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
            kind's fallback (Mooncake's ``mooncake_kv_store.container``), then
            the job container.
        env: Environment for the service process, on top of what the kind injects.
        source: Optional git source to clone before ``build_command`` and
            ``command`` run. Single-node placements only.
        build_command: Argv run once inside the service container, from the
            clone, before ``command`` starts. Only meaningful with ``source``.
        placement: Where the service runs. Default ``head``.
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
        preamble: Shell run inside the container before ``command``
            (``ulimit`` and friends).
        cpus_per_task: Optional ``srun --cpus-per-task``.
        cpu_bind: Optional ``srun --cpu-bind``.
        srun_options: Extra srun options for this service only.
        build_timeout_seconds: Kill ``build_command`` after this many seconds.
    """

    name: str
    type: str = "generic"
    command: list[str] | None = None
    args: list[str] = field(default_factory=list)
    container: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    source: ServiceSourceConfig | None = None
    build_command: list[str] | None = None
    placement: ServicePlacementConfig = field(default_factory=ServicePlacementConfig)
    start: str | None = None
    readiness: ServiceReadinessConfig | None = None
    inherit_discovery_env: bool = True
    critical: bool | None = None
    preamble: str | None = None
    cpus_per_task: int | None = None
    cpu_bind: str | None = None
    srun_options: dict[str, str] = field(default_factory=dict)
    # Wall-clock budget for build_command; the build srun is killed when it runs out
    # so a hung build cannot hold the allocation until walltime.
    build_timeout_seconds: int = 1800

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
        if self.command is None and kind.default_command is None:
            raise ValidationError(f"{label}.command is required for type {self.type!r}")
        if any(not str(part).strip() for part in [*(self.command or []), *self.args]):
            raise ValidationError(f"{label}.command/args must not contain empty arguments")
        if self.build_command is not None and not self.build_command:
            raise ValidationError(f"{label}.build_command, if set, must be non-empty (omit it entirely instead)")
        if self.source is not None and self.placement.node not in SINGLE_NODE_PLACEMENTS:
            raise ValidationError(
                f"{label}.source requires a single-node placement (head or infra); got placement.node="
                f"{self.placement.node!r}"
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

    # -- effective values (type defaults applied) ------------------------------

    @property
    def effective_command(self) -> list[str]:
        """``command`` plus ``args``, with the kind's default command when none is written."""
        from srtctl.services.registry import get_service_kind

        base = self.command if self.command is not None else list(get_service_kind(self.type).default_command or ())
        return [*base, *self.args]

    @property
    def effective_start(self) -> str:
        from srtctl.services.registry import get_service_kind

        return self.start if self.start is not None else get_service_kind(self.type).default_start

    @property
    def effective_critical(self) -> bool:
        from srtctl.services.registry import get_service_kind

        return self.critical if self.critical is not None else get_service_kind(self.type).default_critical
