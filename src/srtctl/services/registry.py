# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service kinds: what ``services[].type`` selects.

A kind supplies defaults (command, start phase, criticality) and the environment
a service of that kind needs at launch. It never launches anything itself; the
``ServiceStageMixin`` does that uniformly for every kind. Register a new kind
with :func:`register_service`, the same pattern as ``@register_benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.services.config import ServiceConfig


@dataclass(frozen=True)
class ServiceLaunchContext:
    """Everything a kind may need to compute one service instance's environment and templates."""

    runtime: RuntimeContext
    node: str
    node_ip: str
    node_id: int  # position of ``node`` in runtime.nodes.worker, or the instance index for head/infra
    index: int  # instance index within this service (0..n-1)
    role: str  # the service's placement.node value
    # Every node this service runs on, in placement order (its pool when it owns or rides one), and
    # their IPs. Instance 0 is the rendezvous for a service that forms its own cluster. Empty means
    # "just this node" (previews, single-instance construction in tests).
    nodes: tuple[str, ...] = ()
    node_ips: tuple[str, ...] = ()

    @classmethod
    def preview(cls, node: str = "<node>") -> ServiceLaunchContext:
        """A context for rendering commands without a job (dry-run): placeholders stand in for runtime values."""
        from types import SimpleNamespace

        runtime = SimpleNamespace(
            nodes=SimpleNamespace(head="<head>", infra="<infra>", worker=()),
            head_node_ip="<head_ip>",
            infra_node_ip="<infra_ip>",
        )
        return cls(runtime=runtime, node=node, node_ip="<node_ip>", node_id=0, index=0, role="<role>")  # type: ignore[arg-type]

    def template_vars(self) -> dict[str, str]:
        """Placeholders substituted into command, args, env values, and preamble."""
        from srtctl.ports import MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_MASTER_PORT

        pool_nodes = self.nodes or (self.node,)
        pool_ips = self.node_ips or (self.node_ip,)
        return {
            "node": self.node,
            "node_ip": self.node_ip,
            "node_id": str(self.node_id),
            "index": str(self.index),
            "role": self.role,
            "head_node": self.runtime.nodes.head,
            "head_ip": self.runtime.head_node_ip,
            "infra_node": self.runtime.nodes.infra,
            "infra_ip": self.runtime.infra_node_ip,
            # The service's own node set: instance 0 is the rendezvous of a self-forming cluster
            # (torchrun --master-addr, a Ray head), which the job head is not when the service
            # runs on a pool next to engine roles.
            "pool_node": pool_nodes[0],
            "pool_ip": pool_ips[0],
            "pool_nodes": ",".join(pool_nodes),
            "pool_ips": ",".join(pool_ips),
            "pool_node_count": str(len(pool_nodes)),
            "gpus_per_node": str(self.runtime.gpus_per_node),
            "master_port": str(MOONCAKE_MASTER_PORT),
            "metadata_port": str(MOONCAKE_HTTP_METADATA_PORT),
        }


class ServiceKind:
    """Base for a registered service type. Subclass, set the class attributes, override hooks as needed."""

    type_name: ClassVar[str] = ""
    # Argv used when the recipe omits ``command``. None means ``command`` is required.
    default_command: ClassVar[tuple[str, ...] | None] = None
    default_start: ClassVar[str] = "after_frontend"
    default_critical: ClassVar[bool] = False
    # Where the service runs when the recipe gives no ``placement``.
    default_placement: ClassVar[str] = "head"
    # TCP ports that must answer before the service counts as ready, when the recipe
    # gives no ``readiness`` (all of them, in order). Empty: launch is enough.
    default_readiness_ports: ClassVar[tuple[int, ...]] = ()
    default_readiness_timeout: ClassVar[int] = 120
    # Whether the srun wraps the command in bash (exports, preamble). Distroless
    # images (the exporters) have no shell.
    use_bash_wrapper: ClassVar[bool] = True
    # Infra-class kinds may reserve the infra node (``placement.node: dedicated``)
    # and may point at an already-running instance (``external``).
    supports_dedicated: ClassVar[bool] = False
    supports_external: ClassVar[bool] = False
    # ``options`` keys this kind understands.
    option_keys: ClassVar[tuple[str, ...]] = ()
    # True when the kind assembles its own command in ``build_command`` (etcd, the
    # exporters, ...), so the recipe need not give one.
    builds_command: ClassVar[bool] = False

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        """Whole-recipe checks for one service (raise ``marshmallow.ValidationError``)."""

    def container_fallback(self, config: SrtConfig) -> str | None:
        """Image to use when the service sets no ``container``; None falls through to the job container."""
        return None

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        """The argv to launch on ``ctx.node`` (placeholders already substituted by the stage)."""
        return list(service.effective_command)

    def preamble(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> str | None:
        """Shell the kind runs before the command (data-dir setup and the like); the recipe's ``preamble`` follows."""
        return None

    def default_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        """Environment the kind provides; the recipe's ``env`` overrides it."""
        return {}

    def forced_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        """Environment srtctl owns for this kind; it overrides the recipe's ``env``."""
        return {}

    def host_native(self, service: ServiceConfig) -> bool:
        """True to run the command straight on the node, with no container (a static host binary)."""
        return False

    def prepare(self, service: ServiceConfig, runtime: RuntimeContext) -> None:
        """Write anything the command needs into the run's log dir; called once per service, before launch."""

    def skip_reason(self, service: ServiceConfig, runtime: RuntimeContext) -> str | None:
        """A reason not to launch this service in this job (a missing host binary); None to launch."""
        return None


_SERVICE_KINDS: dict[str, ServiceKind] = {}


def register_service(name: str):
    """Class decorator registering a :class:`ServiceKind` under ``services[].type: <name>``."""

    def decorator(cls: type[ServiceKind]) -> type[ServiceKind]:
        cls.type_name = name
        _SERVICE_KINDS[name] = cls()
        return cls

    return decorator


def get_service_kind(name: str) -> ServiceKind:
    try:
        return _SERVICE_KINDS[name]
    except KeyError:
        raise ValueError(f"Unknown service type {name!r}. Known: {', '.join(list_service_types())}") from None


def list_service_types() -> list[str]:
    return sorted(_SERVICE_KINDS)
