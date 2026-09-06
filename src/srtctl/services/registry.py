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

    def template_vars(self) -> dict[str, str]:
        """Placeholders substituted into command, args, env values, and preamble."""
        from srtctl.ports import MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_MASTER_PORT

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

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        """Whole-recipe checks for one service (raise ``marshmallow.ValidationError``)."""

    def container_fallback(self, config: SrtConfig) -> str | None:
        """Image to use when the service sets no ``container``; None falls through to the job container."""
        return None

    def default_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        """Environment the kind provides; the recipe's ``env`` overrides it."""
        return {}

    def forced_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        """Environment srtctl owns for this kind; it overrides the recipe's ``env``."""
        return {}


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
