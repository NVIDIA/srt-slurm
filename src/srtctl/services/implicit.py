# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The services a recipe implies, merged with the ones it declares.

Three things used to be launched by bespoke stages with their own placement
knobs, readiness loops, and no dry-run output: etcd and NATS for the Dynamo
frontend, the Mooncake master for ``backend.mooncake_kv_store``, and the DCGM and
node exporters tachometer scrapes. They are services now. This module derives
the implicit ones from the rest of the recipe, lets a declared entry of the same
name take over (or drop it with ``enabled: false``), and hands the effective
list to the service stage and to dry-run, which marks the implicit ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from srtctl.ports import ETCD_CLIENT_PORT, NATS_PORT
from srtctl.services.config import ServiceConfig, ServicePlacementConfig

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

ETCD_SERVICE_NAME = "etcd"
NATS_SERVICE_NAME = "nats"
MOONCAKE_MASTER_SERVICE_NAME = "mooncake-master"
DCGM_EXPORTER_SERVICE_NAME = "dcgm-exporter"
NODE_EXPORTER_SERVICE_NAME = "node-exporter"


@dataclass(frozen=True)
class EffectiveService:
    """One service the job will run, with where it came from."""

    service: ServiceConfig
    implicit: bool
    reason: str = ""


def _infra_placement(config: SrtConfig) -> ServicePlacementConfig:
    return ServicePlacementConfig(node="dedicated" if config.infra.etcd_nats_dedicated_node else "infra")


def implied_services(config: SrtConfig) -> list[EffectiveService]:
    """Services the rest of the recipe asks for without naming them."""
    implied: list[EffectiveService] = []

    if config.frontend.type == "dynamo":
        placement = _infra_placement(config)
        implied.append(
            EffectiveService(
                ServiceConfig(name=ETCD_SERVICE_NAME, type="etcd", placement=placement),
                implicit=True,
                reason="frontend.type dynamo",
            )
        )
        nats_options = {}
        if config.infra.nats_max_payload_mb is not None:
            nats_options["max_payload_mb"] = config.infra.nats_max_payload_mb
        implied.append(
            EffectiveService(
                ServiceConfig(name=NATS_SERVICE_NAME, type="nats", placement=placement, options=nats_options),
                implicit=True,
                reason="frontend.type dynamo",
            )
        )

    mooncake_cfg = getattr(config.backend, "mooncake_kv_store", None)
    if mooncake_cfg is not None:
        options = {}
        store_config = getattr(mooncake_cfg, "store_config", None)
        if store_config:
            options["store_config"] = dict(store_config)
        implied.append(
            EffectiveService(
                ServiceConfig(
                    name=MOONCAKE_MASTER_SERVICE_NAME,
                    type="mooncake-master",
                    container=mooncake_cfg.container,
                    args=list(mooncake_cfg.master_extra_args or []),
                    placement=_infra_placement(config),
                    options=options,
                ),
                implicit=True,
                reason="backend.mooncake_kv_store",
            )
        )

    tachometer = config.observability.tachometer
    if config.observability.tachometer_enabled:
        # The power-telemetry path launches and owns its own DCGM exporter.
        dcgm = None if config.telemetry.enabled else tachometer.resolved_dcgm_exporter
        if dcgm is not None:
            implied.append(
                EffectiveService(
                    ServiceConfig(
                        name=DCGM_EXPORTER_SERVICE_NAME,
                        type="dcgm-exporter",
                        container=dcgm.container_image,
                        command=dcgm.command.format(port=dcgm.port).split() if dcgm.command else None,
                        options={"port": dcgm.port, "collect_interval_ms": tachometer.collect_interval_ms},
                    ),
                    implicit=True,
                    reason="observability.tachometer default exporters",
                )
            )
        node = tachometer.resolved_node_exporter
        if node is not None:
            implied.append(
                EffectiveService(
                    ServiceConfig(
                        name=NODE_EXPORTER_SERVICE_NAME,
                        type="node-exporter",
                        container=node.container_image,
                        command=node.command.format(port=node.port).split() if node.command else None,
                        options={"port": node.port},
                    ),
                    implicit=True,
                    reason="observability.tachometer default exporters",
                )
            )
    return implied


def effective_services(config: SrtConfig) -> list[EffectiveService]:
    """Implicit services first (a declared one of the same name replaces it), then the declared ones.

    Disabled entries (``enabled: false``) are dropped, which is how a recipe
    switches an implicit service off.
    """
    declared = {service.name: service for service in config.services}
    effective: list[EffectiveService] = []
    seen: set[str] = set()
    for implied in implied_services(config):
        override = declared.get(implied.service.name)
        chosen = override if override is not None else implied.service
        seen.add(chosen.name)
        if chosen.enabled:
            effective.append(EffectiveService(chosen, implicit=override is None, reason=implied.reason))
    for service in config.services:
        if service.name in seen or not service.enabled:
            continue
        effective.append(EffectiveService(service, implicit=False))
    return effective


def find_service(config: SrtConfig, name: str) -> ServiceConfig | None:
    for entry in effective_services(config):
        if entry.service.name == name:
            return entry.service
    return None


def _declared_external(config: SrtConfig, name: str) -> str | None:
    """The ``external`` address of a declared service, if any (implicit services never have one)."""
    for service in getattr(config, "services", None) or ():
        if service.name == name and service.enabled:
            return service.external or None
    return None


def discovery_env(config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
    """``ETCD_ENDPOINTS`` and ``NATS_SERVER`` for this job: the infra node, or an ``external`` endpoint."""
    etcd_url = _declared_external(config, ETCD_SERVICE_NAME) or f"http://{runtime.nodes.infra}:{ETCD_CLIENT_PORT}"
    nats_url = _declared_external(config, NATS_SERVICE_NAME) or f"nats://{runtime.nodes.infra}:{NATS_PORT}"
    return {"ETCD_ENDPOINTS": etcd_url, "NATS_SERVER": nats_url}


def uses_discovery_plane(config: SrtConfig) -> bool:
    """Whether this job runs (or points at) etcd and NATS at all."""
    return any(entry.service.type in ("etcd", "nats") for entry in effective_services(config))
