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
GMS_SERVICE_NAME = "gms"
DCGM_EXPORTER_SERVICE_NAME = "dcgm-exporter"
NODE_EXPORTER_SERVICE_NAME = "node-exporter"
PROCESS_EXPORTER_SERVICE_NAME = "process-exporter"


@dataclass(frozen=True)
class EffectiveService:
    """One service the job will run, with where it came from."""

    service: ServiceConfig
    implicit: bool
    reason: str = ""


def infra_placement(config: SrtConfig) -> ServicePlacementConfig:
    """Where the discovery plane and other infra services run: the infra node, or a dedicated one."""
    return ServicePlacementConfig(node="dedicated" if config.infra.etcd_nats_dedicated_node else "infra")


def nats_implied_reasons(config: SrtConfig) -> list[str]:
    """Why a Dynamo job needs NATS, or empty when nothing rides on it.

    The request plane defaults to ``tcp`` and KV events default to direct ZMQ, so
    NATS is implied only by ``dynamo.request_plane: nats``, ``dynamo.event_plane: nats``,
    or a v1 ``infra.nats_max_payload_mb`` (a knob that only means anything with NATS).
    """
    if getattr(config.frontend, "type", None) != "dynamo":
        return []
    dynamo = getattr(config, "dynamo", None)
    infra = getattr(config, "infra", None)
    reasons = [
        f"dynamo.{field} nats"
        for field, value in (
            ("request_plane", getattr(dynamo, "request_plane", None)),
            ("event_plane", getattr(dynamo, "event_plane", None)),
        )
        if value == "nats"
    ]
    if getattr(infra, "nats_max_payload_mb", None) is not None:
        reasons.append("infra.nats_max_payload_mb")
    return reasons


def runs_nats(config: SrtConfig) -> bool:
    """Whether a NATS service is effective: declared (and enabled) or implied."""
    for entry in getattr(config, "services", None) or []:
        if getattr(entry, "type", None) == "nats":
            return bool(getattr(entry, "enabled", True))
    return bool(nats_implied_reasons(config))


def implied_services(config: SrtConfig) -> list[EffectiveService]:
    """Services the rest of the recipe asks for without naming them."""
    implied: list[EffectiveService] = []

    # The frontend brings its own discovery plane (Dynamo: etcd, and NATS when a
    # plane rides on it); a services-only job has no frontend. Imported lazily:
    # the frontend implementations import this module.
    from srtctl.frontends import FRONTEND_NONE, get_frontend

    if config.frontend.type != FRONTEND_NONE:
        implied.extend(get_frontend(config.frontend.type).implied_services(config))

    if getattr(config.backend, "failover", None) is not None:
        # The kind's defaults are the placement: every worker node, one instance per worker.
        implied.append(
            EffectiveService(ServiceConfig(name=GMS_SERVICE_NAME, type="gms"), implicit=True, reason="engine.failover")
        )

    mooncake_cfg = getattr(config.backend, "mooncake_kv_store", None)
    if mooncake_cfg is not None:
        options = {}
        store_config = getattr(mooncake_cfg, "store_config", None)
        if store_config:
            options["store_config"] = dict(store_config)
        devices = getattr(mooncake_cfg, "device_names_by_gpu", None)
        if devices:
            options["device_names_by_gpu"] = list(devices)
        implied.append(
            EffectiveService(
                ServiceConfig(
                    name=MOONCAKE_MASTER_SERVICE_NAME,
                    type="mooncake-master",
                    container=mooncake_cfg.container,
                    args=list(mooncake_cfg.master_extra_args or []),
                    placement=infra_placement(config),
                    options=options,
                ),
                implicit=True,
                reason="backend.mooncake_kv_store",
            )
        )

    tachometer = config.observability.tachometer
    if config.observability.tachometer_enabled:
        # When power telemetry brings its own DCGM exporter it launches and owns it, and the
        # telemetry stage scrapes that one; CPU-only power telemetry leaves tachometer's.
        power_owns_dcgm = config.telemetry.enabled and config.telemetry.dcgm_exporter is not None
        dcgm = None if power_owns_dcgm else tachometer.resolved_dcgm_exporter
        if dcgm is not None:
            implied.append(
                EffectiveService(
                    ServiceConfig(
                        name=DCGM_EXPORTER_SERVICE_NAME,
                        type="dcgm-exporter",
                        placement=ServicePlacementConfig(node="compute"),
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
                        placement=ServicePlacementConfig(node="compute"),
                        container=node.container_image,
                        command=node.command.format(port=node.port).split() if node.command else None,
                        options={"port": node.port},
                    ),
                    implicit=True,
                    reason="observability.tachometer default exporters",
                )
            )
        proc = tachometer.resolved_process_exporter
        if proc is not None:
            # `binary` set (the default) is the host-native launch; a container_image
            # without a binary is the container launch. Both spellings map onto the
            # service: container -> container mode, options.binary -> host-native.
            options: dict = {"port": proc.port}
            if proc.binary:
                options["binary"] = proc.binary
            implied.append(
                EffectiveService(
                    ServiceConfig(
                        name=PROCESS_EXPORTER_SERVICE_NAME,
                        type="process-exporter",
                        container=(proc.container_image or None) if not proc.binary else None,
                        command=proc.command.format(port=proc.port).split() if proc.command else None,
                        options=options,
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
    """``ETCD_ENDPOINTS`` (and ``NATS_SERVER`` when a NATS service runs) for this job.

    Each points at the infra node, or at the ``external`` address of a declared service.
    ``NATS_SERVER`` is omitted when no NATS service is effective, so nothing is handed an
    address that no process listens on.
    """
    env = {
        "ETCD_ENDPOINTS": _declared_external(config, ETCD_SERVICE_NAME)
        or f"http://{runtime.nodes.infra}:{ETCD_CLIENT_PORT}"
    }
    if runs_nats(config):
        env["NATS_SERVER"] = (
            _declared_external(config, NATS_SERVICE_NAME) or f"nats://{runtime.nodes.infra}:{NATS_PORT}"
        )
    return env


def uses_discovery_plane(config: SrtConfig) -> bool:
    """Whether this job runs (or points at) etcd and NATS at all."""
    return any(entry.service.type in ("etcd", "nats") for entry in effective_services(config))
