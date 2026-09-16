# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parses ``cpu-power-exporter`` ``/metrics`` bodies into CPU power readings.

The exporter resolves DCGM-vs-ACPI once at startup and only ever serves one
metric family for its process lifetime, so a scrape body should never
contain both. If it ever did, ACPI wins here: it reports per-channel detail
(``total``/``cpu_rail``/``soc``/``dram``) while DCGM reports only one
already-aggregated value per socket, so ACPI is the more informative source
when both exist.

Rail vocabulary comes from :mod:`srtctl.core.power.cpu_rails`; nothing here
knows a firmware label by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

from srtctl.core.power.cpu_rails import (
    ACPI_RAIL_KINDS,
    COMPONENT_RAIL_KINDS,
    DCGM_KIND,
    TOTAL_KIND,
    classify_acpi_label,
    normalize_kind,
    sensor_name,
)

DCGM_METRIC = "cpu_power_dcgm_watts"
ACPI_METRIC = "cpu_power_acpi_watts"


@dataclass(frozen=True)
class CpuReading:
    """One sensor's power draw within a single scrape."""

    source: str
    sensor: str
    socket_id: int
    power_w: float
    kind: str  # cpu_rails.DCGM_KIND for dcgm; an ACPI_RAIL_KINDS member for acpi


@dataclass(frozen=True)
class SocketPower:
    """One socket's readings for one scrape, pivoted for the wide CSV row.

    ``power_w`` is the authoritative per-socket figure (ACPI ``total``
    envelope, or the DCGM value). ``rails`` holds the ACPI component rails
    by kind; it is empty for DCGM.
    """

    socket_id: int
    sensor: str
    power_w: float
    rails: dict[str, float]


@dataclass(frozen=True)
class ParsedCpuScrape:
    """Readings from one node's single scrape, plus the derived per-host total."""

    mode: str = "unknown"  # "dcgm" | "acpi" | "unknown"
    readings: tuple[CpuReading, ...] = ()
    sockets: tuple[SocketPower, ...] = ()
    total_power_w: float | None = None


def parse_cpu_scrape(text: str) -> ParsedCpuScrape:
    """Parse one exporter ``/metrics`` body into publishable CPU readings."""
    try:
        families = list(text_string_to_metric_families(text))
    except Exception:  # noqa: BLE001 - malformed exposition from a third-party parser
        return ParsedCpuScrape()

    acpi_readings = _parse_acpi(families)
    if acpi_readings:
        sockets = _pivot_sockets(acpi_readings, TOTAL_KIND)
        total = sum(s.power_w for s in sockets) if sockets else None
        return ParsedCpuScrape(mode="acpi", readings=tuple(acpi_readings), sockets=sockets, total_power_w=total)

    dcgm_readings = _parse_dcgm(families)
    if dcgm_readings:
        sockets = _pivot_sockets(dcgm_readings, DCGM_KIND)
        total = sum(s.power_w for s in sockets) if sockets else None
        return ParsedCpuScrape(mode="dcgm", readings=tuple(dcgm_readings), sockets=sockets, total_power_w=total)

    return ParsedCpuScrape()


def _pivot_sockets(readings: list[CpuReading], primary_kind: str) -> tuple[SocketPower, ...]:
    """One :class:`SocketPower` per socket that has a ``primary_kind`` reading.

    A socket whose primary rail is missing from the scrape is dropped rather
    than published with a component rail masquerading as its power: that is
    exactly the number-mixing this layout exists to prevent.
    """
    by_socket: dict[int, dict[str, CpuReading]] = {}
    for reading in readings:
        by_socket.setdefault(reading.socket_id, {}).setdefault(reading.kind, reading)
    sockets: list[SocketPower] = []
    for socket_id in sorted(by_socket):
        kinds = by_socket[socket_id]
        primary = kinds.get(primary_kind)
        if primary is None:
            continue
        rails = {kind: kinds[kind].power_w for kind in COMPONENT_RAIL_KINDS if kind in kinds}
        sockets.append(SocketPower(socket_id=socket_id, sensor=primary.sensor, power_w=primary.power_w, rails=rails))
    return tuple(sockets)


def _parse_dcgm(families) -> list[CpuReading]:
    readings: list[CpuReading] = []
    for family in families:
        for sample in family.samples:
            if sample.name != DCGM_METRIC:
                continue
            socket_id = _parse_socket(sample.labels.get("socket"))
            if socket_id is None:
                continue
            value = sample.value
            if not math.isfinite(value) or value < 0:
                continue
            readings.append(
                CpuReading(
                    source="dcgm",
                    sensor=sensor_name(DCGM_KIND, socket_id),
                    socket_id=socket_id,
                    power_w=value,
                    kind=DCGM_KIND,
                )
            )
    return readings


def _parse_acpi(families) -> list[CpuReading]:
    readings: list[CpuReading] = []
    for family in families:
        for sample in family.samples:
            if sample.name != ACPI_METRIC:
                continue
            labels = sample.labels
            kind = normalize_kind(labels.get("type"))
            socket_id = _parse_socket(labels.get("socket"))
            if socket_id is None or kind is None:
                inferred = classify_acpi_label(labels.get("oem_info") or "")
                if inferred is not None:
                    kind, socket_id = inferred
            # An unclassified rail (the exporter's "other" kind) has no
            # numeric socket, and socket_id is not nullable in the CSV.
            if socket_id is None or kind not in ACPI_RAIL_KINDS:
                continue
            value = sample.value
            if not math.isfinite(value) or value < 0:
                continue
            readings.append(
                CpuReading(
                    source="acpi",
                    sensor=sensor_name(kind, socket_id),
                    socket_id=socket_id,
                    power_w=value,
                    kind=kind,
                )
            )
    return readings


def _parse_socket(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None
