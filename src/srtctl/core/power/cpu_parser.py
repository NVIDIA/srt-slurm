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

from srtctl.core.power.cpu_rails import ACPI_RAIL_KINDS, DCGM_KIND, classify_acpi_label, normalize_kind, sensor_name
from srtctl.core.power.cpu_sample import CpuSample, RailReading, node_total_watts, pivot_socket_samples

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
class ParsedCpuScrape:
    """Readings from one node's single scrape, pivoted per socket, plus the derived per-host total.

    ``readings`` is every classified sensor value as scraped; ``sockets`` is
    the same data pivoted through ``cpu_sample.pivot_socket_samples`` (one
    :class:`CpuSample` per socket that has its primary rail); and
    ``total_power_w`` is ``node_total_watts`` over those sockets.
    """

    mode: str = "unknown"  # "dcgm" | "acpi" | "unknown"
    readings: tuple[CpuReading, ...] = ()
    sockets: tuple[CpuSample, ...] = ()
    total_power_w: float | None = None


def _scrape(mode: str, readings: list[CpuReading]) -> ParsedCpuScrape:
    sockets = pivot_socket_samples(mode, (RailReading(r.socket_id, r.kind, r.sensor, r.power_w) for r in readings))
    return ParsedCpuScrape(
        mode=mode, readings=tuple(readings), sockets=sockets, total_power_w=node_total_watts(sockets)
    )


def parse_cpu_scrape(text: str) -> ParsedCpuScrape:
    """Parse one exporter ``/metrics`` body into publishable CPU readings."""
    try:
        families = list(text_string_to_metric_families(text))
    except Exception:  # noqa: BLE001 - malformed exposition from a third-party parser
        return ParsedCpuScrape()

    acpi_readings = _parse_acpi(families)
    if acpi_readings:
        return _scrape("acpi", acpi_readings)

    dcgm_readings = _parse_dcgm(families)
    if dcgm_readings:
        return _scrape("dcgm", dcgm_readings)

    return ParsedCpuScrape()


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
