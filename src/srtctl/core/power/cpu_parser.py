# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parses ``cpu-power-exporter`` ``/metrics`` bodies into CPU power readings.

The exporter resolves DCGM-vs-ACPI once at startup and only ever serves one
metric family for its process lifetime, so a scrape body should never
contain both. If it ever did, ACPI wins here: it reports the socket envelope
plus every component rail, while DCGM can only report the rails it has
fields for (1130 = ``cpu_rail``, 1132 = ``soc``; no envelope), so ACPI is
the more informative source when both exist.

DCGM bodies come in two shapes. Newer exporters label each sample with the
DCGM ``field_id`` it read; older ones publish only field 1130 with no label.
Both go through :func:`cpu_sample.dcgm_rail_readings`, so an unlabelled
sample is filed exactly like a labelled 1130.

Rail vocabulary comes from :mod:`srtctl.core.power.cpu_rails`; nothing here
knows a firmware label by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

from srtctl.core.power.cpu_rails import (
    ACPI_RAIL_KINDS,
    DCGM_FIELD_RAIL_KINDS,
    DCGM_PRIMARY_FIELD_ID,
    classify_acpi_label,
    normalize_kind,
    sensor_name,
)
from srtctl.core.power.cpu_sample import (
    CpuSample,
    RailReading,
    dcgm_rail_readings,
    node_total_watts,
    pivot_socket_samples,
)

DCGM_METRIC = "cpu_power_dcgm_watts"
DCGM_FIELD_LABEL = "field_id"
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
    # socket -> field_id -> watts. An unlabelled sample (older exporter) is
    # field 1130; a labelled sample names the field. First value per
    # (socket, field) wins, as in the pivot.
    by_socket: dict[int, dict[int, float]] = {}
    for family in families:
        for sample in family.samples:
            if sample.name != DCGM_METRIC:
                continue
            socket_id = _parse_socket(sample.labels.get("socket"))
            if socket_id is None:
                continue
            field_id = _parse_field_id(sample.labels.get(DCGM_FIELD_LABEL))
            if field_id is None or field_id not in DCGM_FIELD_RAIL_KINDS:
                continue
            value = sample.value
            if not math.isfinite(value) or value < 0:
                continue
            by_socket.setdefault(socket_id, {}).setdefault(field_id, value)
    return [
        CpuReading(source="dcgm", sensor=reading.sensor, socket_id=socket_id, power_w=reading.watts, kind=reading.kind)
        for socket_id, by_field in sorted(by_socket.items())
        for reading in dcgm_rail_readings(socket_id, by_field)
    ]


def _parse_field_id(raw: str | None) -> int | None:
    """DCGM field id from the ``field_id`` label; absent means the legacy single-field body (1130)."""
    if raw is None or raw == "":
        return DCGM_PRIMARY_FIELD_ID
    try:
        return int(raw)
    except ValueError:
        return None


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
