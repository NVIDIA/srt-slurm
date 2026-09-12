# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parses ``cpu-power-exporter`` ``/metrics`` bodies into CPU power readings.

The exporter resolves DCGM-vs-ACPI once at startup and only ever serves one
metric family for its process lifetime, so a scrape body should never
contain both. If it ever did, ACPI wins here: it reports per-channel detail
(``total``/``cpu_rail``/``soc``/``dram``) while DCGM reports only one
already-aggregated value per socket, so ACPI is the more informative source
when both exist.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

DCGM_METRIC = "cpu_power_dcgm_watts"
ACPI_METRIC = "cpu_power_acpi_watts"
TOTAL_KIND = "total"
_ACPI_KINDS = {"total", "cpu_rail", "soc", "dram"}
_ACPI_DOMAIN_PATTERNS = (
    (
        "total",
        re.compile(r"\b(?:Grace|Total(?:\s+Input)?)\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b", re.IGNORECASE),
    ),
    (
        "cpu_rail",
        re.compile(
            r"\bCPU(?:\s+Rail)?(?:\s+Input)?\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "soc",
        re.compile(
            r"\b(?:SoC\s+Rail(?:\s+Input)?|SysIO)\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dram",
        re.compile(r"\bDRAM(?:\s+Input)?\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class CpuReading:
    """One sensor's power draw within a single scrape."""

    source: str
    sensor: str
    socket_id: int
    power_w: float
    kind: str  # "" for dcgm; "total" | "cpu_rail" | "soc" | "dram" for acpi


@dataclass(frozen=True)
class ParsedCpuScrape:
    """Readings from one node's single scrape, plus the derived per-host total."""

    mode: str = "unknown"  # "dcgm" | "acpi" | "unknown"
    readings: tuple[CpuReading, ...] = ()
    total_power_w: float | None = None


def parse_cpu_scrape(text: str) -> ParsedCpuScrape:
    """Parse one exporter ``/metrics`` body into publishable CPU readings."""
    try:
        families = list(text_string_to_metric_families(text))
    except Exception:  # noqa: BLE001 - malformed exposition from a third-party parser
        return ParsedCpuScrape()

    acpi_readings = _parse_acpi(families)
    if acpi_readings:
        total = _sum_by_kind(acpi_readings, TOTAL_KIND)
        return ParsedCpuScrape(mode="acpi", readings=tuple(acpi_readings), total_power_w=total)

    dcgm_readings = _parse_dcgm(families)
    if dcgm_readings:
        total = sum(reading.power_w for reading in dcgm_readings)
        return ParsedCpuScrape(mode="dcgm", readings=tuple(dcgm_readings), total_power_w=total)

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
                    sensor=f"CPU{socket_id}:cpuPowerUsageW",
                    socket_id=socket_id,
                    power_w=value,
                    kind="",
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
            kind = labels.get("type") or ""
            socket_id = _parse_socket(labels.get("socket"))
            if socket_id is None or kind not in _ACPI_KINDS:
                inferred = _classify_acpi_domain(labels.get("oem_info") or "")
                if inferred is not None:
                    kind, socket_id = inferred
            # An unclassified rail (the exporter's "other" kind) has no
            # numeric socket, and socket_id is not nullable in the CSV.
            if socket_id is None or kind not in _ACPI_KINDS:
                continue
            value = sample.value
            if not math.isfinite(value) or value < 0:
                continue
            readings.append(
                CpuReading(
                    source="acpi",
                    sensor=labels.get("oem_info") or "",
                    socket_id=socket_id,
                    power_w=value,
                    kind=kind,
                )
            )
    return readings


def _sum_by_kind(readings: list[CpuReading], kind: str) -> float | None:
    matching = [reading.power_w for reading in readings if reading.kind == kind]
    return sum(matching) if matching else None


def _classify_acpi_domain(oem_info: str) -> tuple[str, int] | None:
    """Recover the semantic rail and socket from a firmware OEM label."""
    for kind, pattern in _ACPI_DOMAIN_PATTERNS:
        match = pattern.search(oem_info)
        if match is not None:
            return kind, int(match.group(1))
    return None


def _parse_socket(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None
