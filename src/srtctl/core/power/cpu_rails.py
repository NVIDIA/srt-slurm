# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for CPU power rail naming.

Every producer and consumer of CPU power data derives its vocabulary from
here so a label rename cannot silently change a number downstream:

* ``cpu_power_exporter`` classifies hwmon OEM labels into the ``type`` label
  it publishes;
* ``cpu_parser`` classifies scraped samples (by ``type`` first, OEM label as
  fallback) into ``CpuReading.kind``;
* ``cpu_power`` (host collector) classifies hwmon labels into sensor names
  and the wide per-socket CSV columns;
* ``power_energy_report`` reads the wide columns directly and only falls
  back to :func:`classify_sensor` for legacy long-format CSVs.

Rail kinds
----------
``total``
    The complete CPU-side socket envelope (Grace exposes it as
    ``Grace Power Socket N``; other platforms as ``Total Power socket N``).
    This is the per-socket ``power_w`` figure and the only kind that sums
    into ``total_power_w``.
``cpu_rail`` / ``soc`` / ``dram``
    Component rails. Reference breakdowns only; they are *not* additive to
    ``total`` (real traces show total ~93-104 W vs cpu_rail+soc ~53-58 W).
``dcgm``
    Not an ACPI rail: the origin tag for DCGM-mode socket power. DCGM has no
    CPU power backend of its own -- its sysmon module reads the same ACPI
    hwmon channels by ``power1_oem_info`` label (NVIDIA/DCGM
    ``modules/sysmon/DcgmSystemMonitor.cpp``), so each DCGM field *is* one
    ACPI rail; see :data:`DCGM_FIELD_RAIL_KINDS`. Field 1130 is
    ``CPU Power Socket N`` = ``cpu_rail``; no DCGM field reports the
    ``Grace Power Socket N`` envelope (1131 is only its cap). DCGM-mode
    ``power_w`` is therefore the CPU rail, roughly half the ACPI envelope,
    and is not comparable with ACPI-mode ``power_w``.
"""

from __future__ import annotations

import re

TOTAL_KIND = "total"
DCGM_KIND = "dcgm"
OTHER_KIND = "other"

# DCGM CPU-entity power fields and the ACPI rail each one reads. Verified
# against NVIDIA/DCGM (DcgmSystemMonitor.cpp: label prefix -> file map;
# DcgmModuleSysmon.cpp: field id -> getter). Keyed by field id so a producer
# only ever names the field, never the label.
DCGM_PRIMARY_FIELD_ID = 1130  # DCGM_FI_DEV_CPU_POWER_WATTS: the value filed as DCGM-mode power_w
DCGM_FIELD_RAIL_KINDS: dict[int, str] = {
    1130: "cpu_rail",  # DCGM_FI_DEV_CPU_POWER_WATTS       <- "CPU Power Socket N"   power1_average
    1132: "soc",  # DCGM_FI_DEV_SYSIO_POWER_UTIL_CURRENT <- "SysIO Power Socket N" power1_average
}
DCGM_FIELD_NAMES: dict[int, str] = {
    1130: "DCGM_FI_DEV_CPU_POWER_WATTS",
    1132: "DCGM_FI_DEV_SYSIO_POWER_UTIL_CURRENT",
}
DCGM_FIELD_HWMON_LABELS: dict[int, str] = {
    1130: "CPU Power Socket N",
    1132: "SysIO Power Socket N",
}
DCGM_POWER_FIELD_IDS: tuple[int, ...] = tuple(sorted(DCGM_FIELD_RAIL_KINDS))

# Component rails, in wide-CSV column order.
COMPONENT_RAIL_KINDS: tuple[str, ...] = ("cpu_rail", "soc", "dram")
# Every kind an ACPI channel can classify to.
ACPI_RAIL_KINDS: frozenset[str] = frozenset((TOTAL_KIND, *COMPONENT_RAIL_KINDS))

# Wide-CSV column name per component rail. ``total`` has no column of its
# own: it *is* ``power_w``.
RAIL_COLUMNS: dict[str, str] = {kind: f"{kind}_w" for kind in COMPONENT_RAIL_KINDS}
RAIL_COLUMN_NAMES: tuple[str, ...] = tuple(RAIL_COLUMNS[kind] for kind in COMPONENT_RAIL_KINDS)

# Sensor-name suffix per kind (``CPU<socket>:<suffix>``). The host collector
# has always written these; the scraper wrote raw OEM labels before v2.
SENSOR_SUFFIXES: dict[str, str] = {
    TOTAL_KIND: "cpuSidePowerUsageW",
    "cpu_rail": "cpuRailPowerUsageW",
    "soc": "socPowerUsageW",
    "dram": "dramPowerUsageW",
    DCGM_KIND: "cpuPowerUsageW",
}
_KIND_BY_SUFFIX = {suffix: kind for kind, suffix in SENSOR_SUFFIXES.items()}

# Legacy ``type`` label vocabulary emitted by older cpu_power_exporter builds.
LEGACY_TYPE_ALIASES: dict[str, str] = {"grace": TOTAL_KIND, "cpu": "cpu_rail", "sysio": "soc"}

# Firmware OEM label -> rail kind. Ordered: the first match wins, so the
# specific Grace forms precede the generic ones and "CPU Power Socket N"
# (a Grace *component* rail) is matched after every "total" form.
ACPI_LABEL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (TOTAL_KIND, re.compile(r"\bGrace\s+Power\s+Socket\s+(\d+)\b", re.IGNORECASE)),
    (TOTAL_KIND, re.compile(r"\bTotal(?:\s+Input)?\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b", re.IGNORECASE)),
    ("cpu_rail", re.compile(r"\bCPU\s+Rail(?:\s+Input)?\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b", re.IGNORECASE)),
    ("soc", re.compile(r"\bSoC\s+Rail(?:\s+Input)?\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b", re.IGNORECASE)),
    ("dram", re.compile(r"\bDRAM(?:\s+Input)?\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b", re.IGNORECASE)),
    ("cpu_rail", re.compile(r"\bCPU(?:\s+Input)?\s+Power(?:\s+in\s+uW)?\s+Socket\s+(\d+)\b", re.IGNORECASE)),
    ("soc", re.compile(r"\bSysIO\s+Power\s+Socket\s+(\d+)\b", re.IGNORECASE)),
)


def classify_acpi_label(label: str) -> tuple[str, int] | None:
    """Firmware OEM label -> ``(kind, socket_id)``; None for unrecognised rails (NVSwitch, module, ...)."""
    for kind, pattern in ACPI_LABEL_PATTERNS:
        match = pattern.search(label)
        if match is not None:
            return kind, int(match.group(1))
    return None


def normalize_kind(raw: str | None) -> str | None:
    """Canonical kind for an exporter ``type`` label, accepting legacy aliases; None if unknown."""
    if not raw:
        return None
    if raw in ACPI_RAIL_KINDS:
        return raw
    return LEGACY_TYPE_ALIASES.get(raw)


def sensor_name(kind: str, socket_id: int) -> str:
    return f"CPU{socket_id}:{SENSOR_SUFFIXES[kind]}"


def classify_sensor(sensor: str) -> str:
    """Rail kind for a ``sensor`` CSV cell from any writer version.

    Recognises the ``CPU<n>:<suffix>`` form (host collector; scraper v2+)
    and raw OEM labels (scraper v1). Anything else is ``other``.
    """
    _, _, suffix = sensor.partition(":")
    if suffix in _KIND_BY_SUFFIX:
        return _KIND_BY_SUFFIX[suffix]
    classified = classify_acpi_label(sensor)
    return classified[0] if classified is not None else OTHER_KIND


# Preference when a legacy long-format CSV holds several rails for one socket
# and exactly one must feed the per-socket power series.
LEGACY_RAIL_PREFERENCE: tuple[str, ...] = (TOTAL_KIND, DCGM_KIND, *COMPONENT_RAIL_KINDS, OTHER_KIND)


def legacy_rail_rank(sensor: str) -> int:
    return LEGACY_RAIL_PREFERENCE.index(classify_sensor(sensor))
