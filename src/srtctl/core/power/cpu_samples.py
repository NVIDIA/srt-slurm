# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cpu/samples.csv`` writer and reader.

Best-effort by design: unlike the GPU pipeline's ``samples.py``, there is no
strict re-validation pass here and no reason-code contract with the shared
``Reason`` class. A malformed or missing file is simply unavailable data,
never a job-failing condition.

The writer emits schema v2 (one row per socket, component rails as columns).
The reader accepts v2 and the legacy v1 long format (one row per rail); v1
rows surface with empty ``rails`` since nothing pivots them here -- callers
that need one figure per socket use ``cpu_rails.classify_sensor``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from srtctl.core.power.contract import (
    CPU_SAMPLES_HEADER,
    CPU_SAMPLES_HEADER_V1,
    CPU_SCHEMA_VERSION,
    CPU_SCHEMA_VERSION_V1,
)
from srtctl.core.power.cpu_rails import COMPONENT_RAIL_KINDS


@dataclass(frozen=True)
class CpuSampleRow:
    """One persisted CPU power observation for one socket in one scrape."""

    timestamp_unix: float
    hostname: str
    source: str
    sensor: str
    socket_id: int
    power_w: float
    total_power_w: float | None
    rails: dict[str, float] = field(default_factory=dict)  # component kind -> watts (ACPI only)
    schema_version: int = CPU_SCHEMA_VERSION

    def to_csv(self) -> list[Any]:
        return [
            self.schema_version,
            repr(self.timestamp_unix),
            self.hostname,
            self.source,
            self.sensor,
            self.socket_id,
            repr(self.power_w),
            *("" if kind not in self.rails else repr(self.rails[kind]) for kind in COMPONENT_RAIL_KINDS),
            "" if self.total_power_w is None else repr(self.total_power_w),
        ]


class CpuSampleWriter:
    """Append-only ``cpu/samples.csv`` writer owned by the CPU collector thread."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.row_count = 0
        handle = open(path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        try:
            writer = csv.writer(handle)
            writer.writerow(CPU_SAMPLES_HEADER)
            handle.flush()
        except BaseException:
            handle.close()
            raise
        self._handle: TextIO | None = handle
        self._writer = writer

    @property
    def closed(self) -> bool:
        return self._handle is None

    def append(self, rows: Iterable[CpuSampleRow]) -> None:
        if self._handle is None:
            raise ValueError("cpu/samples.csv writer is closed")
        for row in rows:
            self._writer.writerow(row.to_csv())
            self.row_count += 1

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        self._handle.close()
        self._handle = None


def read_cpu_samples(path: Path) -> tuple[tuple[CpuSampleRow, ...], tuple[str, ...]]:
    """Best-effort parse of persisted CPU samples (v1 or v2). Never raises."""
    if not path.is_file():
        return (), ("cpu_samples_csv_missing",)

    reasons: list[str] = []
    rows: list[CpuSampleRow] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return (), ("cpu_samples_csv_header_mismatch",)
            if header == list(CPU_SAMPLES_HEADER):
                expected_version = CPU_SCHEMA_VERSION
            elif header == list(CPU_SAMPLES_HEADER_V1):
                expected_version = CPU_SCHEMA_VERSION_V1
            else:
                return (), ("cpu_samples_csv_header_mismatch",)
            columns = header
            for raw in reader:
                row = _parse_row(raw, columns, expected_version)
                if row is None:
                    reasons.append("cpu_samples_csv_malformed")
                    continue
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error):
        reasons.append("cpu_samples_csv_malformed")

    return tuple(rows), tuple(dict.fromkeys(reasons))


def _parse_row(raw: list[str], columns: list[str], expected_version: int) -> CpuSampleRow | None:
    if len(raw) != len(columns):
        return None
    cell = dict(zip(columns, raw, strict=True))
    try:
        schema_version = int(cell["schema_version"])
        timestamp_unix = float(cell["timestamp_unix"])
        socket_id = int(cell["socket_id"])
        power_w = float(cell["power_w"])
        total_power_w = float(cell["total_power_w"]) if cell["total_power_w"] else None
        rails = {kind: float(cell[f"{kind}_w"]) for kind in COMPONENT_RAIL_KINDS if cell.get(f"{kind}_w", "") != ""}
    except ValueError:
        return None

    hostname, source, sensor = cell["hostname"], cell["source"], cell["sensor"]
    if schema_version != expected_version or not hostname or not source or not sensor:
        return None
    return CpuSampleRow(
        timestamp_unix=timestamp_unix,
        hostname=hostname,
        source=source,
        sensor=sensor,
        socket_id=socket_id,
        power_w=power_w,
        total_power_w=total_power_w,
        rails=rails,
        schema_version=schema_version,
    )
