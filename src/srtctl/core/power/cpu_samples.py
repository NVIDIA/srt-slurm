# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cpu/samples.csv`` writer and reader.

Best-effort by design: unlike the GPU pipeline's ``samples.py``, there is no
strict re-validation pass here and no reason-code contract with the shared
``Reason`` class. A malformed or missing file is simply unavailable data,
never a job-failing condition.

The writer emits schema v2 (one row per socket, component rails as columns).
The reader accepts v2 and the legacy v1 long format (one row per rail). A v1
row is one *rail*, not one socket, so it cannot be a ``CpuSample``; it is
returned as a :class:`LegacyCpuRailRow` and callers that need one figure per
socket go through ``cpu_rails.classify_sensor``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from srtctl.core.power.contract import (
    CPU_SAMPLES_HEADER,
    CPU_SAMPLES_HEADER_V1,
    CPU_SCHEMA_VERSION,
    CPU_SCHEMA_VERSION_V1,
)
from srtctl.core.power.cpu_rails import COMPONENT_RAIL_KINDS
from srtctl.core.power.cpu_sample import CpuSample


@dataclass(frozen=True)
class CpuSampleRow:
    """One persisted v2 row: a :class:`CpuSample` placed at a time on a host, with the node total."""

    timestamp_unix: float
    hostname: str
    sample: CpuSample
    total_power_w: float | None
    schema_version: int = CPU_SCHEMA_VERSION

    # Column views, so readers of the row never re-derive what the sample already knows.
    @property
    def source(self) -> str:
        return self.sample.source

    @property
    def sensor(self) -> str:
        return self.sample.sensor

    @property
    def socket_id(self) -> int:
        return self.sample.socket_id

    @property
    def power_w(self) -> float:
        return self.sample.power_w

    @property
    def rails(self) -> dict[str, float]:
        return self.sample.rails

    def to_csv(self) -> list[Any]:
        rails = self.rails
        return [
            self.schema_version,
            repr(self.timestamp_unix),
            self.hostname,
            self.source,
            self.sensor,
            self.socket_id,
            repr(self.power_w),
            *("" if kind not in rails else repr(rails[kind]) for kind in COMPONENT_RAIL_KINDS),
            "" if self.total_power_w is None else repr(self.total_power_w),
        ]


@dataclass(frozen=True)
class LegacyCpuRailRow:
    """One persisted v1 row: a single rail reading, before rows were pivoted per socket."""

    timestamp_unix: float
    hostname: str
    source: str
    sensor: str
    socket_id: int
    power_w: float
    total_power_w: float | None
    schema_version: int = CPU_SCHEMA_VERSION_V1

    @property
    def rails(self) -> dict[str, float]:
        return {}


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


CpuCsvRow = CpuSampleRow | LegacyCpuRailRow


def read_cpu_samples(path: Path) -> tuple[tuple[CpuCsvRow, ...], tuple[str, ...]]:
    """Best-effort parse of persisted CPU samples (v1 or v2). Never raises."""
    if not path.is_file():
        return (), ("cpu_samples_csv_missing",)

    reasons: list[str] = []
    rows: list[CpuCsvRow] = []
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


def _parse_row(raw: list[str], columns: list[str], expected_version: int) -> CpuCsvRow | None:
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
    if expected_version == CPU_SCHEMA_VERSION_V1:
        return LegacyCpuRailRow(
            timestamp_unix=timestamp_unix,
            hostname=hostname,
            source=source,
            sensor=sensor,
            socket_id=socket_id,
            power_w=power_w,
            total_power_w=total_power_w,
        )
    try:
        sample = CpuSample.from_columns(source=source, socket_id=socket_id, power_w=power_w, sensor=sensor, rails=rails)
    except ValueError:  # unknown source label
        return None
    return CpuSampleRow(
        timestamp_unix=timestamp_unix,
        hostname=hostname,
        sample=sample,
        total_power_w=total_power_w,
        schema_version=schema_version,
    )
