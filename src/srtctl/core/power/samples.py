# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``samples.csv`` writer, reader, and observed-device derivation.

Terminal validation re-reads the persisted bytes rather than trusting in-memory
state, so the reader here is deliberately strict about types and ordering.
"""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from srtctl.core.power.contract import SAMPLES_HEADER, SCHEMA_VERSION, Reason, dedupe


@dataclass(frozen=True)
class SampleRow:
    """One persisted observation of one GPU."""

    timestamp_unix: float
    scrape_seq: int
    hostname: str
    gpu_index: int
    gpu_uuid: str
    power_w: float
    schema_version: int = SCHEMA_VERSION

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.scrape_seq, self.hostname, self.gpu_index)

    def to_csv(self) -> list[Any]:
        return [
            self.schema_version,
            repr(self.timestamp_unix),
            self.scrape_seq,
            self.hostname,
            self.gpu_index,
            self.gpu_uuid,
            repr(self.power_w),
        ]


@dataclass(frozen=True)
class ObservedDevice:
    """A device as it actually appeared in the persisted samples."""

    hostname: str
    gpu_index: int
    gpu_uuids: tuple[str, ...]
    first_sample_time_unix: float
    last_sample_time_unix: float
    sample_times: tuple[float, ...] = field(default=(), repr=False)

    @property
    def key(self) -> tuple[str, int]:
        return (self.hostname, self.gpu_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "gpu_index": self.gpu_index,
            "gpu_uuids": list(self.gpu_uuids),
            "first_sample_time_unix": self.first_sample_time_unix,
            "last_sample_time_unix": self.last_sample_time_unix,
        }


class SampleWriter:
    """Append-only ``samples.csv`` writer owned by the collector thread."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.row_count = 0
        handle = open(path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        try:
            writer = csv.writer(handle)
            writer.writerow(SAMPLES_HEADER)
            handle.flush()
        except BaseException:
            handle.close()
            raise
        self._handle: TextIO | None = handle
        self._writer = writer

    @property
    def closed(self) -> bool:
        return self._handle is None

    def append(self, rows: Iterable[SampleRow]) -> None:
        if self._handle is None:
            raise ValueError("samples.csv writer is closed")
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


def read_samples(path: Path) -> tuple[tuple[SampleRow, ...], tuple[str, ...]]:
    """Parse persisted samples strictly, returning valid rows and reason codes."""
    if not path.is_file():
        return (), (Reason.SAMPLES_CSV_MISSING,)

    reasons: list[str] = []
    rows: list[SampleRow] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != list(SAMPLES_HEADER):
                return (), (Reason.SAMPLES_CSV_HEADER_MISMATCH,)
            for raw in reader:
                row = _parse_row(raw)
                if row is None:
                    reasons.append(Reason.SAMPLES_CSV_MALFORMED)
                    continue
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error):
        # NOTE: a corrupt byte or oversized field is malformed data, not a crash.
        reasons.append(Reason.SAMPLES_CSV_MALFORMED)

    seen: set[tuple[int, str, int]] = set()
    unique: list[SampleRow] = []
    for row in rows:
        if row.key in seen:
            reasons.append(Reason.DUPLICATE_SAMPLE_ROW)
            continue
        seen.add(row.key)
        unique.append(row)

    if _has_non_monotonic_device(unique):
        reasons.append(Reason.TIMESTAMP_NON_MONOTONIC)

    return tuple(unique), dedupe(reasons)


def derive_observed_devices(rows: Sequence[SampleRow]) -> list[ObservedDevice]:
    """Collapse persisted rows into per-device identity and sample bounds."""
    ordered: dict[tuple[str, int], list[SampleRow]] = {}
    for row in rows:
        ordered.setdefault((row.hostname, row.gpu_index), []).append(row)

    devices: list[ObservedDevice] = []
    for key in sorted(ordered):
        device_rows = sorted(ordered[key], key=lambda row: row.scrape_seq)
        times = tuple(row.timestamp_unix for row in device_rows)
        devices.append(
            ObservedDevice(
                hostname=key[0],
                gpu_index=key[1],
                gpu_uuids=tuple(dict.fromkeys(row.gpu_uuid for row in device_rows)),
                first_sample_time_unix=min(times),
                last_sample_time_unix=max(times),
                sample_times=tuple(sorted(times)),
            )
        )
    return devices


def _parse_row(raw: list[str]) -> SampleRow | None:
    if len(raw) != len(SAMPLES_HEADER):
        return None
    try:
        schema_version = int(raw[0])
        timestamp_unix = float(raw[1])
        scrape_seq = int(raw[2])
        gpu_index = int(raw[4])
        power_w = float(raw[6])
    except ValueError:
        return None

    hostname, gpu_uuid = raw[3], raw[5]
    if schema_version != SCHEMA_VERSION or not hostname or not gpu_uuid:
        return None
    if not math.isfinite(timestamp_unix) or not math.isfinite(power_w) or power_w < 0:
        return None
    if scrape_seq < 0 or gpu_index < 0:
        return None
    return SampleRow(
        timestamp_unix=timestamp_unix,
        scrape_seq=scrape_seq,
        hostname=hostname,
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
        power_w=power_w,
        schema_version=schema_version,
    )


def _has_non_monotonic_device(rows: Sequence[SampleRow]) -> bool:
    per_device: dict[tuple[str, int], list[SampleRow]] = {}
    for row in rows:
        per_device.setdefault((row.hostname, row.gpu_index), []).append(row)
    for device_rows in per_device.values():
        ordered = sorted(device_rows, key=lambda row: row.scrape_seq)
        for previous, current in itertools.pairwise(ordered):
            if current.timestamp_unix < previous.timestamp_unix:
                return True
    return False
