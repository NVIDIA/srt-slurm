# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental parser for SGLang prefill/decode batch-log lines.

This module is intentionally narrow: it walks worker log files in a
``logs/`` directory and extracts numeric metrics from
``Prefill batch, ...`` / ``Decode batch, ...`` lines. It does not
aggregate, plot, or interpret the data.

A :class:`LogState` is kept across snapshot ticks so re-parsing a
multi-hour benchmark every minute stays cheap — only newly appended
bytes are read on each ``refresh()``.

A line we parse looks like (one of two timestamp variants observed in
SGLang scheduler logs)::

    p0\\x1b[2m2026-04-27T23:03:15.250907Z\\x1b[0m ... Prefill batch,
        #new-seq: 1, #new-token: 256, #cached-token: 0,
        full token usage: 0.00, #running-req: 0, #queue-req: 0,
        #prealloc-req: 0, #inflight-req: 0,
        input throughput (token/s): 0.00,

    [2025-11-04 05:31:43 DP0 TP0 EP0] Decode batch, #running-req: 1,
        #full token: 7424, full token usage: 0.00,
        ..., gen throughput (token/s): 0.03, #queue-req: 0,
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

PREFILL_METRICS: dict[str, str] = {
    "#new-seq": r"#new-seq:\s*([\d.]+)",
    "#new-token": r"#new-token:\s*([\d.]+)",
    "#cached-token": r"#cached-token:\s*([\d.]+)",
    "full token usage": r"full token usage:\s*([\d.]+)",
    "#running-req": r"#running-req:\s*([\d.]+)",
    "#queue-req": r"#queue-req:\s*([\d.]+)",
    "#prealloc-req": r"#prealloc-req:\s*([\d.]+)",
    "#inflight-req": r"#inflight-req:\s*([\d.]+)",
    "input throughput (token/s)": r"input throughput \(token/s\):\s*([\d.]+)",
}

DECODE_METRICS: dict[str, str] = {
    "#running-req": r"#running-req:\s*([\d.]+)",
    "#full token": r"#full token:\s*([\d.]+)",
    "full token usage": r"full token usage:\s*([\d.]+)",
    "#prealloc-req": r"#prealloc-req:\s*([\d.]+)",
    "#transfer-req": r"#transfer-req:\s*([\d.]+)",
    "#retracted-req": r"#retracted-req:\s*([\d.]+)",
    "gen throughput (token/s)": r"gen throughput \(token/s\):\s*([\d.]+)",
    "#queue-req": r"#queue-req:\s*([\d.]+)",
}

PREFILL_KEYWORD = "Prefill batch"
DECODE_KEYWORD = "Decode batch"

# Compiled lazily.
_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}

# Two timestamp variants seen across SGLang versions.
_TS_ANSI = re.compile(r"\[2m(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}\.\d+)")
_TS_BRACKET = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _pattern(raw: str) -> re.Pattern[str]:
    cached = _PATTERN_CACHE.get(raw)
    if cached is None:
        cached = re.compile(raw)
        _PATTERN_CACHE[raw] = cached
    return cached


def _parse_timestamp(line: str) -> datetime | None:
    m = _TS_ANSI.search(line)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return None
    m = _TS_BRACKET.search(line)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Per-file series
# ---------------------------------------------------------------------------


@dataclass
class FileSeries:
    """Time-series accumulated from one worker log file.

    Mutated in place by :func:`parse_file_incremental` so successive
    snapshot ticks only read newly appended bytes (using ``byte_offset``
    as the resume point).
    """

    path: Path
    timestamps: list[datetime] = field(default_factory=list)
    metrics: dict[str, list[float | None]] = field(default_factory=dict)
    byte_offset: int = 0

    @property
    def empty(self) -> bool:
        return not self.timestamps

    @property
    def label(self) -> str:
        """Short legend label for plotting.

        Tries to extract just the worker index from typical SLURM-named
        files such as ``slurm-gb300-133-181_prefill_w0.out`` -> ``"p0"``
        or ``..._decode_w3.out`` -> ``"d3"``. Falls back to the full
        filename stem when it can't infer a short form.
        """
        stem = self.path.name
        for ext in (".out", ".err"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        # Match ``<anything>_(prefill|decode|agg)_w<n>``.
        m = re.search(r"_(prefill|decode|agg)_w(\d+)$", stem)
        if m:
            return f"{m.group(1)[0]}{m.group(2)}"
        return stem


def parse_file_incremental(series: FileSeries, keyword: str, metrics_def: dict[str, str]) -> int:
    """Append rows parsed from new bytes in ``series.path`` onto ``series``.

    Returns the number of new rows appended this call.
    """
    for name in metrics_def:
        series.metrics.setdefault(name, [])

    try:
        size = series.path.stat().st_size
    except FileNotFoundError:
        return 0
    if size <= series.byte_offset:
        return 0

    new_rows = 0
    try:
        with open(series.path, errors="replace") as f:
            f.seek(series.byte_offset)
            for line in f:
                if keyword not in line:
                    continue
                ts = _parse_timestamp(line)
                if ts is None:
                    continue

                values: dict[str, float | None] = {}
                any_value = False
                for name, raw in metrics_def.items():
                    m = _pattern(raw).search(line)
                    if m is None:
                        values[name] = None
                        continue
                    try:
                        values[name] = float(m.group(1))
                        any_value = True
                    except ValueError:
                        values[name] = None

                if not any_value:
                    continue

                series.timestamps.append(ts)
                for name in metrics_def:
                    series.metrics[name].append(values[name])
                new_rows += 1

            series.byte_offset = f.tell()
    except OSError as e:
        logger.warning("failed to read %s: %s", series.path, e)
        return 0

    return new_rows


# ---------------------------------------------------------------------------
# Log directory discovery + state
# ---------------------------------------------------------------------------


def _classify_log_files(log_dir: Path) -> tuple[list[Path], list[Path]]:
    """Find prefill/decode worker logs in ``log_dir``.

    Filenames containing ``prefill`` go to the prefill list, ``decode``
    to the decode list. ``_agg_`` files (aggregated mode) are added to
    both since they carry both prefill and decode batch lines.
    """
    if not log_dir.is_dir():
        return [], []

    prefill: list[Path] = []
    decode: list[Path] = []
    agg: list[Path] = []
    for entry in sorted(os.listdir(log_dir)):
        if not (entry.endswith(".out") or entry.endswith(".err")):
            continue
        path = log_dir / entry
        if "prefill" in entry:
            prefill.append(path)
        elif "decode" in entry:
            decode.append(path)
        elif "_agg_" in entry:
            agg.append(path)

    return prefill + agg, decode + agg


@dataclass
class LogState:
    """Per-log-dir incremental parse state.

    A single instance is created when the snapshotter starts and reused
    across all ticks: file discovery is re-run on every refresh (workers
    can come up late), but each file's bytes are only read once.
    """

    log_dir: Path
    prefill_files: dict[Path, FileSeries] = field(default_factory=dict)
    decode_files: dict[Path, FileSeries] = field(default_factory=dict)

    def refresh(self) -> tuple[int, int]:
        """Re-discover files and parse newly appended bytes.

        Returns ``(new_prefill_rows, new_decode_rows)`` parsed this call.
        """
        pf_paths, dc_paths = _classify_log_files(self.log_dir)

        new_pf = 0
        for p in pf_paths:
            s = self.prefill_files.setdefault(p, FileSeries(path=p))
            new_pf += parse_file_incremental(s, PREFILL_KEYWORD, PREFILL_METRICS)

        new_dc = 0
        for p in dc_paths:
            s = self.decode_files.setdefault(p, FileSeries(path=p))
            new_dc += parse_file_incremental(s, DECODE_KEYWORD, DECODE_METRICS)

        return new_pf, new_dc

    @property
    def has_data(self) -> bool:
        return any(not s.empty for s in self.prefill_files.values()) or any(
            not s.empty for s in self.decode_files.values()
        )
