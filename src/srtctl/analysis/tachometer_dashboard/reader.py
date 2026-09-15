# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read only Tachometer capture files and reduce them without discarding identity.

String dictionaries and NumPy operate on batches, never a Python loop per raw row.
Dense bin accumulators use temporary memory-mapped files under the output directory;
only the small series catalog and previous-sample state live in Python memory.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CORE_COLUMNS = {
    "metric_name",
    "metric_name_clean",
    "scraper_endpoint",
    "metric_value",
    "histogram_bucket_lower",
    "histogram_bucket_upper",
    "histogram_sum",
    "histogram_count",
    "timestamp_ns",
    "time_since_start",
}
POINT_COLUMNS = ["bin", "mean", "min", "max", "last", "delta", "observed_s", "samples", "resets", "gaps"]
BATCH_SIZE = 262_144
_LABEL = re.compile(r"\s*([^\s=,]+)\s*=\s*")


def parse_metric_name(encoded: str) -> tuple[str, dict[str, str]]:
    """Parse quoted Prometheus labels and Tachometer's unquoted node labels.

    Quoted commas and the Prometheus escapes \\n, \\" and \\\\ are supported. A
    malformed name is rejected rather than silently merging unlike identities.
    """
    brace = encoded.find("{")
    if brace < 0:
        return encoded, {}
    if not encoded.endswith("}"):
        raise ValueError(f"Unterminated metric labels: {encoded!r}")
    body, pos, labels = encoded[brace + 1 : -1], 0, {}
    while pos < len(body):
        match = _LABEL.match(body, pos)
        if match is None:
            raise ValueError(f"Malformed metric labels: {encoded!r}")
        key, pos = match[1], match.end()
        if pos < len(body) and body[pos] == '"':
            pos += 1
            chars = []
            while pos < len(body):
                char = body[pos]
                pos += 1
                if char == '"':
                    break
                if char == "\\":
                    if pos == len(body):
                        raise ValueError(f"Unterminated label escape: {encoded!r}")
                    char = body[pos]
                    pos += 1
                    char = {"n": "\n", '"': '"', "\\": "\\"}.get(char, "\\" + char)
                chars.append(char)
            else:
                raise ValueError(f"Unterminated quoted label: {encoded!r}")
            value = "".join(chars)
        else:
            end = body.find(",", pos)
            end = len(body) if end < 0 else end
            value, pos = body[pos:end].strip(), end
        if key in labels:
            raise ValueError(f"Duplicate label {key!r}: {encoded!r}")
        labels[key] = value
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos < len(body):
            if body[pos] != ",":
                raise ValueError(f"Missing comma in metric labels: {encoded!r}")
            pos += 1
    return encoded[:brace], labels


def discover_sources(path: Path) -> list[Path]:
    """Find one raw capture leaf, never inspect sidecar configs/logs/exports.

    A compacted final file supersedes intermediate Parquet files in its leaf.
    An Arrow tail is retained; reduction verifies whether it overlaps the final.
    Multiple capture leaves require an explicit leaf to avoid mixing runs.
    """
    path = Path(path)
    if path.is_file():
        if path.suffix not in {".parquet", ".arrow"}:
            raise ValueError("Expected a raw .parquet or .arrow capture file")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    direct = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {".parquet", ".arrow"})
    if not direct:
        found = sorted(set(path.rglob("*.parquet")) | set(path.rglob("*.arrow")))
        leaves = {p.parent for p in found}
        if len(leaves) > 1:
            raise ValueError("Multiple raw capture directories found; select one capture leaf explicitly")
        direct = found
    if not direct:
        raise FileNotFoundError(f"No raw .parquet/.arrow files under {path}")
    finals = [p for p in direct if p.name == "final.parquet"]
    return (finals or [p for p in direct if p.suffix == ".parquet"]) + [p for p in direct if p.suffix == ".arrow"]


def _batches(path: Path, columns: list[str] | None = None) -> Iterator[pa.RecordBatch]:
    if path.suffix == ".parquet":
        yield from pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE, columns=columns)
    elif path.suffix == ".arrow":
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                if columns is not None:
                    batch = batch.select(columns)
                for start in range(0, batch.num_rows, BATCH_SIZE):
                    yield batch.slice(start, BATCH_SIZE)
    else:
        raise ValueError(f"Unsupported raw capture format: {path}")


def _schema(path: Path) -> pa.Schema:
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).schema_arrow
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_stream(source).schema


def _source_snapshot(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": str(stat.st_mtime_ns),
        "ctime_ns": str(stat.st_ctime_ns),
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _assert_unchanged(path: Path, snapshot: dict[str, Any]) -> None:
    try:
        current = _source_snapshot(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{path}: raw capture changed or disappeared; select immutable input files") from exc
    if any(snapshot[key] != value for key, value in current.items()):
        raise ValueError(f"{path}: raw capture changed while reading; select immutable input files")


def _source_info(path: Path) -> dict[str, Any]:
    snapshot = _source_snapshot(path)
    schema = _schema(path)
    missing = {"metric_name", "metric_value", "scraper_endpoint"} - set(schema.names)
    if missing:
        raise ValueError(f"{path}: missing raw columns {sorted(missing)}")
    columns = [c for c in ("timestamp_ns", "time_since_start") if c in schema.names]
    if not columns:
        raise ValueError(f"{path}: capture has neither timestamp_ns nor time_since_start")
    ranges: dict[str, list[float | int]] = {}
    valid_counts = dict.fromkeys(columns, 0)
    rows = 0
    anchor_min = anchor_max = None
    for batch in _batches(path, columns):
        rows += batch.num_rows
        for col in columns:
            values = batch.column(col)
            # These Arrow kernels are registered dynamically; call_function is
            # statically visible and uses the same default kernel options.
            valid = pc.call_function(
                "and", [pc.call_function("is_valid", [values]), pc.call_function("is_finite", [values])]
            )
            values = pc.call_function("filter", [values, valid])
            valid_counts[col] += len(values)
            if len(values):
                mm = pc.call_function("min_max", [values]).as_py()
                if col not in ranges:
                    ranges[col] = [mm["min"], mm["max"]]
                else:
                    ranges[col] = [min(ranges[col][0], mm["min"]), max(ranges[col][1], mm["max"])]
        if "timestamp_ns" in columns and "time_since_start" in columns:
            timestamps = pc.cast(batch.column("timestamp_ns"), pa.int64()).to_numpy(zero_copy_only=False)
            elapsed = pc.cast(batch.column("time_since_start"), pa.float64()).to_numpy(zero_copy_only=False)
            paired = np.isfinite(timestamps) & (timestamps > 0) & np.isfinite(elapsed)
            if np.any(paired):
                anchors = timestamps[paired].astype(np.int64) - np.rint(elapsed[paired] * 1e9).astype(np.int64)
                low, high = int(anchors.min()), int(anchors.max())
                anchor_min = low if anchor_min is None else min(anchor_min, low)
                anchor_max = high if anchor_max is None else max(anchor_max, high)
    _assert_unchanged(path, snapshot)
    if anchor_min is not None:
        assert anchor_max is not None  # Both extrema are populated from the same valid batch.
        if anchor_max - anchor_min > 1_000_000:
            raise ValueError(f"{path}: inconsistent raw epoch/relative time origins (spread exceeds 1 ms)")
    if not rows:
        raise ValueError(f"{path}: empty capture")
    # Positive epoch values are required; zero/null columns indicate older relative captures.
    epoch = valid_counts.get("timestamp_ns") == rows and ranges.get("timestamp_ns", [0])[0] > 0
    col = "timestamp_ns" if epoch else "time_since_start"
    if valid_counts.get(col) != rows:
        raise ValueError(f"{path}: incomplete timestamps; cannot infer missing times from external files")
    if col == "timestamp_ns":
        start, end = map(int, ranges[col])
    else:
        start, end = (round(float(v) * 1e9) for v in ranges[col])
    return {
        "name": path.name,
        "path": str(path),
        "format": path.suffix[1:],
        **snapshot,
        "rows": rows,
        "clock": "epoch" if epoch else "relative",
        "start_ns": str(start),
        "end_ns": str(end),
        "time_column": col,
        "origin_anchor_min_ns": str(anchor_min) if anchor_min is not None else None,
        "origin_anchor_max_ns": str(anchor_max) if anchor_max is not None else None,
    }


@dataclass
class _Series:
    endpoint: str
    metadata: dict[str, str]
    labels: dict[str, str]
    index: int
    previous_t: int | None = None
    previous_v: float | None = None
    cadence: list[float] = field(default_factory=list)
    ambiguous: bool = False


class _Metric:
    def __init__(self, name: str, metric_id: str, path: Path, bin_count: int) -> None:
        self.name, self.id, self.path, self.bin_count = name, metric_id, path, bin_count
        self.kind = "scalar"
        self.series: dict[tuple, _Series] = {}
        self.capacity = 0
        self.data: np.memmap | None = None
        self.warnings: set[str] = set()
        self.duplicate_rows = 0
        self.conflicting_rows = 0

    def add_series(self, key: tuple, endpoint: str, metadata: dict, labels: dict) -> _Series:
        if key in self.series:
            return self.series[key]
        index = len(self.series)
        if index == self.capacity:
            self.close()
            self.capacity = max(4, self.capacity * 2)
            with self.path.open("ab") as fh:
                fh.truncate(self.capacity * self.bin_count * 9 * 8)
            self.data = np.memmap(self.path, mode="r+", dtype="float64", shape=(self.capacity, self.bin_count, 9))
        state = _Series(endpoint, metadata, labels, index)
        self.series[key] = state
        return state

    def close(self) -> None:
        if self.data is not None:
            self.data.flush()
            # NumPy provides no public memmap.close(); this instance owns its mapping.
            self.data._mmap.close()  # ty: ignore[unresolved-attribute]
            self.data = None


def _update(
    metric: _Metric, state: _Series, times: np.ndarray, values: np.ndarray, origin: int, step: int, counter: bool
) -> None:
    if state.previous_t is not None and times[0] < state.previous_t:
        raise ValueError(
            f"Out-of-order samples for {metric.name} at {state.endpoint}; sort each full series by timestamp"
        )
    before_t = state.previous_t if state.previous_t is not None else int(times[0])
    before_v = state.previous_v if state.previous_v is not None else float(values[0])
    dt = np.diff(times, prepend=before_t)
    dv = np.diff(values, prepend=before_v)
    duplicated = dt == 0
    if state.previous_t is None:
        duplicated[0] = False
    metric.duplicate_rows += int(np.count_nonzero(duplicated))
    conflicts = duplicated & (dv != 0)
    if np.any(conflicts):
        metric.conflicting_rows += int(np.count_nonzero(conflicts))
        state.ambiguous = True
        assert metric.data is not None
        metric.data[state.index, :, 4:6] = 0
        metric.warnings.add(
            "Different raw values share an identical full identity and timestamp; affected series retain raw observations but have no counter rates."
        )
    counter = counter and not state.ambiguous
    positive_dt = dt[dt > 0].astype(np.float64) / 1e9
    if len(state.cadence) < 64:
        state.cadence.extend(positive_dt[: 64 - len(state.cadence)].tolist())
    gap_limit = 3 * float(np.median(state.cadence)) if state.cadence else math.inf
    gaps = dt / 1e9 > gap_limit
    resets = (dv < 0) & (dt > 0) if counter else np.zeros(len(times), dtype=bool)
    valid = (dt > 0) & ~gaps & ~resets
    bins = ((times - origin) // step).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
    ends = np.r_[starts[1:], len(bins)]
    which = bins[starts]
    assert metric.data is not None
    accumulator = metric.data[state.index]
    old = accumulator[which].copy()
    new_count = ends - starts
    new_min, new_max = np.minimum.reduceat(values, starts), np.maximum.reduceat(values, starts)
    old[:, 0] += np.add.reduceat(values, starts)
    old[:, 1] = np.where(old[:, 6] > 0, np.minimum(old[:, 1], new_min), new_min)
    old[:, 2] = np.where(old[:, 6] > 0, np.maximum(old[:, 2], new_max), new_max)
    old[:, 3] = values[ends - 1]
    if counter:
        old[:, 4] += np.add.reduceat(np.where(valid, dv, 0), starts)
        old[:, 5] += np.add.reduceat(np.where(valid, dt / 1e9, 0), starts)
    old[:, 6] += new_count
    old[:, 7] += np.add.reduceat(resets.astype(np.int64), starts)
    old[:, 8] += np.add.reduceat(gaps.astype(np.int64), starts)
    accumulator[which] = old
    state.previous_t, state.previous_v = int(times[-1]), float(values[-1])


def _dictionary(array: pa.Array) -> tuple[list[str], np.ndarray]:
    encoded = pc.call_function("dictionary_encode", [pc.fill_null(pc.cast(array, pa.string()), "")])
    return encoded.dictionary.to_pylist(), encoded.indices.to_numpy(zero_copy_only=False)


def _process_batch(
    batch: pa.RecordBatch,
    info: dict,
    metrics: dict[str, _Metric],
    directory: Path,
    origin: int,
    step: int,
    bin_count: int,
    is_counter: Callable[[str], bool],
    name_cache: dict[str, tuple[str, dict[str, str]]],
) -> int:
    values = pc.cast(batch.column("metric_value"), pa.float64()).to_numpy(zero_copy_only=False)
    finite = np.isfinite(values)
    rejected = int(len(values) - np.count_nonzero(finite))
    if not np.all(finite):
        batch = batch.filter(pa.array(finite))
        values = values[finite]
    if not len(values):
        return rejected
    time_values = batch.column(info["time_column"]).to_numpy(zero_copy_only=False)
    times = time_values.astype(np.int64) if info["clock"] == "epoch" else np.rint(time_values * 1e9).astype(np.int64)
    identity_columns = ["metric_name", "scraper_endpoint"] + sorted(set(batch.schema.names) - CORE_COLUMNS)
    dictionaries, codes = zip(*[_dictionary(batch.column(col)) for col in identity_columns], strict=True)
    # Canonicalize equivalent label order before grouping, not after updating state.
    canonical_names: dict[tuple, int] = {}
    name_lookup = []
    inline_bounds = []
    for encoded in dictionaries[0]:
        if encoded not in name_cache:
            name_cache[encoded] = parse_metric_name(encoded)
        base, parsed = name_cache[encoded]
        inline_bound = math.nan
        if "le" in parsed:
            try:
                inline_bound = float(parsed["le"])
            except ValueError as exc:
                raise ValueError(f"Invalid histogram bound in {encoded!r}") from exc
            if math.isnan(inline_bound):
                raise ValueError(f"Invalid histogram bound in {encoded!r}")
            base = base.removesuffix("_bucket")
        inline_bounds.append(inline_bound)
        key = (base, tuple(sorted(parsed.items())))
        name_lookup.append(canonical_names.setdefault(key, len(canonical_names)))
    dimensions = list(codes)
    dimensions[0] = np.asarray(name_lookup, dtype=np.int64)[codes[0]]
    # Most captures retain le inline. Where only the explicit upper column survives,
    # include that raw bound in identity too, so distinct buckets cannot merge.
    uppers = (
        pc.cast(batch.column("histogram_bucket_upper"), pa.float64()).to_numpy(zero_copy_only=False)
        if "histogram_bucket_upper" in batch.schema.names
        else np.full(len(values), np.nan)
    )
    inline = np.asarray(inline_bounds, dtype=np.float64)[codes[0]]
    has_inline = ~np.isnan(inline)
    conflicting = has_inline & ~np.isnan(uppers) & (inline != uppers)
    if np.any(conflicting):
        first_conflict = int(np.flatnonzero(conflicting)[0])
        encoded = dictionaries[0][codes[0][first_conflict]]
        raise ValueError(
            f"Conflicting histogram bounds in {encoded!r}: inline le differs from "
            f"histogram_bucket_upper={uppers[first_conflict]!r}; select unambiguous raw data"
        )
    column_bound = ~has_inline & ~np.isnan(uppers)
    dimensions.extend((column_bound, np.where(column_bound, uppers, 0)))
    varying = [col for col in dimensions if np.any(col != col[0])]
    order = np.lexsort((times, *varying))
    ordered_dimensions = [col[order] for col in varying]
    changed = np.zeros(max(0, len(order) - 1), dtype=bool)
    for col in ordered_dimensions:
        changed |= col[1:] != col[:-1]
    starts = np.r_[0, np.flatnonzero(changed) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends, strict=True):
        indices = order[start:end]
        first = indices[0]
        fields = [dictionary[code[first]] for dictionary, code in zip(dictionaries, codes, strict=True)]
        encoded, endpoint = fields[:2]
        if encoded not in name_cache:
            name_cache[encoded] = parse_metric_name(encoded)
        name, parsed_labels = name_cache[encoded]
        labels = dict(parsed_labels)
        histogram = "le" in labels or not np.isnan(uppers[first])
        if histogram and "le" not in labels:
            labels["le"] = repr(float(uppers[first]))
        if histogram and name.endswith("_bucket"):
            name = name[:-7]
        metadata = {col: value for col, value in zip(identity_columns[2:], fields[2:], strict=True) if value != ""}
        metric = metrics.get(name)
        if metric is None:
            metric_id = f"m{len(metrics):04d}"
            metric = metrics[name] = _Metric(name, metric_id, directory / f"{metric_id}.f64", bin_count)
        if histogram:
            metric.kind = "histogram"
            metric.warnings.add(
                "Attached histogram sum/count/lower columns are not used; bucket samples retain their raw labels."
            )
        for label_key in metadata.keys() & labels.keys():
            if metadata[label_key] != labels[label_key]:
                metric.warnings.add(
                    f"Inline label {label_key!r} conflicts with raw metadata; both identities are retained separately."
                )
        key = (endpoint, tuple(sorted(metadata.items())), tuple(sorted(labels.items())))
        state = metric.add_series(key, endpoint, metadata, labels)
        _update(metric, state, times[indices], values[indices], origin, step, histogram or is_counter(name))
    return rejected


def _write_metric(metric: _Metric, out_dir: Path, label_values: dict[str, set[str]]) -> dict:
    payload = f"{metric.id}.json.gz"
    labels: set[str] = set()
    assert metric.data is not None
    with gzip.open(out_dir / payload, "wt", encoding="utf-8", compresslevel=6) as out:
        header = {"name": metric.name, "kind": metric.kind, "columns": POINT_COLUMNS}
        out.write(json.dumps(header, separators=(",", ":"))[:-1] + ',"series":[')
        for index, state in enumerate(metric.series.values()):
            if index:
                out.write(",")
            label_values.setdefault("scraper_endpoint", set()).add(state.endpoint)
            for prefix, scope in (("capture", state.metadata), ("label", state.labels)):
                for key, value in scope.items():
                    labels.add(key)
                    labels.add(f"{prefix}.{key}")
                    label_values.setdefault(key, set()).add(value)
                    label_values.setdefault(f"{prefix}.{key}", set()).add(value)
            matrix = metric.data[state.index]
            indices = np.flatnonzero(matrix[:, 6] > 0)
            subset = matrix[indices]
            # Process one series at a time rather than constructing an entire run's JSON.
            points = np.column_stack((indices, subset[:, 0] / subset[:, 6], subset[:, 1:])).tolist()
            for point in points:
                point[0] = int(point[0])
                if point[6] == 0:
                    point[5], point[6] = None, None
                point[7:] = [int(value) for value in point[7:]]
            series = {
                "id": f"{metric.id}s{index}",
                "endpoint": state.endpoint,
                "metadata": state.metadata,
                "labels": state.labels,
                "ambiguous": state.ambiguous,
                "points": points,
            }
            out.write(json.dumps(series, separators=(",", ":"), allow_nan=False))
        out.write("]}")
    if metric.duplicate_rows:
        metric.warnings.add(
            f"{metric.duplicate_rows:,} repeated timestamp rows retained as raw observations ({metric.conflicting_rows:,} had conflicting values)."
        )
    return {
        "id": metric.id,
        "name": metric.name,
        "kind": metric.kind,
        "series_count": len(metric.series),
        "endpoints": sorted({series.endpoint for series in metric.series.values()}),
        "duplicate_rows": metric.duplicate_rows,
        "conflicting_rows": metric.conflicting_rows,
        "payload": payload,
        "labels": sorted(labels),
        "warnings": sorted(metric.warnings),
    }


def reduce_capture(
    paths: list[Path],
    out_dir: Path,
    resolution_s: float = 10,
    is_counter: Callable[[str], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Build a metric catalog and compressed payloads exclusively from raw captures.

    Rates use measured consecutive intervals, never cross resets or observed gaps
    exceeding three times the series' median initial cadence. No counter type is
    guessed: scalar deltas require ``is_counter``; bucket deltas are intrinsic.
    Selected files must remain unchanged throughout both passes. File stat records
    detect concurrent writes/replacements; they do not certify capture completeness.
    """
    if not math.isfinite(resolution_s) or resolution_s <= 0:
        raise ValueError("resolution_s must be finite and positive")
    step = round(resolution_s * 1e9)
    if step < 1:
        raise ValueError("resolution_s must be at least one nanosecond")
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one raw capture file is required")
    if any(path.suffix not in {".parquet", ".arrow"} for path in paths):
        raise ValueError("Only raw .parquet/.arrow inputs are accepted")
    report = progress or (lambda message: None)
    infos = []
    for path in paths:
        report(f"Inspecting raw capture {path.name}")
        infos.append(_source_info(path))
    if len({info["clock"] for info in infos}) != 1:
        raise ValueError("Mixed epoch and relative capture clocks; select files with one raw time origin")
    anchors = [info for info in infos if info["origin_anchor_min_ns"] is not None]
    if anchors:
        lowest = min(int(info["origin_anchor_min_ns"]) for info in anchors)
        highest = max(int(info["origin_anchor_max_ns"]) for info in anchors)
        if highest - lowest > 1_000_000:
            raise ValueError(
                "Mixed raw capture origins: epoch minus relative time differs by more than 1 ms; select files from one Tachometer capture"
            )
    for info in infos:
        _assert_unchanged(Path(info["path"]), info)
    infos.sort(key=lambda info: int(info["start_ns"]))
    for previous, current in pairwise(infos):
        if int(current["start_ns"]) <= int(previous["end_ns"]):
            raise ValueError(
                f"Overlapping raw captures: {previous['name']} and {current['name']}; select one compaction generation plus a non-overlapping Arrow tail"
            )
    origin = int(infos[0]["start_ns"])
    end = max(int(info["end_ns"]) for info in infos)
    bin_count = (end - origin) // step + 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, _Metric] = {}
    name_cache: dict[str, tuple[str, dict[str, str]]] = {}
    total = sum(info["rows"] for info in infos)
    processed = rejected = 0
    label_values: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory(prefix=".tachometer-reduce-", dir=out_dir) as temporary:
        try:
            for info in infos:
                _assert_unchanged(Path(info["path"]), info)
                columns = [
                    name
                    for name in _schema(Path(info["path"])).names
                    if name
                    not in {
                        "metric_name_clean",
                        "histogram_sum",
                        "histogram_count",
                        "histogram_bucket_lower",
                        "time_since_start" if info["clock"] == "epoch" else "timestamp_ns",
                    }
                ]
                for batch in _batches(Path(info["path"]), columns):
                    rejected += _process_batch(
                        batch,
                        info,
                        metrics,
                        Path(temporary),
                        origin,
                        step,
                        bin_count,
                        is_counter or (lambda name: False),
                        name_cache,
                    )
                    processed += batch.num_rows
                    report(f"Reduced {processed:,}/{total:,} raw rows; {len(metrics)} metric families")
            for info in infos:
                _assert_unchanged(Path(info["path"]), info)
            catalog_metrics = []
            for metric in sorted(metrics.values(), key=lambda item: item.name):
                catalog_metrics.append(_write_metric(metric, out_dir, label_values))
                metric.close()
            epoch = infos[0]["clock"] == "epoch"
            return {
                "schema_version": 1,
                "start_ns": str(origin) if epoch else None,
                "end_ns": str(end) if epoch else None,
                "relative_start_s": origin / 1e9 if not epoch else None,
                "duration_s": (end - origin) / 1e9,
                "resolution_s": resolution_s,
                "bin_count": bin_count,
                "source_files": infos,
                "row_count": total,
                "nonfinite_rows": rejected,
                "metrics": catalog_metrics,
                "label_values": {key: sorted(values) for key, values in sorted(label_values.items())},
            }
        finally:
            for metric in metrics.values():
                metric.close()
