# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read all raw Tachometer metric families, preserving labels and sample evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tomli as tomllib
from pyarrow import ipc

from srtctl.analysis.metric_catalog import describe_metric

from .engines import MetricDefinition, engine_metrics

if TYPE_CHECKING:
    from .importer import Importer

METRICS = {
    "gpu_util": MetricDefinition("GPU utilization", "%"),
    "DCGM_FI_DEV_GPU_UTIL": MetricDefinition("GPU utilization", "%"),
    "FI_DEV_FB_USED": MetricDefinition("GPU memory used", "MiB"),
    **engine_metrics(),
    "dynamo_component_inflight_requests": MetricDefinition("Worker in flight", "requests"),
    "dynamo_frontend_inflight_requests": MetricDefinition("Frontend in flight", "requests"),
    "dynamo_frontend_queued_requests": MetricDefinition("Frontend queued", "requests"),
    "dynamo_frontend_router_queue_pending_requests": MetricDefinition("Router pending", "requests"),
    "dynamo_work_handler_queue_depth": MetricDefinition("Handler queue", "requests"),
    "load1": MetricDefinition("Host load (1 min)", "load"),
    "memory_MemAvailable_bytes": MetricDefinition("Host available memory", "bytes"),
}


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


def capture_files(root: Path) -> list[Path]:
    """final.parquet supersedes compacted shards; an Arrow tail can remain."""
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files = sorted(set(root.rglob("*.parquet")) | set(root.rglob("*.arrow")))
    if len({p.parent for p in files}) > 1:
        raise ValueError("Multiple raw metric capture leaves; select one with --metrics")
    final = [p for p in files if p.name == "final.parquet"]
    return (final or [p for p in files if p.suffix == ".parquet"]) + [p for p in files if p.suffix == ".arrow"]


def batches(path: Path) -> Iterator[pa.RecordBatch]:
    # Some historical captures use the .parquet extension for Arrow IPC files.
    with path.open("rb") as stream:
        magic = stream.read(6)
    if magic[:4] == b"PAR1":
        yield from pq.ParquetFile(path).iter_batches(batch_size=131072)
    else:
        with pa.memory_map(str(path), "r") as source:
            if magic == b"ARROW1":
                reader = ipc.open_file(source)
                for index in range(reader.num_record_batches):
                    yield reader.get_batch(index)
            else:
                yield from ipc.open_stream(source)


_VALUE_COLUMNS = {
    "_row",
    "timestamp_ns",
    "time_since_start",
    "metric_value",
    "metric_name",
    "metric_name_clean",
    "histogram_sum",
    "histogram_count",
    "histogram_bucket_lower",
    "histogram_bucket_upper",
}


def _parsed_name(encoded: str, cache: dict) -> tuple[str, dict[str, str], float | None]:
    if encoded not in cache:
        if not isinstance(encoded, str) or not encoded:
            raise ValueError(f"Missing raw metric name: {encoded!r}")
        name, labels = parse_metric_name(encoded)
        if not name:
            raise ValueError(f"Missing raw metric name: {encoded!r}")
        bound = None
        if "le" in labels:
            try:
                bound = float(labels["le"])
            except ValueError as exc:
                raise ValueError(f"Invalid histogram bound in {encoded!r}") from exc
            if math.isnan(bound):
                raise ValueError(f"Invalid histogram bound in {encoded!r}")
        cache[encoded] = name, labels, bound
    return cache[encoded]


def _shape(encoded: str, upper: float | None, cache: dict) -> tuple[str, dict[str, str], bool]:
    name, original, inline = _parsed_name(encoded, cache)
    labels = dict(original)
    if upper is not None:
        if math.isnan(upper):
            raise ValueError(f"Invalid histogram_bucket_upper in {encoded!r}")
        if inline is not None and inline != upper:
            raise ValueError(f"Conflicting histogram bounds in {encoded!r}: le differs from histogram_bucket_upper")
    bound = inline if inline is not None else upper
    histogram = bound is not None
    if bound is not None:
        labels["le"] = "+Inf" if bound == math.inf else "-Inf" if bound == -math.inf else repr(float(bound))
    return name.removesuffix("_bucket") if histogram else name, labels, histogram


def _inventory(batch: pa.RecordBatch, families: dict, parsed: dict) -> None:
    """Catalog full-capture names with Arrow grouping, without expanding every row."""
    if not batch.num_rows:
        return
    if batch["metric_name"].null_count:
        raise ValueError("Missing raw metric name: null in raw capture")
    encoded = pc.call_function("dictionary_encode", [batch["metric_name"]])
    definitions = [_parsed_name(name, parsed) for name in encoded.dictionary.to_pylist()]
    names = pc.take(pa.array([item[0] for item in definitions]), encoded.indices)
    inline = pc.take(pa.array([item[2] for item in definitions], type=pa.float64()), encoded.indices)
    upper = (
        batch["histogram_bucket_upper"] if "histogram_bucket_upper" in batch.schema.names else pa.nulls(batch.num_rows)
    )
    upper = pc.cast(upper, pa.float64())
    if pc.call_function("any", [pc.fill_null(pc.call_function("is_nan", [upper]), False)]).as_py():
        raise ValueError("Invalid histogram_bucket_upper: NaN in raw capture")
    conflict = pc.fill_null(pc.call_function("not_equal", [inline, upper]), False)
    if pc.call_function("any", [conflict]).as_py():
        index = pc.call_function("indices_nonzero", [conflict])[0].as_py()
        raise ValueError(f"Conflicting histogram bounds in {batch['metric_name'][index].as_py()!r}")
    histogram = pc.call_function("or", [pc.call_function("is_valid", [inline]), pc.call_function("is_valid", [upper])])
    without_bucket = pc.call_function(
        "replace_substring_regex", [names], options=pc.ReplaceSubstringOptions(pattern="_bucket$", replacement="")
    )
    names = pc.call_function("if_else", [histogram, without_bucket, names])
    identities = pa.table({"name": names, "endpoint": batch["scraper_endpoint"], "histogram": histogram})
    for row in identities.group_by(identities.column_names, use_threads=False).aggregate([]).to_pylist():
        family = families.setdefault(row["name"], {"endpoints": set(), "histogram": False, "scalar": False})
        family["endpoints"].add(row["endpoint"] or "")
        family["histogram"] |= row["histogram"]
        family["scalar"] |= not row["histogram"]
        if family["histogram"] and family["scalar"]:
            raise ValueError(
                f"Metric family {row['name']!r} mixes scalar values and histogram buckets with incompatible units"
            )


def _description(name: str, endpoints: set[str], histogram: bool) -> dict[str, Any]:
    info = describe_metric(name, sorted(endpoints))
    if override := METRICS.get(name):
        info.update(title=override.label, unit=override.unit)
    info["value_kind"] = "histogram" if histogram else "counter" if info["counter"] else "stored"
    if histogram:
        info["observation_unit"] = info["unit"]
        info["unit"] = "observations"
        info["counter"] = True
        info["description"] += (
            " Displayed values are raw cumulative bucket observations; le is the bucket upper bound."
            " No percentile, rate, or attached sum/count is substituted for a raw sample."
        )
    elif info["counter"]:
        info["description"] += " Displayed values are raw cumulative samples, not a per-second rate."
    return info


def read_metrics(run: Importer) -> list[dict[str, Any]]:
    config_path = run.logs / "tachometer_config.toml"
    endpoints: dict[str, Any] = {}
    config_source = None
    if config_path.exists():
        config_source = run.source(config_path, "collector_config")
        endpoints = {e["name"]: e for e in tomllib.loads(config_path.read_text()).get("endpoints", [])}
    root = run.metrics_path or run.logs / "tachometer/local"
    if run.metrics_path and not root.exists():
        raise ValueError(f"Raw metrics path does not exist: {root}")
    files = capture_files(root)
    if files and not any(p.name == "final.parquet" for p in files):
        run.warnings.append(
            "Raw metrics have no final.parquet; imported available shards/tail, completeness unverified."
        )
    series_by_key: dict[tuple, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = {}
    parsed: dict[str, tuple[str, dict[str, str], float | None]] = {}
    for path in files:
        sid = run.source(path, "tachometer_raw")
        offset = 0
        for batch in batches(path):
            required = {"timestamp_ns", "metric_name", "metric_value", "scraper_endpoint"}
            if not required.issubset(batch.schema.names):
                raise ValueError(f"{path}: raw metrics require {sorted(required)}; UTC alignment cannot be inferred")
            run.audit["metric_rows_scanned"] += batch.num_rows
            _inventory(batch, families, parsed)
            mask = pc.call_function(
                "and",
                [
                    pc.call_function("greater_equal", [batch["timestamp_ns"], run.origin]),
                    pc.call_function("less_equal", [batch["timestamp_ns"], run.origin + int(run.duration * 1e9)]),
                ],
            )
            table = pa.Table.from_batches([batch]).filter(mask)
            table = table.append_column(
                "_row", pc.call_function("add", [pc.call_function("indices_nonzero", [mask]), offset])
            )
            offset += batch.num_rows
            if not table.num_rows:
                continue
            finite = pc.fill_null(pc.call_function("is_finite", [table["metric_value"]]), False)
            run.audit["nonfinite_metric_points"] += (
                table.num_rows - pc.call_function("sum", [pc.cast(finite, pa.int64())]).as_py()
            )
            table = table.filter(finite)
            if not table.num_rows:
                continue
            # Group metadata before converting to Python. Each identity is parsed once
            # per batch, while every recorded time/value/source-row is kept below.
            identity_columns = [name for name in table.column_names if name not in _VALUE_COLUMNS]
            identity_columns += ["metric_name"]
            if "histogram_bucket_upper" in table.column_names:
                identity_columns.append("histogram_bucket_upper")
            grouped = table.group_by(identity_columns, use_threads=False).aggregate(
                [("timestamp_ns", "list"), ("metric_value", "list"), ("_row", "list")]
            )
            for row in grouped.to_pylist():
                name, metric_labels, histogram = _shape(row["metric_name"], row.get("histogram_bucket_upper"), parsed)
                endpoint_name = row["scraper_endpoint"] or ""
                endpoint = endpoints.get(endpoint_name, {})
                raw_host = row.get("hostname")
                host = raw_host or endpoint.get("node_metadata", {}).get("hostname", "")
                gpu = str(row.get("gpu")) if row.get("gpu") is not None else ""
                extra = endpoint.get("gpu_metadata", {}).get(gpu, {})
                role = row.get("worker_role") or extra.get("worker_role", "")
                index = row.get("worker_index")
                index = extra.get("worker_index", "") if index in (None, "") else index
                worker = f"{role}-{index}" if role in ("prefill", "decode", "agg") and index != "" else None
                if worker is None and "frontend" in endpoint_name:
                    worker = "frontend"
                labels = {
                    key: str(row[key])
                    for key in identity_columns
                    if key not in _VALUE_COLUMNS and row[key] not in (None, "")
                }
                labels.update({f"metric.{key}": value for key, value in metric_labels.items()})
                rank, rank_kind = next(
                    (
                        (value, kind)
                        for kind, value in (
                            ("global_rank", row.get("global_rank")),
                            ("global_rank", metric_labels.get("global_rank")),
                            ("rank", row.get("rank")),
                            ("rank", metric_labels.get("rank")),
                        )
                        if value is not None and value != ""
                    ),
                    (None, None),
                )
                key = (name, histogram, host, gpu, tuple(sorted(labels.items())))
                if key not in series_by_key:
                    info = _description(name, families[name]["endpoints"], histogram)
                    series_by_key[key] = {
                        "id": len(series_by_key),
                        "name": name,
                        "label": info["title"],
                        "unit": info["unit"],
                        "value_kind": info["value_kind"],
                        "raw_name": row["metric_name"],
                        "endpoint": endpoint_name,
                        "host": host,
                        "gpu": gpu,
                        "worker": worker,
                        "rank": rank,
                        "rank_kind": rank_kind,
                        "labels": labels,
                        "worker_process": row.get("worker_process") or extra.get("worker_process"),
                        "raw_host": raw_host,
                        "host_source": "raw label" if raw_host else "scraper configuration" if host else "unknown",
                        "host_evidence": config_source,
                        "points": [],
                        "source_ids": set(),
                    }
                    if histogram:
                        series_by_key[key]["observation_unit"] = info["observation_unit"]
                series = series_by_key[key]
                series["points"].extend(
                    [run.t(timestamp), value, sid, source_row]
                    for timestamp, value, source_row in zip(
                        row["timestamp_ns_list"], row["metric_value_list"], row["_row_list"], strict=True
                    )
                )
                series["source_ids"].add(sid)
    result = list(series_by_key.values())
    catalog: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            **_description(name, family["endpoints"], family["histogram"]),
            "series_count": 0,
            "samples": 0,
            "endpoints": sorted(family["endpoints"]),
        }
        for name, family in sorted(families.items())
    }
    for series in result:
        series["points"].sort()
        points, seen = [], set()
        for point in series["points"]:
            identity = tuple(point[:2])
            if identity not in seen:
                points.append(point)
                seen.add(identity)
            else:
                run.audit["duplicate_metric_points"] += 1
        series["points"] = points
        conflicts = []
        conflicting_samples = 0
        for timestamp, samples in groupby(points, key=lambda point: point[0]):
            count = sum(1 for _ in samples)
            if count > 1:
                conflicts.append(timestamp)
                conflicting_samples += count
        series["conflict_timestamps"] = conflicts
        series["conflicting_samples"] = conflicting_samples
        run.audit["conflicting_metric_samples"] += conflicting_samples
        series["source_ids"] = sorted(series["source_ids"])
        entry = catalog[series["name"]]
        entry["series_count"] += 1
        entry["samples"] += len(points)
        if conflicts:
            warning = (
                "Conflicting raw values share a source identity and timestamp; all observations remain in raw points."
            )
            if warning not in entry["quality"]:
                entry["quality"] = " ".join(part for part in (entry["quality"], warning) if part)
        if series["worker"] in run.workers:
            run.workers[series["worker"]]["metrics"].append(series["id"])
    run.metric_catalog = list(catalog.values())
    run.audit["metric_families"] = len(catalog)
    run.audit["metric_points"] = sum(len(series["points"]) for series in result)
    return result
