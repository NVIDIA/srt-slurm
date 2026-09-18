# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the selected gauges from one raw Tachometer capture, preserving labels."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tomli as tomllib
from pyarrow import ipc

if TYPE_CHECKING:
    from .importer import Importer

METRICS = {
    "gpu_util": ("GPU utilization", "%"),
    "DCGM_FI_DEV_GPU_UTIL": ("GPU utilization", "%"),
    "FI_DEV_FB_USED": ("GPU memory used", "MiB"),
    "trtllm_num_requests_running": ("Running requests", "requests"),
    "trtllm_num_requests_waiting": ("Waiting requests", "requests"),
    "trtllm_kv_cache_utilization": ("KV cache utilization", "ratio"),
    "dynamo_component_inflight_requests": ("Worker in flight", "requests"),
    "dynamo_frontend_inflight_requests": ("Frontend in flight", "requests"),
    "dynamo_frontend_queued_requests": ("Frontend queued", "requests"),
    "dynamo_frontend_router_queue_pending_requests": ("Router pending", "requests"),
    "dynamo_work_handler_queue_depth": ("Handler queue", "requests"),
    "load1": ("Host load (1 min)", "load"),
    "memory_MemAvailable_bytes": ("Host available memory", "bytes"),
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
    regex = "^(?:" + "|".join(re.escape(x) for x in METRICS) + ")(?:\\{|$)"
    for path in files:
        sid = run.source(path, "tachometer_raw")
        offset = 0
        for batch in batches(path):
            required = {"timestamp_ns", "metric_name", "metric_value", "scraper_endpoint"}
            if not required.issubset(batch.schema.names):
                raise ValueError(f"{path}: raw metrics require {sorted(required)}; UTC alignment cannot be inferred")
            run.audit["metric_rows_scanned"] += batch.num_rows
            selected_names = pc.call_function(
                "match_substring_regex", [batch["metric_name"]], options=pc.MatchSubstringOptions(regex)
            )
            mask = pc.call_function(
                "and",
                [
                    selected_names,
                    pc.call_function(
                        "and",
                        [
                            pc.call_function("greater_equal", [batch["timestamp_ns"], run.origin]),
                            pc.call_function(
                                "less_equal", [batch["timestamp_ns"], run.origin + int(run.duration * 1e9)]
                            ),
                        ],
                    ),
                ],
            )
            table = pa.Table.from_batches([batch]).append_column(
                "_row", pa.array(range(offset, offset + batch.num_rows))
            )
            offset += batch.num_rows
            for row in table.filter(mask).to_pylist():
                name, metric_labels = parse_metric_name(row["metric_name"])
                value = row["metric_value"]
                if value is None or not math.isfinite(value):
                    run.audit["nonfinite_metric_points"] += 1
                    continue
                endpoint = endpoints.get(row["scraper_endpoint"], {})
                raw_host = row.get("hostname")
                host = raw_host or endpoint.get("node_metadata", {}).get("hostname", "")
                gpu = str(row.get("gpu")) if row.get("gpu") is not None else ""
                extra = endpoint.get("gpu_metadata", {}).get(gpu, {})
                role = row.get("worker_role") or extra.get("worker_role", "")
                index = row.get("worker_index")
                index = extra.get("worker_index", "") if index in (None, "") else index
                worker = f"{role}-{index}" if role in ("prefill", "decode", "aggregated") and index != "" else None
                if worker is None and "frontend" in row["scraper_endpoint"]:
                    worker = "frontend"
                # All non-value columns are part of the identity, including rank,
                # process and labels not known to this version of the importer.
                labels = {
                    k: str(v)
                    for k, v in row.items()
                    if k
                    not in {
                        "_row",
                        "timestamp_ns",
                        "time_since_start",
                        "metric_value",
                        "metric_name",
                        "histogram_sum",
                        "histogram_count",
                        "histogram_bucket_lower",
                        "histogram_bucket_upper",
                    }
                    and v not in (None, "")
                }
                labels.update({f"metric.{k}": v for k, v in metric_labels.items()})
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
                key = (row["metric_name"], host, gpu, tuple(sorted(labels.items())))
                if key not in series_by_key:
                    series_by_key[key] = {
                        "id": len(series_by_key),
                        "name": name,
                        "label": METRICS[name][0],
                        "unit": METRICS[name][1],
                        "raw_name": row["metric_name"],
                        "endpoint": row["scraper_endpoint"],
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
                series = series_by_key[key]
                series["points"].append([run.t(row["timestamp_ns"]), value, sid, row["_row"]])
                series["source_ids"].add(sid)
    result = list(series_by_key.values())
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
        series["source_ids"] = sorted(series["source_ids"])
        if series["worker"] in run.workers:
            run.workers[series["worker"]]["metrics"].append(series["id"])
    run.audit["metric_points"] = sum(len(s["points"]) for s in result)
    return result
