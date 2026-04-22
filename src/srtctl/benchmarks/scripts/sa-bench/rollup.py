#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark-rollup.json and benchmark-rollup.csv from sa-bench results."""

from __future__ import annotations

import csv
import json
from collections import Counter
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "Config",
    "Total GPU Count",
    "Decode GPU Count",
    "Concurrency",
    "Total Token Throughput",
    "Output Token Throughput",
    "Median TTFT",
    "Median TPOT",
    "Median ITL",
    "P90 Decode Running Requests",
    "Output Token Throughput per User",
    "Total Token Throughput per GPU",
]

RUNNING_REQ_PATTERN = re.compile(r"#running-req:\s*(\d+)")


def _get_percentile(percentiles: list, target: float) -> float | None:
    """Extract a specific percentile value from the percentiles list."""
    if not percentiles:
        return None
    for p, v in percentiles:
        if p == target:
            return v
    return None


def _read_job_metadata(log_dir: Path) -> dict[str, Any] | None:
    """Read submit metadata JSON from the output directory when available."""
    output_dir = log_dir.parent
    for metadata_path in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(metadata_path.read_text())
        except Exception as exc:
            print(f"Failed to parse {metadata_path}: {exc}", file=sys.stderr)
            continue
        if data:
            return data
    return None


def _as_int(value: Any) -> int:
    """Coerce to int, treating None / non-numeric as 0.

    Submit metadata serializes optional ResourceConfig fields as JSON null, so
    ``dict.get(key, 0)`` returns ``None`` (not 0) when the key is present but
    unset. ``int(None)`` raises, so every field read here must go through this.
    """
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compute_gpu_counts(resources: dict[str, Any]) -> tuple[int | None, int | None]:
    """Compute total and decode-serving GPU counts from resource settings."""
    gpus_per_node = _as_int(resources.get("gpus_per_node"))
    prefill_nodes = _as_int(resources.get("prefill_nodes"))
    decode_nodes = _as_int(resources.get("decode_nodes"))
    agg_nodes = _as_int(resources.get("agg_nodes"))
    if gpus_per_node <= 0:
        return None, None

    if prefill_nodes > 0 or decode_nodes > 0:
        total_gpu_count = (prefill_nodes + decode_nodes) * gpus_per_node
    elif agg_nodes > 0:
        total_gpu_count = agg_nodes * gpus_per_node
    else:
        total_gpu_count = gpus_per_node

    if decode_nodes > 0:
        return total_gpu_count, decode_nodes * gpus_per_node

    decode_workers = _as_int(resources.get("decode_workers"))
    gpus_per_decode = _as_int(resources.get("gpus_per_decode"))
    if decode_workers > 0 and gpus_per_decode > 0:
        return total_gpu_count, decode_workers * gpus_per_decode

    # Aggregated deployments: all provisioned GPUs serve both prefill and decode.
    agg_workers = _as_int(resources.get("agg_workers"))
    gpus_per_agg = _as_int(resources.get("gpus_per_agg"))
    if agg_workers > 0 and gpus_per_agg > 0:
        return total_gpu_count, agg_workers * gpus_per_agg
    if agg_nodes > 0:
        return total_gpu_count, total_gpu_count

    return total_gpu_count, None


def _extract_p90_decode_running_requests(log_dir: Path, metadata: dict[str, Any] | None) -> int | None:
    """Stream decode logs and compute the nearest-rank P90 of #running-req values."""
    if not metadata or metadata.get("backend_type") != "sglang":
        return None

    resources = metadata.get("resources")
    if resources is None:
        return None
    if not (_as_int(resources.get("prefill_nodes")) > 0 and _as_int(resources.get("decode_nodes")) > 0):
        return None
    if _as_int(resources.get("agg_workers")) > 0:
        return None

    counts: Counter[int] = Counter()
    total = 0

    for decode_log in sorted(log_dir.glob("*decode*.out")):
        try:
            with decode_log.open("r", errors="replace") as f:
                for line in f:
                    match = RUNNING_REQ_PATTERN.search(line)
                    if not match:
                        continue
                    value = int(match.group(1))
                    counts[value] += 1
                    total += 1
        except OSError as exc:
            print(f"Failed to read {decode_log}: {exc}", file=sys.stderr)

    if total == 0:
        return None

    rank = math.ceil(total * 0.9)
    cumulative = 0
    for value in sorted(counts):
        cumulative += counts[value]
        if cumulative >= rank:
            return value

    return None


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """Return numerator / denominator when both values are valid and denominator != 0."""
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _format_csv_value(value: object) -> str:
    """Format CSV values with at most three decimal places for numeric fields."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _build_csv_row(
    data: dict[str, object],
    config_name: str,
    gpu_num: int | None,
    decode_gpu_count: int | None,
    p90_decode_running_requests: int | None,
) -> dict[str, object]:
    """Build one CSV row from a parsed sa-bench result."""
    total_token_throughput = data.get("total_token_throughput")
    median_tpot = data.get("median_tpot_ms")
    row = {
        "Config": config_name,
        "Total GPU Count": gpu_num,
        "Decode GPU Count": decode_gpu_count,
        "Concurrency": data.get("max_concurrency"),
        "Total Token Throughput": total_token_throughput,
        "Output Token Throughput": data.get("output_throughput"),
        "Median TTFT": data.get("median_ttft_ms"),
        "Median TPOT": median_tpot,
        "Median ITL": data.get("median_itl_ms"),
        "P90 Decode Running Requests": p90_decode_running_requests,
        "Output Token Throughput per User": _safe_ratio(1000.0, median_tpot),
        "Total Token Throughput per GPU": _safe_ratio(total_token_throughput, gpu_num),
    }
    return {key: _format_csv_value(value) for key, value in row.items()}


def main(log_dir: Path) -> None:
    """Generate benchmark-rollup.json and benchmark-rollup.csv from sa-bench result files."""
    result_files = sorted(log_dir.glob("sa-bench_*/results_*.json"))
    if not result_files:
        print("No sa-bench results found", file=sys.stderr)
        return

    runs = []
    csv_rows = []
    config = {}
    metadata = _read_job_metadata(log_dir)
    config_name = metadata.get("job_name") if metadata else None
    resources = metadata.get("resources") if metadata else None
    total_gpu_count, decode_gpu_count = _compute_gpu_counts(resources) if resources else (None, None)
    p90_decode_running_requests = _extract_p90_decode_running_requests(log_dir, metadata)

    for result_file in result_files:
        try:
            data = json.loads(result_file.read_text())
        except json.JSONDecodeError as exc:
            print(f"Failed to parse {result_file}: {exc}", file=sys.stderr)
            continue

        if not config:
            config = {
                "model": data.get("model_id"),
                "isl": data.get("random_input_len"),
                "osl": data.get("random_output_len"),
            }

        runs.append({
            "concurrency": data.get("max_concurrency"),
            "throughput_toks": data.get("output_throughput"),
            "request_throughput": data.get("request_throughput"),
            "ttft_mean_ms": data.get("mean_ttft_ms"),
            "ttft_p99_ms": _get_percentile(data.get("percentiles_ttft_ms", []), 99.0),
            "tpot_mean_ms": data.get("mean_tpot_ms"),
            "tpot_p99_ms": _get_percentile(data.get("percentiles_tpot_ms", []), 99.0),
            "itl_mean_ms": data.get("mean_itl_ms"),
            "itl_p99_ms": _get_percentile(data.get("percentiles_itl_ms", []), 99.0),
            "e2el_mean_ms": data.get("mean_e2el_ms"),
            "completed_requests": data.get("completed"),
            "total_input_tokens": data.get("total_input"),
            "total_output_tokens": data.get("total_output"),
        })

        csv_rows.append(
            _build_csv_row(
                data=data,
                config_name=config_name or str(data.get("model_id") or "unknown"),
                gpu_num=total_gpu_count,
                decode_gpu_count=decode_gpu_count,
                p90_decode_running_requests=p90_decode_running_requests,
            )
        )

    if not runs:
        print("No valid sa-bench results found", file=sys.stderr)
        return

    rollup = {
        "benchmark_type": "sa-bench",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": config,
        "runs": runs,
    }

    json_path = log_dir / "benchmark-rollup.json"
    json_path.write_text(json.dumps(rollup, indent=2))
    print(f"Wrote {json_path}")

    csv_rows.sort(key=lambda row: int(row["Concurrency"]) if row["Concurrency"] else -1)
    csv_path = log_dir / "benchmark-rollup.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/logs")
    main(log_dir)
