#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print a formatted benchmark summary from an aiperf artifact directory.

Usage: format_results.py <artifact_dir>

Reads <artifact_dir>/profile_export_aiperf.json and prints a vLLM-style summary.
Exits 0 always — missing or corrupt files produce a warning line instead.

aiperf JSON structure: every metric is a stats object {"unit": ..., "avg": ..., "p50": ..., ...}.
Use dot-notation in _FIELD_MAP to reach subkeys (e.g. "time_to_first_token.avg").
Supports arbitrary depth (e.g. "input_config.loadgen.concurrency").
JSON null values display as "inf" (aiperf uses null to mean "unlimited").
"""

import json
import sys
from pathlib import Path

_LABEL_WIDTH = 40

# Sentinel returned by _lookup when no candidate key path resolves to a value.
# Distinct from None, which means the key exists with a JSON null value (→ "inf").
_MISSING: object = object()

# Maps internal key → ordered list of JSON key paths (first resolved path wins).
# All aiperf metrics live inside stats objects — use ".avg" for averages and
# ".p50"/".p99" for percentiles.
_FIELD_MAP: dict[str, list[str]] = {
    "request_rate":        ["input_config.loadgen.request_rate"],
    "burstiness_factor":   ["burstiness_factor"],
    "max_concurrency":     ["input_config.loadgen.concurrency"],
    "successful_requests": ["request_count.avg"],
    "duration":            ["benchmark_duration.avg"],
    "total_input_tokens":  ["total_isl.avg"],
    "total_output_tokens": ["total_output_tokens.avg", "total_osl.avg"],
    "request_throughput":  ["request_throughput.avg"],
    "output_throughput":   ["output_token_throughput.avg"],
    "total_throughput":    ["total_token_throughput.avg"],
    "ttft_mean":           ["time_to_first_token.avg"],
    "ttft_median":         ["time_to_first_token.p50"],
    "ttft_p99":            ["time_to_first_token.p99"],
    "tpot_mean":           ["inter_token_latency.avg"],
    "tpot_median":         ["inter_token_latency.p50"],
    "tpot_p99":            ["inter_token_latency.p99"],
    "itl_mean":            ["inter_token_latency.avg"],
    "itl_median":          ["inter_token_latency.p50"],
    "itl_p99":             ["inter_token_latency.p99"],
    "e2el_mean":           ["request_latency.avg"],
    "e2el_median":         ["request_latency.p50"],
    "e2el_p99":            ["request_latency.p99"],
}

# Fields that should display as integers (no decimal point).
_INTEGER_FIELDS = {
    "successful_requests",
    "total_input_tokens",
    "total_output_tokens",
    "max_concurrency",
}


def _lookup(data: dict[str, object], keys: list[str]) -> object:
    """Return the first matching value, supporting arbitrary-depth dot-notation.

    Returns _MISSING if no candidate path resolves to a non-dict/list value.
    Returns None if the path resolves to a JSON null value (caller displays as "inf").
    Dict and list values are skipped so the next candidate is tried.
    """
    for key in keys:
        node: object = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                node = _MISSING
                break
            node = node[part]  # type: ignore[index]
        if node is _MISSING or isinstance(node, (dict, list)):
            continue
        return node  # scalar or None (JSON null)
    return _MISSING


def _fmt(val: object, *, integer: bool = False) -> str:
    """Format a single value for display."""
    if val is _MISSING:
        return "N/A"
    if val is None:
        return "inf"  # JSON null means unlimited/no-rate-limit in aiperf
    if isinstance(val, str):
        return val if val.lower() not in ("inf", "infinity") else "inf"
    if isinstance(val, float) and val >= 1e18:
        return "inf"
    if integer:
        return str(int(val))  # type: ignore[arg-type]
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _row(label: str, value: str) -> str:
    """Format one data row: label left-aligned in 40-char column, then space, then value."""
    return f"{label + ':':<{_LABEL_WIDTH}} {value}"


def format_results(artifact_dir: str) -> str:
    """Parse profile_export_aiperf.json in artifact_dir and return formatted summary."""
    path = Path(artifact_dir) / "profile_export_aiperf.json"

    if not path.exists():
        return f"[format_results] Warning: profile_export_aiperf.json not found in {artifact_dir}"

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "[format_results] Warning: could not parse profile_export_aiperf.json"

    if not isinstance(data, dict):
        return "[format_results] Warning: profile_export_aiperf.json is not a JSON object"

    def get(field: str) -> object:
        return _lookup(data, _FIELD_MAP[field])

    def fmt(field: str) -> str:
        return _fmt(get(field), integer=field in _INTEGER_FIELDS)

    burstiness_val = get("burstiness_factor")
    burstiness_str = _fmt(burstiness_val)
    if burstiness_val == 1.0:
        burstiness_str += " (Poisson process)"

    lines = [
        f"Traffic request rate: {_fmt(get('request_rate'))}",
        f"Burstiness factor: {burstiness_str}",
        f"Maximum request concurrency: {fmt('max_concurrency')}",
        "============ Serving Benchmark Result ============",
        _row("Successful requests", fmt("successful_requests")),
        _row("Benchmark duration (s)", fmt("duration")),
        _row("Total input tokens", fmt("total_input_tokens")),
        _row("Total generated tokens", fmt("total_output_tokens")),
        _row("Request throughput (req/s)", fmt("request_throughput")),
        _row("Output token throughput (tok/s)", fmt("output_throughput")),
        _row("Total Token throughput (tok/s)", fmt("total_throughput")),
        "---------------Time to First Token----------------",
        _row("Mean TTFT (ms)", fmt("ttft_mean")),
        _row("Median TTFT (ms)", fmt("ttft_median")),
        _row("P99 TTFT (ms)", fmt("ttft_p99")),
        "-----Time per Output Token (excl. 1st token)------",
        _row("Mean TPOT (ms)", fmt("tpot_mean")),
        _row("Median TPOT (ms)", fmt("tpot_median")),
        _row("P99 TPOT (ms)", fmt("tpot_p99")),
        "---------------Inter-token Latency----------------",
        _row("Mean ITL (ms)", fmt("itl_mean")),
        _row("Median ITL (ms)", fmt("itl_median")),
        _row("P99 ITL (ms)", fmt("itl_p99")),
        "----------------End-to-end Latency----------------",
        _row("Mean E2EL (ms)", fmt("e2el_mean")),
        _row("Median E2EL (ms)", fmt("e2el_median")),
        _row("P99 E2EL (ms)", fmt("e2el_p99")),
        "==================================================",
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("[format_results] Usage: format_results.py <artifact_dir>", file=sys.stderr)
        sys.exit(1)
    print(format_results(sys.argv[1]))


if __name__ == "__main__":
    main()
