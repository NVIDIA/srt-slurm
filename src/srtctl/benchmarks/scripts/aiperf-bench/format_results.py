#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print a formatted benchmark summary from an aiperf artifact directory.

Usage: format_results.py <artifact_dir>

Reads <artifact_dir>/profile_export_aiperf.json and prints a vLLM-style summary.
Exits 0 always — missing or corrupt files produce a warning line instead.
"""

import json
import sys
from pathlib import Path

_LABEL_WIDTH = 40

# Maps internal key → ordered list of JSON key candidates (first match wins).
# Dot-notation (e.g. "time_to_first_token.mean") resolves one level of nesting.
_FIELD_MAP: dict[str, list[str]] = {
    "request_rate": ["request_rate", "traffic_request_rate"],
    "burstiness_factor": ["burstiness_factor", "burstiness"],
    "max_concurrency": ["max_concurrency", "concurrency", "maximum_request_concurrency"],
    "successful_requests": ["num_successful_requests", "successful_requests"],
    "duration": ["duration", "benchmark_duration", "benchmark_duration_s"],
    "total_input_tokens": ["total_input_tokens", "total_prompt_tokens"],
    "total_output_tokens": [
        "total_output_tokens",
        "total_generated_tokens",
        "total_completion_tokens",
    ],
    "request_throughput": ["request_throughput", "requests_per_second"],
    "output_throughput": ["output_token_throughput", "output_tokens_per_second"],
    "total_throughput": ["total_token_throughput", "tokens_per_second"],
    "ttft_mean": [
        "ttft_mean_ms",
        "mean_ttft_ms",
        "time_to_first_token.mean",
        "time_to_first_token_mean_ms",
    ],
    "ttft_median": [
        "ttft_median_ms",
        "median_ttft_ms",
        "time_to_first_token.median",
        "time_to_first_token_p50_ms",
    ],
    "ttft_p99": [
        "ttft_p99_ms",
        "p99_ttft_ms",
        "time_to_first_token.p99",
        "time_to_first_token_p99_ms",
    ],
    "tpot_mean": [
        "tpot_mean_ms",
        "mean_tpot_ms",
        "time_per_output_token.mean",
        "time_per_output_token_mean_ms",
    ],
    "tpot_median": [
        "tpot_median_ms",
        "median_tpot_ms",
        "time_per_output_token.median",
        "time_per_output_token_p50_ms",
    ],
    "tpot_p99": [
        "tpot_p99_ms",
        "p99_tpot_ms",
        "time_per_output_token.p99",
        "time_per_output_token_p99_ms",
    ],
    "itl_mean": [
        "itl_mean_ms",
        "mean_itl_ms",
        "inter_token_latency.mean",
        "inter_token_latency_mean_ms",
    ],
    "itl_median": [
        "itl_median_ms",
        "median_itl_ms",
        "inter_token_latency.median",
        "inter_token_latency_p50_ms",
    ],
    "itl_p99": [
        "itl_p99_ms",
        "p99_itl_ms",
        "inter_token_latency.p99",
        "inter_token_latency_p99_ms",
    ],
    "e2el_mean": [
        "e2el_mean_ms",
        "mean_e2el_ms",
        "end_to_end_latency.mean",
        "end_to_end_latency_mean_ms",
    ],
    "e2el_median": [
        "e2el_median_ms",
        "median_e2el_ms",
        "end_to_end_latency.median",
        "end_to_end_latency_p50_ms",
    ],
    "e2el_p99": [
        "e2el_p99_ms",
        "p99_e2el_ms",
        "end_to_end_latency.p99",
        "end_to_end_latency_p99_ms",
    ],
}

# Fields that should display as integers (no decimal point).
_INTEGER_FIELDS = {
    "successful_requests",
    "total_input_tokens",
    "total_output_tokens",
    "max_concurrency",
}


def _lookup(data: dict[str, object], keys: list[str]) -> float | int | str | None:
    """Return the first matching scalar value from data, supporting one level of dot-notation.

    Dict/list values are skipped so the next candidate key is tried — this handles JSON
    structures where a top-level key holds a stats object rather than a scalar.
    """
    for key in keys:
        if "." in key:
            head, tail = key.split(".", 1)
            nested = data.get(head)
            if isinstance(nested, dict) and tail in nested:
                val = nested[tail]
                if not isinstance(val, (dict, list)):
                    return val  # type: ignore[return-value]
        elif key in data:
            val = data[key]
            if not isinstance(val, (dict, list)):
                return val  # type: ignore[return-value]
    return None


def _fmt(val: float | int | str | None, *, integer: bool = False) -> str:
    """Format a single value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val if val.lower() not in ("inf", "infinity") else "inf"
    if isinstance(val, float) and val >= 1e18:
        return "inf"
    if integer:
        return str(int(val))
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

    def get(field: str):
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
