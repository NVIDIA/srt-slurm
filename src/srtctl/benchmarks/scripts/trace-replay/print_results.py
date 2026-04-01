#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print aiperf benchmark results in a formatted table."""

import json
import sys
from pathlib import Path


def get_metric(data: dict, key: str, stat: str = "avg") -> float | None:
    """Extract a metric value from aiperf JSON output."""
    if key not in data:
        return None
    metric = data[key]
    if isinstance(metric, dict):
        return metric.get(stat)
    return metric


def format_value(value: float | None, width: int = 10) -> str:
    """Format a numeric value for display."""
    if value is None:
        return "N/A".ljust(width)
    if abs(value) >= 1000:
        return f"{value:.2f}".ljust(width)
    elif abs(value) >= 1:
        return f"{value:.2f}".ljust(width)
    else:
        return f"{value:.4f}".ljust(width)


def print_results(artifact_dir: str) -> int:
    """Print benchmark results from aiperf output directory."""
    artifact_path = Path(artifact_dir)

    # Find the profile export JSON file
    json_files = list(artifact_path.glob("profile_export_aiperf.json"))
    if not json_files:
        json_files = list(artifact_path.glob("profile_export*.json"))
        # Filter out raw files
        json_files = [f for f in json_files if "_raw" not in f.name]

    if not json_files:
        print(f"Warning: Could not find results JSON in {artifact_dir}", file=sys.stderr)
        return 1

    json_file = json_files[0]

    try:
        with open(json_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading {json_file}: {e}", file=sys.stderr)
        return 1

    # Extract metrics
    request_count = get_metric(data, "request_count")
    benchmark_duration = get_metric(data, "benchmark_duration")
    total_isl = get_metric(data, "total_isl")
    total_osl = get_metric(data, "total_osl")
    request_throughput = get_metric(data, "request_throughput")
    output_token_throughput = get_metric(data, "output_token_throughput")
    total_token_throughput = get_metric(data, "total_token_throughput")

    # TTFT metrics
    ttft_mean = get_metric(data, "time_to_first_token", "avg")
    ttft_median = get_metric(data, "time_to_first_token", "p50")
    ttft_p99 = get_metric(data, "time_to_first_token", "p99")

    # ITL metrics (inter-token latency, equivalent to TPOT)
    itl_mean = get_metric(data, "inter_token_latency", "avg")
    itl_median = get_metric(data, "inter_token_latency", "p50")
    itl_p99 = get_metric(data, "inter_token_latency", "p99")

    # E2EL metrics (request latency)
    e2el_mean = get_metric(data, "request_latency", "avg")
    e2el_median = get_metric(data, "request_latency", "p50")
    e2el_p99 = get_metric(data, "request_latency", "p99")

    # Print formatted output
    print("============ Serving Benchmark Result ============")
    print(f"Successful requests:                     {format_value(request_count)}")
    print(f"Benchmark duration (s):                  {format_value(benchmark_duration)}")
    print(f"Total input tokens:                      {format_value(total_isl)}")
    print(f"Total generated tokens:                  {format_value(total_osl)}")
    print(f"Request throughput (req/s):              {format_value(request_throughput)}")
    print(f"Output token throughput (tok/s):         {format_value(output_token_throughput)}")
    print(f"Total Token throughput (tok/s):          {format_value(total_token_throughput)}")
    print("---------------Time to First Token----------------")
    print(f"Mean TTFT (ms):                          {format_value(ttft_mean)}")
    print(f"Median TTFT (ms):                        {format_value(ttft_median)}")
    print(f"P99 TTFT (ms):                           {format_value(ttft_p99)}")
    print("-----Time per Output Token (excl. 1st token)------")
    print(f"Mean TPOT (ms):                          {format_value(itl_mean)}")
    print(f"Median TPOT (ms):                        {format_value(itl_median)}")
    print(f"P99 TPOT (ms):                           {format_value(itl_p99)}")
    print("---------------Inter-token Latency----------------")
    print(f"Mean ITL (ms):                           {format_value(itl_mean)}")
    print(f"Median ITL (ms):                         {format_value(itl_median)}")
    print(f"P99 ITL (ms):                            {format_value(itl_p99)}")
    print("----------------End-to-end Latency----------------")
    print(f"Mean E2EL (ms):                          {format_value(e2el_mean)}")
    print(f"Median E2EL (ms):                        {format_value(e2el_median)}")
    print(f"P99 E2EL (ms):                           {format_value(e2el_p99)}")
    print("==================================================")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <artifact_dir>", file=sys.stderr)
        sys.exit(1)

    sys.exit(print_results(sys.argv[1]))
