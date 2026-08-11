#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight GPU performance monitor.

Polls nvidia-smi at a fixed interval and writes:
  - per-second CSV samples   (--output-csv)
  - aggregate summary JSON   (--output-json, written on SIGINT/exit)

Usage:
    python3 perfmon.py --output-csv /logs/perf_samples_node1.csv \\
                       --output-json /logs/perf_summary_node1.json \\
                       --interval 1.0
"""

import argparse
import csv
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

_QUERY = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
_FIELDS = ["gpu", "util_pct", "mem_used_mb", "mem_total_mb", "power_w", "temp_c"]


def _sample() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(_FIELDS):
            rows.append(dict(zip(_FIELDS, parts)))
    return rows


def _summarize(samples: list[dict]) -> dict:
    by_gpu: dict[str, list[dict]] = {}
    for s in samples:
        by_gpu.setdefault(s["gpu"], []).append(s)

    summary = {}
    for gpu_idx, gpu_samples in by_gpu.items():

        def avg(field: str, _s: list[dict] = gpu_samples) -> float | None:
            vals = [float(s[field]) for s in _s if s.get(field, "").strip() not in ("", "[N/A]")]
            return round(sum(vals) / len(vals), 2) if vals else None

        summary[f"gpu_{gpu_idx}"] = {
            "samples": len(gpu_samples),
            "avg_util_pct": avg("util_pct"),
            "avg_mem_used_mb": avg("mem_used_mb"),
            "mem_total_mb": avg("mem_total_mb"),
            "avg_power_w": avg("power_w"),
            "avg_temp_c": avg("temp_c"),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    samples: list[dict] = []
    stop = False

    def handle_sigint(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    with Path(args.output_csv).open("w", newline="") as f:
        writer: csv.DictWriter | None = None
        while not stop:
            ts = datetime.now(timezone.utc).isoformat()
            for row in _sample():
                record = {"timestamp": ts, **row}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                    writer.writeheader()
                writer.writerow(record)
                samples.append(record)
            f.flush()
            time.sleep(args.interval)

    if samples:
        Path(args.output_json).write_text(json.dumps(_summarize(samples), indent=2))


if __name__ == "__main__":
    main()
