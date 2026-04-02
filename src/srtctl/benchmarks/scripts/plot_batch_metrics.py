#!/usr/bin/env python3
"""
Plot prefill/decode batch metrics over time from SGLang worker logs.

Usage:
    # Single run (always regenerates):
    python scripts/plot_batch_metrics.py outputs/1042857-1p1d-tp4/logs

    # All runs under outputs/ (incremental, skip existing):
    python scripts/plot_batch_metrics.py --all

    # All runs under outputs/ (force regenerate):
    python scripts/plot_batch_metrics.py --all --force

    # All runs under a custom outputs dir:
    python scripts/plot_batch_metrics.py --all --outputs-dir /path/to/outputs

    # With downsample:
    python scripts/plot_batch_metrics.py --all --downsample 10
"""

import argparse
import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def parse_timestamp(line: str) -> datetime | None:
    # ANSI-escaped ISO timestamp: \x1b[2m2026-02-26T08:42:59.339496Z\x1b[0m
    match = re.search(r"\[2m(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}\.\d+)", line)
    if match:
        ts_str = f"{match.group(1)} {match.group(2)}"
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return None

    # Plain bracket timestamp: [2025-11-04 05:31:43 DP0 TP0 EP0]
    match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def extract_metric(line: str, pattern: str) -> float | None:
    match = re.search(pattern, line)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


PREFILL_METRICS = {
    "#new-seq": r"#new-seq:\s*([\d.]+)",
    "#new-token": r"#new-token:\s*([\d.]+)",
    "#cached-token": r"#cached-token:\s*([\d.]+)",
    "full token usage": r"full token usage:\s*([\d.]+)",
    "mamba usage": r"mamba usage:\s*([\d.]+)",
    "#running-req": r"#running-req:\s*([\d.]+)",
    "#queue-req": r"#queue-req:\s*([\d.]+)",
    "#prealloc-req": r"#prealloc-req:\s*([\d.]+)",
    "#inflight-req": r"#inflight-req:\s*([\d.]+)",
    "input throughput (token/s)": r"input throughput \(token/s\):\s*([\d.]+)",
}

DECODE_METRICS = {
    "#running-req": r"#running-req:\s*([\d.]+)",
    "#full token": r"#full token:\s*([\d.]+)",
    "full token usage": r"full token usage:\s*([\d.]+)",
    "mamba num": r"mamba num:\s*([\d.]+)",
    "mamba usage": r"mamba usage:\s*([\d.]+)",
    "pre-allocated usage": r"pre-allocated usage:\s*([\d.]+)",
    "#prealloc-req": r"#prealloc-req:\s*([\d.]+)",
    "#transfer-req": r"#transfer-req:\s*([\d.]+)",
    "#retracted-req": r"#retracted-req:\s*([\d.]+)",
    "gen throughput (token/s)": r"gen throughput \(token/s\):\s*([\d.]+)",
    "#queue-req": r"#queue-req:\s*([\d.]+)",
}


def parse_log_file(filepath: str, batch_keyword: str, metrics_def: dict) -> dict:
    """Parse a log file and extract time-series for each metric."""
    data = {"timestamps": []}
    for name in metrics_def:
        data[name] = []

    with open(filepath) as f:
        for line in f:
            if batch_keyword not in line:
                continue

            ts = parse_timestamp(line)
            if ts is None:
                continue

            values = {}
            for name, pattern in metrics_def.items():
                values[name] = extract_metric(line, pattern)

            if all(v is None for v in values.values()):
                continue

            data["timestamps"].append(ts)
            for name in metrics_def:
                data[name].append(values[name])

    return data


def find_log_files(log_dir: str) -> tuple[list[str], list[str]]:
    """Find prefill and decode log files in the directory."""
    prefill_files = []
    decode_files = []
    agg_files = []

    for f in sorted(os.listdir(log_dir)):
        if not (f.endswith(".out") or f.endswith(".err")):
            continue
        path = os.path.join(log_dir, f)
        if "prefill" in f:
            prefill_files.append(path)
        elif "decode" in f:
            decode_files.append(path)
        elif "_agg_" in f:
            agg_files.append(path)

    for path in agg_files:
        prefill_files.append(path)
        decode_files.append(path)

    return prefill_files, decode_files


def downsample(data: dict, factor: int) -> dict:
    if factor <= 1:
        return data
    return {key: values[::factor] for key, values in data.items()}


def compute_elapsed_seconds(timestamps: list[datetime]) -> np.ndarray:
    if not timestamps:
        return np.array([])
    start = timestamps[0]
    return np.array([(t - start).total_seconds() for t in timestamps])


def plot_metrics(
    prefill_data_list: list[tuple[str, dict]],
    decode_data_list: list[tuple[str, dict]],
    output_path: str,
    title_prefix: str = "",
):
    """Create side-by-side plots for prefill (left) and decode (right) metrics."""
    prefill_metric_names = list(PREFILL_METRICS.keys())
    decode_metric_names = list(DECODE_METRICS.keys())
    n_rows = max(len(prefill_metric_names), len(decode_metric_names))

    fig, axes = plt.subplots(n_rows, 2, figsize=(22, 3.5 * n_rows), squeeze=False)
    fig.suptitle(
        f"{title_prefix}Batch Metrics Over Time" if title_prefix else "Batch Metrics Over Time",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )

    colors = plt.cm.tab10.colors

    for row, metric_name in enumerate(prefill_metric_names):
        ax = axes[row, 0]
        has_data = False
        for idx, (label, data) in enumerate(prefill_data_list):
            timestamps = data["timestamps"]
            values = data.get(metric_name, [])
            if not timestamps or not values:
                continue
            elapsed = compute_elapsed_seconds(timestamps)
            valid = [(e, v) for e, v in zip(elapsed, values, strict=False) if v is not None]
            if not valid:
                continue
            has_data = True
            es, vs = zip(*valid, strict=False)
            ax.plot(es, vs, color=colors[idx % len(colors)], linewidth=0.8, alpha=0.85, label=label)

        ax.set_title(f"Prefill: {metric_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Elapsed Time (s)", fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if has_data and len(prefill_data_list) > 1:
            ax.legend(fontsize=7, loc="upper right")

    for row in range(len(prefill_metric_names), n_rows):
        axes[row, 0].set_visible(False)

    for row, metric_name in enumerate(decode_metric_names):
        ax = axes[row, 1]
        has_data = False
        for idx, (label, data) in enumerate(decode_data_list):
            timestamps = data["timestamps"]
            values = data.get(metric_name, [])
            if not timestamps or not values:
                continue
            elapsed = compute_elapsed_seconds(timestamps)
            valid = [(e, v) for e, v in zip(elapsed, values, strict=False) if v is not None]
            if not valid:
                continue
            has_data = True
            es, vs = zip(*valid, strict=False)
            ax.plot(es, vs, color=colors[idx % len(colors)], linewidth=0.8, alpha=0.85, label=label)

        ax.set_title(f"Decode: {metric_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Elapsed Time (s)", fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if has_data and len(decode_data_list) > 1:
            ax.legend(fontsize=7, loc="upper right")

    for row in range(len(decode_metric_names), n_rows):
        axes[row, 1].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_run_title(log_dir: str) -> str:
    parts = os.path.normpath(log_dir).split(os.sep)
    for p in reversed(parts):
        if p != "logs":
            return p + " - "
    return ""


def process_single_run(log_dir: str, downsample_factor: int = 1, output_path: str | None = None) -> bool:
    """Process a single run directory. Returns True if plot was generated."""
    output_path = output_path or os.path.join(log_dir, "batch_metrics.png")

    prefill_files, decode_files = find_log_files(log_dir)
    if not prefill_files and not decode_files:
        return False

    prefill_data_list = []
    for fp in prefill_files:
        data = parse_log_file(fp, "Prefill batch", PREFILL_METRICS)
        if data["timestamps"]:
            data = downsample(data, downsample_factor)
            label = os.path.basename(fp).replace(".out", "").replace(".err", "")
            prefill_data_list.append((label, data))

    decode_data_list = []
    for fp in decode_files:
        data = parse_log_file(fp, "Decode batch", DECODE_METRICS)
        if data["timestamps"]:
            data = downsample(data, downsample_factor)
            label = os.path.basename(fp).replace(".out", "").replace(".err", "")
            decode_data_list.append((label, data))

    if not prefill_data_list and not decode_data_list:
        return False

    title = get_run_title(log_dir)
    plot_metrics(prefill_data_list, decode_data_list, output_path, title_prefix=title)
    return True


def discover_run_dirs(outputs_dir: str) -> list[str]:
    """Find all run directories that have a logs/ subdirectory with worker log files."""
    run_dirs = []
    if not os.path.isdir(outputs_dir):
        return run_dirs

    for entry in sorted(os.listdir(outputs_dir)):
        logs_dir = os.path.join(outputs_dir, entry, "logs")
        if os.path.isdir(logs_dir):
            run_dirs.append(logs_dir)
    return run_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Plot prefill/decode batch metrics from SGLang worker logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s outputs/1042857-1p1d-tp4/logs          # single run (force)
  %(prog)s --all                                   # all runs, incremental
  %(prog)s --all --force                           # all runs, force regenerate
  %(prog)s --all --outputs-dir /path/to/outputs    # custom outputs dir
""",
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=None,
        help="Path to a single logs directory (always force-regenerates)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Process all run directories under outputs/",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regenerate even if batch_metrics.png already exists (only with --all)",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Path to the outputs directory (default: outputs/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output image path (only for single-run mode)",
    )
    parser.add_argument(
        "--downsample",
        "-d",
        type=int,
        default=1,
        help="Downsample factor: keep every Nth data point (useful for large logs)",
    )
    args = parser.parse_args()

    if not args.log_dir and not args.all:
        parser.print_help()
        print("\nError: Please specify a log directory, or use --all to process all outputs", file=sys.stderr)
        sys.exit(1)

    # --- Single run mode ---
    if args.log_dir:
        log_dir = args.log_dir
        if not os.path.isdir(log_dir):
            print(f"Error: Directory does not exist: {log_dir}", file=sys.stderr)
            sys.exit(1)

        output_path = args.output or os.path.join(log_dir, "batch_metrics.png")
        prefill_files, decode_files = find_log_files(log_dir)

        if not prefill_files and not decode_files:
            print(f"Error: No prefill/decode log files found in {log_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(prefill_files)} prefill log(s), {len(decode_files)} decode log(s)")
        ok = process_single_run(log_dir, args.downsample, output_path)
        if ok:
            print(f"Plot saved to: {output_path}")
        else:
            print("Error: No valid batch data parsed", file=sys.stderr)
            sys.exit(1)
        return

    # --- Batch mode (--all) ---
    outputs_dir = args.outputs_dir
    if not os.path.isdir(outputs_dir):
        print(f"Error: Outputs directory does not exist: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    run_dirs = discover_run_dirs(outputs_dir)
    if not run_dirs:
        print(f"Error: No run directories with logs/ found in {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    total = len(run_dirs)
    skipped = 0
    generated = 0
    failed = 0

    print(f"Found {total} run directories (force={args.force})")
    print("=" * 60)

    for i, log_dir in enumerate(run_dirs, 1):
        run_name = os.path.basename(os.path.dirname(log_dir))
        output_path = os.path.join(log_dir, "batch_metrics.png")

        if not args.force and os.path.exists(output_path):
            skipped += 1
            continue

        status_prefix = f"[{i}/{total}]"
        print(f"{status_prefix} {run_name} ...", end=" ", flush=True)

        try:
            ok = process_single_run(log_dir, args.downsample, output_path)
            if ok:
                generated += 1
                print("OK")
            else:
                failed += 1
                print("No log data")
        except Exception as e:
            failed += 1
            print(f"Error: {e}")

    print("=" * 60)
    print(f"Done: generated {generated}, skipped {skipped}, failed/no-data {failed}, total {total}")


if __name__ == "__main__":
    main()
