#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot prefill/decode batch metrics over time from SGLang worker logs.

This CLI uses the same parser and 7x2 per-worker renderer as the live
batch-metrics snapshotter, so post-mortem plots match the in-flight
``batch_metrics.png`` view.

Usage:
    # Single run (always regenerates):
    python src/srtctl/benchmarks/scripts/plot_batch_metrics.py outputs/1042857-1p1d-tp4/logs

    # All runs under outputs/ (incremental, skip existing):
    python src/srtctl/benchmarks/scripts/plot_batch_metrics.py --all

    # All runs under outputs/ (force regenerate):
    python src/srtctl/benchmarks/scripts/plot_batch_metrics.py --all --force

    # All runs under a custom outputs dir:
    python src/srtctl/benchmarks/scripts/plot_batch_metrics.py --all --outputs-dir /path/to/outputs

    # With downsample / smoothing:
    python src/srtctl/benchmarks/scripts/plot_batch_metrics.py --all --downsample 10 --smooth 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from srtctl.analysis.batch_log_parser import LogState
from srtctl.analysis.batch_plot_matrix import default_batch_plot_title, render_batch_plot_matrix


def process_single_run(
    log_dir: str,
    downsample_factor: int = 1,
    output_path: str | None = None,
    smooth_input_window: int = 8,
) -> bool:
    """Process a single logs directory. Returns ``True`` when a plot was generated."""
    log_path = Path(log_dir)
    state = LogState(log_dir=log_path)
    state.refresh()
    if not state.has_data:
        return False

    output = Path(output_path) if output_path else log_path / "batch_metrics.png"
    return render_batch_plot_matrix(
        state=state,
        output_path=output,
        title=default_batch_plot_title(log_path),
        downsample=downsample_factor,
        smooth_input_window=smooth_input_window,
    )


def discover_run_dirs(outputs_dir: str) -> list[str]:
    """Find all run directories that have a logs/ subdirectory."""
    run_dirs = []
    if not os.path.isdir(outputs_dir):
        return run_dirs

    for entry in sorted(os.listdir(outputs_dir)):
        logs_dir = os.path.join(outputs_dir, entry, "logs")
        if os.path.isdir(logs_dir):
            run_dirs.append(logs_dir)
    return run_dirs


def main() -> None:
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
    parser.add_argument(
        "--smooth",
        type=int,
        default=8,
        help="Centered moving-average window for derived prefill input throughput (default: 8; 1 disables)",
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
        ok = process_single_run(log_dir, args.downsample, output_path, args.smooth)
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
            ok = process_single_run(log_dir, args.downsample, output_path, args.smooth)
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
