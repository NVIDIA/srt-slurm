# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Export node-level metrics of a benchmark run to CSV.

Uses ``NodeAnalyzer`` (logic consistent with ``log_parser``) to parse ``run_path`` and
prefill/decode ``*.err`` / ``*.out`` under ``run_path/logs/``, then writes a flat table.

Default output directory: ``<run_path>/logs/node_metrics/``, output filename: ``node_metrics.csv``.

Run from the srt-slurm repository root::

    PYTHONPATH=. python -m analysis.srtlog.export_node_metrics /path/to/run_dir
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .log_parser import NodeAnalyzer

logger = logging.getLogger(__name__)

DEFAULT_CSV_NAME = "node_metrics.csv"


def export_node_metrics(run_path: str, output_dir: str | None = None) -> str | None:
    """Parse node logs in the run directory and export them to CSV.

    Args:
        run_path: Run directory containing Slurm output logs (may contain ``logs/`` subdirectory)
        output_dir: Output directory; defaults to ``<run_path>/logs/node_metrics``

    Returns:
        Absolute path to the written CSV; returns ``None`` if there are no parsing results
    """
    run_path = os.path.abspath(run_path)
    if not os.path.isdir(run_path):
        logger.error("Run path is not a directory: %s", run_path)
        return None

    if output_dir is None:
        out = os.path.join(run_path, "logs", "node_metrics")
    else:
        out = os.path.abspath(output_dir)
    os.makedirs(out, exist_ok=True)

    analyzer = NodeAnalyzer()
    nodes = analyzer.parse_run_logs(run_path)
    if not nodes:
        logger.warning("No node metrics parsed from %s; CSV not written", run_path)
        return None

    df = analyzer._serialize_node_metrics(nodes)
    csv_path = os.path.join(out, DEFAULT_CSV_NAME)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df))
    return csv_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Export node metrics from run logs to CSV.")
    parser.add_argument(
        "run_path",
        help="Path to the run directory (contains or nests logs under logs/)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: <run_path>/logs/node_metrics)",
    )
    args = parser.parse_args(argv)

    path = export_node_metrics(args.run_path, output_dir=args.output_dir)
    if path is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
