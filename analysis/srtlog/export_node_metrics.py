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

import pandas as pd

from .log_parser import NodeAnalyzer
from .models import NodeMetrics

logger = logging.getLogger(__name__)

GEN_THROUGHPUT_SUMMARY_NAME = "gen_throughput.csv"

# Columns for batch metrics only (prefill / decode union; unused cells empty in CSV).
BATCH_CSV_COLUMNS: list[str] = [
    "timestamp",
    "dp",
    "tp",
    "ep",
    "batch_type",
    "new_seq",
    "new_token",
    "cached_token",
    "token_usage",
    "running_req",
    "queue_req",
    "prealloc_req",
    "inflight_req",
    "input_throughput",
    "transfer_req",
    "preallocated_usage",
    "num_tokens",
    "gen_throughput",
]


def _node_batch_csv_filename(node: NodeMetrics) -> str:
    ni = node.node_info
    stem = f'{ni.get("node", "unknown")}_{ni.get("worker_type", "")}_{ni.get("worker_id", "")}'
    return f"{stem}.csv"


def node_batches_to_dataframe(node: NodeMetrics) -> pd.DataFrame:
    """One row per batch line; columns restricted to batch-relevant fields."""
    rows: list[dict] = []
    for batch in node.batches:
        rows.append(
            {
                "timestamp": batch.timestamp,
                "dp": batch.dp,
                "tp": batch.tp,
                "ep": batch.ep,
                "batch_type": batch.batch_type,
                "new_seq": batch.new_seq,
                "new_token": batch.new_token,
                "cached_token": batch.cached_token,
                "token_usage": batch.token_usage,
                "running_req": batch.running_req,
                "queue_req": batch.queue_req,
                "prealloc_req": batch.prealloc_req,
                "inflight_req": batch.inflight_req,
                "input_throughput": batch.input_throughput,
                "transfer_req": batch.transfer_req,
                "preallocated_usage": batch.preallocated_usage,
                "num_tokens": batch.num_tokens,
                "gen_throughput": batch.gen_throughput,
            }
        )

    if not rows:
        return pd.DataFrame(columns=BATCH_CSV_COLUMNS)
    return pd.DataFrame(rows, columns=BATCH_CSV_COLUMNS)


def _gen_throughput_summary_dataframe(nodes: list[NodeMetrics]) -> pd.DataFrame:
    """Per ``running_req``: count / mean / median of ``gen_throughput`` (all nodes, valid pairs only)."""
    pairs: list[tuple[int, float]] = []
    for node in nodes:
        for batch in node.batches:
            if batch.gen_throughput is None or batch.running_req is None:
                continue
            pairs.append((int(batch.running_req), float(batch.gen_throughput)))

    if not pairs:
        return pd.DataFrame(
            columns=["running_req", "sample_count", "gen_throughput_mean", "gen_throughput_median"]
        )

    df = pd.DataFrame(pairs, columns=["running_req", "gen_throughput"])
    summary = (
        df.groupby("running_req", sort=True)["gen_throughput"]
        .agg(["count", "mean", "median"])
        .rename(
            columns={
                "count": "sample_count",
                "mean": "gen_throughput_mean",
                "median": "gen_throughput_median",
            }
        )
        .reset_index()
    )
    return summary


def export_node_metrics(run_path: str, output_dir: str | None = None) -> list[str] | None:
    """解析 run 目录下的节点日志，按节点写出仅含 batch 列的 CSV。

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
        logger.warning("No node metrics parsed from %s; no CSV written", run_path)
        return None

    written: list[str] = []
    for node in nodes:
        name = _node_batch_csv_filename(node)
        csv_path = os.path.join(out, name)
        df = node_batches_to_dataframe(node)
        df.to_csv(csv_path, index=False)
        written.append(csv_path)
        logger.info("Wrote %s (%d batch rows)", csv_path, len(df))

    summary_path = os.path.join(out, GEN_THROUGHPUT_SUMMARY_NAME)
    summary_df = _gen_throughput_summary_dataframe(nodes)
    summary_df.to_csv(summary_path, index=False)
    written.append(summary_path)
    logger.info(
        "Wrote %s (%d running_req groups)",
        summary_path,
        len(summary_df),
    )

    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Export per-node batch metrics from run logs to CSV.")
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

    paths = export_node_metrics(args.run_path, output_dir=args.output_dir)
    if not paths:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
