#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarise an inference-endpoint run into srt-slurm's benchmark-rollup.json.

This client does not use LoadGen; it writes its own report, specified in the
client repo at docs/metrics/report_design.md (github.com/mlcommons/endpoints).
The field list below follows that spec and was cross-checked against real runs
under endpoints-launch/sflow_output:

  performance/result_summary.json   qps, tps, sample counts, and ttft/tpot/latency
                                    /output_sequence_lengths blocks
  accuracy/accuracy_results.json    average_accuracy plus a per-dataset breakdown

Accuracy genuinely lives in a separate file: ``to_json`` does ``payload.pop("accuracy")``
because "result_summary.json is the performance report - accuracy lives only in
the dedicated accuracy report". So both files are read; neither is a fallback
for the other.

Two things only the real runs showed:

* The location moved. A July run (286161) wrote ``result_summary.json`` at the
  top level; August runs (355234, 412798, 474440) write
  ``performance/result_summary.json``. Both are searched, newest layout first.
* Every duration is in **nanoseconds**, including the ``percentiles`` maps. The
  report.txt renders them as milliseconds, so a value copied from the text
  report would be off by 1e6. Token-length series are counts and must not scale.

And two the spec showed that the runs could not: ``input_sequence_lengths`` is
absent from every fixture here (it is empty "when no ISL values were recorded,
for example no tokenizer available for text-only inputs") but is a documented
field, and ``legacy_loadgen_window_duration_ns`` records *which window* produced
qps/tps -- left ``None`` when the native window was used. Two runs' QPS are not
comparable across different windows, so the rollup carries the discriminator.

Because the layout has already moved once, a missing or restructured report is
treated as a rollup without metrics rather than as an error: the run itself
succeeded, and losing the whole record because a key moved would be worse than
recording less than we hoped for. The client takes the same line internally --
"honest incompleteness over crashes".
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Durations in the summary are nanoseconds; srt-slurm's other benchmarks report
# milliseconds, so convert at the boundary rather than leaving units mixed.
NS_PER_MS = 1_000_000

# Where result_summary.json has been seen to live, newest layout first.
SUMMARY_CANDIDATES = ("performance/result_summary.json", "result_summary.json")
ACCURACY_CANDIDATES = ("accuracy/accuracy_results.json",)

# The latency blocks worth lifting into the rollup. The client also emits
# histograms; those stay in the report, since a rollup is a summary.
LATENCY_BLOCKS = ("ttft", "tpot", "latency")

# Token-count series. ISL is documented but empty in every run observed here, so
# it is read defensively rather than assumed present.
TOKEN_BLOCKS = ("input_sequence_lengths", "output_sequence_lengths")
PERCENTILES = ("50.0", "90.0", "95.0", "99.0")


def _first_existing(report_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative in candidates:
        path = report_dir / relative
        if path.is_file():
            return path
    return None


def _load(path: Path) -> dict[str, Any] | None:
    """Return the parsed object, or None if it is unusable.

    A truncated report means the client died mid-write, which is exactly when the
    surrounding rollup is most worth keeping.
    """
    try:
        loaded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _latency_ms(block: dict[str, Any]) -> dict[str, float]:
    """Convert one latency block to milliseconds, keeping the useful percentiles."""
    summary = {
        "min_ms": block["min"] / NS_PER_MS,
        "max_ms": block["max"] / NS_PER_MS,
        "mean_ms": block["avg"] / NS_PER_MS,
        "median_ms": block["median"] / NS_PER_MS,
    }
    percentiles = block.get("percentiles", {})
    for name in PERCENTILES:
        if name in percentiles:
            summary[f"p{_percentile_label(name)}_ms"] = percentiles[name] / NS_PER_MS
    return summary


def _percentile_label(name: str) -> str:
    """Turn the client's "50.0"/"99.9" percentile keys into p50/p99_9 suffixes.

    Trimming with ``rstrip(".0")`` is wrong here: it strips every trailing "." and
    "0" character rather than the ".0" suffix, so "50.0" becomes "5" and "90.0"
    becomes "9" while "95.0" and "99.0" survive unharmed.
    """
    if name.endswith(".0"):
        return name[:-2]
    return name.replace(".", "_")


def summarize_performance(summary: dict[str, Any]) -> dict[str, Any]:
    """Lift the headline throughput numbers and latency blocks out of a summary."""
    result: dict[str, Any] = {
        "qps": summary.get("qps"),
        "tps": summary.get("tps"),
        "samples_issued": summary.get("n_samples_issued"),
        "samples_completed": summary.get("n_samples_completed"),
        "samples_failed": summary.get("n_samples_failed"),
        # state/complete are the client's own verdict on whether the run finished
        # (`complete` is `state == "complete" and no pending tasks`). This is the
        # nearest thing it has to LoadGen's VALID/INVALID.
        "state": summary.get("state"),
        "complete": summary.get("complete"),
        # Which window produced qps/tps: the MLPerf LoadGen "completed" window
        # when set, the native window when None. Runs measured under different
        # windows are not comparable, so keep the discriminator beside them.
        "qps_window": ("legacy_loadgen" if summary.get("legacy_loadgen_window_duration_ns") else "native"),
    }
    if "duration_ns" in summary:
        result["duration_s"] = summary["duration_ns"] / 1_000_000_000
    if summary.get("finish_reason_counts"):
        # Distinguishes a clean run from one that mostly hit the length cap.
        result["finish_reason_counts"] = summary["finish_reason_counts"]

    for name in LATENCY_BLOCKS:
        block = summary.get(name)
        if isinstance(block, dict) and "min" in block:
            result[name] = _latency_ms(block)

    for name in TOKEN_BLOCKS:
        block = summary.get(name)
        if isinstance(block, dict) and "min" in block:
            # Tokens, not nanoseconds - do not scale these.
            result[name] = {
                "min": block.get("min"),
                "max": block.get("max"),
                "mean": block.get("avg"),
                "median": block.get("median"),
            }
    return result


def summarize_accuracy(accuracy: dict[str, Any]) -> dict[str, Any]:
    """Lift the overall score and its per-subset breakdown."""
    result: dict[str, Any] = {"average_accuracy": accuracy.get("average_accuracy")}
    scores = accuracy.get("accuracy_scores")
    if isinstance(scores, list):
        result["datasets"] = [
            {
                "name": entry.get("dataset_name"),
                "score": entry.get("score"),
                "samples": entry.get("total_samples"),
                "complete": entry.get("complete"),
                "subset_scores": (entry.get("breakdown") or {}).get("subset_scores"),
            }
            for entry in scores
            if isinstance(entry, dict)
        ]
    return result


def parse_report(report_dir: Path) -> dict[str, Any]:
    """Summarise whatever the client left in report_dir.

    Never raises for a missing or malformed report: returns the keys it could
    fill and records which files it actually read, so a rollup with no metrics
    can be told apart from one whose metrics were genuinely zero.
    """
    metrics: dict[str, Any] = {}
    sources: list[str] = []

    summary_path = _first_existing(report_dir, SUMMARY_CANDIDATES)
    if summary_path is not None:
        summary = _load(summary_path)
        if summary is not None:
            metrics["performance"] = summarize_performance(summary)
            sources.append(str(summary_path.relative_to(report_dir)))

    accuracy_path = _first_existing(report_dir, ACCURACY_CANDIDATES)
    if accuracy_path is not None:
        accuracy = _load(accuracy_path)
        if accuracy is not None:
            metrics["accuracy"] = summarize_accuracy(accuracy)
            sources.append(str(accuracy_path.relative_to(report_dir)))

    return {"metrics": metrics, "metrics_sources": sources}


def canonical_run_metrics(performance: dict[str, Any]) -> dict[str, Any]:
    """Map this client's metrics onto the flat keys srt-slurm's consumers read.

    `srtctl monitor` renders a run with `r.get("throughput_toks")`,
    `r.get("ttft_mean_ms")` and friends at the top level of the run record (see
    cli/monitor.py::_format_run), and sa-bench's rollup emits exactly those. A
    rollup that only nested its numbers would be collected and then displayed as
    a blank line, so the same names are emitted here rather than a private shape.

    MLPerf-specific values that have no counterpart (accuracy, the qps window,
    the client's completion verdict) stay in the nested `metrics` block.
    """
    ttft = performance.get("ttft") or {}
    tpot = performance.get("tpot") or {}
    latency = performance.get("latency") or {}
    osl = performance.get("output_sequence_lengths") or {}
    canonical = {
        # tps is tokens/s and qps is requests/s, matching sa-bench's split of
        # throughput_toks vs request_throughput.
        "throughput_toks": performance.get("tps"),
        "request_throughput": performance.get("qps"),
        "ttft_mean_ms": ttft.get("mean_ms"),
        "ttft_p99_ms": ttft.get("p99_ms"),
        "tpot_mean_ms": tpot.get("mean_ms"),
        "tpot_p99_ms": tpot.get("p99_ms"),
        "e2el_mean_ms": latency.get("mean_ms"),
        "completed_requests": performance.get("samples_completed"),
        "osl": osl.get("mean"),
    }
    # Drop keys the run genuinely lacks: monitor treats a falsy value as absent,
    # and a null is more honest than a fabricated zero.
    return {k: v for k, v in canonical.items() if v is not None}


def build_rollup(report_dir: Path, *, mode: str, endpoints: list[str], client_config: str, exit_code: int) -> dict[str, Any]:
    """Assemble the record srt-slurm's postprocess and monitor read."""
    run: dict[str, Any] = {
        "mode": mode,
        "endpoints": endpoints,
        "client_config": client_config,
        "exit_code": exit_code,
        "report_dir": str(report_dir),
    }
    if report_dir.is_dir():
        run["report_files"] = sorted(str(p.relative_to(report_dir)) for p in report_dir.rglob("*") if p.is_file())

    parsed = parse_report(report_dir)
    run.update(canonical_run_metrics(parsed["metrics"].get("performance") or {}))
    run.update(parsed)
    return {
        "benchmark_type": "mlperf",
        "client": "inference-endpoint",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs": [run],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="where to write benchmark-rollup.json")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--endpoints", required=True, help="comma-separated")
    parser.add_argument("--client-config", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    rollup = build_rollup(
        args.report_dir,
        mode=args.mode,
        endpoints=[e.strip() for e in args.endpoints.split(",") if e.strip()],
        client_config=args.client_config,
        exit_code=args.exit_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rollup, indent=1))

    metrics = rollup["runs"][0]["metrics"]
    if metrics:
        print(f"[mlperf] wrote {args.output} from {', '.join(rollup['runs'][0]['metrics_sources'])}")
    else:
        # Say so loudly: a rollup with no metrics looks like a clean run at a glance.
        print(f"[mlperf] WARNING: no parseable report under {args.report_dir}; wrote {args.output} without metrics", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
