#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 processor: TRT-LLM ``print_iter_log`` lines -> ``iter_bins.json`` (schema 5).

Per-iteration engine telemetry, one record per forward step per worker, emitted when
the engine config sets ``print_iter_log: true``. It is far denser than the Prometheus
scrape -- thousands of records per worker against a 1 Hz gauge -- and it carries the
one quantity the scrape genuinely cannot express: the COMPOSITION of each scheduled
batch, which is what distinguishes "the engine is busy" from "the engine is busy
doing one request at a time".

Line format (verified against real worker logs)::

    iter = 2, global_rank = 0, rank = 0, num_scheduled_requests = 0,
    kv_cache_util = 0.000, currank_total_requests = 0/1,
    host_step_time = 4.08ms, prev_device_step_time = 243.13ms,
    timestamp = 2026-08-19 14:11:49,
    states = {'num_ctx_requests': 1, 'num_ctx_tokens': 4,
              'num_generation_tokens': 0, 'cached_kv_tokens': 0}

THE TIMEZONE TRAP
-----------------
``timestamp`` here is the worker's LOCAL time, while Dynamo's own logs, the metrics
scrape and the request trace are all UTC. Reading it naively lands engine events
hours away from the frontend events they belong to -- far enough that a per-iteration
series looks like it covers a completely different part of the run.

The offset is NOT hardcoded. It is derived per run: the median iteration timestamp is
compared against the run window supplied by the caller, and the offset is snapped to
whole hours. That is self-correcting across clusters and across DST, and it fails
visibly (a large residual) rather than silently misplacing the series.

OTHER LIMITS, recorded because they bound what the data can support
-------------------------------------------------------------------
* Rank 0 only. Every line reports ``rank = 0``, so this cannot show TP/EP-rank
  imbalance -- for that, the per-request attribution in the request trace is the
  only source.
* ``iter`` restarts at 1 on each engine lifecycle, so it is not a global sequence
  number and must never be used as an x-axis.
* ``prev_device_step_time`` is ``N/A`` on the first iteration of each lifecycle.

Stdlib only (runs under a bare cluster python3).

Usage:
    python3 -m src.ingest.iter_log OUT_PATH LOG [LOG ...] [--window-start-ns N --window-end-ns N]
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

logger = logging.getLogger("iter_log")

_RE_ITER = re.compile(
    r"iter = (?P<iter>\d+),.*?"
    r"num_scheduled_requests = (?P<sched>\d+),\s*"
    r"kv_cache_util = (?P<kv>[0-9.]+),\s*"
    r"currank_total_requests = (?P<cur>\d+)/(?P<tot>\d+),\s*"
    r"host_step_time = (?P<host>[0-9.]+)ms,\s*"
    r"prev_device_step_time = (?P<dev>N/A|[0-9.]+ms),\s*"
    r"timestamp = (?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\s*"
    r"states = (?P<states>\{[^}]*\})"
)

# Metrics carried per bin. Kept explicit rather than "whatever was in states" so a
# consumer can rely on the key set regardless of engine version.
BIN_FIELDS = ("kv_cache_util", "num_scheduled_requests", "num_ctx_requests",
              "num_ctx_tokens", "num_generation_tokens", "cached_kv_tokens",
              "host_step_time_ms", "device_step_time_ms")


def parse_line(line: str) -> dict | None:
    """One ``print_iter_log`` line -> a record, or None if it is not one."""
    m = _RE_ITER.search(line)
    if not m:
        return None
    try:
        states = ast.literal_eval(m.group("states"))
    except (ValueError, SyntaxError):
        states = {}
    dev = m.group("dev")
    return {
        "iter": int(m.group("iter")),
        # Naive local time; the caller converts once the offset is known.
        "local_ts": m.group("ts"),
        "num_scheduled_requests": int(m.group("sched")),
        "kv_cache_util": float(m.group("kv")),
        "currank_requests": int(m.group("cur")),
        "total_requests": int(m.group("tot")),
        "host_step_time_ms": float(m.group("host")),
        "device_step_time_ms": None if dev == "N/A" else float(dev[:-2]),
        "num_ctx_requests": states.get("num_ctx_requests"),
        "num_ctx_tokens": states.get("num_ctx_tokens"),
        "num_generation_tokens": states.get("num_generation_tokens"),
        "cached_kv_tokens": states.get("cached_kv_tokens"),
    }


def _local_to_epoch(ts: str) -> float:
    """Naive 'YYYY-MM-DD HH:MM:SS' -> epoch seconds, interpreted as UTC.

    Deliberately UTC and not the host's zone: the host running the INGEST is not the
    host that wrote the log, so ``mktime`` here would apply the wrong zone. The real
    offset is derived separately from the run window.
    """
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def derive_offset_hours(recs: list[dict], window_start_ns: int | None,
                        window_end_ns: int | None) -> int:
    """Whole-hour offset from log-local time to the run's UTC window.

    Returns 0 when no window is supplied -- an underived offset is better than a
    guessed one, and the caller can still read the series, just unaligned.
    """
    if not recs or not window_start_ns or not window_end_ns:
        return 0
    mid_log = sorted(_local_to_epoch(r["local_ts"]) for r in recs)[len(recs) // 2]
    mid_run = ((window_start_ns + window_end_ns) / 2) / 1e9
    return int(round((mid_run - mid_log) / 3600.0))


def process(out_path: str, log_paths: list[str], window_start_ns: int | None = None,
            window_end_ns: int | None = None, bin_seconds: float = 1.0) -> int:
    """Worker logs -> ``iter_bins.json``. Returns the number of bins written.

    Output shape matches what the renderer's per-iteration panels already consume::

        {"bins": {"<worker>": [{"t": <epoch s>, "<field>": <value>, ...}, ...]},
         "meta": {...}}

    Records are median-reduced into ``bin_seconds`` buckets. Median rather than mean
    because ``host_step_time`` is heavily right-tailed -- a single 9-second step would
    drag a mean bin far above anything the engine typically did.
    """
    per_worker: dict[str, list[dict]] = {}
    for path in log_paths:
        # Worker identity comes from the filename (<node>_<mode>_w<i>.out); the line
        # itself only ever says rank 0 and cannot distinguish workers.
        worker = os.path.basename(path).rsplit(".", 1)[0]
        recs: list[dict] = []
        with open(path, errors="replace") as f:
            for line in f:
                if "iter = " not in line:
                    continue
                rec = parse_line(line)
                if rec:
                    recs.append(rec)
        if recs:
            per_worker[worker] = recs
            logger.info("iter_log: %s -> %d iterations", worker, len(recs))

    if not per_worker:
        logger.warning("iter_log: no print_iter_log lines found in %d file(s)", len(log_paths))
        return 0

    all_recs = [r for rs in per_worker.values() for r in rs]
    offset_h = derive_offset_hours(all_recs, window_start_ns, window_end_ns)
    logger.info("iter_log: derived local->UTC offset of %+d h from the run window", offset_h)

    bins_out: dict[str, list[dict]] = {}
    total = 0
    for worker, recs in per_worker.items():
        buckets: dict[int, list[dict]] = {}
        for r in recs:
            t = _local_to_epoch(r["local_ts"]) + offset_h * 3600
            buckets.setdefault(int(t // bin_seconds), []).append(r)
        rows = []
        for key in sorted(buckets):
            group = buckets[key]
            row = {"t": key * bin_seconds, "n_iters": len(group)}
            for field in BIN_FIELDS:
                vals = sorted(v for v in (g.get(field) for g in group) if v is not None)
                row[field] = vals[len(vals) // 2] if vals else None
            # Batch composition is the point of this source, so the distribution of
            # scheduled-request counts is carried per bin and not just its median: a
            # median of 1 hides whether the engine ever batched at all.
            sched = [g["num_scheduled_requests"] for g in group]
            row["sched_max"] = max(sched)
            # Step-time MAX as well as median. A 1 Hz Prometheus gauge samples the
            # engine once a second and cannot see a stall it did not land on: on the
            # reference run host_step_time reaches 230 s with 82 stalls over 1 s, while
            # the scraped iteration-latency gauge tops out at 9.67 s. The median says
            # what a step normally costs; only the max says the engine ever stopped.
            for _f in ("host_step_time_ms", "device_step_time_ms"):
                _v = [g.get(_f) for g in group if g.get(_f) is not None]
                row[_f + "_max"] = max(_v) if _v else None
            row["sched_hist"] = {str(k): sched.count(k) for k in sorted(set(sched))}
            rows.append(row)
        bins_out[worker] = rows
        total += len(rows)

    payload = {
        "bins": bins_out,
        "meta": {
            "bin_seconds": bin_seconds,
            "offset_hours_applied": offset_h,
            "iterations": len(all_recs),
            "workers": sorted(bins_out),
            "note": "timestamps converted from worker-local to UTC using a run-derived "
                    "whole-hour offset; rank 0 only; iter restarts per engine lifecycle",
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    logger.info("iter_log -> %s: %d bins across %d worker(s), %d iterations",
                out_path, total, len(bins_out), len(all_recs))
    return total


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_path")
    p.add_argument("logs", nargs="+")
    p.add_argument("--window-start-ns", type=int, default=None)
    p.add_argument("--window-end-ns", type=int, default=None)
    p.add_argument("--bin-seconds", type=float, default=1.0)
    a = p.parse_args(argv)
    process(a.out_path, a.logs, a.window_start_ns, a.window_end_ns, a.bin_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
