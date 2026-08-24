#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 2 -- Converter B: Dynamo SPAN_CLOSED JSONL logs -> tempo_traces/<xid>.json.

Ported faithfully from ``spans_to_tempo.py`` (lyris-slo20-otel). Parses
``SPAN_CLOSED`` lines out of the frontend + all worker logs, groups spans by
``trace_id`` (shared via W3C traceparent across frontend/prefill/decode), and
maps each trace to the client's x-request-id (xid) via a span carrying the
``x_request_id`` attribute -- KEYED ON x_request_id, NOT the Dynamo request_id
UUID (when the client sends an ``x-request-id`` header, Dynamo makes the span's
request_id equal it, but we join on the explicit x_request_id attr; this was
the debugged gotcha). Emits per-request Tempo-format JSON that L3
(``src/visualization/build_dynamo_bench_dash.py``) reads.

On an srt-slurm run the input logs are the job's ``<node>_<mode>_w<i>.out`` and
``<node>_frontend_<i>.out`` files, which carry SPAN_CLOSED lines whenever
``observability.enabled`` is set (see ``ANALYTICS_SPAN_ENV`` in ``srtctl.core.schema``).

SPAN_CLOSED line grammar (verified on real perf-on data):
  <label>: {"time":"<ISO Z>", "message":"SPAN_CLOSED", "span_name":..., "span_id":...,
            "parent_id":..., "trace_id":..., "request_id":..., "time.duration_us":N,
            "component":"prefill"|"backend"(handle_payload), "dp_rank"/"overlap_blocks"/
            "phase"/"worker_id"(kv_router.route_request), ...}
No startTimeUnixNano -> start = close_time_ns - duration_us*1000.

Usage: python3 -m src.ingest.traces_spanlog <out_dir> <valid_xids_file> <log> [<log> ...]
  valid_xids_file: newline-separated x_request_id values (from the aiperf profile).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

# Span-envelope keys that are metadata, not passthrough attributes. Everything
# NOT in this set becomes a Tempo span attribute (the attr passthrough).
# `time.busy_us` and `time.idle_us` are deliberately NOT skipped. They are the split
# of a span's wall time into work and waiting, which is the difference between a span
# that is slow because it computed and one that is slow because it blocked -- the
# central question for host-side stalls. Only `time.duration_us` is dropped, because
# it is consumed directly as the span's duration rather than passed through.
_SKIP = {
    "time", "level", "file", "line", "target", "message", "span_id", "parent_id",
    "span_name", "trace_id", "request_id", "time.duration_us",
}


def iso_to_ns(t: str) -> int:
    """Convert an ISO-8601 timestamp (``...Z`` or offset) to wall-epoch ns."""
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return int(datetime.fromisoformat(t).timestamp() * 1e9)


def parse_line(line: str) -> dict | None:
    """Parse one ``SPAN_CLOSED`` log line into a normalized span dict.

    Returns None for lines that are not SPAN_CLOSED records or that lack a
    duration. ``start_ns`` is reconstructed as ``end_ns - duration_us*1000``
    because the log carries only the close time.
    """
    i = line.find("{")
    if i < 0:
        return None
    try:
        x = json.loads(line[i:])
    except Exception:
        return None
    if x.get("message") != "SPAN_CLOSED" or "time.duration_us" not in x or "time" not in x:
        return None
    end_ns = iso_to_ns(x["time"])
    start_ns = end_ns - int(float(x["time.duration_us"]) * 1000)
    attrs = {k: v for k, v in x.items() if k not in _SKIP}
    return {
        "trace_id": x.get("trace_id"), "xid": x.get("x_request_id"),
        "span_id": x.get("span_id", ""), "parent_id": x.get("parent_id", ""),
        "name": x.get("span_name", ""), "start_ns": start_ns, "end_ns": end_ns, "attrs": attrs,
    }


def process(out_dir: str, valid_xids: set[str], log_paths: list[str]) -> int:
    """Parse SPAN_CLOSED logs -> tempo_traces/<xid>.json under ``out_dir``.

    Groups spans by trace_id, resolves each trace's client xid via the
    ``x_request_id`` attr filtered against ``valid_xids``, and writes one
    Tempo-schema JSON per resolved trace. Returns the number of files written.
    """
    os.makedirs(out_dir, exist_ok=True)
    by_trace: dict[str, list[dict]] = {}
    trace_xid: dict[str, str] = {}
    n = 0
    for lg in log_paths:
        try:
            fh = open(lg, errors="replace")
        except FileNotFoundError:
            continue
        with fh:
            for line in fh:
                if "SPAN_CLOSED" not in line:
                    continue
                sp = parse_line(line)
                if not sp or not sp["trace_id"]:
                    continue
                by_trace.setdefault(sp["trace_id"], []).append(sp)
                n += 1
                if sp["xid"] in valid_xids:
                    trace_xid[sp["trace_id"]] = sp["xid"]

    written = 0
    for tid, spans in by_trace.items():
        xid = trace_xid.get(tid)
        if not xid:
            continue
        otlp = [{
            "spanId": s["span_id"], "parentSpanId": s["parent_id"], "name": s["name"],
            "startTimeUnixNano": str(s["start_ns"]), "endTimeUnixNano": str(s["end_ns"]),
            "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in s["attrs"].items()],
            "events": [],
        } for s in spans]
        doc = {"traceID": tid, "batches": [{"resource": {"attributes": []}, "scopeSpans": [{"spans": otlp}]}]}
        with open(os.path.join(out_dir, f"{xid}.json"), "w") as f:
            json.dump(doc, f)
        written += 1

    logging.getLogger("traces_spanlog").info(
        f"parsed {n} spans / {len(by_trace)} traces; wrote {written} "
        f"tempo_traces/<xid>.json (valid xids={len(valid_xids)})"
    )
    return written


def _read_xids(path: str) -> set[str]:
    """Load newline-separated x_request_id values from ``path``."""
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("out_dir", help="Output tempo_traces/ directory (created if absent).")
    parser.add_argument("valid_xids_file", help="Newline-separated valid x_request_id values.")
    parser.add_argument("logs", nargs="+", help="One or more SPAN_CLOSED log files.")
    args = parser.parse_args(argv)
    process(args.out_dir, _read_xids(args.valid_xids_file), args.logs)


if __name__ == "__main__":
    main()
