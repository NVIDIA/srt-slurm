#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 processor: ``raw_prometheus.jsonl`` (RAW capture) -> ``server_metrics_export.jsonl``.

This is the parse half of the L1->L2 split. The L1 capture scraper
(:mod:`srtctl.analysis.metrics_scraper`, started in-job when ``observability.enabled``
is set) curls each ``/metrics`` endpoint and appends the response body VERBATIM as
one JSON line:

    {"timestamp_ns": <int>, "endpoint_url": <str>, "role": "frontend"|"prefill"|"decode",
     "worker_id": <str|null>, "text": "<raw Prometheus exposition text, UNPARSED>"}

``process()`` parses each ``text`` into ``{metric_name: [{"labels": {...}, "value": <num>}]}``
and re-emits the fixed ``server_metrics_export.jsonl`` schema that L3
(``src/visualization/build_dynamo_bench_dash.py``) reads:

    {"timestamp_ns": <int>, "metrics": {"<name>": [{"labels": {...}, "value": <num>}, ...]}}

WHY inject worker_id / dynamo_component
---------------------------------------
The TRT-LLM ``MetricsCollector`` labels its gauges with only ``{model_name, engine_type}``.
The KV panels pair ``trtllm_kv_cache_{used,free,max}_blocks`` by a
**worker_id** label and split by **dynamo_component** (prefill / backend). Those
labels are not guaranteed on the worker's /metrics text, so for every metric scraped from a
worker line we ADD ``worker_id=<raw worker_id>`` and ``dynamo_component`` (``prefill`` role ->
``"prefill"``, ``decode`` role -> ``"backend"``) via ``setdefault`` -- NEVER clobbering a real
scraped label. Frontend lines are left untouched (their ``dynamo_frontend_*`` series already
carry ``worker_type``).

The parsing helpers (``parse_exposition``/``parse_labels``/``parse_value`` + histogram
passthrough) are lifted verbatim from ``scrape_dynamo_metrics.py`` (Converter C), which used to
parse inline while scraping. Splitting capture (RAW) from parse (here) lets the same raw bytes be
re-parsed offline without re-running the job.

Stdlib only (runs under a bare cluster python3).

Usage:
    python3 -m src.ingest.metrics_prometheus RAW_PATH OUT_PATH
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import Dict, List

# metric_name{labels} value [timestamp]   |   metric_name value [timestamp]
_SAMPLE_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+[0-9.eE+-]+)?\s*$')
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

# One parsed metric entry: {"labels": {...}, "value": <num>}
Entry = Dict[str, object]
Metrics = Dict[str, List[Entry]]


def _unescape(v: str) -> str:
    return v.replace('\\\\', '\\').replace('\\"', '"').replace('\\n', '\n')


def parse_labels(block: str | None) -> Dict[str, str]:
    """'{a="1",b="x"}' -> {'a': '1', 'b': 'x'}.  None/'' -> {}."""
    if not block:
        return {}
    return {k: _unescape(v) for k, v in _LABEL_RE.findall(block)}


def parse_value(tok: str):
    """Prometheus value token -> float, or None for non-finite (L3 can't int(NaN))."""
    if tok in ("NaN", "+Inf", "-Inf", "Inf"):
        return None
    try:
        f = float(tok)
    except ValueError:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def parse_exposition(text: str, extra_labels: Dict[str, str] | None = None) -> Metrics:
    """Prometheus exposition text -> {metric_name: [{'labels': {...}, 'value': num}, ...]}.

    Histogram/summary components (``_bucket``/``_sum``/``_count``) are kept verbatim under
    their own series names so nothing is lost; L3 just ignores the ones it doesn't read.
    ``extra_labels`` (worker_id/dynamo_component) are added when absent from the scraped series.
    """
    out: Metrics = {}
    extra_labels = extra_labels or {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == "#":
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name, label_block, val_tok = m.group(1), m.group(2), m.group(3)
        value = parse_value(val_tok)
        if value is None:
            continue
        labels = parse_labels(label_block)
        for k, v in extra_labels.items():
            labels.setdefault(k, v)  # never clobber a real scraped label
        out.setdefault(name, []).append({"labels": labels, "value": value})
    return out


# role (from the raw line) -> dynamo_component label value. frontend -> no injection.
_ROLE_COMPONENT = {"prefill": "prefill", "decode": "backend"}


def _extra_labels_for(role: str, worker_id) -> Dict[str, str]:
    """Labels to inject for a raw line's role. Frontend lines get nothing."""
    component = _ROLE_COMPONENT.get(role)
    if component is None:
        return {}
    extra = {"dynamo_component": component}
    if worker_id is not None:
        extra["worker_id"] = str(worker_id)
    return extra


def _dedup(entries: List[Entry]) -> List[Entry]:
    """Drop exact (labels, value) duplicates within one metric, preserving first-seen order.

    Both frontend ports (e.g. :8333 and the DYN_SYSTEM_PORT :8082) serve identical metrics, so
    without this a series would be double-listed per timestamp."""
    seen, uniq = set(), []
    for e in entries:
        key = (tuple(sorted(e["labels"].items())), e["value"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def process(raw_path: str, out_path: str) -> int:
    """Parse ``raw_prometheus.jsonl`` -> ``server_metrics_export.jsonl``.

    All raw lines sharing a ``timestamp_ns`` are merged into one output line (one scrape sweep
    across every endpoint), with per-metric (labels, value) dedup. Returns lines written.
    """
    # Preserve first-seen timestamp order; merge every endpoint at that timestamp.
    by_ts: "dict[int, Metrics]" = {}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = rec["timestamp_ns"]
            extra = _extra_labels_for(rec.get("role", ""), rec.get("worker_id"))
            merged = by_ts.setdefault(ts, {})
            for name, entries in parse_exposition(rec.get("text", ""), extra).items():
                merged.setdefault(name, []).extend(entries)

    lines_written = 0
    with open(out_path, "w") as out:
        for ts, merged in by_ts.items():
            for name in merged:
                merged[name] = _dedup(merged[name])
            out.write(json.dumps({"timestamp_ns": ts, "metrics": merged}) + "\n")
            lines_written += 1
    return lines_written


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("raw_path", help="input raw_prometheus.jsonl (RAW L1 capture)")
    ap.add_argument("out_path", help="output server_metrics_export.jsonl (fixed L2 schema)")
    args = ap.parse_args(argv)
    n = process(args.raw_path, args.out_path)
    logging.getLogger("metrics_prometheus").info(f"[metrics_prometheus] wrote {n} lines -> {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
