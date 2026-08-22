# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 2 (L2) ingest processors: RAW capture bytes -> fixed intermediate schemas.

The component perf dashboard is built as three clean layers::

    L1 srtctl.analysis.metrics_scraper   in-job scrapers  -> RAW bytes (unparsed)
       + Dynamo SPAN_CLOSED worker/frontend logs
       + the benchmark client's own per-request export
    L2 src/ingest/                       processors       RAW -> intermediate schema
    L3 src/visualization/build_dynamo_bench_dash.py       bundle  -> single-file HTML

This package is L2. Each processor reads one RAW source (or an already-in-schema
source) and writes exactly one of the three FIXED intermediate schemas that L3
reads. The schemas are frozen -- do NOT change them here; changing them breaks L3.

Vendored from the ``dynamo-benchmark-perf-dashboard`` repo (commit
``22f49fea243e43403690b38e70a8d4092dec4cc8``). Only the processors reachable from
the component dashboard are carried over; see ``docs/component-dashboard.md``.

Intermediate schemas (L2 outputs, join key ``x_request_id``, timeline = wall-epoch ns)
--------------------------------------------------------------------------------------
1. ``profile_export.jsonl`` -- AIPerf client, one JSON per line::

     {"metadata": {"x_request_id", "conversation_id", "turn_index",
                   "request_start_ns", "request_end_ns"},
      "metrics": {"time_to_first_token": {"value"}, "inter_token_latency": {"value"},
                  "input_sequence_length": {"value"}, "output_sequence_length": {"value"},
                  "request_latency": {"value"}, "output_token_throughput_per_user": {"value"}}}

2. ``server_metrics_export.jsonl`` -- AIPerf metrics, one JSON per line::

     {"timestamp_ns": int,
      "metrics": {"<name>": [{"labels": {...}, "value": <num>}, ...], ...}}
     (histograms flattened to <name>_bucket / _sum / _count series)

3. ``tempo_traces/<x_request_id>.json`` -- Tempo/OTLP, one file per request::

     {"traceID", "batches": [{"resource": {"attributes": []},
       "scopeSpans": [{"spans": [{"spanId", "parentSpanId", "name",
         "startTimeUnixNano", "endTimeUnixNano",
         "attributes": [{"key", "value": {"stringValue"}}], "events": []}]}]}]}

RAW L1->L2 contract (capture emits RAW; the processor parses)
------------------------------------------------------------------------------
``raw_prometheus.jsonl`` -- written in-job by :mod:`srtctl.analysis.metrics_scraper`
when ``observability.enabled`` is set. One JSON per line, the /metrics response
body verbatim::

    {"timestamp_ns": int, "endpoint_url": str,
     "role": "frontend"|"prefill"|"decode", "worker_id": str|null,
     "text": "<raw Prometheus exposition text, UNPARSED>"}

``metrics_prometheus.process`` is the parse half: it parses each ``text``, merges
every endpoint sharing a ``timestamp_ns`` into one scrape sweep, injects
``worker_id``/``dynamo_component`` from the line's role, and re-emits schema 2.
Splitting capture (RAW) from parse (here) lets the same raw bytes be re-parsed
offline without re-running the job.

Processor registry
------------------
``PROCESSORS[axis][name]`` -> a callable. Three axes, one per intermediate schema:

    client  -> profile_export.jsonl
      "aiperf"  passthrough: input is ALREADY schema 1 (AIPerf's own writer);
                stitch shards / copy into the bundle.        (aiperf_passthrough)
    traces  -> tempo_traces/<xid>.json
      "spanlog" Dynamo SPAN_CLOSED logs -> schema 3.         (traces_spanlog.process)
    metrics -> server_metrics_export.jsonl
      "prometheus" raw_prometheus.jsonl -> schema 2.         (metrics_prometheus.process)
      "aiperf-json" AIPerf server_metrics_export.json -> schema 2.
                                                          (metrics_aiperf_json.process)
                   The only server-metrics source on runs predating
                   `observability.enabled`, which is where the before/after
                   pairs for already-landed fixes live.
    request_trace -> request_trace.jsonl
      "dynamo"  dynamo-request-trace -> schema 4.            (request_trace.process)
    iter_log -> iter_bins.json
      "trtllm"  print_iter_log worker lines -> schema 5.     (iter_log.process)

The upstream repo also registers an ``agentperf`` client processor and a live
``tempo`` trace scraper. Neither is reachable from an srt-slurm run -- srt-slurm's
clients are AIPerf-based and no Tempo backend is deployed -- so both were left
behind rather than vendored as dead code.

Sibling processor call contracts (used by ``ingest.py``; keep these stable):
    aiperf_passthrough(inputs, out_path) -> {"written", "shards"}  (defined here)
        inputs = a file path, a glob string, or a list of shard paths (stitched).
    traces_spanlog.process(out_dir, valid_xids: set[str], log_paths: list[str]) -> int
    metrics_prometheus.process(raw_path, out_path) -> int

Imports are LAZY (resolved on first call) so importing this registry never pulls
in a sibling module's deps until that processor actually runs. Everything here is
stdlib-only so it runs under a bare cluster ``python3``.
"""

from __future__ import annotations

import glob
import importlib
import os
import shutil
from typing import Callable

__all__ = ["PROCESSORS", "get_processor", "aiperf_passthrough"]


def _lazy(module_suffix: str, attr: str = "process") -> Callable:
    """A callable that imports ``src.ingest.<module_suffix>`` on first invocation
    and forwards to its ``attr`` (default ``process``). Keeps the registry cheap
    to import even when a sibling processor is missing/heavier."""

    def _call(*args, **kwargs):
        mod = importlib.import_module(f"{__name__}.{module_suffix}")
        return getattr(mod, attr)(*args, **kwargs)

    _call.__name__ = _call.__qualname__ = f"{module_suffix}.{attr}"
    _call._target = (f"{__name__}.{module_suffix}", attr)  # type: ignore[attr-defined]
    return _call


def aiperf_passthrough(inputs, out_path) -> dict:
    """Client passthrough: the input is ALREADY ``profile_export.jsonl`` (AIPerf's
    built-in writer emits schema 1 directly), so there is nothing to convert.

    Stitch the input(s) into the bundle at ``out_path``. ``inputs`` may be a single
    file, a glob string (e.g. ``run/profile_export.w*.jsonl``), or a list of shard
    paths; shards are concatenated in sorted order. Returns ``{"written", "shards"}``
    (``written`` = total non-blank JSON lines carried over).
    """
    if isinstance(inputs, (str, os.PathLike)):
        s = os.fspath(inputs)
        paths = sorted(glob.glob(s)) if glob.has_magic(s) else [s]
    else:
        paths = [os.fspath(p) for p in inputs]
    if not paths:
        raise FileNotFoundError(f"no aiperf profile inputs matched: {inputs!r}")
    # Single concrete file -> straight copy; multiple shards -> concatenate.
    written = 0
    if len(paths) == 1 and os.path.abspath(paths[0]) != os.path.abspath(os.fspath(out_path)):
        shutil.copyfile(paths[0], out_path)
        with open(out_path) as f:
            written = sum(1 for ln in f if ln.strip())
    else:
        with open(out_path, "w") as o:
            for p in paths:
                with open(p) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        o.write(line if line.endswith("\n") else line + "\n")
                        written += 1
    return {"written": written, "shards": paths}


# axis -> source name -> processor callable (all lazy except the local passthrough).
PROCESSORS: dict[str, dict[str, Callable]] = {
    "client": {
        "aiperf": aiperf_passthrough,
    },
    "traces": {
        "spanlog": _lazy("traces_spanlog"),
    },
    "metrics": {
        "prometheus": _lazy("metrics_prometheus"),
        "aiperf-json": _lazy("metrics_aiperf_json"),
    },
    "request_trace": {
        "dynamo": _lazy("request_trace"),
    },
    "iter_log": {
        "trtllm": _lazy("iter_log"),
    },
}


def get_processor(axis: str, name: str) -> Callable:
    """Return the L2 processor callable for ``(axis, name)``.

    ``axis`` is one of ``client`` / ``traces`` / ``metrics`` (one per intermediate
    schema); ``name`` selects the source within that axis. Raises ``KeyError`` with
    the valid options when either is unknown.
    """
    try:
        by_name = PROCESSORS[axis]
    except KeyError:
        raise KeyError(f"unknown ingest axis {axis!r}; valid: {sorted(PROCESSORS)}") from None
    try:
        return by_name[name]
    except KeyError:
        raise KeyError(
            f"unknown {axis!r} processor {name!r}; valid: {sorted(by_name)}"
        ) from None
