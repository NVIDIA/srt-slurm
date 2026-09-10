#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 ingest orchestrator -- makes the layered flow explicit and stops at the bundle.

    L1 srtctl.analysis.metrics_scraper  -> raw_prometheus.jsonl
       Dynamo workers/frontend (stdout) -> SPAN_CLOSED lines in <node>_*.out
       the benchmark client              -> its own per-request profile_export.jsonl
    L2 src/ingest/  per-source processors     RAW -> the 3 fixed intermediate schemas
    -------------------------------------------------------------------------------
    (this script drives L1 artifacts through L2 into an output *bundle* dir, then
     writes a dashboard.yaml pointing at it)
    -------------------------------------------------------------------------------
    L3 src/visualization/build_dynamo_bench_dash.py   bundle -> single-file HTML

Given a ``--run-dir`` of raw artifacts and flags selecting which sources to ingest,
this runs the L2 processors (via the ``src.ingest`` registry) into a self-contained
bundle::

    <bundle>/profile_export.jsonl          (client axis  -> schema 1)
    <bundle>/tempo_traces/<xid>.json        (traces axis  -> schema 3)
    <bundle>/server_metrics_export.jsonl    (metrics axis -> schema 2)
    <bundle>/dashboard.yaml                 (generated sidecar, sources: point here)

then STOPS. It never imports or invokes L3 -- the user runs the presentation layer
themselves::

    python3 -m src.visualization.build_dynamo_bench_dash <bundle> out.html

For an srt-slurm run directory, ``--run-dir`` is the job's ``logs/`` directory: the
scraper writes ``raw_prometheus.jsonl`` there, and the worker/frontend logs that
carry the SPAN_CLOSED lines are the ``<node>_<mode>_w<i>.out`` / ``<node>_frontend_<i>.out``
files beside it.

Baked-in optimizations (ported from render_fast.sh, the hand-tuned ~71 GB perf-ON
render prep):
  * shard-stitch  -- the client passthrough accepts a shard glob and stitches it in
    sorted order.
  * parallel pre-grep -- SPAN_CLOSED lines are grepped out of the (multi-GB) worker
    logs into compact ``*.spans`` files, in parallel, BEFORE the trace processor
    walks them. Turns a 71 GB scan into a few-MB one for the Python span parser.
  * server_metrics dedup -- a final idempotent (labels, value)-dedup fold over the
    produced ``server_metrics_export.jsonl`` (the frontend serves identical metrics
    on two ports; drop the duplicate series per scrape).

Stdlib only (runs under a bare cluster python3). ``grep`` is used for the pre-grep
fast path with a pure-Python fallback when it is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Runnable both as ``python3 -m src.ingest.ingest`` and as a bare script path. The
# latter puts only ``src/ingest/`` on sys.path, so the repo root -- which is what
# makes ``src.ingest`` importable -- has to be added explicitly.
if __package__ in (None, ""):  # pragma: no cover - only on the bare-script path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest import get_processor  # noqa: E402

# ---------------------------------------------------------------------------
# logging -- one tagged line per layer step
# ---------------------------------------------------------------------------


def _log(tag: str, msg: str) -> None:
    logging.getLogger("ingest").info("[%s] %s", tag, msg)


# ---------------------------------------------------------------------------
# owned helpers (baked-in render_fast optimizations + bundle bookkeeping)
# ---------------------------------------------------------------------------


# Every client-export layout seen in the wild. AgentX nests under `agentic/<conc>/`
# (with or without an `aiperf_artifacts/` level, which differs per harness build);
# stock AIPerf uses `artifacts/<run>/`.
# Where each harness puts AIPerf's server-metrics export, mirroring CLIENT_PATTERNS.
AIPERF_METRIC_PATTERNS = [
    "agentic/*/aiperf_artifacts/server_metrics_export.json",
    "agentic/*/server_metrics_export.json",
    "artifacts/*/server_metrics_export.json",
]

# AIPerf's per-scrape jsonl sibling (written with --server-metrics-formats json jsonl),
# mirroring AIPERF_METRIC_PATTERNS. One line per (scrape, endpoint), values verbatim.
AIPERF_JSONL_PATTERNS = [
    "agentic/*/aiperf_artifacts/server_metrics_export.jsonl",
    "agentic/*/server_metrics_export.jsonl",
    "artifacts/*/server_metrics_export.jsonl",
]

# Where the Tachometer leg leaves its parquet, in preference order: the post-run
# compaction writes final.parquet under the storage leaf (tachometer/raw/scrape);
# tachometer/final holds the direct-host explicit-compact output; tachometer/local
# holds out-N/incomplete-N leftovers that only matter when compaction never ran.
TACHOMETER_PATTERNS = [
    "tachometer/raw/scrape/final.parquet",
    "tachometer/raw/scrape/*.parquet",
    "tachometer/final/final.parquet",
    "tachometer/local/final.parquet",
    "tachometer/local/*.parquet",
]

CLIENT_PATTERNS = [
    "agentic/*/aiperf_artifacts/profile_export.jsonl",
    "agentic/*/profile_export.jsonl",
    "artifacts/*/profile_export.jsonl",
]


def resolve_inputs(pattern, run_dir: Path) -> list[str]:
    """Resolve a source flag to a sorted list of concrete files.

    ``pattern`` may be absolute or relative to ``run_dir``, and may contain glob
    metacharacters (e.g. ``results.w*.jsonl`` -> every shard). A relative concrete
    path is resolved under ``run_dir``.
    """
    import glob as _glob

    p = os.fspath(pattern)
    if not os.path.isabs(p):
        p = os.path.join(run_dir, p)
    if _glob.has_magic(p):
        return sorted(_glob.glob(p))
    return [p] if os.path.exists(p) else []


def extract_xids(profile_path: str | Path) -> set[str]:
    """Read the join keys (``metadata.x_request_id``) out of a profile_export.jsonl.

    These select which traces to keep (the trace processor only writes traces whose
    resolved xid is a valid client request), mirroring render_fast.sh's ``xids.txt``.
    """
    xids: set[str] = set()
    with open(profile_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            xid = json.loads(line).get("metadata", {}).get("x_request_id")
            if xid:
                xids.add(xid)
    return xids


def _grep_span_closed(log_path: str, out_path: str) -> int:
    """Extract SPAN_CLOSED lines from one log into ``out_path``; return line count.

    Uses ``grep -h SPAN_CLOSED`` (fast on multi-GB logs) with a pure-Python fallback
    when grep is unavailable. grep exit 1 = no matches (an empty .spans file, fine).
    """
    grep = shutil.which("grep")
    if grep:
        with open(out_path, "w") as o:
            rc = subprocess.run([grep, "-h", "SPAN_CLOSED", log_path], stdout=o).returncode
        if rc not in (0, 1):  # 2+ = real grep error -> fall through to Python scan
            grep = None
    if not grep:
        with open(log_path, errors="replace") as f, open(out_path, "w") as o:
            for line in f:
                if "SPAN_CLOSED" in line:
                    o.write(line)
    with open(out_path) as f:
        return sum(1 for _ in f)


def pregrep_spans(log_paths: list[str], spans_dir: Path, jobs: int = 4) -> list[str]:
    """Parallel pre-grep: SPAN_CLOSED lines from each big log -> compact ``*.spans``.

    Returns the list of NON-EMPTY .spans files (empty logs are dropped so the trace
    processor is not handed dead inputs). Ported from render_fast.sh's ``xargs -P4``
    pre-grep stage.
    """
    spans_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []

    def _one(lg: str) -> tuple[str, int]:
        stem = Path(lg).stem
        dst = str(spans_dir / f"{stem}.spans")
        return dst, _grep_span_closed(lg, dst)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futs = [pool.submit(_one, lg) for lg in log_paths]
        for fut in as_completed(futs):
            dst, n = fut.result()
            _log("L2 traces", f"  pre-grep {Path(dst).name}: {n} SPAN_CLOSED lines")
            if n:
                out.append(dst)
    return sorted(out)


def dedup_server_metrics(path: str | Path) -> tuple[int, int]:
    """In-place, idempotent (labels, value)-dedup of a server_metrics_export.jsonl.

    Both frontend ports serve identical ``dynamo_frontend_*`` series, so a series can
    be double-listed per scrape. Drops exact duplicates within each metric per line,
    preserving order. Returns (lines_in, lines_out). Ported from render_fast.sh's
    Converter-C fold; safe to run even when the processor already deduped.
    """
    path = Path(path)
    nin = nout = 0
    tmp = path.with_suffix(path.suffix + ".dedup.tmp")
    with open(path) as f, open(tmp, "w") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nin += 1
            d = json.loads(line)
            for name, entries in d.get("metrics", {}).items():
                seen, uniq = set(), []
                for e in entries:
                    key = (json.dumps(e.get("labels", {}), sort_keys=True), e.get("value"))
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(e)
                d["metrics"][name] = uniq
            o.write(json.dumps(d) + "\n")
            nout += 1
    os.replace(tmp, path)
    return nin, nout


def parse_worker_spec(spec: str) -> tuple[str, dict]:
    """``role=parallelism:rank:count`` -> (role, {parallelism, rank, worker_count}).

    e.g. ``prefill=dep:4:6`` -> ("prefill", {"parallelism": "dep", "rank": 4,
    "worker_count": 6}). ``role=parallelism:rank`` defaults worker_count to 1.
    """
    role, _, rest = spec.partition("=")
    role = role.strip()
    parts = [p.strip() for p in rest.split(":") if p.strip()]
    if not role or len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"bad --worker {spec!r}; want ROLE=PARALLELISM:RANK[:COUNT], e.g. prefill=dep:4:6"
        )
    parallelism, rank = parts[0], int(parts[1])
    count = int(parts[2]) if len(parts) > 2 else 1
    return role, {"parallelism": parallelism, "rank": rank, "worker_count": count}


def probe_warmup_end_ns(
    run_dir: Path, patterns: list[str]
) -> tuple[int | None, str | None, int | None, int | None]:
    """First PROFILING request start (ns) from the client export, without copying it.

    The load generator's warmup phase is tagged per record as
    ``metadata.benchmark_phase``. The renderer reads that directly when the client leg
    is in the bundle -- but the export is routinely 700 MB+ and is often not copied,
    and the boundary is one integer. So it is probed here and recorded in
    dashboard.yaml, which keeps the marker available to a metrics-only bundle.

    Streamed with a substring pre-filter so the JSON parser only sees candidate lines;
    a full parse of every record would cost minutes on a 700 MB export for one number.
    Returns ``(None, None)`` when the harness tags no phase -- which is not a failure,
    it means this run has no warmup boundary to mark.
    """
    srcs: list[str] = []
    for pat in patterns:
        srcs.extend(resolve_inputs(pat, run_dir))
    srcs = sorted(dict.fromkeys(srcs))
    if not srcs:
        return None, None, None, None
    prof_min = warm_max = None
    span_lo = span_hi = None
    scanned = 0
    for src in srcs:
        with open(src) as f:
            for line in f:
                if '"benchmark_phase"' not in line:
                    continue
                scanned += 1
                try:
                    md = json.loads(line).get("metadata", {})
                except Exception:
                    continue
                st = md.get("request_start_ns")
                if st:
                    span_lo = st if span_lo is None else min(span_lo, st)
                    span_hi = st if span_hi is None else max(span_hi, st)
                ph = md.get("benchmark_phase")
                if ph == "profiling":
                    v = md.get("request_start_ns")
                    if v:
                        prof_min = v if prof_min is None else min(prof_min, v)
                elif ph:
                    v = md.get("request_end_ns") or md.get("request_start_ns")
                    if v:
                        warm_max = v if warm_max is None else max(warm_max, v)
    if prof_min:
        return prof_min, "first profiling request", span_lo, span_hi
    if warm_max:
        return warm_max, "last warmup request (profiling phase never began)", span_lo, span_hi
    return None, None, span_lo, span_hi


def generate_dashboard_yaml(
    *,
    name: str,
    description: str,
    mode: str,
    framework: str,
    block_size: int,
    workers: dict[str, dict],
    have_aiperf: bool,
    have_traces: bool,
    have_metrics: bool,
    have_request_trace: bool = False,
    warmup_end_ns: int | None = None,
    warmup_source: str | None = None,
    metrics_source: str | None = None,
) -> str:
    """Render a dashboard.yaml (skeleton fields: name/description/mode/framework/
    topology/sources) whose ``sources:`` point at the bundle's own files (so the
    yaml lives in the bundle and paths are plain filenames). Only sources that were
    actually produced are emitted."""
    lines: list[str] = []
    lines.append("# Generated by src/ingest/ingest.py (L2). Render with:")
    lines.append("#   python3 -m src.visualization.build_dynamo_bench_dash <this-dir> out.html")
    lines.append(f"name: {name}")
    lines.append("description: |")
    for dl in (description or "").splitlines() or [""]:
        lines.append(f"  {dl}")
    lines.append(f"mode: {mode}")
    lines.append(f"framework: {framework}")
    lines.append("topology:")
    lines.append(f"  block_size: {block_size}   # fallback; server_metrics tokens_per_block wins")
    lines.append("  workers:")
    if workers:
        for role, w in workers.items():
            lines.append(
                f"    {role}: {{parallelism: {w['parallelism']}, "
                f"rank: {w['rank']}, worker_count: {w['worker_count']}}}"
            )
    else:
        # No --worker given: emit a role-appropriate placeholder to fill in.
        if mode == "agg":
            lines.append("    agg: {parallelism: tep, rank: 4, worker_count: 1}   # TODO: set from your run")
        else:
            lines.append("    prefill: {parallelism: dep, rank: 4, worker_count: 1}  # TODO: set from your run")
            lines.append("    decode:  {parallelism: tep, rank: 4, worker_count: 1}  # TODO: set from your run")
    if warmup_end_ns:
        # Carried so a bundle WITHOUT the client leg can still mark the phase
        # boundary. Absent when the harness tags no phase -- which the renderer must
        # read as "no boundary known", not as "warmup ended at t=0".
        lines.append("warmup:")
        lines.append(f"  end_ns: {int(warmup_end_ns)}")
        if warmup_source:
            lines.append(f"  source: {warmup_source}")
    lines.append("sources:")
    if have_aiperf:
        lines.append("  aiperf_profile: profile_export.jsonl")
    if have_traces:
        lines.append("  tempo_traces:   tempo_traces")
    if have_metrics:
        # metrics_source is only set for the sources whose provenance is not obvious
        # from the run dir (tachometer / aiperf per-scrape jsonl); the annotation is a
        # comment, so the sources: entry itself stays byte-identical for L3.
        sm_line = "  server_metrics: server_metrics_export.jsonl"
        if metrics_source:
            sm_line += f"   # source: {metrics_source}"
        lines.append(sm_line)
    if have_request_trace:
        lines.append("  request_trace:  request_trace.jsonl")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# per-axis L2 steps
# ---------------------------------------------------------------------------


def run_client(args, run_dir: Path, bundle: Path) -> bool:
    """L2 client axis -> profile_export.jsonl. Returns whether it was produced."""
    if args.client == "none":
        _log("L2 client", "skipped (--client none)")
        return False
    # Two client layouts exist and neither is a superset of the other, so both are
    # tried rather than making the caller know which harness ran:
    #   AgentX   <log_dir>/agentic/conc_<N>/aiperf_artifacts/profile_export.jsonl
    #   AIPerf   <log_dir>/artifacts/<model>_<workload>_<ts>/profile_export.jsonl
    # AgentX nests one level deeper AND shards by concurrency level. A sweep that ran
    # several concurrencies yields several files; they are stitched, which puts each
    # phase in sequence on the run-relative time axis rather than averaging them
    # together -- the phases stay visually separable, and no phase is silently dropped.
    patterns = [args.client_input] if args.client_input else list(CLIENT_PATTERNS)
    inputs: list[str] = []
    for pat in patterns:
        inputs.extend(resolve_inputs(pat, run_dir))
    inputs = sorted(dict.fromkeys(inputs))
    if not inputs:
        _log("L2 client", f"WARN no client inputs matched {patterns} under {run_dir} -- skipping")
        return False
    _log("L1", f"client raw: {len(inputs)} file(s) matching {patterns} (shard-stitch)")
    out = bundle / "profile_export.jsonl"
    proc = get_processor("client", args.client)
    # agentperf/passthrough both accept a shard list and stitch it in sorted order.
    summary = proc(inputs if len(inputs) > 1 else inputs[0], str(out))
    _log("L2 client", f"{args.client} -> {out.name}: {summary}")
    return out.exists()


def run_traces(args, run_dir: Path, bundle: Path, profile_path: Path | None) -> bool:
    """L2 traces axis -> tempo_traces/<xid>.json. Returns whether any trace produced."""
    if args.traces == "none":
        _log("L2 traces", "skipped (--traces none)")
        return False
    out_dir = bundle / "tempo_traces"

    if args.traces == "spanlog":
        if profile_path is None or not profile_path.exists():
            _log("L2 traces", "WARN no profile_export.jsonl -> cannot resolve xids; skipping spanlog")
            return False
        # srt-slurm names its worker/frontend logs <node>_<mode>_w<i>.out and
        # <node>_frontend_<i>.out, so the SPAN_CLOSED lines land in *.out, not *.log.
        patterns = args.span_logs or ["*.out"]
        logs: list[str] = []
        for pat in patterns:
            logs.extend(resolve_inputs(pat, run_dir))
        logs = sorted(dict.fromkeys(logs))  # de-dup, keep order
        if not logs:
            _log("L2 traces", f"WARN no span logs matched {patterns} under {run_dir}; skipping")
            return False
        _log("L1", f"trace raw: {len(logs)} SPAN_CLOSED log(s)")
        xids = extract_xids(profile_path)
        _log("L2 traces", f"{len(xids)} valid xids from {profile_path.name}")
        spans = pregrep_spans(logs, bundle / "spans", jobs=args.jobs)
        if not spans:
            _log("L2 traces", "WARN no SPAN_CLOSED lines found in any log; skipping")
            return False
        proc = get_processor("traces", "spanlog")
        written = proc(str(out_dir), xids, spans)
        _log("L2 traces", f"spanlog -> {out_dir.name}/: {written} trace files")
        return written > 0

    return False


def run_metrics(args, run_dir: Path, bundle: Path) -> bool:
    """L2 metrics axis -> server_metrics_export.jsonl. Returns whether produced."""
    out = bundle / "server_metrics_export.jsonl"

    if args.metrics == "none":
        # A pre-existing server_metrics_export.jsonl may still be passed through.
        if args.server_metrics:
            srcs = resolve_inputs(args.server_metrics, run_dir)
            if srcs:
                shutil.copyfile(srcs[0], out)
                nin, nout = dedup_server_metrics(out)
                _log("L2 metrics", f"passthrough+dedup {Path(srcs[0]).name}: {nin} -> {nout} lines")
                return True
        _log("L2 metrics", "skipped (--metrics none)")
        return False

    mode = args.metrics
    if mode == "auto":
        # Pick the source the run actually captured, best first: the tachometer
        # parquet (whole-window, per-replica, every endpoint) wins over the in-job
        # raw_prometheus.jsonl, which wins over AIPerf's own exports (per-scrape
        # jsonl over the aggregate json). An `observability.enabled` run has
        # raw_prometheus.jsonl; a run without it usually still has AIPerf's own export,
        # because the frontend's /metrics surface exists regardless of that knob and
        # the benchmark scrapes it. Choosing here rather than at every call site is
        # what lets one ingest command work on all of them.
        tach: list[str] = []
        for pat in ([args.tachometer_parquet] if args.tachometer_parquet else TACHOMETER_PATTERNS):
            tach.extend(resolve_inputs(pat, run_dir))
        raw = resolve_inputs(args.raw_prometheus or "raw_prometheus.jsonl", run_dir)
        if tach:
            mode = "tachometer"
        elif raw:
            mode = "prometheus"
        else:
            found_jsonl: list[str] = []
            for pat in AIPERF_JSONL_PATTERNS:
                found_jsonl.extend(resolve_inputs(pat, run_dir))
            if found_jsonl:
                mode = "aiperf-jsonl"
            else:
                found: list[str] = []
                for pat in AIPERF_METRIC_PATTERNS:
                    found.extend(resolve_inputs(pat, run_dir))
                mode = "aiperf-json" if found else "prometheus"
        _log("L2 metrics", f"auto-selected source: {mode}")

    if mode == "tachometer":
        # The in-job Tachometer scraper's parquet: the whole-window per-replica
        # capture of every /metrics endpoint. First pattern with a hit wins, so
        # final.parquet is preferred over shards/leftovers.
        patterns = ([args.tachometer_parquet] if args.tachometer_parquet
                    else list(TACHOMETER_PATTERNS))
        srcs: list[str] = []
        for pat in patterns:
            srcs = resolve_inputs(pat, run_dir)
            if srcs:
                break
        if not srcs:
            _log("L2 metrics", f"WARN no tachometer parquet matched {patterns} under {run_dir}; skipping")
            return False
        shards = f" (+{len(srcs) - 1} shard(s))" if len(srcs) > 1 else ""
        _log("L1", f"metrics raw: {srcs[0]}{shards} (tachometer parquet)")
        proc = get_processor("metrics", "tachometer")
        n = proc(srcs if len(srcs) > 1 else srcs[0], str(out))
        nin, nout = dedup_server_metrics(out)
        _log("L2 metrics", f"tachometer -> {out.name}: {n} timestamps, dedup {nin} -> {nout} lines")
        args._metrics_source = "tachometer parquet"
        return out.exists()

    if mode == "aiperf-jsonl":
        # AIPerf's per-scrape export (--server-metrics-formats json jsonl): raw
        # readings per (scrape, endpoint), so no timeslice reconstruction needed.
        patterns = ([args.aiperf_server_metrics_jsonl] if args.aiperf_server_metrics_jsonl
                    else list(AIPERF_JSONL_PATTERNS))
        srcs = []
        for pat in patterns:
            srcs.extend(resolve_inputs(pat, run_dir))
        srcs = sorted(dict.fromkeys(srcs))
        if not srcs:
            _log("L2 metrics", f"WARN no aiperf per-scrape jsonl matched {patterns} under {run_dir}; skipping")
            return False
        _log("L1", f"metrics raw: {srcs[0]} (AIPerf per-scrape server_metrics_export.jsonl)")
        proc = get_processor("metrics", "aiperf-jsonl")
        n = proc(srcs[0], str(out))
        nin, nout = dedup_server_metrics(out)
        _log("L2 metrics", f"aiperf-jsonl -> {out.name}: {n} timestamps, dedup {nin} -> {nout} lines")
        args._metrics_source = "aiperf per-scrape jsonl"
        return out.exists()

    if mode == "aiperf-json":
        # AIPerf's own export. The only server-metrics source on a run that predates
        # `observability.enabled`, so it is what makes the historical before/after
        # corpus renderable without re-running anything.
        patterns = ([args.aiperf_server_metrics] if args.aiperf_server_metrics
                    else list(AIPERF_METRIC_PATTERNS))
        srcs: list[str] = []
        for pat in patterns:
            srcs.extend(resolve_inputs(pat, run_dir))
        srcs = sorted(dict.fromkeys(srcs))
        if not srcs:
            _log("L2 metrics", f"WARN no aiperf server metrics matched {patterns} under {run_dir}; skipping")
            return False
        _log("L1", f"metrics raw: {srcs[0]} (AIPerf server_metrics_export.json)")
        proc = get_processor("metrics", "aiperf-json")
        n = proc(srcs[0], str(out))
        nin, nout = dedup_server_metrics(out)
        _log("L2 metrics", f"aiperf-json -> {out.name}: {n} timestamps, dedup {nin} -> {nout} lines")
        return out.exists()

    # metrics == prometheus: parse RAW -> schema 2.
    pattern = args.raw_prometheus or "raw_prometheus.jsonl"
    srcs = resolve_inputs(pattern, run_dir)
    if not srcs:
        _log("L2 metrics", f"WARN no raw prometheus matched {pattern!r} under {run_dir}; skipping")
        return False
    _log("L1", f"metrics raw: {srcs[0]} (raw_prometheus.jsonl contract)")
    proc = get_processor("metrics", "prometheus")
    n = proc(srcs[0], str(out))
    # Idempotent dedup fold (render_fast Converter-C); the processor also dedups,
    # this guarantees a clean artifact regardless of source.
    nin, nout = dedup_server_metrics(out)
    _log("L2 metrics", f"prometheus -> {out.name}: {n} scrapes, dedup {nin} -> {nout} lines")
    return out.exists()


def run_request_trace(args, run_dir: Path, bundle: Path) -> bool:
    """L2 request-trace axis -> request_trace.jsonl. Returns whether produced.

    The frontend's per-request record. It is the only source of KV-transfer cost --
    the Prometheus ``trtllm_kv_transfer_*`` family is declared but never sampled -- and
    the only one carrying ``session_id``, so both the per-request waterfall and the
    per-session view depend on it.

    Dynamo writes it to the path in ``DYN_REQUEST_TRACE_FILE_PATH``, which srt-slurm
    sets to ``<log_dir>/dynamo-request-trace`` (no extension, despite being JSON lines).
    """
    if args.request_trace == "none":
        _log("L2 req-trace", "skipped (--request-trace none)")
        return False
    pattern = args.request_trace_input or "dynamo-request-trace"
    srcs = resolve_inputs(pattern, run_dir)
    if not srcs:
        _log("L2 req-trace", f"WARN no request trace matched {pattern!r} under {run_dir}; skipping")
        return False
    out = bundle / "request_trace.jsonl"
    _log("L1", f"request trace raw: {srcs[0]}")
    proc = get_processor("request_trace", "dynamo")
    n = proc(srcs[0], str(out))
    _log("L2 req-trace", f"dynamo -> {out.name}: {n} requests")
    return n > 0


def run_iter_log(args, run_dir: Path, bundle: Path,
                 fallback_start_ns: int | None = None,
                 fallback_end_ns: int | None = None) -> bool:
    """L2 per-iteration axis -> iter_bins.json. Returns whether produced.

    Parses TRT-LLM's ``print_iter_log`` lines out of the worker logs. This is the only
    source for per-step batch COMPOSITION -- the Prometheus stream reports how busy the
    engine was, not whether "busy" meant one request at a time or many.

    The run window is passed through so the processor can derive the log's local->UTC
    offset instead of hardcoding one; TRT-LLM stamps worker-local time while every
    other source in the bundle is UTC.
    """
    if args.iter_log == "none":
        _log("L2 iter-log", "skipped (--iter-log none)")
        return False
    patterns = args.iter_log_input or ["*_prefill_w*.out", "*_decode_w*.out", "*_agg_w*.out"]
    logs: list[str] = []
    for pat in patterns:
        logs.extend(resolve_inputs(pat, run_dir))
    logs = sorted(dict.fromkeys(logs))
    if not logs:
        _log("L2 iter-log", f"WARN no worker logs matched {patterns} under {run_dir}; skipping")
        return False
    start_ns = end_ns = None
    sm = bundle / "server_metrics_export.jsonl"
    if sm.exists():
        # The metrics stream is the bundle's UTC anchor: its first and last scrape
        # bracket the run, which is what the offset is derived against.
        try:
            with open(sm) as f:
                first = f.readline()
                start_ns = json.loads(first)["timestamp_ns"] if first.strip() else None
            with open(sm) as f:
                for line in f:
                    if line.strip():
                        end_ns = json.loads(line)["timestamp_ns"]
        except Exception as e:  # noqa: BLE001 - anchoring is best-effort
            _log("L2 iter-log", f"WARN could not read the run window: {e}")
    if start_ns is None and fallback_start_ns is not None:
        # No metrics stream to anchor against -- a trtllm-serve run has no Dynamo
        # endpoint at all. The client export's own span is UTC and covers the same
        # run, so it anchors the offset just as well. Without it the offset silently
        # defaults to 0 and the engine timeline sits hours away from every other
        # source, which only shows up when something tries to join them.
        start_ns, end_ns = fallback_start_ns, fallback_end_ns
        _log("L2 iter-log", "UTC anchor from the client export (no metrics stream)")
    out = bundle / "iter_bins.json"
    _log("L1", f"iter-log raw: {len(logs)} worker log(s)")
    proc = get_processor("iter_log", "trtllm")
    n = proc(str(out), logs, start_ns, end_ns)
    _log("L2 iter-log", f"trtllm -> {out.name}: {n} bins")
    return n > 0


def run_engine_configs(run_dir: Path, bundle: Path) -> list[str]:
    """Copy the run's resolved engine configs into the bundle, verbatim.

    L3 reads ``<bundle>/trtllm_config_{prefill,decode}.yaml`` for the in-flight-batch
    ceilings drawn on the Engine tab, and falls back to its ``--max-batch-*`` CLI
    defaults when they are absent. srt-slurm writes exactly those filenames, but into
    the run's log dir (``backends/trtllm.py``: ``runtime.log_dir / f"trtllm_config_{mode}.yaml"``),
    which is one level above the bundle -- so without this copy the fallback always won.

    That fallback is not a cosmetic default. On AgentX run 2739690 the real decode
    ``max_batch_size`` is 1 while the CLI default is 256, so every decode in-flight
    panel was drawn against a ceiling 256x too high and read as "nowhere near
    saturated" when the engine was in fact pinned at its limit. Prefill happened to
    match (128) which is exactly what makes the decode error easy to miss.

    Globbed rather than enumerated so aggregated-mode runs (``trtllm_config_agg*.yaml``)
    are carried across without a second code path.
    """
    copied: list[str] = []
    for src in sorted(Path(run_dir).glob("trtllm_config_*.yaml")):
        shutil.copyfile(src, bundle / src.name)
        copied.append(src.name)
    if copied:
        _log("L2 engine-cfg", f"copied {len(copied)} engine config(s): {', '.join(copied)}")
    else:
        _log("L2 engine-cfg", "no trtllm_config_*.yaml in run dir; Engine tab will use --max-batch-* defaults")
    return copied


_BENCH_ERROR_MARKERS = ("Error:", "ERROR:", "Traceback", "command not found",
                        "No such file or directory", "Permission denied",
                        # Not every fatal message is prefixed "Error:". These are real
                        # lines from real failures whose absence left the banner able to
                        # say only "exit 1": the post-run env guard writes a bare
                        # "<VAR> is required when ...", and a phase abort writes
                        # "Benchmark aborted: <reason>".
                        "is required when", "Benchmark aborted", "ProfileAborted")

# AIPerf's own phase census. This is the single most informative line about what the
# benchmark actually did, and it is in aiperf.log rather than benchmark.out -- so a run
# whose benchmark.out carries no error marker at all (arm 2752189: zero markers, exit 1)
# otherwise yields a banner that reports a failure with no reason attached.
_PHASE_RE = "Phase (warmup|profiling) complete"


def run_benchmark_status(run_dir: Path, bundle: Path) -> dict | None:
    """Why the benchmark produced what it produced -> ``benchmark_status.json``.

    A dashboard built from a run whose benchmark never served anything is not wrong,
    but it is mute: it reports ``client=False traces=False N=0`` and leaves the reader
    to guess between a dead deployment, a workload that never started, and a run cut
    short. Those have completely different responses, and the answer is sitting in
    ``benchmark.out`` and the sweep log the whole time.

    This is not hypothetical. Eight ablation arms in one session produced dashboards
    that could not explain their own emptiness -- four with ``exit 127`` (the benchmark
    script was not mounted) and four with ``exit 1`` eleven seconds after the workers
    went healthy (``Error: KV_OFFLOADING must be set for agentic benchmarks``). Both
    are environment faults, and neither is visible in any panel.

    Captures the reported exit code, the first few error-shaped lines, and a bounded
    tail. Bounded because ``benchmark.out`` is a ``set -x`` trace and can be large; the
    diagnosis is always near the end.
    """
    src_dir = Path(run_dir)
    bench = src_dir / "benchmark.out"
    sweeps = sorted(src_dir.glob("sweep_*.log"))
    if not bench.exists() and not sweeps:
        return None

    status: dict = {"exit_code": None, "errors": [], "tail": []}

    # The sweep log is srtctl's own account and carries the exit code verbatim.
    for sw in sweeps:
        try:
            with open(sw, errors="replace") as fh:
                for line in fh:
                    if "Benchmark failed with exit code" in line:
                        status["exit_code"] = line.strip().split()[-1]
        except OSError:
            pass

    if bench.exists():
        try:
            with open(bench, errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            lines = []
        status["tail"] = [ln.rstrip("\n")[:400] for ln in lines[-40:]]
        # Error-shaped lines anywhere, not just the tail: a script can fail early and
        # then print pages of cleanup noise after it.
        #
        # Two shapes of noise have to be handled or the captured message is useless:
        #
        #   * `set -x` echoes every command, so each message appears twice -- once as
        #     `+ echo 'Error: ...'` and once as its own output. The trace line is
        #     dropped, because a reader shown `+ echo 'Error: ...'` is being shown the
        #     mechanism instead of the message.
        #   * a message can be a HEADER whose payload is on the following lines.
        #     `check_env_vars` prints "The following required environment variables are
        #     not set:" and then one `  - NAME` line per variable. Capturing only the
        #     header names the failure class and withholds the one detail that makes it
        #     actionable -- which variable. Observed on v3 arm 2752146, where the
        #     answer was FRAMEWORK / PRECISION / RESULT_FILENAME / DURATION.
        collecting = False
        for ln in lines:
            raw = ln.rstrip("\n")
            if raw.lstrip().startswith("+"):      # set -x trace, not output
                continue
            # AIPerf frames fatal messages in a Rich box, so the raw line arrives as
            # "|    Error: ProfileAborted                     |" -- box rule, message,
            # padding, box rule. Stripped here so the banner shows the message rather
            # than the frame it happened to be drawn in.
            s = raw.strip().strip("\u2502\u2503|").strip()
            s = re.sub(r"\s{2,}", " ", s)[:400]
            if any(m in raw for m in _BENCH_ERROR_MARKERS):
                if s and s not in status["errors"]:
                    status["errors"].append(s)
                collecting = True
            elif collecting and s.startswith("-"):
                # Continuation of the message above (a bullet list of names/reasons).
                if s not in status["errors"]:
                    status["errors"].append(s)
            elif s:
                collecting = False
            if len(status["errors"]) >= 12:
                break

    # AIPerf's phase census, from its own log. Carries the completed/cancelled split that
    # distinguishes "the workload never started" from "it ran and was cut short" -- and
    # those have opposite fixes. On the v4 arms it separated two failure modes that looked
    # identical from the exit code alone: a0/a1 aborted in WARMUP (cancelled 4 and 6, so
    # profiling never ran), while a2/a3 passed warmup clean and had PROFILING time out
    # with ~47 requests still in flight.
    status["phases"] = []
    for lg in sorted(src_dir.glob("agentic/*/aiperf_artifacts/logs/aiperf.log")):
        try:
            with open(lg, errors="replace") as fh:
                for line in fh:
                    if re.search(_PHASE_RE, line):
                        # Keep the census, drop the log preamble before it.
                        status["phases"].append(line.split(" - ")[-1].strip()[:300])
        except OSError:
            pass

    with open(bundle / "benchmark_status.json", "w") as f:
        json.dump(status, f)
    if status["exit_code"] or status["errors"] or status["phases"]:
        _log("L2 bench-status", f"benchmark exit={status['exit_code']} "
                                f"first error: {(status['errors'] or ['-'])[0][:110]}")
        for ph in status["phases"]:
            _log("L2 bench-status", f"  {ph[:140]}")
    else:
        _log("L2 bench-status", "benchmark reported no error")
    return status


# Log-only signals: patterns whose ONLY evidence is a worker or frontend log line, with
# no metric family behind them. Each entry is (id, regex, why it matters).
#
# Kept as a fixed, named set rather than "whatever looked interesting", so the leg is a
# standardised artifact that means the same thing on every run -- the same reason the
# panel spec is declarative.
_LOG_SIGNALS = [
    ("worker_crash", r"CUDA error|device-side assert|Segmentation fault|core dumped|"
                     r"terminate called|std::bad_alloc",
     "a worker died; the crash line is the only artifact of it"),
    ("torch_recompile", r"recompile_limit|torch\._dynamo hit config\.cache_size_limit|"
                        r"falling back to eager",
     "Dynamo recompilation storm: silently converts compiled steps to eager"),
    ("oom", r"CUDA out of memory|OutOfMemoryError|Killed process|oom-kill",
     "memory exhaustion, which no gauge survives to report"),
    ("kv_block_not_found", r"Block not found during remove|Failed to find block to remove",
     "router KV index desync; the counter for it reads a constant 0"),
    ("kv_transfer_timeout", r"kv transfer.*timed out|KV_TRANSFER_TIMEOUT|transfer timeout",
     "KV-transfer timeouts, whose counter family is declared but never sampled"),
    ("nccl_error", r"NCCL WARN|NCCL error|ncclInternalError|ncclUnhandledCudaError",
     "collective failure; no NCCL metric family exists at all"),
    ("engine_stall", r"num_fitting_reqs=0|may not have enough kvCache",
     "the engine could not fit a request, distinct from being merely busy"),
    ("etcd_disconnect", r"etcd.*reconnect|lease.*lost|watch channel closed",
     "control-plane instability that reads downstream as worker flapping"),
    ("request_cancelled", r"request cancelled|client disconnected|connection reset",
     "client-side aborts, which never appear as server errors"),
]


def run_log_signals(run_dir: Path, bundle: Path, max_samples: int = 2) -> dict:
    """Distil log-only signals into ``log_signals.json`` (bounded).

    Several documented issues have their ONLY evidence in a log line -- a worker crash, a
    Dynamo recompilation storm, a KV-index desync -- with no metric family behind them.
    Those were unreachable from a bundle not because the instrumentation is missing but
    because the bundle did not carry the logs, and the logs are far too large to carry
    whole (43 MB frontend + 40 MB prefill on one run, multi-GB at DYN_LOG=debug).

    So this carries the ANSWER rather than the evidence: per named signal, a count, the
    first and last timestamp, the per-file breakdown, and up to `max_samples` verbatim
    lines. Bounded by construction -- the output is a few KB regardless of input size --
    and enough to answer "did this happen, how often, when did it start".

    A count of zero is kept, not dropped. "This run had no worker crashes" is a different
    and more useful statement than the signal's absence from the file.
    """
    import re as _re
    pats = [(sid, _re.compile(rx, _re.I), why) for sid, rx, why in _LOG_SIGNALS]
    out: dict = {sid: {"count": 0, "why": why, "first_ts": None, "last_ts": None,
                       "by_file": {}, "samples": []}
                 for sid, _, why in pats}
    # ISO-8601 or "YYYY-MM-DD HH:MM:SS"; whichever the emitting component uses.
    ts_re = _re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[.\d]*Z?|\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")

    logs = sorted(Path(run_dir).glob("*.out"))
    for path in logs:
        name = path.name
        try:
            fh = open(path, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                for sid, rx, _why in pats:
                    if rx.search(line):
                        e = out[sid]
                        e["count"] += 1
                        e["by_file"][name] = e["by_file"].get(name, 0) + 1
                        m = ts_re.search(line)
                        if m:
                            if e["first_ts"] is None:
                                e["first_ts"] = m.group(1)
                            e["last_ts"] = m.group(1)
                        if len(e["samples"]) < max_samples:
                            e["samples"].append(line.strip()[:300])
                        break   # one signal per line; the first match wins

    with open(bundle / "log_signals.json", "w") as f:
        json.dump({"signals": out, "files_scanned": [p.name for p in logs]}, f)
    fired = {k: v["count"] for k, v in out.items() if v["count"]}
    _log("L2 log-signals", f"scanned {len(logs)} log(s); "
                           + (f"fired: {fired}" if fired else "no log-only signal fired"))
    return out


# Run lifecycle milestones, in the order they must occur. Each is (id, regex).
# Fixed and ordered so the leg means the same thing on every run.
_LIFECYCLE = [
    ("infra_start", r"Starting infrastructure services"),
    ("workers_start", r"Starting backend workers"),
    ("frontend_start", r"Starting frontend layer"),
    ("model_ready", r"Model is ready\."),
    ("benchmark_start", r"Server is healthy - starting benchmark"),
    ("benchmark_end", r"Benchmark (?:completed|failed)"),
    ("run_end", r"(?:✗ Sweep failed|✓ Sweep complete|End:)"),
]


def run_lifecycle(run_dir: Path, bundle: Path) -> dict:
    """Time-to-ready and terminal cause -> ``run_lifecycle.json`` (PERF-45).

    Two questions no panel could answer, because both live in srtctl's own sweep log
    rather than in any metric or client artifact:

    * **How long until the deployment could serve?** On run 2752632 the workers start at
      02:23:19 and the model is ready at 02:34:09 -- **650 s** during which the job holds
      28 GPUs and serves nothing. That is a first-order cost of every run and it is
      invisible in a dashboard whose x-axis starts at the first request.
    * **Why did the run end?** srtctl writes its own verdict, and a reader looking at an
      empty or truncated dashboard needs it before anything else.

    The readiness GAP matters as much as the total: `model_ready` is when the health
    check passes, but the frontend has been accepting connections since `frontend_start`.
    Requests arriving in that window meet a router with nowhere to place them.

    Derived durations are emitted alongside the raw timestamps so a consumer never has to
    re-parse them, and a milestone that never happened is recorded as null rather than
    omitted -- "the model never became ready" is the single most useful thing a failed
    run can say.
    """
    import re as _re
    from datetime import datetime as _dt

    sweeps = sorted(Path(run_dir).glob("sweep_*.log"))
    if not sweeps:
        _log("L2 lifecycle", "no sweep_*.log; time-to-ready and terminal cause unavailable")
        return {}

    pats = [(mid, _re.compile(rx)) for mid, rx in _LIFECYCLE]
    ts_re = _re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
    marks: dict = {mid: None for mid, _ in _LIFECYCLE}
    terminal = None
    for sw in sweeps:
        try:
            fh = open(sw, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                m = ts_re.match(line)
                for mid, rx in pats:
                    # First occurrence wins: these are one-shot transitions, and a
                    # retry loop would otherwise overwrite the real start time.
                    if marks[mid] is None and rx.search(line):
                        marks[mid] = m.group(1) if m else "unknown"
                if "Sweep failed" in line or "Sweep complete" in line:
                    terminal = line.strip()[:200]

    def _delta(a, b):
        try:
            return round((_dt.strptime(marks[b], "%Y-%m-%d %H:%M:%S")
                          - _dt.strptime(marks[a], "%Y-%m-%d %H:%M:%S")).total_seconds(), 1)
        except Exception:
            return None

    out = {
        "milestones": marks,
        "terminal_cause": terminal,
        "durations_s": {
            # workers up -> health check passes: the GPUs-held-but-idle window.
            "time_to_ready": _delta("workers_start", "model_ready"),
            # frontend accepting -> router able to place: requests here have nowhere to go.
            "readiness_gap": _delta("frontend_start", "model_ready"),
            "benchmark": _delta("benchmark_start", "benchmark_end"),
            "total": _delta("infra_start", "run_end"),
        },
    }
    with open(bundle / "run_lifecycle.json", "w") as f:
        json.dump(out, f)
    d = out["durations_s"]
    _log("L2 lifecycle", f"time_to_ready={d['time_to_ready']}s readiness_gap={d['readiness_gap']}s "
                         f"benchmark={d['benchmark']}s | terminal: {terminal or 'unknown'}")
    return out


def run_host_samples(run_dir: Path, bundle: Path) -> dict:
    """``host_samples.jsonl`` -> ``host_series.json``: host and per-process health.

    Closes the three issues that live below the application: host CPU saturation
    (PERF-37), a load generator that has become the bottleneck (PERF-38), and
    file-descriptor exhaustion (PERF-40). None is expressible in any published metric.

    The sampler writes CUMULATIVE jiffie counters, so the busy percentage is derived
    HERE by differencing consecutive samples. That keeps the sampler's interval out of
    the stored numbers -- a percentage computed at capture time would silently bake in
    whatever cadence that run happened to use.

    Per-process CPU is likewise a delta over the same window, expressed as percent of a
    single core, so >100 means the process is genuinely using more than one.
    """
    src = Path(run_dir) / "host_samples.jsonl"
    per_node = sorted(Path(run_dir).glob("host_samples_*.jsonl"))
    if not src.exists() and not per_node:
        _log("L2 host", "no host_samples*.jsonl; host CPU / fd / client-bottleneck "
                        "signals unavailable for this run")
        return {}

    out: dict = {}
    if src.exists():
        out = _host_series_from_rows(_read_host_rows(src)) or {}

    # Per-node samplers (observability.host_sampler_all_nodes) write one file per
    # node; keyed by hostname so worker nodes and a dedicated frontend node are
    # separable downstream. The orchestrator-node series stays at the top level
    # for backward compatibility with existing consumers.
    hosts: dict[str, dict] = {}
    for path in per_node:
        series = _host_series_from_rows(_read_host_rows(path))
        if series:
            hosts[series["host"] or path.stem.removeprefix("host_samples_")] = series
    if hosts:
        out.setdefault("hosts", {}).update(hosts)
    if not out:
        _log("L2 host", "host sample files present but none had >= 2 rows")
        return {}

    with open(bundle / "host_series.json", "w") as f:
        json.dump(out, f)
    peak_cpu = max((v for _, v in out.get("host_cpu_pct", [])), default=None)
    _log("L2 host", f"{out.get('samples', 0)} samples on {out.get('host')} (+{len(hosts)} remote node(s)): "
                    f"peak host CPU {peak_cpu}%, "
                    f"{len(out.get('procs', {}))} process(es) tracked")
    return out


def _read_host_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _host_series_from_rows(rows: list[dict]) -> dict | None:
    """Difference cumulative counters into rate series for one node's samples."""
    if len(rows) < 2:
        return None

    host_cpu, fds, conns, mem, runq, blocked_series = [], [], [], [], [], []
    procs: dict = {}
    for prev, cur in zip(rows, rows[1:]):
        # Rate denominators prefer the monotonic clock: NTP steps can make
        # wall-clock dt negative (pair dropped) or inflated (rates deflated).
        # Wall-clock t stays as the series x-axis for cross-source alignment.
        if cur.get("t_mono") is not None and prev.get("t_mono") is not None:
            dt = cur["t_mono"] - prev["t_mono"]
        else:
            dt = (cur.get("t") or 0) - (prev.get("t") or 0)
        if dt <= 0:
            continue
        t = cur["t"]
        pb, pt = prev.get("cpu_busy_jiffies"), prev.get("cpu_total_jiffies")
        cb, ct = cur.get("cpu_busy_jiffies"), cur.get("cpu_total_jiffies")
        if None not in (pb, pt, cb, ct) and ct > pt:
            host_cpu.append([t, round(100.0 * (cb - pb) / (ct - pt), 2)])
        m = cur.get("mem") or {}
        if m.get("MemTotal") and m.get("MemAvailable") is not None:
            mem.append([t, round(100.0 * (1 - m["MemAvailable"] / m["MemTotal"]), 2)])
        if cur.get("established_conns") is not None:
            conns.append([t, cur["established_conns"]])
        if cur.get("procs_running") is not None:
            runq.append([t, cur["procs_running"]])
        if cur.get("procs_blocked") is not None:
            blocked_series.append([t, cur["procs_blocked"]])

        prev_by_pid = {p["pid"]: p for p in (prev.get("procs") or [])}
        for p in cur.get("procs") or []:
            q = prev_by_pid.get(p["pid"])
            key = f"{p.get('name', 'proc')}:{p['pid']}"
            e = procs.setdefault(key, {"cpu_pct": [], "rss_kb": [], "ctx_invol_rate": [],
                                       "open_fds": [], "threads": [],
                                       "run_delay_ms_per_s": [], "migrations_rate": [],
                                       "affinity_ncpus": []})
            if q and p.get("cpu_jiffies") is not None and q.get("cpu_jiffies") is not None:
                # Jiffies are 1/100 s on Linux; /dt gives percent of ONE core.
                e["cpu_pct"].append([t, round((p["cpu_jiffies"] - q["cpu_jiffies"]) / dt, 1)])
            if q and p.get("ctx_invol") is not None and q.get("ctx_invol") is not None:
                e["ctx_invol_rate"].append(
                    [t, round((p["ctx_invol"] - q["ctx_invol"]) / dt, 1)])
            if q and p.get("run_delay_ns") is not None and q.get("run_delay_ns") is not None:
                # ms of run-queue wait accumulated per wall second: the direct
                # scheduler-contention rate that CPU pinning is the remedy for.
                e["run_delay_ms_per_s"].append(
                    [t, round((p["run_delay_ns"] - q["run_delay_ns"]) / 1e6 / dt, 2)])
            if q and p.get("nr_migrations") is not None and q.get("nr_migrations") is not None:
                e["migrations_rate"].append(
                    [t, round((p["nr_migrations"] - q["nr_migrations"]) / dt, 1)])
            if p.get("affinity_ncpus") is not None:
                e["affinity_ncpus"].append([t, p["affinity_ncpus"]])
            if p.get("rss_kb") is not None:
                e["rss_kb"].append([t, p["rss_kb"]])
            if p.get("open_fds") is not None:
                e["open_fds"].append([t, p["open_fds"]])
            if p.get("threads") is not None:
                e["threads"].append([t, p["threads"]])
        if cur.get("procs"):
            tot = sum((p.get("open_fds") or 0) for p in cur["procs"])
            fds.append([t, tot])

    fd_limit = rows[-1].get("fd_limit")
    peak_fds = max((v for _, v in fds), default=0)
    return {
        "host_cpu_pct": host_cpu,
        "host_mem_used_pct": mem,
        "established_conns": conns,
        "procs_runnable": runq,
        "procs_blocked": blocked_series,
        "open_fds_total": fds,
        "fd_limit": fd_limit,
        # The number that matters for PERF-40: how close the run came to the ceiling.
        # Reported even when comfortable, because "we were at 3%" is the answer that
        # rules the hypothesis out.
        "fd_headroom_pct": (round(100.0 * peak_fds / fd_limit, 2)
                            if fd_limit else None),
        "procs": procs,
        "samples": len(rows),
        "host": rows[-1].get("host"),
    }


def run_provenance(run_dir: Path, bundle: Path) -> list[str]:
    """Copy the run's provenance files into the bundle: what actually ran.

    ``config.yaml`` is the RESOLVED recipe, which is the only record of the run's
    frontend/worker environment -- and therefore the only way to prove, after the fact,
    that two runs differed in exactly one variable. An A/B whose single-variable claim
    rests on the submitter's memory is not an A/B.

    ``fingerprint_<role>_w<i>.json`` carries per-worker ground truth: ``frameworks``
    (the real inner ``tensorrt_llm`` / ``dynamo`` versions), ``cuda_version``,
    ``nccl_version``, ``gpu``, ``pip_packages``. This matters because container tags
    routinely disagree with what they bundle -- an image tagged 1.1.0-rc3 shipping
    1.3.0rc11 is an observed case, not a hypothetical. It is also per-worker, so a
    deployment where prefill and decode ended up on different builds is visible here
    and nowhere else.

    ``resource_snapshot.json`` records the allocation the numbers were produced on.

    All small (~35 KB per fingerprint), so they are copied verbatim rather than
    summarised: a provenance record that has already been filtered cannot answer the
    question nobody thought to ask when writing the filter.
    """
    copied: list[str] = []
    src_dir = Path(run_dir)
    for pattern in ("config.yaml", "resource_snapshot.json", "fingerprint_*.json"):
        for src in sorted(src_dir.glob(pattern)):
            shutil.copyfile(src, bundle / src.name)
            copied.append(src.name)
    if copied:
        _log("L2 provenance", f"copied {len(copied)} provenance file(s): "
                              f"{', '.join(copied[:4])}{' ...' if len(copied) > 4 else ''}")
    else:
        _log("L2 provenance", "no config.yaml / fingerprint_*.json in the run dir; two "
                              "bundles cannot be compared for what actually differed")
    return copied


def run_client_summary(run_dir: Path, bundle: Path, patterns: list[str] | None = None) -> str | None:
    """Copy AIPerf's run-level summary (``profile_export_aiperf.json``) into the bundle.

    ``profile_export.jsonl`` is per-request and is what every panel is built from. This
    is the sibling AIPerf writes ONCE per concurrency, and it carries three things the
    per-request stream cannot express:

    * ``theoretical_prefix_cache_hit`` -- the ceiling the WORKLOAD offered. Without it
      the engine's measured reuse is a number with nothing to be measured against: on
      run 2751593 the workload offered 94.7% and the engine achieved 65.8%, and the
      29-point gap is the finding. Reported alone, 65.8% invites the reader to supply
      their own expectation.
    * validity -- ``error_summary``, ``was_cancelled``, ``branch_stats``. An agentic run
      whose child branches errored or were truncated produced numbers that should not
      be compared against a clean run, and nothing in the per-request stream says so.
    * ``effective_concurrency`` -- what the client actually sustained, as against what
      was offered. srt-slurm takes the offered value from the ``CONC`` environment
      variable, which never reaches any artifact.

    Its ABSENCE is itself a signal: AIPerf writes it at the end of a concurrency, so a
    run killed by its wall clock has none. Reference run 2750618 is exactly that case.

    Copied verbatim rather than parsed here, so the renderer reads one authority and
    this layer cannot silently reinterpret a schema it does not own.
    """
    pats = patterns or ["agentic/*/aiperf_artifacts/profile_export_aiperf.json",
                        "artifacts/*/profile_export_aiperf.json"]
    found = sorted(p for pat in pats for p in Path(run_dir).glob(pat))
    if not found:
        _log("L2 client-summary", "no profile_export_aiperf.json (run may have been cut "
                                  "short before AIPerf wrote its summary); no workload "
                                  "cache ceiling or validity flags will be shown")
        return None
    # Last by sort order = highest concurrency shard, matching the per-request leg.
    src = found[-1]
    shutil.copyfile(src, bundle / "profile_export_aiperf.json")
    _log("L2 client-summary", f"copied {src.name} from {src.parent.name}"
                              + (f" ({len(found)} shards, took the last)" if len(found) > 1 else ""))
    return str(src)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="directory of RAW L1 artifacts")
    p.add_argument("--out", "--bundle", dest="out", default=None,
                   help="output bundle dir (default: <run-dir>/ingest_bundle)")

    # yaml / topology
    p.add_argument("--name", default=None, help="dashboard.yaml name (default: run-dir basename)")
    p.add_argument("--description", default="", help="free-text header description")
    p.add_argument("--mode", choices=["agg", "disagg"], default="disagg")
    p.add_argument("--framework", choices=["trtllm", "vllm"], default="trtllm")
    p.add_argument("--block-size", type=int, default=512, help="topology.block_size fallback")
    p.add_argument("--worker", action="append", default=[], metavar="ROLE=PAR:RANK:COUNT",
                   help="topology worker pool, repeatable (e.g. prefill=dep:4:6 decode=tep:4:1)")

    # client axis
    p.add_argument("--client", choices=["aiperf", "none"], default="aiperf",
                   help="client source (aiperf->passthrough; the export is already schema 1)")
    p.add_argument("--client-input", default=None,
                   help="client input path/glob (default: artifacts/*/profile_export.jsonl)")

    # traces axis
    p.add_argument("--traces", choices=["spanlog", "none"], default="spanlog")
    p.add_argument("--span-logs", action="append", default=[], metavar="GLOB",
                   help="SPAN_CLOSED log path/glob, repeatable (default: *.out, srt-slurm worker/frontend logs)")

    # request-trace axis
    p.add_argument("--request-trace", choices=["dynamo", "none"], default="dynamo")
    p.add_argument("--request-trace-input", default=None,
                   help="dynamo-request-trace path/glob (default: dynamo-request-trace)")

    # per-iteration axis
    p.add_argument("--iter-log", choices=["trtllm", "none"], default="trtllm")
    p.add_argument("--iter-log-input", action="append", default=[], metavar="GLOB",
                   help="worker log path/glob carrying print_iter_log lines "
                        "(default: *_prefill_w*.out, *_decode_w*.out, *_agg_w*.out)")

    # metrics axis
    p.add_argument("--metrics", choices=["auto", "tachometer", "prometheus", "aiperf-jsonl", "aiperf-json", "none"],
                   default="auto",
                   help="metrics source; 'auto' prefers the tachometer parquet, then "
                        "raw_prometheus.jsonl, then AIPerf's per-scrape jsonl, then "
                        "AIPerf's aggregate json")
    p.add_argument("--raw-prometheus", default=None,
                   help="raw_prometheus.jsonl path/glob (default: raw_prometheus.jsonl)")
    p.add_argument("--tachometer-parquet", default=None,
                   help="tachometer parquet path/glob for --metrics tachometer "
                        "(default: tachometer/raw/scrape/final.parquet, then shards/local leftovers)")
    p.add_argument("--aiperf-server-metrics-jsonl", default=None,
                   help="AIPerf per-scrape server_metrics_export.jsonl path/glob for --metrics aiperf-jsonl "
                        "(default: agentic/*/ then artifacts/*/)")
    p.add_argument("--aiperf-server-metrics", default=None,
                   help="AIPerf server_metrics_export.json path/glob for --metrics aiperf-json "
                        "(default: agentic/*/ then artifacts/*/)")
    p.add_argument("--server-metrics", default=None,
                   help="pre-existing server_metrics_export.jsonl to pass through (with --metrics none)")

    p.add_argument("--jobs", type=int, default=4, help="parallelism for the SPAN_CLOSED pre-grep")
    return p


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        _log("L1", f"ERROR run-dir not found: {run_dir}")
        return 2
    bundle = Path(args.out).resolve() if args.out else run_dir / "ingest_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    name = args.name or run_dir.name
    workers = dict(parse_worker_spec(w) for w in args.worker)

    t0 = time.time()
    _log("L1", f"run-dir={run_dir}")
    _log("L1", f"bundle ={bundle}")

    have_aiperf = run_client(args, run_dir, bundle)
    profile_path = bundle / "profile_export.jsonl" if have_aiperf else None
    have_traces = run_traces(args, run_dir, bundle, profile_path)
    have_metrics = run_metrics(args, run_dir, bundle)
    have_req_trace = run_request_trace(args, run_dir, bundle)

    # Probed here, before the iteration log, for two reasons: it supplies the warmup
    # boundary for a bundle without the client leg, AND its time span is the fallback
    # UTC anchor for the iter-log offset. Skipped when the client leg is in the bundle
    # -- the renderer reads benchmark_phase off it directly and the metrics stream
    # already anchors the offset.
    warmup_ns = warmup_src = None
    client_lo = client_hi = None
    if not have_aiperf:
        warmup_ns, warmup_src, client_lo, client_hi = probe_warmup_end_ns(
            run_dir, list(CLIENT_PATTERNS))
        if warmup_ns:
            _log("L2 warmup", f"phase boundary {warmup_ns} ns ({warmup_src})")
        else:
            _log("L2 warmup", "no benchmark_phase tag found; no warmup marker")

    run_iter_log(args, run_dir, bundle, client_lo, client_hi)
    run_engine_configs(run_dir, bundle)
    run_client_summary(run_dir, bundle)
    run_provenance(run_dir, bundle)
    run_benchmark_status(run_dir, bundle)
    run_log_signals(run_dir, bundle)
    run_lifecycle(run_dir, bundle)
    run_host_samples(run_dir, bundle)

    yaml_text = generate_dashboard_yaml(
        name=name,
        description=args.description,
        mode=args.mode,
        framework=args.framework,
        block_size=args.block_size,
        workers=workers,
        have_aiperf=have_aiperf,
        have_traces=have_traces,
        have_metrics=have_metrics,
        have_request_trace=have_req_trace,
        warmup_end_ns=warmup_ns,
        warmup_source=warmup_src,
        metrics_source=getattr(args, "_metrics_source", None),
    )
    yaml_path = bundle / "dashboard.yaml"
    yaml_path.write_text(yaml_text)
    _log("L3-prep", f"wrote {yaml_path}")

    _log("done", f"bundle ready in {time.time() - t0:.1f}s: "
                  f"aiperf={have_aiperf} traces={have_traces} metrics={have_metrics} "
                  f"request_trace={have_req_trace}")
    _log("next", f"python3 -m src.visualization.build_dynamo_bench_dash {bundle} {bundle / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
