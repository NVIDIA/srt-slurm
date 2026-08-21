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
    lines.append("sources:")
    if have_aiperf:
        lines.append("  aiperf_profile: profile_export.jsonl")
    if have_traces:
        lines.append("  tempo_traces:   tempo_traces")
    if have_metrics:
        lines.append("  server_metrics: server_metrics_export.jsonl")
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
    patterns = [args.client_input] if args.client_input else [
        "agentic/*/aiperf_artifacts/profile_export.jsonl",
        "artifacts/*/profile_export.jsonl",
    ]
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


def run_iter_log(args, run_dir: Path, bundle: Path) -> bool:
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
    p.add_argument("--metrics", choices=["prometheus", "none"], default="prometheus")
    p.add_argument("--raw-prometheus", default=None,
                   help="raw_prometheus.jsonl path/glob (default: raw_prometheus.jsonl)")
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
    run_iter_log(args, run_dir, bundle)
    run_engine_configs(run_dir, bundle)
    run_client_summary(run_dir, bundle)
    run_provenance(run_dir, bundle)

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
