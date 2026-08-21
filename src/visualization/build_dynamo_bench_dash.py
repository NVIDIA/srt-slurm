#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo benchmark dashboard -- component observability (overview / router / engine /
frontend) rendered from an ingest bundle: profile_export.jsonl + tempo_traces/ +
server_metrics_export.jsonl, as produced by ``src/ingest/ingest.py``.

A single-file HTML with four component tabs, each a set of D3 panels. Charts use
FIXED viewBox widths (not clientWidth, which is 0 for panels built inside
display:none tabs), so hidden-tab panels lay out correctly.

Usage:
    python3 -m src.visualization.build_dynamo_bench_dash <bundle_dir> <out.html>
                                       [--d3 d3.v7.min.js | --d3-cdn]
                                       [--max-batch-prefill N] [--max-batch-decode N]
                                       [--frontend-log PATH]

<bundle_dir> is an ingest bundle (profile_export.jsonl, tempo_traces/,
server_metrics_export.jsonl, and optionally dashboard.yaml for the header labels).
D3 is inlined from the vendored sibling ``d3.v7.min.js`` by default, so the output
is a single self-contained file that survives being copied off the cluster or
synced to S3. ``--d3-cdn`` loads it from the CDN instead (smaller file, needs
network at view time).

Vendored from the ``dynamo-benchmark-perf-dashboard`` repo (commit
``22f49fea243e43403690b38e70a8d4092dec4cc8``). Comments below that cite
``dashboard.py``, ``render_fast.sh`` or ``src/common/*`` are provenance from that
repo -- those files are deliberately NOT vendored here (see
``docs/component-dashboard.md``); the reasoning they record still applies.
"""
import argparse
import glob
import json
import logging
import math
import os
from collections import Counter, defaultdict, deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDORED_D3 = os.path.join(_HERE, "d3.v7.min.js")

# Repo root, so `src.visualization` / `src.ingest` resolve whether this is run with
# -m or as a bare script path.
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from src.visualization.panels import evaluate as _evaluate_panels  # noqa: E402

_ap = argparse.ArgumentParser(description="Render the Dynamo benchmark component dashboard from an ingest bundle.")
_ap.add_argument("bundle_dir", nargs="?", help="ingest bundle dir (profile_export.jsonl, tempo_traces/, server_metrics_export.jsonl). Optional: omit for a log-only build.")
_ap.add_argument("out_html", nargs="?", help="output HTML path")
_ap.add_argument("--d3", default=None, help=f"path to d3.v7.min.js to inline (default: the vendored {os.path.basename(_VENDORED_D3)} beside this script)")
_ap.add_argument("--d3-cdn", action="store_true", help="load D3 from the CDN instead of inlining it (smaller file, needs network to view)")
_ap.add_argument("--max-batch-prefill", type=int, default=128, help="prefill max_batch_size ceiling for the in-flight panel")
_ap.add_argument("--max-batch-decode", type=int, default=256, help="decode max_batch_size ceiling for the in-flight panel")
_ap.add_argument("--include-warmup", action="store_true", help="keep AgentX cache-warmup requests (default: profiling phase only)")
_ap.add_argument("--gpus", type=int, default=None, help="total GPU count for tok/s/GPU (default: sum(rank x worker_count) from the bundle's dashboard.yaml)")
_ap.add_argument("--dump-json", default=None, metavar="PATH",
                 help="also write the page's underlying DATA payload as indented JSON. The HTML embeds "
                      "this same object, so the dump is the machine-readable form of everything the "
                      "dashboard shows -- for diffing two runs, for CI assertions, and for reading a "
                      "result on a machine with no browser.")
_ap.add_argument("--frontend-log", default=None, metavar="PATH",
                 help="Dynamo frontend log (INFO level) to populate the 'Log analysis' tab. "
                      "Accepts raw container stdout or sflow-wrapped. Tab is omitted if not given.")
_args = _ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
_log = logging.getLogger("dynamo-bench-dash")

# Two invocation shapes, so a run WITHOUT observability (no traces, no Prometheus
# scrape -- just a frontend log) still produces a dashboard:
#     <bundle_dir> <out.html> [--frontend-log L]   -> up to 5 tabs
#     <out.html> --frontend-log L                  -> Log analysis only
if _args.out_html is None:
    _args.bundle_dir, _args.out_html = None, _args.bundle_dir
if _args.out_html is None:
    _ap.error("an output HTML path is required")
if _args.bundle_dir is None and not _args.frontend_log:
    _ap.error("nothing to render: give an ingest bundle, --frontend-log, or both")

SRC, OUT = _args.bundle_dir, _args.out_html


# Engine ceilings for the in-flight panels. The run's own resolved engine config is
# authoritative; the --max-batch-* CLI flags are only a fallback for bundles that
# have none. Resolved HERE, before anything consumes them, because the ceiling is
# what the in-flight series is drawn against: on AgentX run 2739690 the real decode
# max_batch_size is 1 against a CLI default of 256, so a decode engine pinned at its
# limit rendered as 0.4% utilised. Prefill happening to match the default (128) is
# what makes that error easy to miss.
def _cfg_max_batch(mode):
    if not SRC:
        return None, None
    for cand in (os.path.join(SRC, f"trtllm_config_{mode}.yaml"),
                 os.path.join(SRC, "..", "..", f"trtllm_config_{mode}.yaml")):
        try:
            import yaml as _y
            c = _y.safe_load(open(cand)) or {}
            if c.get("max_batch_size") is not None:
                return int(c["max_batch_size"]), int(c.get("max_num_tokens") or 0)
        except Exception:
            pass
    return None, None
def _host_series():
    """Host and per-process health from ``host_series.json`` (PERF-37/38/40)."""
    if not SRC:
        return None
    try:
        with open(os.path.join(SRC, "host_series.json")) as fh:
            return json.load(fh) or None
    except Exception:
        return None


def _lifecycle():
    """Time-to-ready and terminal cause from ``run_lifecycle.json`` (PERF-45)."""
    if not SRC:
        return None
    try:
        with open(os.path.join(SRC, "run_lifecycle.json")) as fh:
            return json.load(fh) or None
    except Exception:
        return None


def _log_signals():
    """Log-only signals from ``log_signals.json``: the ones with no metric behind them.

    Returned whole, including zero counts. "This run had no worker crashes" is a
    statement; the signal simply being absent from the payload is not.
    """
    if not SRC:
        return None
    try:
        with open(os.path.join(SRC, "log_signals.json")) as fh:
            return (json.load(fh) or {}).get("signals") or None
    except Exception:
        return None


def _benchmark_status():
    """Why the benchmark produced what it produced, from ``benchmark_status.json``.

    An empty dashboard is not wrong, but it is mute: `client=False traces=False N=0`
    leaves the reader to guess between a dead deployment, a workload that never
    started, and a run cut short -- three different responses. Eight ablation arms in
    one session hit exactly this, four failing before the benchmark script was even
    reachable and four eleven seconds after the workers went healthy.
    """
    if not SRC:
        return None
    for cand in (os.path.join(SRC, "benchmark_status.json"),):
        try:
            with open(cand) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not d.get("exit_code") and not d.get("errors"):
            return None
        return {"exit_code": d.get("exit_code"),
                # 6, not 4: a check_env_vars failure is a header plus one line per
                # missing variable, and truncating to 4 drops the variables.
                "errors": (d.get("errors") or [])[:6],
                # AIPerf's own completed/cancelled census per phase. Frequently the ONLY
                # account of what happened: arm 2752189 exited 1 with zero error-marker
                # lines in benchmark.out, so without this the banner could say nothing
                # beyond the exit code.
                "phases": (d.get("phases") or [])[:4]}
    return None


def _flat_config():
    """The run's resolved recipe, flattened to {dotted.path: scalar}.

    Flattened rather than cherry-picked because the interesting key is whichever one
    an experiment happened to move, and a hand-maintained list of "settings worth
    recording" is guaranteed to omit exactly that one the first time it matters.

    This is also the only record of the FRONTEND environment. The worker fingerprints
    capture prefill and decode, but settings like DYN_TOKENIZER_CACHE live on the
    frontend and appear in no fingerprint at all -- so without this, an ablation on a
    frontend variable has no artifact proving what it changed.

    `name` is dropped: every arm of an ablation renames itself, and leaving it in makes
    a clean single-variable diff report two differences instead of one.
    """
    if not SRC:
        return None
    for cand in (os.path.join(SRC, "config.yaml"),
                 os.path.join(SRC, "..", "..", "config.yaml")):
        try:
            import yaml as _y
            with open(cand) as fh:
                doc = _y.safe_load(fh) or {}
        except Exception:
            continue
        flat = {}

        def _walk(node, prefix=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    _walk(v, f"{prefix}{k}.")
            elif isinstance(node, list):
                # Joined rather than indexed: a reordered list is not a config change,
                # and per-index keys would report one for every shifted element.
                flat[prefix.rstrip(".")] = ",".join(str(x) for x in node)
            else:
                flat[prefix.rstrip(".")] = node
        _walk(doc)
        # Identity fields, dropped so a comparison reports TREATMENTS and not
        # bookkeeping. Every arm of an ablation necessarily renames itself and writes to
        # its own result file; leaving these in means a genuinely single-variable
        # comparison always reports two differences, and the SINGLE-VARIABLE verdict can
        # never fire -- which trains the reader to ignore the count.
        for k in ("name", "benchmark.env.RESULT_FILENAME"):
            flat.pop(k, None)
        return flat
    return None


def _provenance():
    """What actually ran: framework versions, GPU, and per-worker agreement.

    Container tags routinely disagree with the wheels they bundle -- an image tagged
    1.1.0-rc3 shipping tensorrt_llm 1.3.0rc11 is an observed case. The fingerprints
    record the versions found INSIDE the running worker, which is the only trustworthy
    answer, and they are per-worker, so a deployment whose prefill and decode landed on
    different builds is visible here and nowhere else. That mismatch is silent in every
    other panel: both halves run, and the numbers simply mean something different than
    the reader assumes.

    Returns None when no fingerprints reached the bundle, rather than a partial record
    that would read as agreement.
    """
    if not SRC:
        return None
    fps = sorted(glob.glob(os.path.join(SRC, "fingerprint_*.json")))
    if not fps:
        return None
    per_worker, frameworks = {}, {}
    for path in fps:
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue
        worker = os.path.basename(path)[len("fingerprint_"):-len(".json")]
        fw = d.get("frameworks") or {}
        per_worker[worker] = {
            "frameworks": fw,
            "cuda": d.get("cuda_version"), "nccl": d.get("nccl_version"),
            "python": d.get("python_version"), "gpu": d.get("gpu"),
            "hostname": d.get("hostname"),
        }
        for k, v in fw.items():
            frameworks.setdefault(k, set()).add(v)
    if not per_worker:
        return None
    cfg = _flat_config()
    # A framework name mapping to more than one version means the workers are not
    # running the same build. Reported as a first-class disagreement rather than
    # collapsed to "the version", which would silently pick one arbitrarily.
    disagree = {k: sorted(v) for k, v in frameworks.items() if len(v) > 1}
    return {"workers": per_worker,
            "frameworks": {k: sorted(v)[0] for k, v in frameworks.items()},
            "framework_disagreement": disagree or None,
            "n_workers": len(per_worker),
            "config": cfg}


def _client_summary():
    """AIPerf's run-level summary: the workload's cache ceiling, and run validity.

    Returns a compact dict, or None when AIPerf never wrote one -- which happens when
    a run is killed before the concurrency finishes, and is itself worth reporting
    rather than papering over.

    `theoretical_prefix_cache_hit` is the number that makes the engine's measured
    reuse interpretable. The engine hit rate alone is a bare figure that each reader
    scores against a private expectation; against the ceiling the workload actually
    offered, it becomes a gap with a size.
    """
    if not SRC:
        return None
    for cand in (os.path.join(SRC, "profile_export_aiperf.json"),
                 os.path.join(SRC, "..", "..", "profile_export_aiperf.json")):
        try:
            with open(cand) as fh:
                d = json.load(fh)
        except Exception:
            continue
        bs = d.get("branch_stats") or {}
        # `avg` is AIPerf's own aggregate for the whole concurrency; the nested shape
        # ({unit, avg, count, sum}) is stable across the 1.x schema.
        def _avg(key):
            v = d.get(key)
            return v.get("avg") if isinstance(v, dict) else v
        return {
            "schema": d.get("schema_version"),
            "aiperf_version": d.get("aiperf_version"),
            "theoretical_prefix_cache_hit": _avg("theoretical_prefix_cache_hit"),
            "effective_concurrency": _avg("effective_concurrency"),
            "request_count": _avg("request_count"),
            "was_cancelled": d.get("was_cancelled"),
            "n_errors": len(d.get("error_summary") or []),
            "branch_errored": bs.get("children_errored"),
            "branch_truncated": bs.get("children_truncated"),
            "branch_spawned": bs.get("children_spawned"),
        }
    return None


def _cfg_tokens_per_block():
    """Engine KV block size from the run's own config, prefill or decode.

    Either mode answers: the two engines pool the same KV block size, and a run may
    ship only one of the two configs. Nested under `kv_cache_config` in current
    TRT-LLM configs, top-level in older ones, so both are accepted.
    """
    if not SRC:
        return None
    for mode in ("decode", "prefill"):
        for cand in (os.path.join(SRC, f"trtllm_config_{mode}.yaml"),
                     os.path.join(SRC, "..", "..", f"trtllm_config_{mode}.yaml")):
            try:
                import yaml as _y
                c = _y.safe_load(open(cand)) or {}
                v = (c.get("kv_cache_config") or {}).get("tokens_per_block",
                                                         c.get("tokens_per_block"))
                if v:
                    return int(v)
            except Exception:
                pass
    return None
CLIENT_SUMMARY = _client_summary()
PROVENANCE = _provenance()
BENCH_STATUS = _benchmark_status()
LOG_SIGNALS = _log_signals()
LIFECYCLE = _lifecycle()
HOST = _host_series()
if HOST:
    _pk = max((v for _, v in (HOST.get("host_cpu_pct") or [])), default=None)
    _log.info(f"host telemetry: {HOST.get('samples')} samples on {HOST.get('host')}, "
              f"peak CPU {_pk}%, fd headroom {HOST.get('fd_headroom_pct')}% of "
              f"{HOST.get('fd_limit')}, {len(HOST.get('procs') or {})} process(es)")
else:
    _log.info("host telemetry: absent (host_series.json not in the bundle)")
if LIFECYCLE:
    _d = LIFECYCLE.get("durations_s") or {}
    _log.info(f"lifecycle: time_to_ready={_d.get('time_to_ready')}s "
              f"readiness_gap={_d.get('readiness_gap')}s | {LIFECYCLE.get('terminal_cause')}")
if LOG_SIGNALS:
    _fired = {k: v["count"] for k, v in LOG_SIGNALS.items() if v.get("count")}
    _log.info(f"log-only signals: {_fired}" if _fired
              else "log-only signals: none fired (all counts zero)")
if BENCH_STATUS:
    _log.warning(f"benchmark reported failure: exit={BENCH_STATUS['exit_code']}; "
                 f"{BENCH_STATUS['errors'][0] if BENCH_STATUS['errors'] else 'no error line captured'}")
if PROVENANCE:
    _log.info(f"provenance: {PROVENANCE['n_workers']} worker fingerprint(s), frameworks="
              f"{PROVENANCE['frameworks']}")
    if PROVENANCE["framework_disagreement"]:
        _log.warning(f"workers are NOT running the same build: "
                     f"{PROVENANCE['framework_disagreement']}")
else:
    _log.info("provenance: no fingerprint_*.json in the bundle; framework versions "
              "unknown and two bundles cannot be compared for what differed")
if CLIENT_SUMMARY:
    _log.info(f"client summary: workload prefix-cache ceiling "
              f"{CLIENT_SUMMARY['theoretical_prefix_cache_hit']}%, effective concurrency "
              f"{CLIENT_SUMMARY['effective_concurrency']}, {CLIENT_SUMMARY['n_errors']} error(s), "
              f"cancelled={CLIENT_SUMMARY['was_cancelled']}")
else:
    _log.info("client summary: no profile_export_aiperf.json in the bundle -- no workload "
              "cache ceiling to compare the engine's measured reuse against")
CFG_PF_BATCH, CFG_PF_TOK = _cfg_max_batch("prefill")
CFG_DE_BATCH, CFG_DE_TOK = _cfg_max_batch("decode")
MAX_BATCH_PF = CFG_PF_BATCH if CFG_PF_BATCH else _args.max_batch_prefill
MAX_BATCH_DE = CFG_DE_BATCH if CFG_DE_BATCH else _args.max_batch_decode
_log.info(f"engine ceilings: prefill max_batch_size={MAX_BATCH_PF} max_num_tokens={CFG_PF_TOK} | "
          f"decode max_batch_size={MAX_BATCH_DE} max_num_tokens={CFG_DE_TOK} "
          f"(source: {'run config' if CFG_DE_BATCH else '--max-batch-* defaults'})")

# Per-source availability. A tab is rendered only when its inputs exist -- an
# empty tab is worse than an absent one, because it reads as "this run had no
# queueing" rather than "this run was not instrumented".
def _have(*parts):
    return bool(SRC) and os.path.exists(os.path.join(SRC, *parts))
HAS_CLIENT = _have("profile_export.jsonl")
HAS_TRACES = bool(SRC) and bool(glob.glob(os.path.join(SRC, "tempo_traces", "*.json")))
HAS_SM = _have("server_metrics_export.jsonl")
HAS_RT = _have("request_trace.jsonl")
_log.info(f"sources: client={HAS_CLIENT} traces={HAS_TRACES} server_metrics={HAS_SM} "
          f"request_trace={HAS_RT} frontend_log={bool(_args.frontend_log)}")

# Loaded here, before any aggregate is computed, because the run-level waterfall needs
# it: KV transfer is a phase no span reports and no Prometheus series samples, so
# without this the waterfall has a silent hole between prefill and decode compute --
# on the reference runs that hole is the median 79% of the decode span.
rt_all=[]
if HAS_RT:
    for _l in open(os.path.join(SRC,"request_trace.jsonl")):
        _l=_l.strip()
        if _l: rt_all.append(json.loads(_l))
RT_BY_XID={r["x_request_id"]:r for r in rt_all if r.get("x_request_id")}


def _load_meta(bundle_dir):
    """Header labels (run name + topology summary) and the run's GPU count from the
    bundle's dashboard.yaml, falling back to the bundle dir name when the yaml is
    absent/unreadable. `rank` is GPUs-per-worker (TEP/DEP)."""
    if not bundle_dir:                       # log-only build: no bundle to read
        return "", "", None
    src = os.path.basename(os.path.normpath(bundle_dir)) or bundle_dir
    topo = ""
    gpus = None
    ypath = os.path.join(bundle_dir, "dashboard.yaml")
    if os.path.exists(ypath):
        try:
            import yaml
            cfg = yaml.safe_load(open(ypath)) or {}
            src = cfg.get("name", src)
            t = cfg.get("topology") or {}
            parts = [f"{role} {w.get('worker_count','?')}×{w.get('parallelism','?')}{w.get('rank','')}"
                     for role, w in (t.get("workers") or {}).items()]
            # block_size from the yaml is only ingest's --block-size fallback. The
            # authoritative value comes from the metrics stream and is substituted
            # later; emitting the fallback here would put a third, wrong number in the
            # header while the panels use the right one.
            if t.get("block_size"):
                parts.append("block __BLK__")
            topo = " · ".join(parts)
            gpus = sum(int(w.get("rank") or 0) * int(w.get("worker_count") or 0)
                       for w in (t.get("workers") or {}).values()) or None
        except Exception:
            pass
    return src, topo, gpus


META_SRC, META_TOPO, META_GPUS = _load_meta(SRC)
GPUS = _args.gpus or META_GPUS or 1
if not (_args.gpus or META_GPUS):
    _log.warning("no GPU count from --gpus or dashboard.yaml topology; tok/s/GPU falls back to 1 GPU")

def pct(v, p):
    if not v: return 0.0
    v = sorted(v); k=(len(v)-1)*p/100.0; lo=int(k); hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(k-lo)

# ============================ 1. traces + client ============================
# AgentX writes its cache-warmup phase into the SAME profile_export.jsonl as the
# measured phase, tagged metadata.benchmark_phase. Including it skews every
# percentile: on run 2674375 warmup was 1180 requests at p99 43.5s vs profiling
# 918 at p99 88.5s -- mixing them roughly halves the reported tail. Records with
# no benchmark_phase (plain, non-AgentX AIPerf runs) are always kept.
# The phase boundary is captured HERE rather than re-derived, because this loop is
# already the only place that sees warmup records -- the filter below drops them.
# Preferred boundary is the first PROFILING request's start; the last warmup end is
# the fallback for a run whose profiling phase never began.
client={}; _skipped_phase=0
_phase_seen={}   # every phase value in the file, kept or dropped
_prof_min_start=None; _warm_max_end=None
for l in (open(os.path.join(SRC,"profile_export.jsonl")) if HAS_CLIENT else []):
    l=l.strip()
    if not l: continue
    r=json.loads(l); m=r["metadata"]; mt=r["metrics"]
    _ph=m.get("benchmark_phase")
    _phase_seen[_ph or "(none)"]=_phase_seen.get(_ph or "(none)",0)+1
    if _ph=="profiling":
        _s=m.get("request_start_ns")
        if _s: _prof_min_start=_s if _prof_min_start is None else min(_prof_min_start,_s)
    elif _ph is not None:
        _e=m.get("request_end_ns") or m.get("request_start_ns")
        if _e: _warm_max_end=_e if _warm_max_end is None else max(_warm_max_end,_e)
    if not _args.include_warmup and _ph not in (None,"profiling"):
        _skipped_phase+=1; continue
    g=lambda k:(mt.get(k) or {}).get("value")
    client[m["x_request_id"]]={"start_ns":m.get("request_start_ns") or 0,"ttft":g("time_to_first_token"),
        "itl":g("inter_token_latency"),"isl":g("input_sequence_length"),"osl":g("output_sequence_length"),
        "e2e":g("request_latency")}
def dms(s): return (int(s["endTimeUnixNano"])-int(s["startTimeUnixNano"]))/1e6
def at(s,k):
    for a in s.get("attributes",[]):
        if a["key"]==k: return a["value"].get("stringValue")
    return None
rows=[]; hp_ev={"prefill":[],"decode":[]}; trace_start=None; trace_end=None
spans_raw={}   # xid -> raw span list, kept for the per-request drill-down (§4)
for f in (glob.glob(os.path.join(SRC,"tempo_traces","*.json")) if HAS_TRACES else []):
    xid=os.path.basename(f)[:-5]; c=client.get(xid)
    if not c: continue
    sp=json.load(open(f))["batches"][0]["scopeSpans"][0]["spans"]
    spans_raw[xid]=sp
    by={}
    for s in sp: by.setdefault(s["name"],[]).append(s)
    def mx(name,phase=None,comp=None):
        best=None
        for s in by.get(name,[]):
            if phase and at(s,"phase")!=phase: continue
            if comp and at(s,"component")!=comp: continue
            d=dms(s)
            if best is None or d>best[0]: best=(d,s)
        return best
    sched=mx("kv_router.schedule"); rr_pf=mx("kv_router.route_request",phase="Prefill")
    rr_de=mx("kv_router.route_request",phase="Decode"); hp_pf=mx("handle_payload",comp="prefill")
    hp_de=mx("handle_payload",comp="backend"); http=mx("http-request"); selw=mx("kv_router.select_worker")
    hashm=sum(dms(s) for s in by.get("kv_router.compute_block_hashes",[])+by.get("kv_router.compute_seq_hashes",[]))
    findm=sum(dms(s) for s in by.get("kv_router.find_matches",[]))
    routem=(rr_pf[0] if rr_pf else 0)+(rr_de[0] if rr_de else 0)
    rows.append({"xid":xid,"ts_ns":c["start_ns"],"ttft":c["ttft"] or 0,"e2e":c["e2e"] or 0,"isl":c["isl"] or 0,
        "osl":c["osl"] or 0,"adm":sched[0] if sched else 0,"pf":hp_pf[0] if hp_pf else 0,
        "de":hp_de[0] if hp_de else 0,"rhash":hashm,"rfind":findm,"rroute":routem,
        "ov":int(at(rr_pf[1],"overlap_blocks") or 0) if rr_pf else 0,
        "wpf":at(rr_pf[1],"worker_id") if rr_pf else None,
        # dp_rank off the SAME span as overlap_blocks. The router's free-text selector
        # line carries a richer cost breakdown but has no request_id on it, so it
        # cannot be joined per request -- this span is the joinable substitute.
        "rank":at(rr_pf[1],"dp_rank") if rr_pf else None})
    # Requests-in-flight occupancy window per worker role: min(start)..max(end) across
    # EVERY handle_payload on that component, not just the longest one (mx() above).
    # This matches gpu_occupancy.compute_requests_in_flight, which spans the envelope so
    # a request split across several spans still counts as occupying the worker throughout.
    for _comp,_role in (("prefill","prefill"),("backend","decode")):
        _ss=[s for s in by.get("handle_payload",[]) if at(s,"component")==_comp]
        if _ss: hp_ev[_role] += [(min(int(s["startTimeUnixNano"]) for s in _ss),1),
                                 (max(int(s["endTimeUnixNano"])   for s in _ss),-1)]
    if http:
        a=int(http[1]["startTimeUnixNano"]); b=int(http[1]["endTimeUnixNano"])
        trace_start=a if trace_start is None else min(trace_start,a); trace_end=b if trace_end is None else max(trace_end,b)
rows.sort(key=lambda r:r["ts_ns"]); N=len(rows)
if _skipped_phase: _log.info(f"dropped {_skipped_phase} non-profiling (warmup) requests; {len(client)} kept")

# ============================ 2. server_metrics stream ======================
scr=[]  # list of (ts_ns, metrics_dict)
if HAS_SM:
    for l in open(os.path.join(SRC,"server_metrics_export.jsonl")):
        l=l.strip()
        if not l: continue
        d=json.loads(l); scr.append((d["timestamp_ns"], d["metrics"]))
    scr.sort(key=lambda x:x[0])
sm_start=scr[0][0] if scr else None
sm_end=scr[-1][0] if scr else None

# The run window is whichever source actually exists. The iter log has to be in
# here, not just traces/scrapes: it is the ONLY leg a trtllm-serve run produces (no
# Dynamo frontend means no /metrics surface and no spans at all), and it is present
# on any run whose recipe sets `print_iter_log` regardless of `observability.enabled`.
# Left out, `run_dur` collapses to 1 ns, the `t > run_dur + 60` guard below discards
# every bin, and a run with 175,800 real iterations renders as an empty page while
# only logging "0pts" -- which is how it read before this was fixed.
def _load_iter_bins(src):
    p=os.path.join(src,"iter_bins.json") if src else ""
    if not os.path.exists(p): return None
    try: return json.load(open(p))
    except Exception: return None

# Loaded ONCE here, before the run window, because three separate consumers need it
# and two of them run earlier than the panel section: the window itself, and the KV
# fallbacks below.
_IB=_load_iter_bins(SRC)

def _iter_bins_window(ib):
    if not ib: return None,None
    ts=[r["t"] for rows in (ib.get("bins") or {}).values() for r in rows if r.get("t")]
    if not ts: return None,None
    return int(min(ts)*10**9), int(max(ts)*10**9)

_it_start,_it_end=_iter_bins_window(_IB)
_starts=[t for t in (trace_start, sm_start, _it_start) if t is not None]
_ends=[t for t in (trace_end, sm_end, _it_end) if t is not None]
run_start=min(_starts) if _starts else 0
run_end=max(_ends) if _ends else run_start+1
run_dur=(run_end-run_start)/1e9

def relt(ns): return round((ns-run_start)/1e9,1)
for r in rows: r["ts"]=relt(r["ts_ns"]);

# ---- end of the load generator's warmup phase -----------------------------------
# Optional, and deliberately so: many harnesses have no warmup phase at all, and a
# marker invented for them would be a lie. Resolved once here, from the highest-
# confidence source available, and carried as an OPTIONAL field -- absent means "no
# warmup boundary is known", never "warmup ended at t=0".
#
# Precedence:
#   1. the client export's own `benchmark_phase` (authoritative; the harness said so)
#   2. `warmup_end_ns` recorded in dashboard.yaml by ingest, for bundles built
#      without the client leg -- the export is routinely ~700 MB and is not copied
#   3. nothing
def _yaml_warmup_ns(bundle_dir):
    if not bundle_dir: return None
    ypath=os.path.join(bundle_dir,"dashboard.yaml")
    if not os.path.exists(ypath): return None
    try:
        import yaml as _y
        return ((_y.safe_load(open(ypath)) or {}).get("warmup") or {}).get("end_ns")
    except Exception:
        return None

WARMUP_END_S=None; WARMUP_SRC=None
_w_ns=_prof_min_start or _warm_max_end
if _w_ns: WARMUP_SRC="profile_export.jsonl (benchmark_phase)"
else:
    _w_ns=_yaml_warmup_ns(SRC)
    if _w_ns: WARMUP_SRC="dashboard.yaml (warmup.end_ns)"
if _w_ns:
    _v=relt(int(_w_ns))
    # A boundary outside the rendered window is not plottable and would draw a line
    # on the axis edge implying the whole run was one phase. Dropped, with a warning.
    if 0 < _v < run_dur:
        WARMUP_END_S=_v
        _log.info(f"warmup ends at t+{_v:.1f}s of {run_dur:.0f}s "
                  f"({_v/run_dur*100:.0f}% of the run) — source: {WARMUP_SRC}")
    else:
        _log.warning(f"warmup boundary t+{_v:.1f}s falls outside the run window "
                     f"(0..{run_dur:.0f}s); no marker drawn")
        WARMUP_SRC=None
# ---- metric extractors ----
def gsingle(m,name):
    a=m.get(name)
    return a[0]["value"] if a else None
def _entries(m, name):
    """Entries for a metric family in one scrape, tolerating the `_total` suffix.

    Prometheus counters conventionally end in `_total`, and the ingest layer keeps the
    scraped name verbatim -- so a lookup written without the suffix silently finds
    nothing and the caller reads a hard zero. That is not hypothetical: five families
    here (`dynamo_frontend_requests` and all four `dynamo_frontend_tokenizer_cache_*`)
    were being read without it, which pinned the tokenizer-cache KPI to 0.0% on a run
    whose real hit rate was ~99%.

    Resolving centrally rather than fixing each call site, because the failure is
    invisible at the call site: a missing family and an all-zero family look identical
    downstream.
    """
    return m.get(name) or m.get(name + "_total")


def ggroup(m,name,by="dynamo_component"):
    out=defaultdict(list)
    for e in (_entries(m,name) or []):
        out[e.get("labels",{}).get(by)].append(e["value"])
    return out
def hsc(m,name):
    """(sum,count) for a single-series histogram, across both on-disk shapes.

    `src/ingest` parses a Prometheus histogram into ONE structured entry that
    carries `sum`/`count`/`buckets` alongside its labels. Raw text scraped
    without that parsing instead yields two flat families, `<name>_sum` and
    `<name>_count`. Reading only the flat form silently returns (None,None) on
    every modern bundle -- which renders as an all-empty panel rather than an
    error, so prefer the structured form and keep the flat one as fallback.
    """
    s=c=None
    for e in m.get(name,[]):
        if "sum" in e:   s=e["sum"]   if s is None else s+e["sum"]
        if "count" in e: c=e["count"] if c is None else c+e["count"]
    if s is not None or c is not None:
        return s,c
    fs=m.get(name+"_sum"); fc=m.get(name+"_count")
    return (fs[0]["value"] if fs else None, fc[0]["value"] if fc else None)
def hsc_by(m,name,by="stage"):
    S={}; C={}
    for e in m.get(name,[]):
        if "sum" not in e and "count" not in e: continue
        k=e.get("labels",{}).get(by)
        if "sum" in e:   S[k]=S.get(k,0)+e["sum"]
        if "count" in e: C[k]=C.get(k,0)+e["count"]
    if S or C: return S,C
    for e in m.get(name+"_sum",[]): S[e["labels"].get(by)]=e["value"]
    for e in m.get(name+"_count",[]): C[e["labels"].get(by)]=e["value"]
    return S,C

def peak_with(m_name,key="count"):
    """The scrape holding the largest cumulative value of `m_name`.

    Two reasons not to just use `scr[-1]`: the stream interleaves one line per
    endpoint, so the final line is whichever endpoint was scraped last and often
    lacks the metric entirely; and components zero their counters as they shut
    down, so the run's last frontend scrape reports 0. The peak scrape is the
    true end-of-run total.
    """
    best={}; bv=-1.0
    for _,m in scr:
        ents=_entries(m,m_name)
        if not ents: continue
        tot=0.0
        for e in ents:
            v=e.get(key,e.get("value"))
            if v is not None: tot+=v
        if tot>bv: bv=tot; best=m
    return best

def times(): return [relt(ts) for ts,_ in scr]
T=times()

def gauge_series(name,agg="single",by="dynamo_component",group=None,scale=1.0):
    out=[]
    for (ts,m) in scr:
        if agg=="single":
            v=gsingle(m,name); out.append([relt(ts), None if v is None else round(v*scale,3)])
        elif agg in ("max","mean","sum"):
            g=ggroup(m,name,by); vals=g.get(group,[]) if group is not None else [x for vs in g.values() for x in vs]
            if not vals: out.append([relt(ts),None]); continue
            v={"max":max,"mean":lambda a:sum(a)/len(a),"sum":sum}[agg](vals)
            out.append([relt(ts),round(v*scale,3)])
    return out

def hist_mean_series(name,by=None,group=None,scale_ms=1000.0):
    """interval-mean = Δsum/Δcount between scrapes, in ms (scale_ms: sum in seconds→ms=1000; _ms metrics→1)."""
    out=[]; prev=None
    for (ts,m) in scr:
        if by:
            S,C=hsc_by(m,name,by); s=S.get(group); c=C.get(group)
        else:
            s,c=hsc(m,name)
        if prev and s is not None and c is not None and prev[0] is not None and c>prev[1]:
            out.append([relt(ts), round((s-prev[0])/(c-prev[1])*scale_ms,3)])
        else:
            out.append([relt(ts),None])
        if s is not None and c is not None: prev=(s,c)
    return out

def _csum(m,name,label=None,val=None):
    """Total of a counter/gauge family in one scrape, or None if absent here."""
    ents=_entries(m,name)
    if not ents: return None
    tot=0.0
    for e in ents:
        if label is not None and e.get("labels",{}).get(label)!=val: continue
        v=e.get("value")
        if v is not None: tot+=v
    return tot

def counter_rate(name,label=None,val=None,window_s=30.0):
    """Δvalue/Δt for a cumulative counter, in units/second, over a trailing window.

    Two things this guards against:

    * Scrapes that do not carry `name` are skipped, NOT read as zero. The stream
      interleaves one line per endpoint, so a frontend-only counter is missing
      from all four worker lines; treating those as 0 manufactures a
      reset-then-respike sawtooth whose peaks are pure artefact.
    * `window_s` fixes rate quantisation. This bundle scrapes every ~0.3 s while
      requests arrive at ~0.5/s, so a *consecutive-scrape* difference is 0 in
      most intervals and 1/0.3 = 3.33 in the rest -- a staircase of the sampling
      grid rather than a throughput curve. Differencing across a trailing window
      recovers the real rate (this is Prometheus' rate()[w] idiom).
    """
    hist=[]; out=[]
    for (ts,m) in scr:
        rt=relt(ts); tot=_csum(m,name,label,val)
        if tot is None: out.append([rt,None]); continue
        hist.append((rt,tot))
        cut=rt-window_s
        while len(hist)>2 and hist[1][0]<cut: hist.pop(0)
        t0,v0=hist[0]
        out.append([rt, round(max(0.0,tot-v0)/(rt-t0),3)] if rt>t0 else [rt,None])
    return out

def counter_cum(name,label=None,val=None):
    """Raw cumulative counter value per scrape (absent scrape -> None)."""
    return [[relt(ts), _csum(m,name,label,val)] for ts,m in scr]

def hist_final_mean_by(name,by="stage",scale_ms=1000.0):
    S,C=hsc_by(peak_with(name),name,by)
    return sorted(([k, round(S[k]/C[k]*scale_ms,3)] for k in S if C.get(k)), key=lambda x:-x[1])

def hist_final_mean(name,scale_ms=1000.0):
    s,c=hsc(peak_with(name),name)
    return round(s/c*scale_ms,3) if s is not None and c else None

# ---- FRONTEND server_metrics ----
# NOTE ON NAMES: the raw Prometheus text publishes these counters with a
# `_total` suffix (dynamo_frontend_requests_total, ..._tokenizer_cache_hits_total).
# `src/ingest` applies the OpenMetrics convention of stripping it, so the names
# below -- which read the INGESTED bundle -- are deliberately suffix-free.
fe = {
  # Panel 1 "Requests". requests_total is cumulative; it is plotted as a rate so
  # it shares one linear axis with the three instantaneous gauges beside it.
  "req_rate":  counter_rate("dynamo_frontend_requests"),
  "inflight":  gauge_series("dynamo_frontend_inflight_requests",agg="sum",group=None),
  "queued":    gauge_series("dynamo_frontend_queued_requests",agg="sum",group=None),
  "active":    gauge_series("dynamo_frontend_active_requests",agg="sum",group=None),
  # Panel 2 "Tokenizer": the four L1 prefix-cache counters, raw cumulative.
  "tk_hits":       counter_cum("dynamo_frontend_tokenizer_cache_hits"),
  "tk_misses":     counter_cum("dynamo_frontend_tokenizer_cache_misses"),
  "tk_cached":     counter_cum("dynamo_frontend_tokenizer_cache_cached_tokens"),
  "tk_uncached":   counter_cum("dynamo_frontend_tokenizer_cache_uncached_tokens"),
}
# Not plotted on this tab any more, but still feeds the shared KPI strip.
def cval(name):
    # Resolve through _entries on BOTH lookups: peak_with finds the right scrape, but
    # indexing the returned scrape by the un-suffixed name misses it a second time.
    a=_entries(peak_with(name,key="value"), name)
    return a[0]["value"] if a else 0
tk_h=cval("dynamo_frontend_tokenizer_cache_hits"); tk_m=cval("dynamo_frontend_tokenizer_cache_misses")
fe["tok_cache_hit_pct"]=round(100*tk_h/max(1,tk_h+tk_m),1)

# ---- ROUTER server_metrics ----
ro = {
  # Panel 1 "Router overhead": histogram -> interval mean (Δsum/Δcount), ms.
  "oh_total_ms":   hist_mean_series("dynamo_router_overhead_total_ms",scale_ms=1.0),
  # Panel 2 "Router queue": gauge, summed over worker_type/policy_class labels.
  "queue_pending": gauge_series("dynamo_frontend_router_queue_pending_requests",agg="sum",group=None),
}

# ---- ENGINE server_metrics ----
en = {
  # gpu_cache_usage_percent is a 0-1 FRACTION per dp-rank; peak worker (max across ranks) ×100 = the KV util %.
  # (Mean across ranks would be idle-rank-diluted toward 0, so use max = the busiest worker.)
  "kvutil_pf": gauge_series("dynamo_component_gpu_cache_usage_percent",agg="max",by="dynamo_component",group="prefill",scale=100),
  "kvutil_de": gauge_series("dynamo_component_gpu_cache_usage_percent",agg="max",by="dynamo_component",group="backend",scale=100),
  "inflight_pf": gauge_series("dynamo_component_inflight_requests",agg="mean",by="dynamo_component",group="prefill"),
  "inflight_de": gauge_series("dynamo_component_inflight_requests",agg="mean",by="dynamo_component",group="backend"),
  "max_num_seqs": (gsingle(scr[-1][1],"dynamo_frontend_model_max_num_seqs") if scr else None) or 128,
  "max_batch_pf": MAX_BATCH_PF,   # prefill max_batch_size ceiling (--max-batch-prefill)
  "max_batch_de": MAX_BATCH_DE,   # decode max_batch_size ceiling (--max-batch-decode)
}
# true cache-hit (trtllm_kv_cache_hit_rate gauge, per ctx worker) — mean over scrapes where present
def _reuse_enabled_components():
    """Components whose engine config actually enables KV block reuse.

    A worker with ``kv_cache_config.enable_block_reuse: false`` reports a hard 0 for
    trtllm_kv_cache_hit_rate -- correctly, because it does no reuse. Averaging that 0
    in with the components that DO reuse does not produce a run-level hit rate; it
    produces a number that falls as the deployment adds decode workers. On the
    reference 1P3D run the true prefill hit rate is ~0.65-0.76 while the naive mean
    over four components reports ~0.16.

    Returns None when no engine config is available, which means "cannot tell" -- the
    caller then keeps every component rather than silently dropping data.
    """
    keep=set()
    seen_cfg=False
    for mode,comp in (("prefill","prefill"),("decode","backend"),("aggregated","backend")):
        for cand in (os.path.join(SRC or "", f"trtllm_config_{mode}.yaml"),):
            if not cand or not os.path.exists(cand): continue
            try:
                import yaml as _y
                c=_y.safe_load(open(cand)) or {}
            except Exception:
                continue
            seen_cfg=True
            if ((c.get("kv_cache_config") or {}).get("enable_block_reuse")) is not False:
                keep.add(comp)
    return keep if seen_cfg else None

_REUSE_COMPONENTS=_reuse_enabled_components()
if _REUSE_COMPONENTS is not None:
    _log.info(f"KV hit rate: averaging only components with block reuse enabled: "
              f"{sorted(_REUSE_COMPONENTS) or 'NONE'}")
else:
    _log.warning("KV hit rate: no engine config in the bundle, so reuse-disabled "
                 "components cannot be excluded; the series may understate the true rate")

def _iter_reuse_series():
    """Context-phase block reuse per 1 s bin, from the per-iteration log.

    A fallback for ``trtllm_kv_cache_hit_rate``, which only exists if a WORKER
    ``/metrics`` endpoint was scraped. It frequently was not -- AIPerf's own export
    scrapes the frontend only, and a trtllm-serve run has no Dynamo endpoint at all --
    and the panel then drew a full-length series of nulls, i.e. an empty chart that
    reads as "reuse is zero" rather than "reuse was not captured".

    ``cached_kv_tokens / (cached_kv_tokens + num_ctx_tokens)`` over context workers.
    Validated against the client's own Prompt Cache Read % on three runs of the same
    point: 95.88 / 95.76 / 94.27 % here vs 95.6-97.2 % reported. Close, but NOT the
    same estimator -- this is token-weighted over prefill workers, so it is labelled
    as the derived series it is rather than presented as the gauge.
    """
    if not _IB: return []
    acc={}
    for w,rows in (_IB.get("bins") or {}).items():
        if "prefill" not in w and "ctx" not in w: continue
        for r in rows:
            c=r.get("cached_kv_tokens") or 0; t=r.get("num_ctx_tokens") or 0
            if c+t<=0: continue
            k=relt(int(r["t"])*10**9)
            a=acc.setdefault(k,[0,0]); a[0]+=c; a[1]+=c+t
    return [[k, round(v[0]/v[1]*100,2)] for k,v in sorted(acc.items())
            if 0<=k<=run_dur+60 and v[1]]


def kvhit_series():
    out=[]
    for ts,m in scr:
        g=ggroup(m,"trtllm_kv_cache_hit_rate",by="dynamo_component")
        if _REUSE_COMPONENTS is not None:
            g={k:v for k,v in g.items() if k in _REUSE_COMPONENTS}
        vals=[x for vs in g.values() for x in vs]
        out.append([relt(ts), round(sum(vals)/len(vals)*100,2) if vals else None])
    return out
def _iter_kvutil_series(role):
    """Peak-worker KV pool occupancy per 1 s bin, from the per-iteration log.

    Fallback for ``dynamo_component_gpu_cache_usage_percent`` on the same grounds as
    the reuse fallback: the gauge lives on the worker endpoint, which is frequently
    not scraped. `max` across workers of that role, matching the gauge's own
    peak-worker aggregation so the two are read the same way.
    """
    if not _IB: return []
    acc={}
    for w,rows in (_IB.get("bins") or {}).items():
        if role not in w: continue
        for r in rows:
            v=r.get("kv_cache_util")
            if v is None: continue
            k=relt(int(r["t"])*10**9)
            if not (0<=k<=run_dur+60): continue
            acc[k]=max(acc.get(k,0.0), v*100)
    return [[k,round(v,2)] for k,v in sorted(acc.items())]


# The gauge is absent whenever no worker endpoint was scraped. Falling back keeps the
# panel honest: before this it emitted a full-length series of Nones, which draws an
# empty axis -- indistinguishable from "utilisation was zero".
en["kvutil_src"]="dynamo_component_gpu_cache_usage_percent"
if not [p for p in en["kvutil_pf"] if p[1] is not None] and \
   not [p for p in en["kvutil_de"] if p[1] is not None]:
    _pf,_de=_iter_kvutil_series("prefill"),_iter_kvutil_series("decode")
    if _pf or _de:
        _grid=sorted({k for s in (_pf,_de) for k,_ in s})
        _pfm,_dem=dict(_pf),dict(_de)
        en["kvutil_pf"]=[[k,_pfm.get(k)] for k in _grid]
        en["kvutil_de"]=[[k,_dem.get(k)] for k in _grid]
        en["kvutil_src"]="iter_bins.json (kv_cache_util)"
        _log.info(f"KV utilisation: gauge absent, derived from the iteration log "
                  f"({len(_grid)} bins)")

en["true_hit_pct"]=kvhit_series()
en["true_hit_src"]="trtllm_kv_cache_hit_rate"
if not [p for p in en["true_hit_pct"] if p[1] is not None]:
    _fb=_iter_reuse_series()
    if _fb:
        en["true_hit_pct"]=_fb; en["true_hit_src"]="iter_bins.json (cached_kv_tokens)"
        _log.info(f"KV hit rate: gauge absent, derived from the iteration log "
                  f"({len(_fb)} bins)")
en["reuse_components"]=sorted(_REUSE_COMPONENTS) if _REUSE_COMPONENTS is not None else None
_hitvals=[p[1] for p in en["true_hit_pct"] if p[1] is not None]
en["true_hit_kpi"]=round(pct(_hitvals,50),1) if _hitvals else None

# ============================ 3. trace aggregates ===========================
def col(k): return [r[k] for r in rows]
def band(sortkey,p):
    v=sorted(rows,key=lambda r:r[sortkey]); lo=max(0,int(len(v)*(p-5)/100)); hi=max(lo+1,int(len(v)*min(100,p+5)/100)); sub=v[lo:hi]
    # kvt comes from the request trace, not from spans: the decode `handle_payload`
    # span starts AFTER the KV cache has transferred, so a spans-only waterfall places
    # prefill compute directly against decode compute and silently omits the wait
    # between them. 0.0 when the run is aggregated (no transfer) or the trace is absent.
    _kvt=[(RT_BY_XID.get(r["xid"],{}) or {}).get("kv_transfer_ms") or 0 for r in sub]
    return {"adm":pct([r["adm"] for r in sub],50),"pf":pct([r["pf"] for r in sub],50),
            "kvt":pct(_kvt,50),
            "de":pct([r["de"] for r in sub],50),"route":pct([r["rroute"]+r["rhash"]+r["rfind"] for r in sub],50),
            "ttft":pct([r["ttft"] for r in sub],50),"e2e":pct([r["e2e"] for r in sub],50)}
waterfall={p:band("e2e",p) for p in (50,90,95,99)}
NB=360
def binseries(fn):
    bk=[[] for _ in range(NB)]
    for r in rows:
        i=min(NB-1,int(r["ts"]/run_dur*NB)) if run_dur else 0; bk[i].append(fn(r))
    return [[round(i*run_dur/NB,1), round(pct(b,50),1) if b else None, round(pct(b,99),1) if b else None] for i,b in enumerate(bk)]
ttft_series=binseries(lambda r:r["ttft"])
def dist(v): return {p:round(pct(v,p),3) for p in (50,90,95,99)}
router_compute=[r["rhash"]+r["rfind"]+r["rroute"] for r in rows]
ov=[r["ov"] for r in rows]; ov_max=max(ov) if ov else 0; OVB=30; ov_hist=[0]*OVB
for x in ov:
    if ov_max: ov_hist[min(OVB-1,int(x/(ov_max+1)*OVB))]+=1
worker_pf=Counter(r["wpf"] for r in rows if r["wpf"])

# ============ 3b. Overview load rows (ported from dashboard.py occupancy) ====
# Throughput + the four in-flight rows that dashboard.py renders in its stacked
# `occupancy` panel, recomputed against this bundle so the component Overview
# answers "was the system loaded?" next to "how did TTFT move?". Semantics follow
# src/common/gpu_occupancy.py; the two deliberate departures are noted inline.
TPUT_WIN_S=10.0
NBW=720           # time bins for the metric-driven rows (65k scrapes -> a plottable series)
_clamp=lambda t: round(min(max(t,0.0),run_dur),1)

def _rolling_tput(gpus,window_s=TPUT_WIN_S):
    """Output tok/s/GPU: each request's OSL is attributed at COMPLETION (arrival +
    client e2e), then trailing-window sum / window seconds / GPU count.
    Mirrors gpu_occupancy.compute_rolling_throughput."""
    evs=sorted(((r["ts_ns"]-run_start)/1e9+(r["e2e"] or 0)/1000.0, r["osl"] or 0)
               for r in rows if r["ts_ns"])
    out=[];win=deque();s=0.0
    for t,tok in evs:
        win.append((t,tok)); s+=tok
        while win and win[0][0]<t-window_s: s-=win.popleft()[1]
        out.append([_clamp(t),round(s/window_s/max(gpus,1),1)])
    return out

def _in_flight(events):
    """Concurrency step function [[s,count],...] from (+1 at start, -1 at end) events."""
    d=defaultdict(int)
    for ns,dl in events: d[_clamp((ns-run_start)/1e9)]+=dl
    out=[];cur=0
    for t in sorted(d):
        cur+=d[t]; out.append([t,cur])
    return out

def _worker_bins(name,worker_type,scale=1.0):
    """Per-worker gauge series bucketed to NBW bins holding the bin PEAK.

    Two departures from gpu_occupancy._wid_label, both required by this run's shape:
      * keyed by worker_id AND dp_rank -- one prefill worker_id here reports under
        dp_rank 0..3 (DEP 4), so keying on worker_id alone interleaves four ranks
        into a single zig-zag series;
      * bucketed, because this bundle holds 65,405 scrapes: raw would be ~65 points
        per horizontal pixel and would dominate the HTML payload. Peak (not mean)
        per bin, so contention spikes survive the downsample.
    """
    acc=defaultdict(lambda:[None]*NBW)
    for ts,m in scr:
        i=min(NBW-1,int(relt(ts)/run_dur*NBW)) if run_dur>0 else 0
        for e in m.get(name,[]):
            L=e.get("labels",{})
            if L.get("worker_type")!=worker_type: continue
            k=f"{(L.get('worker_id') or '?')[-6:]}·dp{L.get('dp_rank','0')}"
            v=e["value"]*scale; b=acc[k]
            if b[i] is None or v>b[i]: b[i]=v
    step=run_dur/NBW if run_dur>0 else 1
    return {k:[[round(i*step,1),round(v,1)] for i,v in enumerate(b) if v is not None]
            for k,b in sorted(acc.items())}

def _mdc_block_size():
    """Tokens per block for the ROUTER's block accounting.

    dynamo_frontend_worker_active_decode_blocks is set by WorkerLoadMetrics::observe from
    ActiveSequences::active_blocks(), whose block_size is the ModelDeploymentCard's
    kv_cache_block_size -- the same value exported as dynamo_frontend_model_kv_cache_block_size
    (32 on run 2674375). The ENGINE's trtllm_kv_cache_tokens_per_block (256 here) sizes a
    different pool; gpu_occupancy._block_size_from_metrics prefers it, which overstates decode
    tokens 8x on this run, so it is deliberately NOT used.
    """
    for _,m in scr:
        a=m.get("dynamo_frontend_model_kv_cache_block_size")
        if a and a[0].get("value"): return int(a[0]["value"])
    return None

BLK=_mdc_block_size()

def _block_size_mismatch():
    """Router block size vs the engine's tokens-per-block, as a first-class check.

    These are two different pools sized independently, and when they disagree the
    router's KV events reference blocks the engine cannot match. The consequence is
    not subtle -- on the reference run it produces a chain of 540 "Block contains N
    tokens" errors, 2,835 "Failed to find block to remove", and 35,198 "Block not
    found during remove", the last of which is 73% of the entire frontend log (94% at
    higher concurrency).

    It is invisible in the metrics: the counter built for exactly this failure,
    `dynamo_component_kv_cache_events_applied{status="block_not_found"}`, reads a
    constant 0. So the only way to surface it is to compare the two configured sizes
    directly, which is what this does.

    TWO sources for the engine side, in this order:
      1. the scrape's `trtllm_kv_cache_tokens_per_block` gauge
      2. `tokens_per_block` in the run's own resolved `trtllm_config_<mode>.yaml`

    The fallback is not decoration. The gauge is only exported when the engine
    publishes its KV-cache family, so on a run whose workers are up but not yet
    publishing -- exactly the e2e self-build run 2751593 -- source 1 is empty and the
    check reported "cannot tell" for a value that was sitting in the bundle all along.
    The config is the same authority `_cfg_max_batch` already trusts for the engine
    ceilings, and it is present whenever the ingest ran at all.

    Always returns (router_blk, engine_tpb, source), either side possibly None. The two
    sides are reported INDEPENDENTLY on purpose: an earlier version returned nothing
    unless both were known, which threw away a perfectly good engine value whenever the
    router gauge was missing. "engine 256, router unknown" is a useful thing to show;
    silence is not. The mismatch VERDICT is what requires both, and that is decided by
    the caller.
    """
    eng=None
    for _,m in scr:
        for e in (_entries(m,"trtllm_kv_cache_tokens_per_block") or []):
            v=e.get("value")
            if isinstance(v,(int,float)) and v>0: eng=int(v); break
        if eng: break
    src="scrape" if eng else None
    if not eng:
        eng=_cfg_tokens_per_block()
        src="engine config" if eng else None
    return (BLK,eng,src)

_BLK_ROUTER,_BLK_ENGINE,_BLK_SRC=_block_size_mismatch()
if _BLK_ROUTER and _BLK_ENGINE and _BLK_ROUTER!=_BLK_ENGINE:
    _log.warning(f"KV block-size mismatch: router indexes {_BLK_ROUTER}-token blocks "
                 f"while the engine pools {_BLK_ENGINE}-token blocks (engine size via "
                 f"{_BLK_SRC}); router KV events will reference blocks the engine "
                 f"cannot match")
elif _BLK_ROUTER and _BLK_ENGINE:
    _log.info(f"KV block size agrees across router and engine: {_BLK_ROUTER} tokens "
              f"(engine size via {_BLK_SRC})")
else:
    _log.warning(f"KV block-size agreement UNKNOWN (router={_BLK_ROUTER} "
                 f"engine={_BLK_ENGINE}); reported as unknown rather than as agreement")
if BLK is None: _log.warning("no dynamo_frontend_model_kv_cache_block_size in the stream; "
                             "decode tokens-in-flight stays in BLOCKS")
load={"tput":_rolling_tput(GPUS),
      "rif_pf":_in_flight(hp_ev["prefill"]),"rif_de":_in_flight(hp_ev["decode"]),
      "tif_pf":_worker_bins("dynamo_frontend_worker_active_prefill_tokens","prefill"),
      "tif_de":_worker_bins("dynamo_frontend_worker_active_decode_blocks","decode",scale=BLK or 1),
      "gpus":GPUS,"blk":BLK,"win_s":TPUT_WIN_S,"bins":NBW,
      "blk_router":_BLK_ROUTER,"blk_engine":_BLK_ENGINE,"blk_src":_BLK_SRC}
_log.info(f"load rows: tput={len(load['tput'])}pts gpus={GPUS} "
          f"rif_pf={len(load['rif_pf'])} rif_de={len(load['rif_de'])} "
          f"tif_pf={len(load['tif_pf'])}series tif_de={len(load['tif_de'])}series block={BLK}")

# ==================== 4. per-request drill-down (TTFT percentiles) ==========
# The waterfall above medians each stage independently across a p+-5 band, so a
# "p99 bar" is a composite no single request experienced. This series is the
# complement: NEAREST-RANK, so every plotted point IS an observed request whose
# span breakdown can be opened.
#
# Window is a REQUEST COUNT, not a time span. At this run's 0.29 req/s a 10 s
# trailing window holds ~5 requests, and nearest-rank with n=5 puts p95 and p99
# both on the max -- the two lines were identical at 100% of points. n>=20
# separates them; 50 gives ranks 25/48/50.
DRILL_WINDOW_N=50
def _drill_series(rows,window_n=DRILL_WINDOW_N,pcts=(50,95,99)):
    pts=[(r["ts_ns"],r["ttft"],r["xid"]) for r in rows if r["ts_ns"] and r["ttft"]]
    if not pts: return {},{}
    t0=pts[0][0]
    out={f"p{p}":[] for p in pcts}
    for i in range(len(pts)):
        win=sorted((v,x) for _,v,x in pts[max(0,i-window_n+1):i+1]); n=len(win)
        for p in pcts:
            v,x=win[max(0,min(n-1,math.ceil(p/100*n)-1))]
            out[f"p{p}"].append([round((pts[i][0]-t0)/1e6,1),round(v,1),x])
    return out,{k:_extrema(v) for k,v in out.items()}

def _extrema(series,min_sep=12,prom=.12):
    ys=[p[1] for p in series]
    if len(ys)<5: return []
    rng=(max(ys)-min(ys)) or 1.0; out=[]; last=-10**9
    for i in range(1,len(ys)-1):
        if i-last<min_sep: continue
        w=ys[max(0,i-5):i+6]
        peak=ys[i]>=max(w) and (ys[i]-min(w))/rng>prom
        low =ys[i]<=min(w) and (max(w)-ys[i])/rng>prom
        if peak or low: out.append([i,"peak" if peak else "low"]); last=i
    return out

def _ordered_spans(sp):
    """DFS from the structural root, NOT sorted by timestamp.

    The three emitting processes (frontend / prefill / decode worker) have
    unsynchronised clocks: measured over 800 traces the root `http-request` is
    not the earliest span in 10.4% of them (max skew 66 ms). Time-ordering
    therefore buried the root mid-list for ~1 trace in 10. The tree is
    unambiguous (exactly 1 structural root), so DFS is stable regardless.
    """
    by={s["spanId"]:s for s in sp}; kids={}
    for s in sp:
        p=s.get("parentSpanId")
        if p in by: kids.setdefault(p,[]).append(s)
    roots=[s for s in sp if s.get("parentSpanId") not in by]
    t0=min(int(s["startTimeUnixNano"]) for s in sp); out=[]
    def emit(s,d,g=0):
        # `role` marks the ONE row standing for each phase's segment, so the
        # renderer can cut the axis at first token without knowing this source's
        # span names. The log source (src/ingest/frontend_infolog_parser.py) sets
        # the same field on its own rows -- that is what makes the panel shared.
        _comp=at(s,"component")
        _role=({"prefill":"prefill","backend":"decode"}.get(_comp)
               if s["name"]=="handle_payload" else None)
        out.append({"name":s["name"],"t":round((int(s["startTimeUnixNano"])-t0)/1e6,3),
                    "d":round(dms(s),3),"depth":d,
                    "opaque":s["name"]=="handle_payload","comp":_comp,"role":_role})
        if g>24: return
        for c in sorted(kids.get(s["spanId"],[]),key=lambda c:int(c["startTimeUnixNano"])):
            emit(c,d+1,g+1)
    for r in sorted(roots,key=lambda r:int(r["startTimeUnixNano"])): emit(r,0)
    # A root span contains all its children by construction, so a recorded root
    # start LATER than a child's is a clock artifact, not causality. Left as-is
    # it indents the total bar: on trace 25fbac32 http-request starts 12.2 ms
    # after its own router children, which is invisible on the full-request axis
    # (0.46 px) but a clear 15.7 px offset once the axis is cut at first token.
    # Anchor the root at the origin and keep its END fixed.
    if out and out[0]["depth"] == 0 and out[0]["t"] > 0:
        out[0]["d"] = round(out[0]["d"] + out[0]["t"], 3); out[0]["t"] = 0.0
    return out

# TRT-LLM per-iteration engine telemetry (print_iter_log: true), pre-binned to
# 1s by parse_iterlog.py. Optional: absent unless that parser has been run.
# The engine ceilings it captions against are resolved at the top of this file,
# before any panel consumes them.
iter_series={}; iter_bins=None
if _IB:
    _ib=_IB; iter_bins=_ib
    for _wkey,_rows in (_ib.get("bins") or {}).items():
        out={k:[] for k in ("kv_cache_util","num_scheduled_requests","num_ctx_requests","num_generation_tokens")}
        for r in _rows:
            t=relt(int(r["t"])*10**9)                    # epoch s -> run-relative s
            if t<0 or t>run_dur+60: continue
            for k in out:
                if r.get(k) is not None: out[k].append([t,r[k]])
        iter_series[_wkey]=out
    _log.info(f"iter-log series: "+", ".join(f"{k}={len(v['kv_cache_util'])}pts" for k,v in iter_series.items()))

drill_series,drill_extrema=_drill_series(rows)
# Whole-run reference: the rolling lines say "how the tail moved"; these say
# "where it landed overall", so a spike can be read against the run's own number.
_ttfts=[r["ttft"] for r in rows if r["ttft"]]
drill_global={f"p{q}":round(pct(_ttfts,q),1) for q in (50,95,99)}
_reps={p[2] for v in drill_series.values() for p in v}
drill_spans={x:_ordered_spans(spans_raw[x]) for x in _reps if x in spans_raw}
_cl=lambda x,k:(client.get(x) or {}).get(k)
drill_req={x:{"ttft":_cl(x,"ttft"),"e2e":_cl(x,"e2e"),"isl":_cl(x,"isl"),"osl":_cl(x,"osl")} for x in _reps}
_log.info(f"drill-down: {len(drill_series.get('p50',[]))} pts/line, "
          f"{len(_reps)} representative requests, {len(drill_spans)} with spans")

# ============ 4b. Log analysis: the SAME drill payload, from the INFO log =====
# Second producer of the per-request stage IR. The span source above needs Tempo +
# a tracing backend; this one needs only the frontend log at default INFO level.
# Both feed the identical renderer (`drillPanel` in the JS below) because both emit
# {name,t,d,depth,opaque,comp,role} rows -- so the TTFT line and its breakdown are
# one implementation with two back-ends, not two look-alike panels.
logdrill=None
if _args.frontend_log:
    import sys as _sys
    # Repo root (src/visualization/ -> src/ -> root): that is the path entry that
    # makes the `src` namespace package -- and therefore `src.ingest` -- importable
    # when this file is run as a bare script rather than with -m.
    _sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
    from src.ingest.frontend_infolog_parser import parse_frontend_log
    _fl=parse_frontend_log(_args.frontend_log)
    _fst=_fl["stats"]
    _log.info("frontend log: "+", ".join(f"{k}={v}" for k,v in _fst.items()))
    _ok={k:v for k,v in _fl["requests"].items()
         if v["status"]=="success" and v["ttft"] and v["ts_ns"]}
    # Same warmup exclusion as the rest of the dashboard, via the x_request_id
    # pivot: `client` was already phase-filtered at load, so intersecting on its
    # keys transfers that decision to the log source. Only applied when the two
    # artifacts are actually the same run -- a small overlap means they are not,
    # and silently emptying the tab would be worse than leaving it unfiltered.
    # SAME-RUN ENFORCEMENT. The other tabs are built from the bundle; this one from
    # a log file. Mixing runs produces a page whose header and Overview describe one
    # workload and whose Log analysis tab describes another -- a build-time warning
    # is invisible to whoever opens the HTML, so this is a hard failure instead.
    _ov=len(set(_ok)&set(client))
    _matched=bool(client) and _ov >= .10*max(1,len(_ok))
    if _matched:
        _ok={k:v for k,v in _ok.items() if k in client}
        _log.info(f"log analysis: phase-filtered via x_request_id ({_ov} matched client records)")
    elif client:
        _ap.error(
            f"--frontend-log is from a different run than the bundle.\n"
            f"  bundle          : {META_SRC} ({len(client)} client records)\n"
            f"  frontend log    : {os.path.basename(_args.frontend_log)} ({len(_ok)} requests)\n"
            f"  x_request_id overlap: {_ov}\n"
            f"Pass the frontend log belonging to this bundle's run, or build the log "
            f"on its own:\n"
            f"  python3 -m src.visualization.build_dynamo_bench_dash <out.html> "
            f"--frontend-log {_args.frontend_log}")
    elif SRC:
        # Bundle present but no client records to compare against -- cannot verify.
        _log.warning("log analysis: bundle has no profile_export.jsonl, so same-run "
                     "correspondence with the frontend log could not be verified")
    _lrows=sorted(({"ts_ns":v["ts_ns"],"ttft":v["ttft"],"xid":k} for k,v in _ok.items()),
                  key=lambda r:r["ts_ns"])
    if _lrows:
        _ls,_lex=_drill_series(_lrows)
        # A log-sourced run can be two orders of magnitude larger than a traced one
        # (51,315 requests here vs 918 with spans). The rolling percentiles are
        # computed over EVERY request first -- decimation only thins what is
        # PLOTTED, so the lines keep their true shape. Uncapped, the AgentPerf log
        # produced a 45.8 MB page and ~154k SVG circles against a 1000px axis.
        # Every surviving point is still a real request, so click-to-drill holds.
        _MAXP=1500
        _np=len(next(iter(_ls.values()),[]))
        if _np>_MAXP:
            _k=math.ceil(_np/_MAXP)
            _ls={kk:vv[::_k] for kk,vv in _ls.items()}
            _lex={kk:_extrema(vv) for kk,vv in _ls.items()}   # indices shifted by decimation
            _log.info(f"log analysis: decimated {_np}->{len(next(iter(_ls.values())))} plotted "
                      f"points/line (stride {_k}); percentiles still computed over all {_np}")
        _lreps={p[2] for v in _ls.values() for p in v}
        _lt=[r["ttft"] for r in _lrows]
        logdrill={"series":_ls,"extrema":_lex,
                  "spans":{x:_fl["stages"][x] for x in _lreps if x in _fl["stages"]},
                  "req":{x:{kk:_ok[x][kk] for kk in ("ttft","e2e","isl","osl")} for x in _lreps},
                  "window_n":DRILL_WINDOW_N,
                  "global":{f"p{q}":round(pct(_lt,q),1) for q in (50,95,99)},
                  "stats":_fst,
                  "n":len(_lrows),
                  "routing":bool(_fst["routing_joinable"]),
                  "matched":_matched,"overlap":_ov,"bundle":META_SRC,
                  "src":os.path.basename(_args.frontend_log)}
        _log.info(f"log analysis: {len(_lrows)} requests, "
                  f"p50={logdrill['global']['p50']}ms p99={logdrill['global']['p99']}ms, "
                  f"routing_join={'yes' if logdrill['routing'] else 'no'}")

        # How much of the run's traffic the KV-routing panels actually describe.
        #
        # Computed HERE, into the payload, rather than formatted inside the page's JS.
        # The HTML is frequently unreadable where it lands -- a headless node, a CI log,
        # an S3 prefix -- so a caveat that exists only in rendered DOM is invisible to
        # every consumer that reads the JSON, which is the primary artifact. It also
        # makes the caveat testable: asserting on the HTML text cannot distinguish a
        # rendered string from the template literal that would produce it.
        #
        # Session-affinity pins a request to the worker already holding its
        # conversation, and a pinned request never reaches the KV scorer. On the e2e
        # run 2751593 that was 248 of 274 decisions -- 90.5% -- so the KV-routing
        # panels there describe under a tenth of the traffic.
        _tot=_fst.get("selector_without_request_id") or 0
        if _tot:
            _pin=_fst.get("selector_decisions_pinned") or 0
            ro["coverage"]={"decisions":_tot,"pinned":_pin,
                            "pct_pinned":round(100.0*_pin/_tot,1)}
            _log.info(f"router coverage: {_pin}/{_tot} decisions pinned by session affinity "
                      f"({ro['coverage']['pct_pinned']}%); the KV-routing panels describe the "
                      f"remaining {100-ro['coverage']['pct_pinned']:.1f}%")
    else:
        _log.warning("log analysis: no usable requests in the frontend log; tab omitted")

# ---- load balance per worker, from the log ---------------------------------
# Absolute rates, one line per worker: a worker holding 50% of 2 req/s is a very
# different situation from 50% of 20 req/s, and share-only charts erase that.
#
# Three attribution paths, in order of preference per ROLE:
#   1. worker ids on the completion record  -> worker granularity, token-weighted
#   2. selector joined on request_id        -> (worker, dp_rank), token-weighted
#   3. selector decisions alone             -> (worker, dp_rank), COUNTS ONLY
# Path 3 has no tokens but is the only route to dp_rank on Dynamo 1.3, and it is
# what reveals prefill dp_rank 1 dropping to zero traffic mid-run on 2663744.
# Selection is per role and picks whichever path resolves the most distinct
# workers -- prefill is frequently a single worker across four dp-ranks, where
# path 1 would render one flat line and say nothing.
def _balance_from_log(fl, keep):
    ok={k:v for k,v in fl["requests"].items() if k in keep and v["ts_ns"]}
    if not ok: return None
    routes=fl.get("routes") or {}
    sev=[e for e in (fl.get("sel_events") or []) if e[1]]
    t0=min([v["ts_ns"] for v in ok.values()]+[e[0] for e in sev] or [0])
    t1=max([v["ts_ns"] for v in ok.values()]+[e[0] for e in sev] or [0])
    span=max(1.0,(t1-t0)/1e9)
    # Snap to a readable interval rather than span/300 (which yields e.g. 14s).
    # Smallest that keeps the line under ~300 points.
    B=next((b for b in (5,10,15,30,60,120,300,600) if span/b<=300), 900)
    roles={}
    for role,wkey in (("prefill","pf_worker"),("decode","de_worker")):
        cands=[]
        # path 1
        p1=[(v["ts_ns"],str(v[wkey])[-6:],v["isl"] or 0,v["osl"] or 0)
            for v in ok.values() if v.get(wkey)]
        if p1: cands.append(("completion",True,p1))
        # path 2
        p2=[(ok[k]["ts_ns"],f"{str(routes[k][role][0])[-6:]} dp{routes[k][role][1] or '?'}",
             ok[k]["isl"] or 0, ok[k]["osl"] or 0)
            for k in ok if k in routes and role in routes[k]]
        if p2: cands.append(("join",True,p2))
        # path 3
        p3=[(ts,f"{str(w)[-6:]} dp{dp or '?'}",0,0) for ts,wt,w,dp in sev if wt==role]
        if p3: cands.append(("selector",False,p3))
        if not cands: continue
        src,has_tok,pts=max(cands,key=lambda c:(len({p[1] for p in c[2]}), c[1]))
        keys=sorted({p[1] for p in pts})
        acc={m:{k:defaultdict(float) for k in keys} for m in ("req","in","out")}
        for ts,k,i,o in pts:
            b=int((ts-t0)/1e9//B)
            acc["req"][k][b]+=1; acc["in"][k][b]+=i; acc["out"][k][b]+=o
        nb=int(span//B)+1
        roles[role]={"src":src,"has_tokens":has_tok,"keys":keys,
            "series":{m:{k:[[round(b*B,1),round(acc[m][k].get(b,0.0)/B,4)] for b in range(nb)]
                         for k in keys} for m in ("req","in","out")},
            "totals":{k:{m:round(sum(acc[m][k].values()),1) for m in ("req","in","out")} for k in keys}}
    return {"bucket_s":B,"roles":roles} if roles else None

if logdrill is not None:
    logdrill["balance"]=_balance_from_log(_fl,set(_ok))
    if logdrill["balance"]:
        for _r,_v in logdrill["balance"]["roles"].items():
            _log.info(f"load balance [{_r}]: {len(_v['keys'])} workers via {_v['src']}"
                      f"{'' if _v['has_tokens'] else ' (counts only, no tokens)'}")

# ============ 5. request-trace: per-request waterfall + per-session view =====
# The frontend's own per-request record (schema 4, see src/ingest/request_trace.py).
# This is the ONLY source for a gapless TTFT decomposition and the ONLY one carrying
# session_id, so both of the non-time-series dashboard entities are built from it.
#
# One code path, no per-run branching: every request yields the same four bands and
# every session the same row shape, whatever the topology. A field that is constant
# across the whole run (queue_depth and decode_dp_rank both read 0 on the reference
# runs) is reported as constant rather than special-cased away -- "this run never
# queued" is a finding, and hiding the field would make it unaskable.

# The waterfall is fixed and ordered. Each band is (key, label), and they sum to
# total_ms by construction -- verified 0/560 negative residuals on the reference runs.
RT_BANDS = [("prefill_wait_ms", "admission + routing"),
            ("prefill_ms",      "prefill compute"),
            ("kv_transfer_ms",  "KV transfer"),
            ("steady_decode_ms","steady decode")]
# Carried per request so a card can show what the request WAS, not just how long it took.
RT_ATTRS = ["isl","osl","cached_tokens","kv_hit_rate","queue_depth","finish_reason",
            "prefill_worker_id","prefill_dp_rank","decode_worker_id","decode_dp_rank",
            "turn_index","prefix_reuse_ratio","client_ttft_ms","clean_itl_ms",
            "avg_itl_ms","total_ms"]

rt_rows=list(rt_all)   # loaded at the top; see the HAS_RT block
# Same warmup exclusion as every other source, via the x_request_id pivot. `client`
# was already phase-filtered at load, so intersecting transfers that decision here.
# Only applied when the two artifacts are the same run -- a tiny overlap means they
# are not, and silently emptying the view is worse than leaving it whole.
if rt_rows and client:
    _keep=[r for r in rt_rows if r.get("x_request_id") in client]
    if len(_keep) >= .10*max(1,len(rt_rows)):
        _log.info(f"request trace: phase-filtered via x_request_id ({len(_keep)} of {len(rt_rows)} kept)")
        rt_rows=_keep
    else:
        _log.warning(f"request trace: only {len(_keep)}/{len(rt_rows)} rows match the client "
                     f"export; leaving unfiltered (different run?)")

_SPAN_BY_XID={r["xid"]:r for r in rows}
rt_requests={}; rt_sessions=[]; rt_const={}; rt_belief=None
if rt_rows:
    _t0=min(r["received_ms"] for r in rt_rows if r.get("received_ms"))
    for r in rt_rows:
        _bands=[[k,l,r.get(k)] for k,l in RT_BANDS]
        # Routing OUTCOME, joined from the prefill kv_router.route_request span.
        # Deliberately labelled an outcome and not a rationale: the router's own cost
        # comparison lives on a free-text selector line that carries no request_id, so
        # "which worker was chosen, with how much prefix overlap" is joinable per
        # request while "why it beat the alternatives" is not. Claiming the latter
        # would be inventing an explanation.
        _sp=_SPAN_BY_XID.get(r["x_request_id"]) or {}
        _routing={"overlap_blocks":_sp.get("ov"),"worker_id":_sp.get("wpf"),
                  "dp_rank":_sp.get("rank"),
                  "router_ms":round((_sp.get("rhash") or 0)+(_sp.get("rfind") or 0)
                                    +(_sp.get("rroute") or 0),3) if _sp else None,
                  "admission_ms":round(_sp["adm"],3) if _sp.get("adm") is not None else None}
        # Router belief vs engine reality, per request. The router scores candidates on
        # overlap_blocks; the engine reports what it actually reused as cached_tokens.
        # In blocks-to-tokens terms they should be the same number, and a divergence is
        # the signature of a router confidently routing on a prefix the engine does not
        # have -- reported in the field as high advertised reuse on traffic with none.
        # Computed every run rather than only when suspected, because the failure is
        # silent: the routing still "works", it is just routing on fiction.
        _ob=_sp.get("ov"); _ct=r.get("cached_tokens")
        if _ob is not None and _ct is not None and BLK:
            _pred=_ob*BLK
            _routing["overlap_tokens"]=_pred
            _routing["belief_error"]=0.0 if (_pred==0 and _ct==0) else round(abs(_pred-_ct)/max(1,_ct),4)
        rt_requests[r["x_request_id"]]={
            "t":round((r.get("received_ms") or _t0)-_t0,1),
            "session_id":r.get("session_id"),
            "bands":_bands,
            "attrs":{k:r.get(k) for k in RT_ATTRS},
            "routing":_routing if _sp else None,
        }
    # A field that never varies carries no information for THIS run; report it once
    # here rather than drawing a flat line per panel, so the reader learns it was
    # constant instead of inferring it from a chart that looks broken.
    for k in RT_ATTRS+[b[0] for b in RT_BANDS]:
        _vals={json.dumps(r.get(k)) for r in rt_rows}
        if len(_vals)==1: rt_const[k]=rt_rows[0].get(k)

    _by_sess={}
    for r in rt_rows:
        if r.get("session_id"): _by_sess.setdefault(r["session_id"],[]).append(r)
    for _sid,_rs in _by_sess.items():
        _rs.sort(key=lambda r:(r.get("turn_index") if r.get("turn_index") is not None else 0,
                               r.get("received_ms") or 0))
        _span_ms=(max((r.get("received_ms") or 0)+(r.get("total_ms") or 0) for r in _rs)
                  - min(r.get("received_ms") or 0 for r in _rs))
        _busy=sum(r.get("total_ms") or 0 for r in _rs)
        rt_sessions.append({
            "session_id":_sid,
            "t":round((min(r.get("received_ms") or _t0 for r in _rs))-_t0,1),
            "turns":len(_rs),
            "span_ms":round(_span_ms,1),
            "busy_ms":round(_busy,1),
            # Wall-clock the session existed but was not being served. On the reference
            # run one session idled 562s of 661s: a session-level latency number that
            # ignores this describes the harness's think time, not the server.
            "idle_ms":round(max(0.0,_span_ms-_busy),1),
            "ttft_ms":[r.get("client_ttft_ms") for r in _rs],
            "kv_hit":[r.get("kv_hit_rate") for r in _rs],
            "isl":[r.get("isl") for r in _rs],
            "reuse":[r.get("prefix_reuse_ratio") for r in _rs],
            "xids":[r.get("x_request_id") for r in _rs],
            "decode_workers":sorted({r.get("decode_worker_id") for r in _rs if r.get("decode_worker_id") is not None}),
            "prefill_ranks":sorted({r.get("prefill_dp_rank") for r in _rs if r.get("prefill_dp_rank") is not None}),
        })
    rt_sessions.sort(key=lambda s:s["t"])
    _pin=sum(1 for s in rt_sessions if len(s["decode_workers"])==1)
    _log.info(f"request trace: {len(rt_requests)} requests, {len(rt_sessions)} sessions, "
              f"{_pin}/{len(rt_sessions)} pinned to a single decode worker, "
              f"{len(rt_const)} constant field(s): {sorted(rt_const)}")
    _errs=[v["routing"]["belief_error"] for v in rt_requests.values()
           if (v.get("routing") or {}).get("belief_error") is not None]
    if _errs:
        _bad=[e for e in _errs if e>0.02]
        rt_belief={"n":len(_errs),"disagree":len(_bad),
                   "worst":round(max(_errs),4),"threshold":0.02}
        _log.info(f"router belief vs engine reality: {len(_errs)-len(_bad)}/{len(_errs)} agree "
                  f"(overlap_blocks x {BLK} == cached_tokens within 2%), worst error {max(_errs):.1%}")
    else:
        rt_belief=None

# ============ 6. declarative time-series panels ==============================
# One generic evaluator over the panel table in panels.py. Every panel is a data row;
# none of them has bespoke code, so none can drift into being run-specific.
panels=_evaluate_panels(scr)

# The warmup boundary rides on the PANEL ENTITY, not on a global the renderer reaches
# for. A panel is then self-describing -- it carries its own series, its own units and
# its own phase marker -- so any consumer that can draw one panel can draw the marker,
# and a panel whose x-axis is not run-time simply never sets the field.
# Applied just before the payload is assembled, NOT here: the derived panels
# (batch composition, step stalls, rank balance) are appended to `panels` further
# down, and marking at this point would silently skip every one of them.
def _mark_warmup(ps):
    if WARMUP_END_S is None: return ps
    for _p in ps.values():
        _p["warmup_end_s"]=WARMUP_END_S
    return ps

# ---- derived panels: in-flight balance across ranks and workers -------------
# Emitted in the SAME shape as the spec panels so the generic renderer draws them
# with no extra code -- a different SOURCE, not a different kind of panel.
#
# Why this cannot come from the metrics stream: engine-side per-rank series are a
# replicated broadcast (identical across ranks in every sweep), so splitting an
# engine gauge by rank renders a flat family of lines and reads as "balanced" on a
# run that was not. The request trace carries the rank that actually served each
# request, so occupancy per rank can be reconstructed from it instead.
#
# Why INSTANTANEOUS rather than a whole-run total, in the reporter's own words:
#   "Over the whole benchmark phase, I don't see a large imbalance between DP ranks.
#    However, there is a large instantaneously imbalance."
# A run-total request count per rank is exactly the aggregate that hides this, which
# is why the panel is max-minus-min occupancy over time and not a bar chart.
#
# The spread statistic follows the documented method -- "using running-request and
# looking at min/max between DP ranks" -- with the normalised form alongside, since
# max-min of 4 means something different at 5 in flight than at 500.
def _inflight_spread(rows, field, nbins=240):
    live=[r for r in rows if r.get(field) is not None and r.get("received_ms") and r.get("total_ms")]
    if len({r[field] for r in live})<2: return None,None
    t0=min(r["received_ms"] for r in live); t1=max(r["received_ms"]+r["total_ms"] for r in live)
    if t1<=t0: return None,None
    step=(t1-t0)/nbins
    keys=sorted({r[field] for r in live})
    spread=[]; norm=[]
    for i in range(nbins):
        t=t0+i*step
        counts={k:0 for k in keys}
        for r in live:
            if r["received_ms"]<=t<r["received_ms"]+r["total_ms"]: counts[r[field]]+=1
        hi,lo=max(counts.values()),min(counts.values())
        mean=sum(counts.values())/len(counts)
        spread.append([round(i*step/1000,1),hi-lo])
        # Normalised only where there is load; (hi-lo)/0 is undefined, and reporting
        # perfect balance for an idle window would dilute the very spikes being hunted.
        if mean>0: norm.append([round(i*step/1000,1),round((hi-lo)/mean,3)])
    return spread,norm

# ---- derived panel: batch composition, from the per-iteration log ----------
# The scrape stream says how BUSY the engine was; only the per-iteration log says
# what "busy" consisted of. An engine scheduling one request per step at high
# occupancy is saturated in a completely different way from one scheduling many, and
# the two are indistinguishable in every aggregate gauge.
if iter_series and 'iter_bins' in globals() and iter_bins:
    _bf={}; _mx={}
    for _w,_rows in iter_bins.get("bins",{}).items():
        _fr=[];_m=[]
        for _r in _rows:
            _h={int(k):v for k,v in (_r.get("sched_hist") or {}).items()}
            _active=sum(v for k,v in _h.items() if k>=1)
            _batched=sum(v for k,v in _h.items() if k>=2)
            _t=relt(int(_r["t"])*10**9)
            if _t<0 or _t>run_dur+60: continue
            # Fraction is over ACTIVE iterations only: an idle step did not decline to
            # batch, it had nothing to batch, and counting it would read as a batching
            # failure during a quiet period.
            if _active: _fr.append([round(_t,1),round(_batched/_active,4)])
            _m.append([round(_t,1),_r.get("sched_max") or 0])
        if _fr: _bf[_w]=_fr
        if _m: _mx[_w]=_m
    if _bf:
        panels["en_batched_fraction"]={
          "tab":"engine","title":"Batched iterations, share of active steps",
          "unit":"ratio","kind":"derived","split_by":None,
          "why":"Of the steps that had work, the share that scheduled more than one "
                "request. A value pinned near zero means the engine is stepping one "
                "request at a time -- the same wall-clock occupancy as a healthy engine, "
                "at a fraction of the throughput.",
          "source":["iter_bins.json"],
          "caveat":"From rank 0 only, so it cannot show per-rank divergence. Idle steps "
                   "are excluded from the denominator; a quiet period is not a batching "
                   "failure.",
          "issues":["PERF-batch-starvation"],"series":_bf}
    _st={}
    for _w,_rows in iter_bins.get("bins",{}).items():
        _pts=[]
        for _r in _rows:
            _v=_r.get("host_step_time_ms_max")
            _t=relt(int(_r["t"])*10**9)
            if _v is None or _t<0 or _t>run_dur+60: continue
            _pts.append([round(_t,1),_v/1000.0])
        if _pts: _st[_w]=_pts
    if _st:
        panels["en_step_stall"]={
          "tab":"engine","title":"Worst engine step per second",
          "unit":"s","kind":"derived","split_by":None,
          "why":"The longest forward step in each second, from the per-iteration log. A "
                "scraped gauge samples once a second and cannot see a stall it did not "
                "land on, so this is the only view in which a multi-second engine stop "
                "is visible at all.",
          "source":["iter_bins.json"],
          "caveat":"Rank 0 only, and a maximum rather than a typical value -- read it "
                   "beside the median step time, not instead of it.",
          "issues":["PERF-iteration-stall"],"series":_st}
        _pk=max(v for s_ in _st.values() for _,v in s_)
        _log.info(f"engine step stalls: worst single step {_pk:.1f}s across {len(_st)} worker(s)")
    if _mx:
        panels["en_sched_max"]={
          "tab":"engine","title":"Peak scheduled requests per step",
          "unit":"requests","kind":"derived","split_by":None,
          "why":"The most the scheduler ever managed in a single step, against the "
                "engine's configured batch ceiling. A peak far below the ceiling means "
                "the limit is arrival or admission, not engine capacity.",
          "source":["iter_bins.json"],
          "caveat":"Rank 0 only.","issues":["PERF-batch-starvation"],"series":_mx}
        _log.info("batch composition: "+", ".join(
            f"{w}={sum(v for _,v in s)/max(1,len(s)):.2f} batched-share" for w,s in _bf.items()))

for _field,_label,_tab in (("prefill_dp_rank","prefill DP rank","router"),
                           ("decode_worker_id","decode worker","engine")):
    _sp,_nm=_inflight_spread(rt_rows,_field)
    if not _sp: continue
    panels[f"bal_{_field}"]={
      "tab":_tab,
      "title":f"In-flight imbalance across {_label}s",
      "unit":"requests","kind":"derived","split_by":None,
      "why":"Concurrent requests on the busiest minus the least busy, sampled over the "
            "run. A whole-run total per rank averages this away: balance over a phase "
            "and balance at an instant are different properties, and it is the "
            "instantaneous spread that stalls a synchronised step.",
      "source":["request_trace.jsonl"],
      "caveat":"Reconstructed from per-request attribution, not from an engine gauge -- "
               "engine-side per-rank series are a replicated broadcast and cannot show "
               "imbalance at all. Counts a request as occupying its rank for its whole "
               "lifetime, so it measures occupancy, not instantaneous GPU work.",
      "issues":["PERF-dp-imbalance"],
      "series":{"max-min":_sp,**({"normalised (max-min)/mean":_nm} if _nm else {})},
    }
    _peak=max(v for _,v in _sp)
    _log.info(f"balance [{_field}]: peak instantaneous spread {_peak} requests "
              f"across {len({r[_field] for r in rt_rows if r.get(_field) is not None})} keys")
if panels:
    _by_tab={}
    for _pid,_p in panels.items(): _by_tab.setdefault(_p["tab"],[]).append(_pid)
    _log.info("panels: "+", ".join(f"{t}={len(v)}" for t,v in sorted(_by_tab.items()))
              +f" ({sum(len(p['series']) for p in panels.values())} series total)")
    _flat=[pid for pid,p in panels.items()
           if all(len({round(v,9) for _,v in s})==1 for s in p["series"].values())]
    if _flat:
        # Named rather than dropped: a flat queue-depth panel says the run never
        # queued, which is one of the more useful things a run can tell you.
        _log.info(f"panels reading a single constant value: {sorted(_flat)}")

DATA={
 "logdrill":logdrill,
 "panels":_mark_warmup(panels),
 "rt":{"requests":rt_requests,"sessions":rt_sessions,"bands":RT_BANDS,"const":rt_const,"belief":rt_belief},
 "iter":iter_series,
 "iter_cfg":{"pf_batch":CFG_PF_BATCH,"pf_tok":CFG_PF_TOK,"de_batch":CFG_DE_BATCH,"de_tok":CFG_DE_TOK},
 "drill":{"series":drill_series,"extrema":drill_extrema,"spans":drill_spans,
          "req":drill_req,"window_n":DRILL_WINDOW_N,"global":drill_global},
 # Which tabs have inputs. A tab with no data is DROPPED, not rendered empty:
 # an empty Router tab reads as "this run had no queueing" rather than "this run
 # was not instrumented", which is the more dangerous of the two misreadings.
 # Engine is the one tab with two independent producers: the scrape stream and the
 # per-iteration log. Gating it on scrapes alone drops it on a run that has full
 # engine telemetry and no /metrics surface -- exactly the trtllm-serve control a
 # Dynamo run has to be compared against.
 "tabs":{"overview":bool(rows),"frontend":bool(scr),"router":bool(scr),
         "engine":bool(scr) or bool(iter_series),"loganalysis":logdrill is not None,
         "session":bool(rt_sessions)},
 "meta":{"n":N,"run_dur_s":round(run_dur,1),"scrapes":len(scr),
   "warmup_end_s":WARMUP_END_S,"warmup_src":WARMUP_SRC,
   "src":META_SRC,
   # Substitute the authoritative block size measured from the metrics stream. The
   # yaml only ever carried ingest's --block-size fallback, and on the reference run
   # that fallback (512) disagreed with both the measured value (32) and the engine
   # config (256) -- three numbers, of which the header was showing the wrong one.
   "topo":(META_TOPO or f"max batch prefill {MAX_BATCH_PF} / decode {MAX_BATCH_DE}")
          .replace("__BLK__", str(BLK) if BLK else "unknown"),
   # Why the population is the size it is. `n` alone cannot distinguish "this run
   # served nothing" from "this run never left warmup", and those call for opposite
   # responses: the first is a broken deployment, the second is a run that needed
   # more wall clock. Verified on the c32 reference run 2750618, where all 118 client
   # records are benchmark_phase=warmup because the job hit its 20-minute limit before
   # profiling began -- a reader seeing only n=0 would conclude the stack was dead.
   #
   # The page says this in its header already; this is the same statement in the
   # payload, which is what anything downstream of the HTML actually reads.
   "phase_filter":{"applied":not _args.include_warmup,
                   "kept":len(client),"dropped":_skipped_phase,
                   "phases":_phase_seen},
   "client_summary":CLIENT_SUMMARY,
   "provenance":PROVENANCE,
   "benchmark_status":BENCH_STATUS,
   "log_signals":LOG_SIGNALS,
   "lifecycle":LIFECYCLE,
   "host":HOST},
 "kpi":{"ttft_p50":round(pct(col("ttft"),50)/1000,1),"ttft_p99":round(pct(col("ttft"),99)/1000,1),
   "adm_p50":round(pct(col("adm"),50)/1000,1),"pf_p50":round(pct(col("pf"),50)/1000,1),
   "reqs":N,"adm_share":round(pct(col("adm"),50)/max(1,pct(col("ttft"),50))*100,0),
   "kv_hit_true":en["true_hit_kpi"],"tok_cache":fe["tok_cache_hit_pct"]},
 "waterfall":waterfall,"ttft_series":ttft_series,"router_compute_dist":dist(router_compute),
 "admission_dist_s":{k:round(v/1000,2) for k,v in dist(col("adm")).items()},
 "pf_svc_dist":{k:round(v/1000,2) for k,v in dist(col("pf")).items()},
 "de_svc_dist":{k:round(v/1000,2) for k,v in dist(col("de")).items()},
 "ov_hist":ov_hist,"ov_max":ov_max,
 "worker_pf":sorted(([str(k)[-6:],v] for k,v in worker_pf.items()),key=lambda x:-x[1]),
 "load":load,"fe":fe,"ro":ro,"en":en,
}
_log.info(f"built: N={N} scrapes={len(scr)} run_dur={run_dur:.0f}s ttft_p50={DATA['kpi']['ttft_p50']}s "
          f"true_kv_hit={en['true_hit_kpi']}% tok_cache={fe['tok_cache_hit_pct']}%")

# Inline the vendored D3 by default: the HTML is routinely read from a laptop after
# being pulled off a cluster or synced to S3, where a CDN fetch is not guaranteed.
# An explicit --d3 wins over the sibling copy; --d3-cdn opts out of inlining entirely.
_d3_path = None if _args.d3_cdn else (_args.d3 or (_VENDORED_D3 if os.path.exists(_VENDORED_D3) else None))
if _d3_path:
    D3_TAG = "<script>" + open(_d3_path).read() + "</script>"
    _log.info(f"D3 inlined from {_d3_path}")
else:
    D3_TAG = '<script src="https://d3js.org/d3.v7.min.js"></script>'
    _log.info("D3 loaded from the CDN; the page needs network access to render")

HTML=r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamo Benchmark Performance Dashboard</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--edge:#2a3038;--ink:#e6edf3;--dim:#8b949e;--grn:#76b900;
--adm:#e5484d;--pf:#4a90d9;--de:#a78bfa;--route:#f2cc60;--cy:#22b8cf}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--edge);background:linear-gradient(180deg,#12161c,#0e1116)}
h1{margin:0;font-size:16px}.sub{color:var(--dim);font-size:12px;margin-top:3px}
.tabs{display:flex;gap:4px;padding:8px 16px 0;border-bottom:1px solid var(--edge);position:sticky;top:0;background:var(--bg);z-index:5}
.tab{padding:8px 16px;cursor:pointer;border:1px solid transparent;border-bottom:none;border-radius:7px 7px 0 0;color:var(--dim);font-weight:600}
.tab.on{background:var(--panel);color:var(--grn);border-color:var(--edge)}
.view{display:none;padding:16px 20px 60px}.view.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:14px}
.panel{background:var(--panel);border:1px solid var(--edge);border-radius:10px;padding:12px 14px;min-width:0}
.panel h2{margin:0 0 2px;font-size:13px}.panel .src{color:#6e7681;font-size:11px}
.warn{color:#d29922;font-size:11px}
.kpi .src{color:#6e7681;font-size:10px;margin-top:2px}
.reqcard{border-top:1px solid #21262d;padding:8px 0}
.reqhead{font-size:12px;color:#c9d1d9;margin-bottom:2px}
.reqmeta{font-size:11px;color:#8b949e;margin-top:2px}
.tbl{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.tbl th{text-align:left;color:#8b949e;font-weight:600;border-bottom:1px solid #30363d;padding:4px 8px}
.tbl td{padding:4px 8px;border-bottom:1px solid #21262d;color:#c9d1d9;font-variant-numeric:tabular-nums}
.cap{color:var(--dim);font-size:11px;margin-bottom:8px}
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.kpi{background:var(--panel);border:1px solid var(--edge);border-radius:9px;padding:9px 14px;min-width:110px}
.kpi .v{font-size:22px;font-weight:700;color:var(--grn)}.kpi .l{color:var(--dim);font-size:11px}
.note{background:#1b2230;border:1px solid #2b3a52;border-radius:8px;padding:9px 12px;color:#c9d6e8;font-size:12px;margin:0 0 12px}
.note.hl{background:#14231a;border-color:#2c5a38;color:#cfe8d5}.note b{color:var(--grn)}.note code{color:#f2cc60;font-size:11px}
.lg{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--dim);margin:2px 0 6px}
/* per-request drill-down card (dark theme, matches .panel) */
.drillcard{position:absolute;z-index:40;background:#11161d;border:1px solid var(--edge);border-radius:9px;
  box-shadow:0 10px 34px rgba(0,0,0,.62);padding:11px 14px;display:none;width:fit-content;max-width:94%;color:var(--ink)}
.drillcard.pinned{border-color:#2f81f7;box-shadow:0 10px 34px rgba(47,129,247,.35)}
.drillcard .dtitle{font-weight:600;font-size:12.5px;margin-bottom:2px}
.drillcard .dmeta{color:var(--dim);font-size:11px;margin-bottom:8px}
.drillcard .dpin{float:right;font-size:10.5px;color:#2f81f7}
.drillcard .dkpi{display:flex;gap:10px;font-size:11px;margin-bottom:9px;flex-wrap:wrap}
.drillcard .dkpi span{background:#1b2230;border:1px solid #2b3a52;border-radius:4px;padding:2px 7px}
.drillcard .dkpi b{color:var(--grn)}
.drillcard td{color:#c9d6e8}
.drillcard .dnote{margin-top:8px;font-size:10.5px;color:var(--dim);border-top:1px solid var(--edge);padding-top:6px}
.drillcard .dexp{position:absolute;right:-13px;top:50%;transform:translateY(-50%);width:26px;height:44px;
  border:1px solid var(--edge);border-radius:6px;background:#11161d;cursor:pointer;font-size:14px;color:#2f81f7;
  display:flex;align-items:center;justify-content:center;padding:0}
.drillcard .dexp:hover{background:#1b2230;border-color:#2f81f7}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px}
svg{width:100%;height:auto;display:block;overflow:visible}.axis text{fill:var(--dim);font-size:10px}.axis path,.axis line{stroke:#30363d}
.tip{position:fixed;display:none;background:#0b0e12;border:1px solid var(--edge);border-radius:6px;padding:6px 9px;font-size:11px;pointer-events:none;z-index:99;max-width:300px}
.full{grid-column:1/-1}
</style></head><body>
<header><h1>Dynamo Benchmark Performance Dashboard</h1>
<div class="sub" id="sub"></div></header>
<div class="tabs" id="tabs"></div><div id="views"></div>
<div id="sink" style="display:none"></div><div class="tip" id="tip"></div>
__D3__
<script>
const DATA=__DATA__;
const VW=1000, VH=460;              // FIXED viewBox width for full panels (fixes hidden-tab clientWidth=0)
const C={adm:'#e5484d',pf:'#4a90d9',de:'#a78bfa',route:'#f2cc60',grn:'#76b900',cy:'#22b8cf',dim:'#8b949e'};
const tip=d3.select('#tip'),RD=DATA.meta.run_dur_s;
const fmtS=ms=>ms==null?'—':ms>=1000?(ms/1000).toFixed(1)+'s':ms.toFixed(ms<10?2:0)+'ms';
function sTip(h,e){tip.style('display','block').html(h).style('left',(e.clientX+14)+'px').style('top',(e.clientY-10)+'px')}
function hTip(){tip.style('display','none')}
// Header provenance. `meta.n` counts only PROFILING-phase requests joined across the
// client and trace legs, and it is legitimately 0 on a run that never reached the
// profiling phase -- every record still being warmup. Reporting a bare "0 requests"
// next to a Session tab listing dozens of sessions reads as a broken page, so the
// header says which population it is describing and names the other one.
(function(){
  const M=DATA.meta, RT=DATA.rt||{};
  const nSess=(RT.sessions||[]).length, nReq=Object.keys(RT.requests||{}).length;
  let head=`${M.n.toLocaleString()} profiling requests · ${M.scrapes} metric scrapes`
         + ` · run ${RD}s · ${M.topo} · ${M.src}`;
  if(!M.n && (nReq||nSess)) head += `  —  no request completed the profiling phase;`
         + ` ${nReq} request(s) across ${nSess} session(s) are warmup and are shown on the`
         + ` Session tab, excluded from every percentile above`;
  // A benchmark that failed outranks everything else the header could say. Without it
  // an empty dashboard reads as a dead deployment, when the usual cause is an
  // environment fault that never let the workload start -- and those call for opposite
  // responses. Placed before the validity clause because it subsumes it: if the
  // benchmark never ran, its own validity flags are moot.
  const BS=M.benchmark_status;
  if(BS){
    head += `  —  BENCHMARK FAILED (exit ${BS.exit_code||'?'})`
          + (BS.errors && BS.errors.length ? `: ${BS.errors[0]}` : '')
          // The phase census when there is no error line: it distinguishes "never
          // started" from "ran and was cut short", which have opposite fixes.
          + (!(BS.errors||[]).length && (BS.phases||[]).length
               ? ` — ${BS.phases[BS.phases.length-1]}` : '');
  }
  // Run validity bounds every number on the page, so it belongs in the header rather
  // than on one tab. A cancelled run, a run with client errors, or an agentic run
  // whose child branches errored produced figures that must not be compared against a
  // clean run -- and nothing in the per-request stream says so.
  const LC=M.lifecycle;
  if(LC && LC.durations_s && LC.durations_s.time_to_ready!=null){
    head += `  —  ready in ${LC.durations_s.time_to_ready}s`
          + (LC.durations_s.readiness_gap!=null
               ? ` (frontend accepted ${LC.durations_s.readiness_gap}s before the router could place)` : '');
  }
  const CS=M.client_summary;
  if(CS){
    const bad=[];
    if(CS.was_cancelled) bad.push('client reported the run CANCELLED');
    if(CS.n_errors) bad.push(`${CS.n_errors} client error(s)`);
    if(CS.branch_errored) bad.push(`${CS.branch_errored} agentic branch(es) errored`);
    if(CS.branch_truncated) bad.push(`${CS.branch_truncated} branch(es) truncated`);
    head += bad.length ? `  —  VALIDITY: ${bad.join('; ')}` : `  —  client reports no errors`;
  }
  document.getElementById('sub').textContent=head;
})();
// 'Log analysis' only exists when --frontend-log was supplied; the tab is dropped
// entirely rather than rendered empty, so a bundle-only build looks unchanged.
const ALL_TABS=[['overview','Overview'],['frontend','Frontend'],['router','Router'],
                ['engine','Engine'],['session','Session'],['loganalysis','Log analysis']];
const OK=DATA.tabs||{};
const TABS=ALL_TABS.filter(([id])=>OK[id]);
const te=d3.select('#tabs'),ve=d3.select('#views');
TABS.forEach(([id,l],i)=>{te.append('div').attr('class','tab'+(i===0?' on':'')).attr('data-t',id).text(l).on('click',()=>act(id));
  ve.append('div').attr('class','view'+(i===0?' on':'')).attr('id','v-'+id);});
function act(id){d3.selectAll('.tab').classed('on',false);d3.select(`.tab[data-t=${id}]`).classed('on',true);
  d3.selectAll('.view').classed('on',false);d3.select('#v-'+id).classed('on',true);}
// ---- primitives (all fixed-width; W,H are viewBox coords, CSS scales to container) ----
// Panels for a dropped tab are built into a hidden sink rather than skipped. The
// builders stay untouched (several call .node(), which would throw on an empty
// selection), and the output is simply never shown.
function vsel(id){const v=d3.select('#v-'+id);return v.empty()?d3.select('#sink'):v}
function grid(view){let g=vsel(view).select('.grid');return g.empty()?vsel(view).append('div').attr('class','grid'):g}
function panel(view,title,cap,full){const el=grid(view).append('div').attr('class','panel'+(full?' full':''));
  el.append('h2').html(title);if(cap)el.append('div').attr('class','cap').html(cap);return el}
// items are [value, label, source?]. The optional third element names WHERE the number
// came from -- a KPI with no provenance is indistinguishable from one that was
// computed off the wrong population, which has happened here more than once.
function kpis(view,items){const k=vsel(view).insert('div',':first-child').attr('class','kpis');
  items.forEach(it=>{const c=k.append('div').attr('class','kpi');
    c.append('div').attr('class','v').text(it[0]);
    c.append('div').attr('class','l').text(it[1]);
    if(it[2]) c.append('div').attr('class','src').text(it[2]);})}
function note(view,html,cls){vsel(view).append('div').attr('class','note '+(cls||'')).html(html)}
function svg(el,w,h){return el.append('svg').attr('viewBox',`0 0 ${w} ${h}`)}
function lg(el,items){el.append('div').attr('class','lg').html(items.map(([c,t])=>`<span><span class="sw" style="background:${c}"></span>${t}</span>`).join(''))}

// ---- warmup phase marker ----------------------------------------------------
// Drawn by every time-axis chart, from ONE helper, so the marker cannot be present
// on some tabs and missing on others. The value is an OPTIONAL field on the panel
// entity (`p.warmup_end_s`), falling back to the run-level `DATA.meta.warmup_end_s`
// for the hand-written charts that have no panel entity. Absent => nothing drawn:
// a harness with no warmup phase must not grow an invented boundary.
function wmVal(p){
  const v = (p && p.warmup_end_s!=null) ? p.warmup_end_s : (DATA.meta||{}).warmup_end_s;
  return (v==null || !isFinite(v)) ? null : v;
}
function wmLine(s,x,hTop,hBot,p){
  const v=wmVal(p); if(v==null) return;
  const px=x(v); const [lo,hi]=x.range();
  if(px<lo || px>hi) return;          // boundary outside this chart's window
  s.append('line').attr('x1',px).attr('x2',px).attr('y1',hTop).attr('y2',hBot)
   .attr('stroke','#8b949e').attr('stroke-width',1).attr('stroke-dasharray','3,3').attr('opacity',.75);
  s.append('text').attr('x',px+3).attr('y',hTop+9).attr('fill','#8b949e')
   .attr('font-size','9px').text('warmup ends');
}
function lineChart(el,series,{unit='ms',w=VW,h=220,cols=null,keys=null,legend=null,ymax=null,hline=null,logy=false,panelSpec=null}={}){
  if(legend)lg(el,legend);
  const s=svg(el,w,h),L=56,B=26,R=12;
  const x=d3.scaleLinear().domain([0,RD]).range([L,w-R]);
  const lines = keys ? keys : [[1,C.grn],[2,C.adm]];
  let mx = ymax!=null?ymax:d3.max(series,d=>d3.max(lines.map(k=>d[Array.isArray(k)?k[0]:k]).filter(v=>v!=null)))||1;
  // logy: for co-plotting series whose magnitudes differ by orders of magnitude
  // (e.g. cache hits ~1e3 next to cached tokens ~1e8), where a shared linear
  // axis pins the small series flat onto zero. Zero/negative points are dropped
  // rather than clamped -- log(0) has no position, and inventing one would draw
  // a line through a value never observed.
  const y = logy
    ? d3.scaleLog().domain([Math.max(1e-9,d3.min(series,d=>d3.min(lines.map(k=>d[Array.isArray(k)?k[0]:k]).filter(v=>v!=null&&v>0)))||1), mx*1.6]).range([h-B,8])
    : d3.scaleLinear().domain([0,mx*1.05]).range([h-B,8]);
  s.append('g').attr('class','axis').attr('transform',`translate(0,${h-B})`).call(d3.axisBottom(x).ticks(8).tickFormat(d=>d+'s'));
  wmLine(s,x,8,h-B,panelSpec);
  s.append('g').attr('class','axis').attr('transform',`translate(${L},0)`).call((logy?d3.axisLeft(y).ticks(6,'~s'):d3.axisLeft(y).ticks(5).tickFormat(d=>unit==='ms'?fmtS(d):(unit==='%'?d+'%':d3.format('~s')(d)))));
  if(hline!=null){s.append('line').attr('x1',L).attr('x2',w-R).attr('y1',y(hline)).attr('y2',y(hline)).attr('stroke','#666').attr('stroke-dasharray','5 3');
    s.append('text').attr('x',w-R-3).attr('y',y(hline)-4).attr('text-anchor','end').attr('fill','#8b949e').attr('font-size',10).text('max '+hline);}
  // Draw each line from ONLY the rows where that series actually has a value.
  //
  // This is load-bearing, not a tidy-up. The scrape stream carries one row per
  // ENDPOINT (frontend + prefill + 3x decode), so a frontend-only metric is
  // present on roughly every 5th row and null on the rest -- two non-null points
  // are therefore almost never ADJACENT. d3.line().defined() emits a segment
  // only between consecutive defined points, so filtering with .defined() drew
  // nothing at all (measured: 13,081 non-null points but 2 adjacent pairs, and
  // 0 for router overhead). Compacting per series is what makes these render.
  // Bridging the gaps is correct here: they are a sampling artefact of the
  // interleave, not intervals where the metric was missing.
  lines.forEach(k=>{const idx=Array.isArray(k)?k[0]:k, col=Array.isArray(k)?k[1]:C.grn;
    const pts=series.filter(d=>d[idx]!=null&&(!logy||d[idx]>0));
    if(!pts.length) return;
    s.append('path').datum(pts).attr('fill','none').attr('stroke',col).attr('stroke-width',1.6)
     .attr('d',d3.line().x(d=>x(d[0])).y(d=>y(d[idx])));});
  return s;
}
function distBar(el,d,unit,color,w=VW/2-30){
  const rows=[['p50',d[50]],['p90',d[90]],['p95',d[95]],['p99',d[99]]].filter(r=>r[1]!=null&&r[1]!=='—');
  const h=rows.length*30+8,L=34,s=svg(el,w,h),mx=d3.max(rows,r=>+r[1])||1,x=d3.scaleLinear().domain([0,mx]).range([L,w-70]);
  rows.forEach((r,i)=>{const y=i*30+4;s.append('text').attr('x',2).attr('y',y+15).attr('fill',C.dim).attr('font-size',11).text(r[0]);
   s.append('rect').attr('x',L).attr('y',y).attr('width',Math.max(1,x(+r[1])-L)).attr('height',20).attr('fill',color).attr('opacity',.85).attr('rx',3);
   s.append('text').attr('x',x(+r[1])+6).attr('y',y+15).attr('fill',C.ink).attr('font-size',11).text(r[1]+unit)});
}
function catBar(el,cats,color,unit='',w=VW/2-30){
  const h=Math.max(90,cats.length*22+12),L=100,s=svg(el,w,h),y=d3.scaleBand().domain(cats.map(c=>c[0])).range([4,h-8]).padding(.18),
   x=d3.scaleLinear().domain([0,d3.max(cats,c=>+c[1])||1]).range([L,w-56]);
  cats.forEach(c=>{s.append('text').attr('x',2).attr('y',y(c[0])+y.bandwidth()/2+4).attr('fill',C.dim).attr('font-size',10).text(c[0]);
   s.append('rect').attr('x',L).attr('y',y(c[0])).attr('width',Math.max(1,x(+c[1])-L)).attr('height',y.bandwidth()).attr('fill',color).attr('opacity',.85).attr('rx',2);
   s.append('text').attr('x',x(+c[1])+5).attr('y',y(c[0])+y.bandwidth()/2+4).attr('fill',C.ink).attr('font-size',10).text((+c[1]).toLocaleString()+unit)});
}
function histBar(el,hist,mx,color,w=VW/2-30){
  const h=150,B=22,s=svg(el,w,h),x=d3.scaleBand().domain(d3.range(hist.length)).range([30,w-6]).padding(.12),
   y=d3.scaleLinear().domain([0,d3.max(hist)||1]).range([h-B,6]);
  s.append('g').selectAll('rect').data(hist).join('rect').attr('x',(d,i)=>x(i)).attr('y',d=>y(d)).attr('width',x.bandwidth())
   .attr('height',d=>h-B-y(d)).attr('fill',color).attr('opacity',.85).on('mousemove',(e,d)=>sTip(d.toLocaleString()+' reqs',e)).on('mouseleave',hTip);
  s.append('g').attr('class','axis').attr('transform',`translate(0,${h-B})`).call(d3.axisBottom(d3.scaleLinear().domain([0,mx]).range([30,w-6])).ticks(6));
}

// Multi-series step chart: {label:[[s,val],...]} -> one line per worker/dp-rank, shared
// x = run seconds. Used by the Overview load rows, where series are sampled independently
// (each worker has its own scrape timeline) so they can't be zipped into lineChart's rows.
const PAL=['#76b900','#4a90d9','#a78bfa','#f2cc60','#22b8cf','#e5484d','#f97316','#14b8a6'];
const fmtN=v=>v==null?'—':Math.abs(v)>=1e6?(v/1e6).toFixed(2)+'M':Math.abs(v)>=1e3?(v/1e3).toFixed(1)+'k':''+(Math.round(v*10)/10);
// xmax: the log tab has its own time base -- DATA.meta.run_dur_s is 0 on a
// log-only build, which would collapse the x axis to a point.
function multiLine(el,series,{h=240,unitLabel='',area=false,xmax=null}={}){
  const labels=Object.keys(series).filter(k=>series[k]&&series[k].length);
  if(!labels.length){const e=svg(el,VW,h);e.append('text').attr('x',VW/2).attr('y',h/2).attr('text-anchor','middle')
    .attr('fill',C.dim).attr('font-size',12).text('no data in this run');return e}
  if(labels.length>1)lg(el,labels.map((k,i)=>[PAL[i%PAL.length],k]));
  const s=svg(el,VW,h),L=64,B=26,R=14;
  const x=d3.scaleLinear().domain([0,xmax||RD||d3.max(labels,k=>d3.max(series[k],d=>d[0]))||1]).range([L,VW-R]);
  const mx=d3.max(labels,k=>d3.max(series[k],d=>d[1]))||1;
  const y=d3.scaleLinear().domain([0,mx*1.05]).range([h-B,8]);
  s.append('g').attr('class','axis').attr('transform',`translate(0,${h-B})`).call(d3.axisBottom(x).ticks(8).tickFormat(d=>d+'s'));
  wmLine(s,x,8,h-B,null);
  s.append('g').attr('class','axis').attr('transform',`translate(${L},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(fmtN));
  if(unitLabel)s.append('text').attr('x',6).attr('y',12).attr('font-size',11).attr('fill',C.dim).text(unitLabel);
  const ln=d3.line().defined(d=>d[1]!=null).x(d=>x(d[0])).y(d=>y(d[1])).curve(d3.curveStepAfter);
  labels.forEach((k,i)=>{const col=PAL[i%PAL.length];
    if(area)s.append('path').datum(series[k]).attr('fill',col).attr('opacity',.16)
      .attr('d',d3.area().defined(d=>d[1]!=null).x(d=>x(d[0])).y0(y(0)).y1(d=>y(d[1])).curve(d3.curveStepAfter));
    s.append('path').datum(series[k]).attr('fill','none').attr('stroke',col).attr('stroke-width',1.5).attr('d',ln)});
  // crosshair: snap to the nearest real sample of each series (they don't share timestamps)
  const bis=d3.bisector(d=>d[0]).left;
  const near=(a,t)=>{const j=bis(a,t,1),p=a[j-1],q=a[j];return(!q||Math.abs(t-p[0])<=Math.abs(t-q[0]))?p:q};
  const ch=s.append('line').attr('y1',8).attr('y2',h-B).attr('stroke','#8b949e').attr('stroke-width',1)
    .attr('stroke-dasharray','3 3').style('display','none').attr('pointer-events','none');
  const dots=labels.map((k,i)=>s.append('circle').attr('r',3.2).attr('fill',PAL[i%PAL.length])
    .attr('stroke','#0e1116').attr('stroke-width',1.2).style('display','none').attr('pointer-events','none'));
  s.append('rect').attr('x',L).attr('y',8).attr('width',VW-R-L).attr('height',h-B-8).attr('fill','none').attr('pointer-events','all')
   .on('mousemove',ev=>{const t=x.invert(d3.pointer(ev,s.node())[0]);
     const p0=near(series[labels[0]],t);ch.attr('x1',x(p0[0])).attr('x2',x(p0[0])).style('display',null);
     const body=labels.map((k,i)=>{const p=near(series[k],t);dots[i].attr('cx',x(p[0])).attr('cy',y(p[1])).style('display',null);
       return `<span class="sw" style="background:${PAL[i%PAL.length]}"></span>${labels.length>1?k+': ':''}<b>${fmtN(p[1])}</b>`}).join('<br>');
     sTip(`<b>${p0[0].toFixed(0)}s</b><br>${body}`,ev)})
   .on('mouseleave',()=>{ch.style('display','none');dots.forEach(d=>d.style('display','none'));hTip()});
  return s;
}

// ================= OVERVIEW =================
// No KPI strip and no framing note here: the run's headline numbers are in the
// header sub-line, and the panels below carry their own captions. (DATA.kpi is
// still built -- the Frontend tab's strip and the build log read it.)
// ---- per-request TTFT drill-down (nearest-rank: every point is a real request)
// SHARED per-request TTFT panel. Source-agnostic: it consumes the stage IR
// ({name,t,d,depth,opaque,comp,role}) and knows nothing about where the rows came
// from. Called twice -- once with the Tempo-span payload (Overview) and once with
// the INFO-log payload (Log analysis). Anything source-specific arrives via `cfg`.
function drillPanel(view,D,cfg){
 if(!D||!D.series||!D.series.p50||!D.series.p50.length) return;
 const el=panel(view,cfg.title,cfg.cap,true);
 const CO={p50:'#0284c7',p95:'#d97706',p99:'#dc2626'};
 lg(el,[[CO.p50,`p50 (rank ${Math.ceil(.50*D.window_n)}/${D.window_n})`],
        [CO.p95,`p95 (rank ${Math.ceil(.95*D.window_n)}/${D.window_n})`],
        [CO.p99,`p99 (rank ${Math.ceil(.99*D.window_n)}/${D.window_n})`],
        ['#8b949e','dashed = whole-run percentile'],
        ['#fca5a5','hatched = '+cfg.opaqueLegend]]);
 const wrap=el.append('div').style('position','relative');
 const H2=300,M={top:10,right:18,bottom:28,left:64};
 const all=Object.values(D.series).flat();
 const x=d3.scaleLinear().domain(d3.extent(all,d=>d[0])).range([M.left,VW-M.right]);
 const y=d3.scaleLinear().domain([0,d3.max(all,d=>d[1])]).nice().range([H2-M.bottom,M.top]);
 const s=wrap.append('svg').attr('viewBox',`0 0 ${VW} ${H2}`);
 const fS=ms=>(ms/1000).toFixed(ms>=10000?0:1)+'s';
 const fT=ms=>ms>=1000?(ms/1000).toFixed(2)+'s':ms.toFixed(1)+'ms';
 s.append('g').attr('class','axis').attr('transform',`translate(0,${H2-M.bottom})`)
  .call(d3.axisBottom(x).ticks(8).tickFormat(v=>(v/1000).toFixed(0)+'s'));
 s.append('g').attr('class','axis').attr('transform',`translate(${M.left},0)`)
  .call(d3.axisLeft(y).ticks(6).tickFormat(fS));
 s.append('text').attr('x',6).attr('y',12).attr('font-size',11).attr('fill',C.dim).text('TTFT (s)');
 // whole-benchmark reference lines (drawn first so the rolling series sit on top)
 if(D.global){for(const [k,v] of Object.entries(D.global)){
   if(v==null||v>y.domain()[1])continue;
   s.append('line').attr('x1',M.left).attr('x2',VW-M.right).attr('y1',y(v)).attr('y2',y(v))
    .attr('stroke',CO[k]).attr('stroke-width',1).attr('stroke-dasharray','5 4').attr('opacity',.45);
   s.append('text').attr('x',VW-M.right-2).attr('y',y(v)-4).attr('text-anchor','end')
    .attr('font-size',10).attr('fill',CO[k]).attr('opacity',.9)
    .text(`run ${k} ${(v/1000).toFixed(v>=10000?0:1)}s`);}}
 const ln=d3.line().x(d=>x(d[0])).y(d=>y(d[1]));
 const card=wrap.append('div').attr('class','drillcard');
 let pinned=false,expanded=false,last=null;
 for(const [k,pts] of Object.entries(D.series)){
   s.append('path').datum(pts).attr('fill','none').attr('stroke',CO[k]).attr('stroke-width',1.4).attr('opacity',.85).attr('d',ln);
   const ex=new Map((D.extrema[k]||[]).map(([i,kind])=>[i,kind]));
   s.append('g').selectAll('circle').data(pts.map((p,i)=>({p,i}))).join('circle')
    .attr('cx',d=>x(d.p[0])).attr('cy',d=>y(d.p[1])).attr('r',d=>ex.has(d.i)?4.5:1.6)
    .attr('fill',CO[k]).attr('opacity',d=>ex.has(d.i)?1:.35)
    .attr('stroke',d=>ex.has(d.i)?'#fff':'none').attr('stroke-width',1.2).style('cursor','pointer')
    .on('mouseenter',(e,d)=>{if(!pinned)show(d.p[2],k,ex.get(d.i),e)})
    .on('mouseleave',()=>{if(!pinned)card.style('display','none')})
    .on('click',(e,d)=>{e.stopPropagation();pinned=true;show(d.p[2],k,ex.get(d.i),e);card.classed('pinned',true)});
 }
 document.addEventListener('click',e=>{if(pinned&&!card.node().contains(e.target)){pinned=false;card.classed('pinned',false);card.style('display','none')}});
 function show(xid,pctk,kind,evt){
   const r=D.req[xid],sp=D.spans[xid]||[]; if(!r)return; last=[xid,pctk,kind,evt];
   // Role-based, not name-based: both producers tag exactly one row per phase, so
   // this works for `handle_payload` (spans) and `generation ...` (log) alike.
   const isDe=z=>z.role==='decode';
   const isPf=z=>z.role==='prefill';
   const tag=z=>isPf(z)?' <span style="color:#78716c">(prefill)</span>':isDe(z)?' <span style="color:#78716c">(decode)</span>':'';
   const BW=560,GAP=10,STUB=46,dec=sp.find(isDe);
   const full=Math.max(...sp.map(z=>z.t+z.d),1);
   // Collapsed axis ends at the CLIENT-MEASURED first token, not at the decode
   // span's start. The decode handle_payload begins at handoff and the first
   // token lands inside it -- p50 162ms later, p90 6.4s, max 74.7s across this
   // run's 918 requests. Cutting at decode-start therefore hid almost the whole
   // TTFT path (on b59904ad: axis 0.65s vs TTFT 44.5s).
   const ttftMs=(r.ttft!=null&&r.ttft>0)?r.ttft:null;
   const dur=expanded?full:Math.max(1,Math.min(full,ttftMs??full));
   const g=sp.map(z=>{
     const past=!expanded&&z.t>=dur-.001;                 // starts after the axis
     const cut =!expanded&&!past&&(z.t+z.d)>dur+.001;     // crosses the axis
     if(past)return{z,geo:null,cut:false};
     const left=Math.max(0,BW*z.t/dur);
     return{z,geo:{left,w:Math.max(1,cut?Math.max(1,BW-left):BW*z.d/dur)},cut}});
   const right=Math.max(1,...g.filter(o=>o.geo).map(o=>o.geo.left+o.geo.w));
   const sL=right+GAP, hasStub=g.some(o=>!o.geo), dw=Math.ceil((hasStub?sL+STUB:right)+4);
   const rows=g.map(({z,geo,cut})=>{
     let bar,dl,nl=z.name+tag(z);
     if(!geo){bar=`<svg width="${dw}" height="13"><line x1="${right+2}" y1="0" x2="${right+2}" y2="13" stroke="#e7e5e4"></line>`
       +`<rect x="${sL}" y="2" width="${STUB*.42}" height="9" rx="2" fill="url(#dhatch)" stroke="#dc2626" stroke-width=".8"></rect>`
       +`<text x="${sL+STUB*.47}" y="10.5" font-size="10" fill="#dc2626">⋯</text>`
       +`<rect x="${sL+STUB*.7}" y="2" width="${STUB*.3}" height="9" rx="2" fill="url(#dhatch)" stroke="#dc2626" stroke-width=".8"></rect></svg>`;
       dl=`<b style="color:#dc2626">${fT(z.d)}</b>`;}
     else{bar=`<svg width="${dw}" height="13"><rect x="${geo.left}" y="2" width="${geo.w}" height="9" rx="2" fill="${z.opaque?'url(#dhatch)':'#0284c7'}" ${z.opaque?'stroke="#dc2626" stroke-width=".8"':''}></rect>`
       +(cut?`<path d="M${geo.left+geo.w} 2 l4 2 l-4 2 l4 2 l-4 3" fill="none" stroke="#78716c" stroke-width="1.1"></path>`:'')+`</svg>`;
       dl=cut?`${fT(z.d)} <span style="color:#a8a29e">✂</span>`:fT(z.d);
       if(cut&&z.name==='http-request')nl=`${z.name} <span style="color:#a8a29e">(cut at first token)</span>`;}
     return `<tr><td style="padding-left:${z.depth*12}px;white-space:nowrap;font-size:11.5px">${nl}</td>`
       +`<td style="width:${dw}px">${bar}</td>`
       +`<td style="text-align:right;font-variant-numeric:tabular-nums;font-size:11.5px;padding-left:10px">${dl}</td></tr>`}).join('');
   const deMs=sp.filter(isDe).reduce((a,z)=>a+z.d,0), opMs=sp.filter(z=>z.opaque&&(expanded||!isDe(z))).reduce((a,z)=>a+z.d,0);
   card.html(`<span class="dpin">${pinned?'📌 pinned — click outside to dismiss':'click to pin'}</span>`
    +`<div class="dtitle">Detail ${xid.slice(0,16)}</div>`
    +`<div class="dmeta">${pctk.toUpperCase()} of its ${D.window_n}-request window${kind?` · <b style="color:${CO[pctk]}">${kind==='peak'?'▲ local peak':'▼ local trough'}</b>`:''}</div>`
    +`<div class="dkpi"><span>TTFT <b>${r.ttft==null?'–':fS(r.ttft)}</b></span><span>e2e <b>${r.e2e==null?'–':fS(r.e2e)}</b></span>`
    +`<span>ISL <b>${r.isl??'–'}</b></span><span>OSL <b>${r.osl??'–'}</b></span><span>spans <b>${sp.length}</b></span></div>`
    +`<svg width="0" height="0"><defs><pattern id="dhatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">`
    +`<rect width="6" height="6" fill="#fee2e2"></rect><line x1="0" y1="0" x2="0" y2="6" stroke="#fca5a5" stroke-width="3"></line></pattern></defs></svg>`
    +`<table style="border-collapse:collapse">${rows}</table>`
    +`<div class="dnote">Time axis spans <b>0–${fT(dur)}</b> (${expanded?'full request, absolute scale':cfg.axisNote}). `
    +`Hatched = opaque: ${fT(opMs)} of that is ${cfg.opaqueNote}`
    +(deMs>0?(expanded?` Decode (<b>${fT(deMs)}</b>) at true scale — collapse ◀ to refocus on TTFT.`
                     :` ${cfg.decodeNote.replace('{d}',fT(deMs))} Expand ▶ for absolute lengths.`):'')
    +`</div><button class="dexp">${expanded?'◀':'▶'}</button>`);
   card.select('.dexp').on('click',ev=>{ev.stopPropagation();expanded=!expanded;if(last)show(...last)});
   card.style('display','block').style('width',null);
   const tbl=card.select('table').node();
   if(tbl)card.style('width',Math.max(320,Math.ceil(tbl.getBoundingClientRect().width))+'px');
   const pb=wrap.node().getBoundingClientRect(),cw=card.node().offsetWidth,ch=card.node().offsetHeight;
   card.style('left',Math.max(4,Math.min(pb.width-cw-4,evt.clientX-pb.left+14))+'px')
       .style('top', Math.max(4,Math.min(Math.max(4,pb.height-ch-4),evt.clientY-pb.top+14))+'px');
 }
}

// ---- back-end 1: Tempo spans (Overview) ----
drillPanel('overview',DATA.drill,{
  title:'TTFT p50 / p95 / p99 — every point is a real request',
  cap:`Nearest-rank over a trailing ${(DATA.drill||{}).window_n}-request window, so each point <b>is</b> an observed request rather than an interpolated statistic. Larger dots = local peaks &amp; troughs. Hover for that request's span breakdown · click to pin · click outside to dismiss.`,
  opaqueLegend:'opaque span (queue + compute + KV transfer)',
  axisNote:'up to first token, client-measured TTFT',
  opaqueNote:'queue + compute + KV transfer combined, not separable with current instrumentation.',
  decodeNote:'Decode spans <b>{d}</b> total and is <b>cut</b> at first token — it begins at handoff, so the wait for the first token happens <i>inside</i> it.'});

// ---- back-end 2: frontend INFO log (Log analysis) ----
(function(){const L=DATA.logdrill; if(!L) return;
 const S=L.stats;
 kpis('loganalysis',[
   [L.n.toLocaleString(),'requests charted','frontend log'],
   [(L.global.p50/1000).toFixed(2)+'s','TTFT p50 (log)','request completed'],
   [(L.global.p99/1000).toFixed(2)+'s','TTFT p99 (log)','request completed'],
   [S.records.toLocaleString(),'log records parsed','frontend log'],
   [S.completed.toLocaleString(),'lifecycle completions','request completed'],
   [(S.selector_without_request_id||0).toLocaleString(),'routing decisions seen','selector'],
   [L.routing?'yes':'no','routing joinable per request','selector request_id'],
   [L.matched?'same run':'DIFFERENT','vs. bundle in other tabs','x_request_id overlap']]);
 note('loganalysis',
  `<b>Same panel, different back-end.</b> The Overview TTFT chart is built from Tempo spans; this one is built from
   <code>${L.src}</code> — the Dynamo frontend log at <b>default INFO level</b>. No tracing backend, no DEBUG, no
   observability flags. Both feed the identical renderer because both emit the same per-request stage IR
   (<code>{name, t, d, depth, opaque, comp, role}</code>).
   <br><br>The breakdown here is <b>deliberately shallower</b>. <code>ttft_ms</code> on the INFO
   <code>request completed</code> line is a single number covering routing, scheduler queue, prefill compute and KV
   transfer; the log cannot separate them, so that interval is drawn as <b>one hatched row</b> rather than invented
   sub-stages. The admission+routing prefix is separable only when the router's
   selector line carries a <code>request_id</code> (Dynamo &ge; 2026-08-10); without it TTFT stays a
   single opaque block. Whether this run has it is the <i>routing joinable per request</i> figure above.`);
 // Panel order on this tab is CALL order. TTFT leads: it is the primary question
 // the tab answers, and the load-balance panels below are read against it.
 drillPanel('loganalysis',L,{
   title:'TTFT p50 / p95 / p99 — from the frontend INFO log',
   cap:`Identical nearest-rank construction to the Overview chart (trailing ${L.window_n}-request window), sourced from <code>request received</code> / <code>request completed</code> instead of spans. Hover for the log-derived stage breakdown · click to pin.`,
   opaqueLegend:'opaque interval (not decomposable at INFO level)',
   axisNote:'up to first token, server-measured ttft_ms',
   opaqueNote:'queue + prefill + KV transfer, reported by the frontend as one number.',
   decodeNote:'Generation totals <b>{d}</b> and starts <i>at</i> first token, so it sits entirely beyond this axis.'});

 // ---- load balance per worker -------------------------------------------
 // Absolute rates rather than share: 50% of 2 req/s and 50% of 20 req/s are
 // different situations and a normalised chart cannot tell them apart.
 (function(){const B=L.balance; if(!B) return;
  const MET=[['req','requests/s','req/s'],['in','in-tok/s','input tok/s'],['out','out-tok/s','output tok/s']];
  for(const role of ['prefill','decode']){
    // Gating is a TAB-level rule (no source data -> no tab). At panel level a
    // degenerate value is still a value: a 1P1D deployment has one worker with a
    // real rate, and omitting the panel would read as "no data" when the honest
    // answer is "one worker, nothing to balance". Render it.
    const R=B.roles[role]; if(!R||!R.keys.length) continue;
    const nice=role[0].toUpperCase()+role.slice(1);
    const via={completion:'worker ids on the completion record',
               join:'selector joined to completion on request_id',
               selector:'selector decisions only — <b>counts, no tokens</b>'}[R.src];
    // Whole-run imbalance, and the WORST single bucket. The two diverge sharply:
    // on 2663744 decode the run-level request spread is 1.22x while individual
    // buckets reach 19x, so a run-level number alone reads as "balanced".
    const tot=R.keys.map(k=>R.totals[k].req), sum=d3.sum(tot)||1;
    const runImb=d3.max(tot)/Math.max(1,d3.min(tot));
    const ref=R.series.req[R.keys[0]]||[];
    let worst=1,worstT=0,dark=false;
    for(let i=0;i<ref.length;i++){
      const v=R.keys.map(k=>R.series.req[k][i][1]);
      if(!d3.sum(v)) continue;                       // idle bucket, not an imbalance
      const mn=d3.min(v),mx=d3.max(v);
      if(mn===0){dark=true; worstT=worstT||ref[i][0]; continue}   // a worker went idle
      if(mx/mn>worst){worst=mx/mn; worstT=ref[i][0]}
    }
    // One title and one caption template for every topology. Whatever the run
    // happens to contain -- 1 worker or 12, with or without dp-ranks -- is carried
    // by the DATA, never by branching prose. See CLAUDE.md, "Generalise over the
    // data, do not customise per run".
    const n=R.keys.length;
    const worstTxt=dark
      ? `at least one worker was <b>fully idle</b> while others served (first at ${Math.round(worstT)}s)`
      : `worst single bucket <b>${worst.toFixed(1)}×</b> (at ${Math.round(worstT)}s)`;
    const el=panel('loganalysis',`${nice} load per worker`,
      `Rate per worker in ${B.bucket_s}s buckets, from ${via}. `
      +`<b>${n}</b> worker${n===1?'':'s'}; whole-run request spread <b>${runImb.toFixed(2)}×</b> max:min, `
      +`${worstTxt}. Rates are absolute, not shares, so load level and balance are both readable.`,true);
    const ctl=el.append('div').attr('class','lg').style('gap','6px');
    const box=el.append('div');
    let metric='req';                       // requests: the router's own unit, and
                                            // the only metric all three paths have
    const btns=MET.filter(m=>m[0]==='req'||R.has_tokens).map(([m,lbl])=>
      ctl.append('span').style('cursor','pointer').style('padding','2px 8px')
         .style('border','1px solid #2a3038').style('border-radius','10px')
         .attr('data-m',m).text(lbl).on('click',()=>{metric=m;draw()}));
    if(!R.has_tokens)
      ctl.append('span').style('color','#8b949e').style('font-size','11px')
         .text('· token rates unavailable: this path has no per-request tokens');
    function draw(){
      btns.forEach(b=>b.style('background',b.attr('data-m')===metric?'#22303a':'transparent')
                       .style('color',b.attr('data-m')===metric?'#e6edf3':'#8b949e'));
      box.selectAll('*').remove();
      const series={}; R.keys.forEach(k=>series[k]=R.series[metric][k]);
      const xm=d3.max(R.keys,k=>d3.max(R.series[metric][k],d=>d[0]))||1;
      multiLine(box,series,{unitLabel:MET.find(m=>m[0]===metric)[2],xmax:xm});
      box.append('div').attr('class','dnote').html(
        R.keys.map(k=>`<b>…${k}</b> ${(100*R.totals[k].req/sum).toFixed(1)}% of requests`
          +(R.has_tokens?` · ${(100*R.totals[k].in/Math.max(1,d3.sum(R.keys.map(q=>R.totals[q].in)))).toFixed(1)}% of input tokens`:'')
        ).join(' &nbsp;·&nbsp; '));
    }
    draw();
  }})();
})();

// ---- load rows ported from dashboard.py's stacked `occupancy` panel ----
// Order is deliberate: throughput (what the run produced) -> requests in flight
// (how many were resident) -> tokens in flight (how much KV/token load that was),
// prefill before decode at each level.
(function(){const L=DATA.load; if(!L) return;
 (function(){const el=panel('overview','Throughput (toks/s/gpu)',
   `Output tokens/s per GPU, trailing ${L.win_s}s window, each request's OSL attributed at completion · <b>${L.gpus} GPUs</b> (prefill + decode).`,true);
  multiLine(el,{'tok/s/GPU':L.tput},{unitLabel:'tok/s/GPU',area:true});})();
 (function(){const el=panel('overview','Prefill Requests In Flight',
   'Concurrent requests resident on a <b>prefill</b> worker, from each request&rsquo;s <code>handle_payload(component=prefill)</code> span window. Trace-derived, so it counts real requests rather than a gauge sample.',true);
  multiLine(el,{'requests':L.rif_pf},{unitLabel:'requests',area:true});})();
 (function(){const el=panel('overview','Decode Requests In Flight',
   'Concurrent requests resident on a <b>decode</b> worker, from each request&rsquo;s <code>handle_payload(component=backend)</code> span window.',true);
  multiLine(el,{'requests':L.rif_de},{unitLabel:'requests',area:true});})();
 (function(){const el=panel('overview','Prefill Tokens In Flight',
   `<code>dynamo_frontend_worker_active_prefill_tokens</code>, one line per prefill worker &middot; DEP rank. Peak per bin over ${L.bins} bins.`,true);
  multiLine(el,L.tif_pf,{unitLabel:'tokens'});})();
 (function(){const el=panel('overview','Decode Tokens In Flight',
   `<code>dynamo_frontend_worker_active_decode_blocks</code> &times; <b>${L.blk??'?'}</b> tokens/block (the router&rsquo;s MDC block size, not the engine&rsquo;s), one line per decode worker. Peak per bin over ${L.bins} bins.`,true);
  multiLine(el,L.tif_de,{unitLabel:'tokens'});})();
})();

// ================= ROUTER =================
// The coverage clause is appended only when the frontend-log leg is present, because
// only that leg can count routing decisions. The SENTENCE is fixed; the numbers in it
// are measured per run. Without it these panels read as "how the router chose", when
// on a session-affinity deployment most requests never reach the KV scorer at all --
// they are pinned to the worker that already holds the conversation. A reader drawing
// conclusions about KV-aware routing from a run that barely used it is the failure
// this sentence exists to prevent.
(function(){
 let cov='';
 const CV=DATA.ro.coverage;
 if(CV){
   cov=`<br><br><b>Routing-decision coverage.</b> ${CV.decisions.toLocaleString()} routing decisions were
     observed and <b>${CV.pinned.toLocaleString()} (${CV.pct_pinned.toFixed(1)}%)</b> were pinned by session
     affinity, so they bypassed KV scoring entirely. These panels therefore describe the
     <b>${(100-CV.pct_pinned).toFixed(1)}%</b> of decisions that actually reached the scorer. Read the KV-routing
     panels as a description of that slice, not of the run's whole traffic.`;
 }
 note('router',`Both panels come from the <b>frontend</b> endpoint (<code>:8000/metrics</code>). The KV router runs in-process in the frontend, so every router metric is published there — the prefill and decode endpoints expose none.`+cov,'hl');
})();
(function(){const el=panel('router','Router overhead',
  '<code>dynamo_router_overhead_total_ms</code> — "Total routing overhead per request in milliseconds". A Prometheus histogram, plotted as the <b>interval mean</b> (Δsum/Δcount between consecutive scrapes), so each point is the average routing cost of the requests routed during that interval.',true);
 lg(el,[[C.grn,'router overhead (ms/request)']]);
 lineChart(el,DATA.ro.oh_total_ms,{unit:'ms',keys:[[1,C.grn]]});})();
(function(){const el=panel('router','Router queue',
  '<code>dynamo_frontend_router_queue_pending_requests</code> — "Number of requests pending in the router scheduler queue". A gauge, summed across its <code>worker_type</code> / <code>policy_class</code> labels and read at every scrape.',true);
 lg(el,[[C.adm,'pending requests']]);
 lineChart(el,DATA.ro.queue_pending,{unit:'n',keys:[[1,C.adm]]});})();

// ================= ENGINE =================
// The measured reuse rate is uninterpretable on its own -- every reader scores it
// against a private expectation. AIPerf computes what the WORKLOAD offered, so the
// two together turn a bare percentage into a gap with a size. Fixed sentence,
// measured numbers, and omitted entirely when the client summary is absent.
(function(){
 let ceil='';
 const CS=DATA.meta.client_summary, TH=CS&&CS.theoretical_prefix_cache_hit;
 if(TH!=null){
   const got=(DATA.kpi||{}).kv_hit_true;
   ceil=`<br><br><b>Reuse against the workload's ceiling.</b> The client computes a theoretical
     prefix-cache hit of <b>${TH.toFixed(1)}%</b> for this workload — the most any engine could
     have reused given the prompts actually sent.`
     + (got!=null ? ` The engine measured <b>${got.toFixed(1)}%</b>, a gap of
     <b>${(TH-got).toFixed(1)} points</b>.` : '')
     + ` Read the true-hit panel against that ceiling, not against an absolute
     expectation: a workload with little shared prefix cannot be reused into a high rate,
     and a low rate under a high ceiling is a routing or eviction problem.`;
 }
 note('engine',`Engine occupancy from Prometheus: <b>KV utilisation</b>, <b>in-flight batch vs configured max</b>, and the <b>true block cache-hit</b>. Read together — low KV utilisation with in-flight batch far below the max ceiling means the engines are starved (requests held upstream in admission — see Router), not compute-bound. Caveats (image-gated): no <code>llm_request</code> span (coarse gantt), no per-pool KV, no DCGM GPU-compute util.`+ceil,'hl');
})();
// A series of all-nulls is NOT the same as a series of zeros, and an empty axis
// says the second. Every panel below states which source produced it and, when
// neither the gauge nor the iteration-log fallback exists, says so in words.
function notCaptured(el,what){
  el.append('div').attr('class','cap')
    .style('padding','14px 0').style('color','#d29922')
    .html(`<b>Not captured in this run.</b> ${what} No worker <code>/metrics</code> endpoint `
         +`was scraped and the per-iteration log carries no substitute, so this is absent `
         +`data — not a measured zero.`);
}
function anyVal(s,i){return (s||[]).some(d=>d[i]!=null);}
(function(){const src=DATA.en.kvutil_src||'';
 const el=panel('engine','KV cache utilisation over the run (% — peak worker)',
  `Source: <code>${src}</code>, max across workers per role (the busiest worker). `
 +`Low peaks with in-flight batch far below the ceiling ⇒ the engines are starved, not KV-bound.`
 +(src.startsWith('iter_bins')?' <b>Derived from the per-iteration log</b> because the Prometheus gauge was not scraped; same quantity, denser sampling.':''),true);
 if(!anyVal(DATA.en.kvutil_pf,1)&&!anyVal(DATA.en.kvutil_de,1)){
   notCaptured(el,'<code>dynamo_component_gpu_cache_usage_percent</code> is a worker-side gauge.');return;}
 lg(el,[[C.pf,'prefill'],[C.de,'decode']]);
 lineChart(el,DATA.en.kvutil_pf.map((d,i)=>[d[0],d[1],(DATA.en.kvutil_de[i]||[])[1]]),{unit:'%',keys:[[1,C.pf],[2,C.de]]});})();
(function(){const src=DATA.en.true_hit_src||'';
 const el=panel('engine','True block cache-hit over the run (%)',
  `Source: <code>${src}</code> — the REAL reuse, measured at the engine rather than inferred from the client.`
 +(src.startsWith('iter_bins')?' <b>Derived from the per-iteration log</b> as <code>cached_kv_tokens / (cached_kv_tokens + num_ctx_tokens)</code> over context workers, because <code>trtllm_kv_cache_hit_rate</code> was not scraped. Token-weighted and context-phase only, so it is close to but not identical to the gauge.':''),true);
 if(!anyVal(DATA.en.true_hit_pct,1)){
   notCaptured(el,'<code>trtllm_kv_cache_hit_rate</code> is a worker-side gauge.');return;}
 lineChart(el,DATA.en.true_hit_pct,{unit:'%',keys:[[1,C.grn]]});})();

// ================= FRONTEND =================
// ---- engine iteration telemetry (TRT-LLM print_iter_log) ----
(function(){
 const IT=DATA.iter; if(!IT||!Object.keys(IT).length) return;
 const ROLE={prefill:C.pf, decode:C.de};
 (function(){const el=panel('engine','Engine KV-cache occupancy (utilisation)',
   'Formula, verbatim from TensorRT-LLM <code>_torch/pyexecutor/py_executor.py</code> where the iteration log is emitted:<br><code style="color:#f2cc60">kv_cache_util = 1.0 - kv_stats.free_num_blocks / kv_stats.max_num_blocks</code><br>So this is <b>pool occupancy</b> — the fraction of KV blocks currently allocated — and <b>not</b> the cache hit rate (that is <code>trtllm_kv_cache_hit_rate</code>, the <i>True block cache-hit</i> panel). Sampled once per forward step and median-reduced to 1s bins, so far denser than the 3s Prometheus gauge. Low values are expected at this load: <code>free_gpu_memory_fraction: 0.8</code> sizes the pool for high concurrency (prefill 13,453 blocks / decode 52,767) while the run is concurrency 3 — peak occupancy 17.6% / 3.7%.',true);
  const _wk=Object.keys(IT).sort(), _pal=[C.pf,C.de,C.grn,C.route,C.cy,C.adm];
  lg(el,_wk.map((w,i)=>[_pal[i%_pal.length],w]));
  const ser=_wk.map(w=>IT[w].kv_cache_util); const keys=_wk.map((w,i)=>[w,_pal[i%_pal.length]]);
  const L=54,B=28,h=240,s=svg(el,VW,h);
  const all=ser.flat(); if(!all.length) return;
  const x=d3.scaleLinear().domain([0,RD]).range([L,VW-10]);
  const y=d3.scaleLinear().domain([0,d3.max(all,d=>d[1])*1.1||1]).nice().range([h-B,8]);
  s.append('g').attr('class','axis').attr('transform',`translate(0,${h-B})`).call(d3.axisBottom(x).ticks(8).tickFormat(v=>v+'s'));
  s.append('g').attr('class','axis').attr('transform',`translate(${L},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(v=>(v*100).toFixed(0)+'%'));
  wmLine(s,x,8,h-B,null);
  const ln=d3.line().x(d=>x(d[0])).y(d=>y(d[1]));
  ser.forEach((d,i)=>s.append('path').datum(d).attr('fill','none').attr('stroke',keys[i][1]).attr('stroke-width',1.2).attr('opacity',.9).attr('d',ln));
 })();
 // Per-worker in-flight batch, straight from the TRT-LLM iteration log.
 // Raw integers (max per 1s bin -- a median of [1,2] would be 1.5, a batch the
 // engine never actually ran). No ceiling line: the configured max is stated in
 // the caption instead, because plotting it flattens the series onto the axis.
 const WK=Object.keys(IT).sort();
 const PAL=[C.pf,C.de,C.grn,C.route,C.cy,C.adm];
 function batchPanel(role, ceil, tok){
   // Worker keys are the LOG FILE stem, `<node>_<role>_w<N>` (e.g.
   // `hecate0176_decode_w0`) -- the role is in the middle, not at the front. A
   // startsWith() here matched nothing on every srt-slurm run ever rendered, and
   // because the function returns silently on an empty match, both batch panels
   // were absent rather than broken. Substring match, anchored on the `_role_`
   // separator so a node named e.g. `decode01` cannot be mistaken for a role.
   const wk=WK.filter(w=>w.includes('_'+role+'_')||w.includes('_'+role));
   if(!wk.length) return;
   const all=wk.flatMap(w=>IT[w].num_scheduled_requests);
   if(!all.length) return;
   const ymax=d3.max(all,d=>d[1]);
   const vals=all.map(d=>d[1]).sort((a,b)=>a-b);
   const med=vals[Math.floor(vals.length/2)];
   // The ceiling is drawn whenever it is on a comparable scale to the data. When it
   // is far above (prefill runs a batch of ~1-10 against max_batch_size 256) drawing
   // it to scale squashes every series onto the axis and the panel says nothing, so
   // the axis stays on the data and the ceiling is reported as a utilisation figure.
   const onScale = ceil!=null && ceil>0 && ymax >= ceil*0.25;
   const pctOf = ceil ? v=>` (${(v/ceil*100).toFixed(1)}% of ceiling)` : ()=>'';
   const cap=`<code>num_scheduled_requests</code> from the TRT-LLM per-iteration log, one line per `
     +`${role} worker, raw integer (max per 1s bin), plotted against this run's configured `
     +`<b><code>max_batch_size = ${ceil==null?'?':ceil}</code></b>`+(tok?` (<code>max_num_tokens = ${tok}</code>)`:'')
     +` from <code>trtllm_config_${role}.yaml</code>. `
     +(onScale
        ? `The ceiling is the dashed line — the gap between it and the series is unserved batch capacity.`
        : `The ceiling is <b>off-scale</b> here and is not drawn, because plotting it would squash `
          +`every series onto the axis. Peak batch <b>${ymax}</b>${pctOf(ymax)}, median <b>${med}</b>${pctOf(med)} `
          +`— at this ratio the batch is bounded by something other than <code>max_batch_size</code>`
          +(tok?`, most likely the <code>max_num_tokens = ${tok}</code> budget.`:'.'));
   const el=panel('engine',`${role==='prefill'?'Prefill':'Decode'} in-flight batch size per worker`,cap,true);
   lg(el,wk.map((w,i)=>[PAL[i%PAL.length],w])
        .concat(onScale?[['#d29922',`max_batch_size = ${ceil}`]]:[]));
   const L=54,B=28,h=230,s=svg(el,VW,h);
   const ytop = onScale ? ceil*1.06 : ymax;
   const x=d3.scaleLinear().domain([0,RD]).range([L,VW-10]);
   const y=d3.scaleLinear().domain([0,ytop]).range([h-B,10]);
   s.append('g').attr('class','axis').attr('transform',`translate(0,${h-B})`)
    .call(d3.axisBottom(x).ticks(8).tickFormat(v=>v+'s'));
   s.append('g').attr('class','axis').attr('transform',`translate(${L},0)`)
    .call(d3.axisLeft(y).ticks(Math.min(Math.ceil(ytop),6)).tickFormat(d3.format('d')));
   wmLine(s,x,10,h-B,null);
   if(onScale){
     s.append('line').attr('x1',L).attr('x2',VW-10).attr('y1',y(ceil)).attr('y2',y(ceil))
      .attr('stroke','#d29922').attr('stroke-width',1.4).attr('stroke-dasharray','6,4').attr('opacity',.95);
     s.append('text').attr('x',VW-14).attr('y',y(ceil)-5).attr('text-anchor','end')
      .attr('fill','#d29922').attr('font-size','10px').text(`max_batch_size = ${ceil}`);
   }
   const ln=d3.line().x(d=>x(d[0])).y(d=>y(d[1]));
   wk.forEach((w,i)=>s.append('path').datum(IT[w].num_scheduled_requests).attr('fill','none')
     .attr('stroke',PAL[i%PAL.length]).attr('stroke-width',1.2).attr('opacity',.85).attr('d',ln));
 }
 batchPanel('prefill', DATA.iter_cfg.pf_batch, DATA.iter_cfg.pf_tok);
 batchPanel('decode',  DATA.iter_cfg.de_batch, DATA.iter_cfg.de_tok);
})();

note('frontend',`Both panels come from the <b>frontend</b> endpoint (<code>:8000/metrics</code>). The raw Prometheus text names these counters with a <code>_total</code> suffix; the ingest strips it per the OpenMetrics convention, so the series below are the same metrics under their normalised names.`,'hl');
(function(){const el=panel('frontend','Requests',
  '<code>requests_total</code> with the three instantaneous gauges <code>inflight_requests</code>, <code>queued_requests</code> and <code>active_requests</code>. '
 +'<b>requests_total is cumulative</b> (it climbs to ~2.1k), so plotting it raw would pin the three gauges — which never exceed single digits — flat onto the axis; it is therefore drawn as its <b>rate in requests/second, differenced over a trailing 30 s window</b>. The window is not cosmetic: scrapes are ~0.3 s apart while requests arrive at ~0.5/s, so a scrape-to-scrape difference would be 0 in most intervals and 3.33 in the rest — a staircase of the sampling grid, not throughput. The three gauges are raw instantaneous counts. '
 +'<b>Note:</b> <code>inflight_requests</code> and <code>active_requests</code> are identical in this build (they differ at 1 of 65,405 samples), so those two lines coincide — only <code>queued_requests</code> tracks separately.',true);
 lg(el,[[C.grn,'requests_total (req/s, 30s rate)'],[C.adm,'inflight_requests'],[C.cy,'queued_requests'],[C.pf,'active_requests (= inflight)']]);
 lineChart(el,DATA.fe.req_rate.map((d,i)=>[d[0],d[1],DATA.fe.inflight[i][1],DATA.fe.queued[i][1],DATA.fe.active[i][1]]),
  {unit:'n',keys:[[1,C.grn],[2,C.adm],[3,C.cy],[4,C.pf]]});})();
(function(){const el=panel('frontend','Tokenizer',
  'The four L1 tokenizer prefix-cache counters — <code>tokenizer_cache_{hits,misses,cached_tokens,uncached_tokens}_total</code> — as raw cumulative values. '
 +'<b>Note the log y-axis:</b> the pair counted in <i>requests</i> (hits, misses) and the pair counted in <i>tokens</i> (cached, uncached) differ by roughly five orders of magnitude, so a linear axis would flatten hits and misses onto zero. A log axis cannot plot 0, so each line begins at its first non-zero sample.',true);
 lg(el,[[C.grn,'hits'],[C.adm,'misses'],[C.cy,'cached_tokens'],[C.pf,'uncached_tokens']]);
 lineChart(el,DATA.fe.tk_hits.map((d,i)=>[d[0],d[1],DATA.fe.tk_misses[i][1],DATA.fe.tk_cached[i][1],DATA.fe.tk_uncached[i][1]]),
  {unit:'n',keys:[[1,C.grn],[2,C.adm],[3,C.cy],[4,C.pf]],logy:true});})();

// hash-based tab select (for headless screenshots) + default
// ---- declarative spec panels (src/visualization/panels.py) ------------------
// ONE renderer for every spec panel. It is handed {series,unit,split_by,...} and
// knows nothing else about the signal, so no panel can acquire bespoke drawing
// behaviour -- which is the property the spec table exists to guarantee. The
// caption is assembled from fixed fields (why / source / caveat), never from the
// run's values, so two runs produce byte-identical captions.
(function(){
 const P=DATA.panels; if(!P||!Object.keys(P).length) return;
 const PAL=[C.pf,C.de,C.grn,C.route,C.cy,C.adm,'#d29922','#a371f7','#db6d28','#3fb950'];
 const fmtU=(u)=>{
   if(u==='ratio')  return v=>(v*100).toFixed(0)+'%';
   if(u==='s')      return v=>v>=1?v.toFixed(1)+'s':(v*1000).toFixed(0)+'ms';
   if(u==='ms')     return v=>v>=1000?(v/1000).toFixed(1)+'s':v.toFixed(0)+'ms';
   return d3.format('~s');
 };
 Object.keys(P).sort().forEach(pid=>{
  const p=P[pid], keys=Object.keys(p.series).sort();
  if(!keys.length) return;
  // Fixed-composition caption: purpose, then provenance, then the known trap.
  let cap=p.why
    +`<br><span class="src">source: <code>${p.source.join('</code> + <code>')}</code>`
    +` &middot; ${p.kind}${p.split_by?' &middot; split by <code>'+p.split_by+'</code>':''}</span>`;
  if(p.caveat) cap+=`<br><span class="warn">caveat: ${p.caveat}</span>`;
  const el=panel(p.tab,p.title,cap,keys.length>3);
  // A series that never moves is stated in words. Drawing a flat line invites the
  // reader to conclude the panel is broken, when "it never moved" is the finding.
  const flat=keys.every(k=>{const v=p.series[k].map(d=>d[1]);return Math.max(...v)===Math.min(...v)});
  if(flat){
    const v=p.series[keys[0]][0][1];
    el.append('div').attr('class','note').html(
      `Constant <b>${fmtU(p.unit)(v)}</b> for the whole run across `
      +`${keys.length} series &mdash; this run never exercised this signal.`);
    return;
  }
  // High-cardinality panels (a per-thread runtime gauge is ~144 series) draw the
  // busiest few rather than everything. Drawing all of them is a point cloud and a
  // legend nobody can read; drawing one merged series hides which thread saturated.
  // Ranked by PEAK because saturation is a property of the busiest members, and the
  // omitted count is always stated -- a silently truncated panel reads as complete.
  const MAXS=12; let drawn=keys, omitted=0;
  if(keys.length>MAXS){
    drawn=keys.slice().sort((a,b)=>d3.max(p.series[b],d=>d[1])-d3.max(p.series[a],d=>d[1])).slice(0,MAXS);
    omitted=keys.length-MAXS;
  }
  if(drawn.length>1) lg(el,drawn.map((k,i)=>[PAL[i%PAL.length],p.split_by?`${p.split_by}=${k}`:k]));
  if(omitted) el.append('div').attr('class','note').html(
    `Showing the <b>${MAXS}</b> highest-peak of <b>${keys.length}</b> series; `
    +`<b>${omitted}</b> lower-peak series omitted from the chart.`);
  const L=58,B=28,H=keys.length>3?250:200,s=svg(el,VW,H);
  const all=drawn.flatMap(k=>p.series[k]);
  const x=d3.scaleLinear().domain([0,d3.max(all,d=>d[0])||1]).range([L,VW-12]);
  const y=d3.scaleLinear().domain([0,(d3.max(all,d=>d[1])||1)*1.08]).nice().range([H-B,8]);
  s.append('g').attr('class','axis').attr('transform',`translate(0,${H-B})`)
   .call(d3.axisBottom(x).ticks(8).tickFormat(v=>v.toFixed(0)+'s'));
  s.append('g').attr('class','axis').attr('transform',`translate(${L},0)`)
   .call(d3.axisLeft(y).ticks(5).tickFormat(fmtU(p.unit)));
  wmLine(s,x,8,H-B,p);
  const ln=d3.line().x(d=>x(d[0])).y(d=>y(d[1]));
  drawn.forEach((k,i)=>s.append('path').datum(p.series[k]).attr('fill','none')
    .attr('stroke',PAL[i%PAL.length]).attr('stroke-width',1.2).attr('opacity',.9).attr('d',ln));
 });
})();

// ---- per-request decomposition cards (DATA.rt.requests) --------------------
// Entity type 2. One card shape for every request in every run: the four bands that
// sum to total_ms, what the request WAS, and where it was routed. Nothing here is
// conditional on the run -- a field the run did not produce renders as an em dash
// rather than changing the card's shape.
(function(){
 const RT=DATA.rt||{}, R=RT.requests||{}; const xids=Object.keys(R);
 if(!xids.length) return;
 const BANDC=[C.adm,C.pf,C.route,C.de];
 const el=panel('session','Per-request decomposition',
   'The slowest requests in the run, each broken into the four phases that sum to its '
   +'total. <b>Routing is an outcome, not a rationale</b>: the router\'s cost comparison '
   +'is logged without a request id and cannot be joined, so this shows which worker was '
   +'chosen and with how much prefix overlap, not why it won.'
   +'<br><span class="src">source: <code>request_trace.jsonl</code> + <code>tempo_traces/</code></span>',true);

 if(RT.belief){
   const b=RT.belief, bad=b.disagree;
   el.append('div').attr('class','note'+(bad?' warn':'')).html(
     `Router belief vs engine reality: <b>${b.n-bad}/${b.n}</b> requests agree within `
     +`${(b.threshold*100).toFixed(0)}% (router <code>overlap_blocks</code> x block size `
     +`vs engine <code>cached_tokens</code>), worst error <b>${(b.worst*100).toFixed(1)}%</b>. `
     +(bad?'A disagreement means the router scored a candidate on a prefix the engine does not hold.'
          :'The router scored candidates on a prefix the engine actually had.'));
 }

 const slowest=xids.map(x=>[x,R[x]])
   .sort((a,b)=>(b[1].attrs.total_ms||0)-(a[1].attrs.total_ms||0)).slice(0,12);
 const W=760,BH=26;
 slowest.forEach(([xid,c])=>{
   const tot=c.bands.reduce((a,[,,v])=>a+(v||0),0)||1;
   const row=el.append('div').attr('class','reqcard');
   const a=c.attrs, rt=c.routing;
   row.append('div').attr('class','reqhead').html(
     `<code>${xid.slice(0,8)}</code> &middot; <b>${fmtS(tot)}</b> total`
     +` &middot; ISL ${d3.format('~s')(a.isl||0)} &rarr; OSL ${d3.format('~s')(a.osl||0)}`
     +` &middot; cache hit ${a.kv_hit_rate==null?'&mdash;':(a.kv_hit_rate*100).toFixed(1)+'%'}`
     +` &middot; turn ${a.turn_index==null?'&mdash;':a.turn_index}`);
   const sv=svg(row,W,BH+6); let x0=0;
   c.bands.forEach(([,label,v],i)=>{
     const w=(v||0)/tot*W;
     if(w>0){
       sv.append('rect').attr('x',x0).attr('y',3).attr('width',w).attr('height',BH)
         .attr('fill',BANDC[i%BANDC.length]).attr('opacity',.85)
         .append('title').text(`${label}: ${fmtS(v)}`);
       if(w>54) sv.append('text').attr('x',x0+4).attr('y',3+BH/2+4).attr('fill','#0d1117')
         .attr('font-size',10).text(fmtS(v));
     }
     x0+=w;
   });
   row.append('div').attr('class','reqmeta').html(
     rt ? `routed to worker <code>${String(rt.worker_id??'').slice(-6)||'&mdash;'}</code>`
          +` rank <code>${rt.dp_rank??'&mdash;'}</code>`
          +` &middot; overlap <b>${rt.overlap_blocks==null?'&mdash;':d3.format('~s')(rt.overlap_blocks)}</b> blocks`
          +` &middot; router ${rt.router_ms==null?'&mdash;':rt.router_ms+'ms'}`
          +` &middot; admission ${rt.admission_ms==null?'&mdash;':rt.admission_ms+'ms'}`
        : 'routing not joinable for this run (no span data)');
 });
 lg(el,c_bandLegend());
 function c_bandLegend(){return (RT.bands||[]).map((b,i)=>[BANDC[i%BANDC.length],b[1]])}
 const cst=RT.const||{};
 if(Object.keys(cst).length) el.append('div').attr('class','note').html(
   'Constant for every request in this run: '
   +Object.entries(cst).map(([k,v])=>`<code>${k}</code>=${v}`).join(', ')
   +' &mdash; reported rather than charted.');
})();

// ---- session decomposition (DATA.rt) ---------------------------------------
// One row per session, same columns for every session and every run.
(function(){
 const S=(DATA.rt||{}).sessions||[]; if(!S.length) return;
 const el=panel('session','Sessions',
   'One row per session, ordered by first request. <b>busy</b> is time the server was '
   +'working on this session\'s turns; <b>idle</b> is wall-clock the session existed '
   +'without work in flight &mdash; harness think time, which a session-level latency '
   +'figure would otherwise absorb and attribute to the server.'
   +'<br><span class="src">source: <code>request_trace.jsonl</code></span>',true);
 const t=el.append('table').attr('class','tbl');
 t.append('thead').append('tr').selectAll('th')
  .data(['session','turns','span','busy','idle','TTFT p50','KV hit p50','decode workers','prefill ranks'])
  .join('th').text(d=>d);
 const med=a=>{const v=a.filter(x=>x!=null).sort((p,q)=>p-q);return v.length?v[Math.floor(v.length/2)]:null};
 const tb=t.append('tbody');
 S.forEach(s=>{
   const ttft=med(s.ttft_ms), kv=med(s.kv_hit);
   tb.append('tr').selectAll('td').data([
     s.session_id.slice(0,8), s.turns,
     (s.span_ms/1000).toFixed(1)+'s', (s.busy_ms/1000).toFixed(1)+'s',
     (s.idle_ms/1000).toFixed(1)+'s',
     ttft==null?'-':(ttft/1000).toFixed(2)+'s',
     kv==null?'-':(kv*100).toFixed(1)+'%',
     s.decode_workers.map(w=>String(w).slice(-6)).join(', ')||'-',
     s.prefill_ranks.join(', ')||'-',
   ]).join('td').text(d=>d);
 });
 const cst=(DATA.rt||{}).const||{};
 if(Object.keys(cst).length) note('session',
   'Constant for every request in this run: '
   +Object.entries(cst).map(([k,v])=>`<code>${k}</code>=${v}`).join(', ')
   +'. Reported rather than charted &mdash; a flat line reads as a broken panel, '
   +'whereas "this never varied" is itself a result.');
})();

const h=(location.hash||'').replace('#','');
if(TABS.some(t=>t[0]===h)) act(h); else if(TABS.length) act(TABS[0][0]);
</script></body></html>"""

html=HTML.replace("__D3__",D3_TAG).replace("__DATA__",json.dumps(DATA,separators=(",",":")))
open(OUT,"w").write(html)
_log.info(f"wrote {OUT} ({os.path.getsize(OUT)/1024/1024:.1f} MB)")

# The HTML embeds DATA verbatim, so dumping it is not a second rendering path --
# it is the same payload in a form that can be diffed between two runs, asserted
# on in CI, or read at all on a machine with no browser.
if _args.dump_json:
    with open(_args.dump_json,"w") as _jf:
        json.dump(DATA,_jf,indent=1,sort_keys=True)
    _log.info(f"wrote {_args.dump_json} ({os.path.getsize(_args.dump_json)/1024/1024:.1f} MB) "
              f"-- tabs={[k for k,v in DATA['tabs'].items() if v]}")
