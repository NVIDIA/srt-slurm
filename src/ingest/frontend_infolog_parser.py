# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request records + stage breakdown from a Dynamo frontend log at INFO level.

This is the second producer of the dashboard's per-request **stage IR** (the first
is `_ordered_spans()` in src/visualization/build_dynamo_bench_dash.py, which reads
Tempo spans). Both emit the same structure so a single renderer can draw either:

    StageRow = {
        "name":   str,        # stage label shown in the breakdown table
        "t":      float,      # start offset, ms, relative to request start (0 = start)
        "d":      float,      # duration, ms
        "depth":  int,        # indent level; 0 = the request root
        "opaque": bool,       # True = an unseparable aggregate (renders hatched)
        "comp":   str | None, # component tag: "frontend" / "prefill" / "backend"
        "role":   str | None, # "prefill" / "decode" on the ONE row representing that
                              # phase's segment; the renderer uses it to cut the axis
                              # at first token without knowing either source's names.
    }

The span source produces a deep tree (~20 rows/request). This source produces a
shallow one (3-5 rows) because the frontend log cannot decompose TTFT: `ttft_ms` is
a single number covering routing + queue + prefill + KV transfer. That gap is
represented honestly as one `opaque=True` row rather than invented sub-stages.

Why this source exists at all: it needs **only the frontend log**. No Tempo, no
tracing backend, no DEBUG level, no observability flags. Verified against run
2663744, whose fingerprint records DYN_LOG=info -- the log reproduced aiperf's own
published TTFT percentiles to within ~2%.

Two log flavours are handled:

  * raw container stdout (AgentX / srtctl) -- ANSI-coloured, one record per line;
  * python-logging wrapped (AgentPerf / sflow) -- each line prefixed
    "<ts> - sflow.task.<name> - <LEVEL> - <rank>: ", and records may be TORN across
    two physical lines because concurrent writers interleave into one pipe.

Tearing is not cosmetic. On the reference AgentPerf log 663 records are split (41
mid-ANSI-escape); a naive physical-line parse loses 504 of 51,613 `request
completed` records. That barely moves a percentile (p50 1.293s -> 1.294s) but it
destroys any counter: an in-flight sweep drifts to a phantom 622 concurrent
requests against a true peak of 128. Reassembly is therefore mandatory, not
optional -- see `_iter_records`.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone

# ANSI CSI. Written as a normal (non-raw) string: r"\x1b" is a literal backslash.
_ANSI = re.compile("\x1b\\[[0-9;]*m")
_SFLOW = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ - sflow\.task\.\S+ - \w+ - \d+: ")
_TS = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z)")
_HEAD = re.compile(r"^\d{4}-\d\d-\d\dT\S+Z\s+(?:TRACE|DEBUG|INFO|WARN|ERROR)\b")
_KV = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')

# Target-colon then OPTIONAL whitespace: repaired records can lose the space.
_RE_RECV = re.compile(r"metrics:\s*request received")
_RE_DONE = re.compile(r"metrics:\s*request completed")
_RE_RESP = re.compile(r"service_v2:\s*http response sent")
_RE_SEL = re.compile(r"selector:\s*Selected (?:pinned )?worker")
# Only responses inside the http-request span are real inference. The same Tower
# hook fires for /health and /metrics: on run 2663744, 12,964 of 15,471 `http
# response sent` lines are probe traffic. Counting them overstates requests 6.2x.
_RE_INSPAN = re.compile(r"http-request:")
# The router's cost formula, JSON flavour. On a DYN_LOGGING_JSONL=true run every
# record is JSON, so the text-only _RE_SEL above never matches and the whole router
# family counted as zero -- 597 real lines reported as 0 on the reference run. These
# carry the scoring inputs (worker_id, dp_rank, effective cached blocks, the
# prefill_load_scale/decode_blocks terms) but NO request_id, so they support
# aggregate and per-worker analysis and cannot be joined to a request.
_RE_SEL_JSON_TARGET = re.compile(r"scheduling::selector$")
# Two record shapes share that target, and only one is a decision:
#   "Selected [pinned] worker: worker_type=..., worker_id=N dp_rank=N, logit=..."
#       -> the CHOICE. One per phase per request (prefill + decode).
#   "[Pinned ]Formula for worker_id=N dp_rank=N with N effective cached blocks: ..."
#       -> the per-CANDIDATE score that fed the choice.
# On the reference log: 266 decisions against 329 candidate scores. The "pinned" vs
# fresh split of the decisions is itself the finding -- 26 of 266 (9.8%) were real
# KV-aware selections and the rest were session affinity resolving to an already
# bound worker, which independently matches the span-derived 13/133.
_RE_SEL_DECISION = re.compile(
    r"Selected (?P<pinned>pinned )?worker(?::\s*worker_type=(?P<worker_type>\w+),\s*"
    r"worker_id=(?P<worker_id>\d+)\s+dp_rank=(?P<dp_rank>\d+))?")
_RE_SEL_FORMULA = re.compile(
    r"worker_id=(?P<worker_id>\d+)\s+dp_rank=(?P<dp_rank>\d+)"
    r".*?([\d.]+) effective cached blocks:\s*(?P<cost>[\d.]+)")


def _kv(line):
    """Key/value pairs, FIRST occurrence wins.

    `request_id` and `model` each appear twice per lifecycle line -- once as an
    event field, once inherited from the enclosing `http-request` span -- and the
    two request_id VALUES DIFFER. The first is the internal lifecycle id, which is
    also what the router's selector line emits; the second is the HTTP server's.
    A last-wins dict (the default for every idiomatic parser) silently swaps the
    join key: joining selector->completion drops from 100% to ~0%, which is
    indistinguishable from the genuine Dynamo-1.3 limitation of having no
    request_id on the selector line at all.
    """
    out = {}
    for m in _KV.finditer(line):
        out.setdefault(m.group(1), m.group(2) if m.group(2) is not None else m.group(3))
    return out


def _iter_records(path):
    """Yield normalised single-line Dynamo records, reassembling torn ones.

    Order matters: strip the wrapper, pre-strip complete CSI sequences so an inner
    timestamp header can be recognised, concatenate continuations with NO
    delimiter, then strip CSI again (a sequence split across the tear only becomes
    complete after joining).
    """
    buf = None
    with open(path, errors="replace") as fh:
        for line in fh:
            line = _SFLOW.sub("", line.rstrip("\n").rstrip("\r"))
            probe = _ANSI.sub("", line)
            if _HEAD.match(probe) or probe.lstrip().startswith("{"):
                if buf is not None:
                    yield _ANSI.sub("", buf)
                buf = line
            elif buf is not None:
                buf += line
    if buf is not None:
        yield _ANSI.sub("", buf)


def _json_obj(line):
    """A structured log record, or None if this line is the text flavour."""
    s = line.lstrip()
    if not s.startswith("{"):
        return None
    try:
        d = json.loads(s)
    except ValueError:
        return None
    return d if isinstance(d, dict) and "time" in d and "message" in d else None


def _ts_ns(line):
    m = _TS.search(line)
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _f(d, k):
    v = d.get(k)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _i(d, k):
    v = _f(d, k)
    return None if v is None else int(v)


def parse_frontend_log(path):
    """Parse a frontend log into per-request records and the shared stage IR.

    Returns {"requests": {key: rec}, "stages": {key: [StageRow]}, "stats": {...}}.

    `key` is `x_request_id` when the client supplied an `x-request-id` header
    (which makes the record joinable to the load generator's own artifacts, the
    same key the ingest bundle uses), else the internal request_id.
    """
    recv, done, resp = {}, {}, {}
    sel = defaultdict(list)
    sel_all = []
    n_rec = n_sel_no_id = n_json = 0
    n_sel_pinned = n_sel_scored = 0

    for line in _iter_records(path):
        n_rec += 1

        # ---- JSON-lines flavour -------------------------------------------
        # Runs with observability enabled emit structured logs, where the
        # per-request summary is not `request completed` at all: it is the
        # request_span SPAN_CLOSED record, which carries the same span fields
        # plus time.duration_us. A text-only parser silently finds nothing on
        # exactly the runs that have the most instrumentation.
        j = _json_obj(line)
        if j is not None:
            n_json += 1
            if j.get("message") == "SPAN_CLOSED" and j.get("target") == "request_span":
                rid = j.get("request_id")
                if rid is None:
                    continue
                t_end = _ts_ns(j.get("time", ""))
                dur_ms = _f(j, "time.duration_us")
                dur_ms = None if dur_ms is None else dur_ms / 1000.0
                d = dict(j)
                d["elapsed_ms"] = dur_ms
                # A span that closed is a request that finished; these records
                # carry no `status` field, so successes cannot be told from
                # cancellations here. Treat as success and say so in stats.
                d.setdefault("status", "success")
                done[rid] = (t_end, d)
                if dur_ms is not None and t_end is not None:
                    recv[rid] = (t_end - int(dur_ms * 1e6), d)
                continue
            # NOT a SPAN_CLOSED record. Previously every such line was discarded
            # here, which on a JSONL run silently threw away the entire router
            # family: the text matchers below can never see a JSON line, so the
            # selector counters read 0 while the log held hundreds of records.
            if _RE_SEL_JSON_TARGET.search(j.get("target") or ""):
                msg = j.get("message") or ""
                dec = _RE_SEL_DECISION.search(msg)
                if dec:
                    n_sel_no_id += 1     # decision records carry no request_id at all
                    n_sel_pinned += 1 if dec.group("pinned") else 0
                    sel_all.append((_ts_ns(j.get("time", "")),
                                    dec.group("worker_type") or "",
                                    dec.group("worker_id") or "",
                                    dec.group("dp_rank") or ""))
                elif _RE_SEL_FORMULA.search(msg):
                    n_sel_scored += 1    # a per-candidate score, not a choice
            continue

        # ---- human-readable tracing flavour --------------------------------
        if _RE_SEL.search(line):
            d = _kv(line)
            rid = d.get("request_id")
            # Every selector line is independently timestamped, so per-(worker,
            # dp_rank) DECISION RATES are available even when the line carries no
            # request_id and cannot be joined to tokens. On Dynamo 1.3 this is the
            # only route to dp_rank at all -- and it is what shows prefill dp_rank 1
            # going to zero traffic mid-run on 2663744.
            sel_all.append((_ts_ns(line),
                            (d.get("worker_type") or "").rstrip(","),
                            (d.get("worker_id") or "").rstrip(","),
                            (d.get("dp_rank") or "").rstrip(",")))
            if rid is None:
                n_sel_no_id += 1          # Dynamo < 2026-08-10: not joinable
                continue
            sel[rid].append((_ts_ns(line),
                             (d.get("worker_type") or "").rstrip(","),
                             (d.get("worker_id") or "").rstrip(","),
                             (d.get("dp_rank") or "").rstrip(",")))
            continue
        is_recv, is_done = _RE_RECV.search(line), _RE_DONE.search(line)
        is_resp = _RE_RESP.search(line) and _RE_INSPAN.search(line)
        if not (is_recv or is_done or is_resp):
            continue
        d = _kv(line)
        rid = d.get("request_id")
        if rid is None:
            continue
        t = _ts_ns(line)
        if is_recv:
            recv[rid] = (t, d)
        elif is_done:
            done[rid] = (t, d)
        else:
            # `http response sent` carries NO request_id of its own -- the event
            # fields are only status/latency_ms, so every id on the line is
            # inherited from the span. First-wins therefore yields the HTTP
            # request_id, which does NOT equal the lifecycle id on `request
            # completed`. Key on x_request_id, the one identifier both lines
            # agree on, and fall back to the (mismatched) rid only when the
            # client sent no x-request-id header.
            resp[d.get("x_request_id") or rid] = (t, d)

    requests, stages = {}, {}
    for rid, (t_done, dd) in done.items():
        t_recv = recv.get(rid, (None,))[0]
        ttft = _f(dd, "ttft_ms")
        elapsed = _f(dd, "elapsed_ms")
        key = dd.get("x_request_id") or rid
        requests[key] = {
            "ttft": ttft,
            "e2e": elapsed,
            "isl": _i(dd, "input_tokens"),
            "osl": _i(dd, "output_tokens"),
            "itl": _f(dd, "avg_itl_ms"),
            "status": dd.get("status"),
            "error_type": dd.get("error_type"),
            "ts_ns": t_recv if t_recv is not None else t_done,
            "rid": rid,
            # A only: populated from backend LLMMetricAnnotation via set_worker_info.
            # dynamo-sglang does not supply these, so they are frequently absent.
            "pf_worker": dd.get("prefill_worker_id"),
            "de_worker": dd.get("decode_worker_id"),
        }
        st = _build_stages(rid, key, dd, t_recv, t_done, ttft, elapsed, resp, sel)
        if st:
            stages[key] = st

    joinable = sum(1 for r in requests.values() if r["rid"] in sel)
    # Per-request route, keyed the same way as `requests`, for load-balance
    # attribution WITH tokens. Empty when the selector line carries no request_id.
    routes = {}
    for key, rec in requests.items():
        for ts, wtype, wid, dp in sel.get(rec["rid"], []):
            if wtype:
                routes.setdefault(key, {})[wtype] = (wid, dp)
    return {
        "requests": requests,
        "stages": stages,
        "routes": routes,
        "sel_events": [e for e in sel_all if e[0] is not None],
        "stats": {
            "records": n_rec,
            "json_records": n_json,
            "received": len(recv),
            "completed": len(done),
            "http_responses_in_span": len(resp),
            "selector_with_request_id": len(sel),
            "selector_without_request_id": n_sel_no_id,
            # The pinned/fresh split of routing DECISIONS. A run where almost every
            # decision is "pinned" is one where session affinity, not KV-aware
            # scoring, is choosing the worker -- which makes router-tuning experiments
            # mostly no-ops and is invisible from the hit rate alone.
            "selector_decisions_pinned": n_sel_pinned,
            "selector_candidate_scores": n_sel_scored,
            "routing_joinable": joinable,
            "with_x_request_id": sum(1 for k, r in requests.items() if k != r["rid"]),
            "success": sum(1 for r in requests.values() if r["status"] == "success"),
        },
    }


def _build_stages(rid, key, dd, t_recv, t_done, ttft, elapsed, resp, sel):
    """Emit the shared StageRow list for one request.

    Depth 0 is the request envelope. Depth 1 splits it at first token. Depth 2
    appears only when the router's selector line carried a request_id (Dynamo
    >= 2026-08-10), which is what lets routing be timed separately from the
    opaque remainder.
    """
    if elapsed is None:
        return None
    rows = [{"name": "http-request", "t": 0.0, "d": round(elapsed, 3),
             "depth": 0, "opaque": False, "comp": None, "role": None}]

    # Time to the Response object (Tower's on_response). For SSE this is the header
    # commit, well before any token -- NOT TTFT. Kept because a large value here
    # means pre-stream handler work, which is otherwise invisible.
    if key in resp and t_recv is not None:
        lat = _f(resp[key][1], "latency_ms")
        if lat is not None:
            rows.append({"name": "response-committed", "t": 0.0, "d": round(lat, 3),
                         "depth": 1, "opaque": False, "comp": "frontend", "role": None})

    if ttft is None:
        return rows

    # Routing, when the selector line is joinable. Both decisions land up front
    # (prefill then decode, ~3ms apart on the reference run), so this measures
    # admission+routing overhead and bounds the opaque remainder from below.
    route_end = 0.0
    marks = []
    if t_recv is not None and rid in sel:
        for ts, wtype, wid, dp in sorted((s for s in sel[rid] if s[0]), key=lambda s: s[0]):
            off = (ts - t_recv) / 1e6
            if 0 <= off <= ttft:
                marks.append((off, wtype, wid, dp))
        if marks:
            route_end = marks[-1][0]

    if marks:
        rows.append({"name": "admission + routing", "t": 0.0, "d": round(route_end, 3),
                     "depth": 1, "opaque": False, "comp": "frontend", "role": None})
        for off, wtype, wid, dp in marks:
            rows.append({"name": f"route:{wtype or '?'} (worker {str(wid)[-6:]} dp{dp or '?'})",
                         "t": round(off, 3), "d": 0.0,
                         "depth": 2, "opaque": False, "comp": wtype or None, "role": None})

    # The irreducible part. Everything between routing and the first token --
    # scheduler queue, prefill compute, KV transfer, decode handoff -- is one
    # number in the INFO log. Marked opaque so the renderer hatches it, matching
    # how the span source marks `handle_payload`.
    rows.append({"name": "queue + prefill + KV transfer",
                 "t": round(route_end, 3), "d": round(max(0.0, ttft - route_end), 3),
                 "depth": 1, "opaque": True, "comp": "prefill", "role": "prefill"})

    gen = elapsed - ttft
    if gen > 0:
        rows.append({"name": "generation (post first token)", "t": round(ttft, 3),
                     "d": round(gen, 3), "depth": 1, "opaque": False,
                     "comp": "backend", "role": "decode"})
        itl, osl = _f(dd, "avg_itl_ms"), _i(dd, "output_tokens")
        if itl and osl and osl > 1:
            rows.append({"name": f"{osl - 1} tokens x {itl:.2f}ms avg ITL",
                         "t": round(ttft, 3), "d": round(itl * (osl - 1), 3),
                         "depth": 2, "opaque": False, "comp": "backend", "role": None})
    return rows
