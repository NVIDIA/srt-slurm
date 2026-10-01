#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 processor: Dynamo ``dynamo-request-trace`` -> ``request_trace.jsonl`` (schema 4).

The frontend's own per-request record, written when ``observability.enabled`` sets
``DYN_REQUEST_TRACE`` (see ``ANALYTICS_REQUEST_TRACE_ENV`` in ``srtctl.core.schema``).
srt-slurm has captured this since PR #317 and nothing consumed it. It is the only
source for three things no other leg provides:

1. **KV-transfer cost.** The Prometheus ``trtllm_kv_transfer_*`` family is declared
   but sampled zero times, so the scrape stream cannot measure it at all. Here it is
   a per-request number, and it is not small: p50 is ~79% of the decode span, and it
   exceeds the entire prefill-side TTFT on 92 of 427 requests in the reference hour.
2. **A gapless TTFT waterfall.** Verified 0/560 negative residuals (see below).
3. **Session identity.** ``session_id`` is present on 100% of records, which is what
   makes a per-session view possible at all.

THE TTFT TRAP
-------------
``ttft_ms`` in this file is the PREFILL worker's first token, not the client's. On
disagg the client waits for the decode worker, which cannot start until the KV cache
has transferred. Measured against the AIPerf client export over 557 joined requests:

    client_ttft - ttft_ms                 -> p50 +127 ms, max +85.4 s, wrong on 510/557
    client_ttft - (ttft_ms + kv_transfer) -> p50 +1.4 ms, max +11.2 ms, wrong on 0/557

So the client-facing decomposition is::

    prefill_wait_ms + prefill_ms + kv_transfer_ms  = client TTFT   (+~1.5ms HTTP)
    + steady_decode_ms                             = total_ms

Both derived fields are emitted here rather than left to each consumer, so no panel
can accidentally plot the prefill-side number and label it TTFT.

``avg_itl_ms`` carries the same contamination: it averages over a decode span that
begins at the prefill worker's first token, so it absorbs the KV transfer. The
transfer-free form is emitted as ``clean_itl_ms`` (validated against the client's
``time_to_second_token``: p50 residual -0.10 ms).

FIELDS DELIBERATELY NOT CARRIED
-------------------------------
``replay.input_sequence_hashes`` is the bulk of the file -- 463k entries in a 20-minute
run, 1.9M in an hour -- and would dominate any embedded payload. It is not dropped
though: the one thing it proves, that turn N of a session reuses turn N-1's prefix,
is computed here into ``prefix_reuse_ratio`` and the raw array discarded. That keeps
the expensive input out of the bundle while keeping the signal.

``queue_depth`` and ``decode_dp_rank`` are carried verbatim despite reading a constant
0 in both reference runs. A field that is constant on one topology can be live on
another, and it is the renderer's job to drop a flat series -- not this layer's job to
decide the run had no queueing.

Stdlib only (runs under a bare cluster python3).

Usage:
    python3 -m src.ingest.request_trace IN_PATH OUT_PATH
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger("request_trace")

SCHEMA = "dynamo.request.trace.v1"


def _num(value: Any) -> float | None:
    """Coerce to float, or None for missing/non-finite (a consumer cannot plot NaN)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def flatten(record: dict) -> dict | None:
    """One ``request_end`` record -> one flat schema-4 row. None if unusable.

    Only ``event_type == "request_end"`` is emitted. The schema also permits
    ``tool_start`` / ``tool_end`` / ``tool_error`` from the harness bridge, which is
    not enabled on these runs (0 occurrences in 560 records); they are skipped rather
    than mangled into a request row.
    """
    event = record.get("event") or {}
    if event.get("event_type") != "request_end":
        return None
    req = event.get("request") or {}
    xid = req.get("x_request_id")
    if not xid:
        return None

    worker = req.get("worker") or {}
    replay = req.get("replay") or {}
    finish = req.get("finish_reason_metadata") or {}

    prefill_wait = _num(req.get("prefill_wait_time_ms"))
    prefill = _num(req.get("prefill_time_ms"))
    kv_transfer = _num(req.get("kv_transfer_estimated_latency_ms"))
    ttft_prefill = _num(req.get("ttft_ms"))
    total = _num(req.get("total_time_ms"))
    avg_itl = _num(req.get("avg_itl_ms"))
    osl = req.get("output_tokens")

    # The two corrections that stop a consumer plotting the prefill-side number as
    # TTFT. Both are None-safe: an aggregated (non-disagg) run has no KV transfer,
    # and there client TTFT *is* the prefill-side TTFT.
    client_ttft = ttft_prefill if kv_transfer is None else (
        None if ttft_prefill is None else ttft_prefill + kv_transfer
    )
    steady_decode = None if (total is None or client_ttft is None) else total - client_ttft
    clean_itl = avg_itl
    if avg_itl is not None and kv_transfer is not None and isinstance(osl, int) and osl > 1:
        clean_itl = avg_itl - kv_transfer / (osl - 1)

    return {
        "x_request_id": xid,
        "request_id": req.get("request_id"),
        "session_id": (event.get("agent_context") or {}).get("session_id"),
        "received_ms": req.get("request_received_ms"),
        "model": req.get("model"),
        # measured
        "prefill_wait_ms": prefill_wait,
        "prefill_ms": prefill,
        "kv_transfer_ms": kv_transfer,
        "ttft_prefill_ms": ttft_prefill,
        "total_ms": total,
        "avg_itl_ms": avg_itl,
        # derived -- see the TTFT trap above
        "client_ttft_ms": client_ttft,
        "steady_decode_ms": steady_decode,
        "clean_itl_ms": clean_itl,
        # request shape
        "isl": req.get("input_tokens"),
        "osl": osl,
        "cached_tokens": req.get("cached_tokens"),
        "kv_hit_rate": _num(req.get("kv_hit_rate")),
        "queue_depth": req.get("queue_depth"),
        # attribution
        "prefill_worker_id": worker.get("prefill_worker_id"),
        "prefill_dp_rank": worker.get("prefill_dp_rank"),
        "decode_worker_id": worker.get("decode_worker_id"),
        "decode_dp_rank": worker.get("decode_dp_rank"),
        # context
        "finish_reason": finish.get("finish_reason"),
        "block_size": replay.get("trace_block_size"),
        "n_hash_blocks": len(replay.get("input_sequence_hashes") or []) or None,
        # filled in by _add_prefix_reuse once the whole file is known
        "turn_index": None,
        "prefix_reuse_ratio": None,
    }


def _add_prefix_reuse(rows: list[dict], hashes: dict[str, list]) -> int:
    """Annotate each row with its turn index and prefix reuse against the prior turn.

    Turn order within a session is by ``received_ms`` -- verified to reproduce the
    client's own ``turn_index`` on 33/33 sessions, with turns strictly non-overlapping
    on 518/518 pairs, so this is the real ordering and not an approximation.

    ``prefix_reuse_ratio`` is the fraction of THIS turn's KV blocks that are a literal
    prefix continuation of the previous turn's. It answers the question a cache-hit
    percentage cannot: whether a multi-turn session is actually reusing its own
    context, or whether something (eviction, a routing change) broke the chain between
    two turns. The first turn of a session has no predecessor and is left None rather
    than 0 -- "no previous turn" is not "no reuse".
    """
    by_session: dict[str, list[dict]] = {}
    for row in rows:
        sid = row.get("session_id")
        if sid:
            by_session.setdefault(sid, []).append(row)

    annotated = 0
    for session_rows in by_session.values():
        session_rows.sort(key=lambda r: (r.get("received_ms") or 0))
        prev: list | None = None
        for idx, row in enumerate(session_rows):
            row["turn_index"] = idx
            cur = hashes.get(row["x_request_id"])
            if prev and cur:
                shared = 0
                for a, b in zip(prev, cur):
                    if a != b:
                        break
                    shared += 1
                row["prefix_reuse_ratio"] = round(shared / len(cur), 6) if cur else None
                annotated += 1
            prev = cur or prev
    return annotated


def process(in_path: str, out_path: str) -> int:
    """``dynamo-request-trace`` -> ``request_trace.jsonl``. Returns rows written.

    Two passes conceptually, one pass over the file: rows are flattened while the
    hash arrays are held aside, then the session-derived columns are filled and the
    hashes dropped. Peak memory is the hash arrays (~1.9M ints on the reference hour),
    which is the reason they never reach the bundle.
    """
    rows: list[dict] = []
    hashes: dict[str, list] = {}
    skipped = malformed = 0

    with open(in_path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            row = flatten(record)
            if row is None:
                skipped += 1
                continue
            seq = ((record.get("event") or {}).get("request") or {}).get("replay") or {}
            seq_hashes = seq.get("input_sequence_hashes")
            if seq_hashes:
                hashes[row["x_request_id"]] = seq_hashes
            rows.append(row)

    annotated = _add_prefix_reuse(rows, hashes)
    hashes.clear()

    rows.sort(key=lambda r: (r.get("received_ms") or 0))
    with open(out_path, "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")

    sessions = len({r["session_id"] for r in rows if r.get("session_id")})
    logger.info(
        "request_trace -> %s: %d rows, %d sessions, %d turns with prefix reuse"
        "%s%s",
        out_path, len(rows), sessions, annotated,
        f", {skipped} non-request_end skipped" if skipped else "",
        f", {malformed} malformed lines" if malformed else "",
    )
    return len(rows)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("in_path", help="dynamo-request-trace (JSON lines)")
    p.add_argument("out_path", help="output request_trace.jsonl")
    args = p.parse_args(argv)
    process(args.in_path, args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
