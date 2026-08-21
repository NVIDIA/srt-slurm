# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative time-series panel specification, evaluated by one generic code path.

Every panel is a row in :data:`PANELS` and every row is evaluated by
:func:`evaluate`. There is deliberately no per-panel rendering code: adding a signal
means adding a dict, and no panel can acquire behaviour that another panel lacks.
That is the whole point -- a panel whose title, caption or arithmetic is special-cased
for the run it was written against is the failure mode this layer exists to prevent.

A panel row is::

    {
      "id":     stable key in the emitted payload
      "tab":    which dashboard tab it belongs to
      "title":  run-independent. No worker counts, no model names, no "(1 worker)".
      "unit":   for the axis
      "kind":   how to turn samples into a series -- see KINDS below
      "metrics": the exact metric name(s) from server_metrics_export.jsonl
      "split_by": label key to break the series out by, or None
      "why":    what this panel is FOR, as a diagnostic question. Fixed prose, never
                interpolated with run values -- the numbers live in the chart.
      "issues": PERF ids this panel is meant to surface (provenance, not display)
      "caveat": a known way this panel can mislead, or None
    }

KINDS
-----
``gauge``        plot the sample as-is.
``counter_rate`` monotonic counter -> per-second rate via successive differences.
                 Resets (a restart) are dropped rather than plotted as a negative
                 spike, which would read as a throughput collapse that never happened.
``hist_mean``    a Prometheus histogram's ``_sum``/``_count`` pair -> mean per scrape.
                 Emitted as an INTERVAL mean (delta sum / delta count), not the
                 cumulative mean: the cumulative form flattens over a long run and
                 hides exactly the late-run degradation these panels exist to catch.
``ratio``        two counters -> ``a / (a + b)`` on the interval delta.

CAVEATS ENCODED HERE, NOT LEFT TO THE READER
--------------------------------------------
* Engine-side ``dp_rank`` is a REPLICATED BROADCAST -- every rank reports identical
  values (byte-identical in 380/380 sweeps on the reference run). Splitting an engine
  metric by rank yields a perfectly flat family of lines that reads as "no imbalance"
  when the run in fact had a 12x spread. No panel here splits an engine metric by
  ``dp_rank``; per-worker splits use ``worker_id``, and true per-rank data only exists
  frontend-side.
* Several queue gauges read a constant 0 on a healthy run. They are still specified:
  "this run never queued" is a finding, and queue depth is the single strongest TTFT
  predictor when it is NOT zero.
* Metrics are registered lazily, at first use. A family absent from the first ~3
  minutes is not a gap and must not be read as an outage.
"""
from __future__ import annotations

KINDS = ("gauge", "counter_rate", "hist_mean", "ratio")

PANELS: list[dict] = [
    # ---------------------------------------------------------------- frontend
    {
        "id": "fe_ttft_mean", "tab": "frontend", "unit": "s", "kind": "hist_mean",
        "title": "Client-observed TTFT (frontend)",
        "metrics": ["dynamo_frontend_time_to_first_token_seconds"], "split_by": None,
        "why": "What the caller actually waited for. Diverging from the engine-side "
               "TTFT is the definition of Dynamo-added overhead.",
        "issues": ["PERF-router-ttft", "PERF-frontend-tax"], "caveat": None,
    },
    {
        "id": "fe_inflight", "tab": "frontend", "unit": "requests", "kind": "gauge",
        "title": "In-flight requests", "metrics": ["dynamo_frontend_inflight_requests"],
        "split_by": None,
        "why": "True concurrency reaching the server, independent of what the load "
               "generator believes it is offering.",
        "issues": ["PERF-client-bottleneck"], "caveat": None,
    },
    {
        "id": "fe_queued", "tab": "frontend", "unit": "requests", "kind": "gauge",
        "title": "Queued requests", "metrics": ["dynamo_frontend_queued_requests"],
        "split_by": None,
        "why": "Admission backlog ahead of routing.",
        "issues": ["PERF-admission-queue"],
        "caveat": "Reads a constant 0 on runs that never queued; that is a finding, "
                  "not a broken panel.",
    },
    {
        "id": "fe_tokenize_mean", "tab": "frontend", "unit": "s", "kind": "hist_mean",
        "title": "Tokenization time", "metrics": ["dynamo_frontend_tokenize_seconds"],
        "split_by": None,
        "why": "Tokenization is timed inclusive of queue wait for the blocking pool, "
               "so it can grow until it accounts for essentially all of TTFT.",
        "issues": ["PERF-tokenize-ttft"], "caveat": None,
    },
    {
        "id": "fe_tok_cache_hit", "tab": "frontend", "unit": "ratio", "kind": "ratio",
        "title": "Tokenizer cache hit ratio",
        "metrics": ["dynamo_frontend_tokenizer_cache_hits_total",
                    "dynamo_frontend_tokenizer_cache_misses_total"],
        "split_by": None,
        "why": "A cold tokenizer cache re-encodes prompts that were already seen, "
               "turning prompt length directly into TTFT.",
        "issues": ["PERF-tokenize-ttft"], "caveat": None,
    },
    {
        "id": "fe_evloop_delay", "tab": "frontend", "unit": "s", "kind": "hist_mean",
        "title": "Event-loop delay", "metrics": ["dynamo_frontend_event_loop_delay_seconds"],
        "split_by": None,
        "why": "Async starvation. When the loop is delayed, every request pays it "
               "regardless of engine state.",
        "issues": ["PERF-tokio-starvation"], "caveat": None,
    },
    {
        "id": "fe_evloop_stalls", "tab": "frontend", "unit": "stalls/s", "kind": "counter_rate",
        "title": "Event-loop stalls", "metrics": ["dynamo_frontend_event_loop_stall_total"],
        "split_by": None,
        "why": "Discrete stall events, which a mean delay can average away.",
        "issues": ["PERF-tokio-starvation"], "caveat": None,
    },
    {
        "id": "fe_tokio_busy", "tab": "frontend", "unit": "ratio", "kind": "gauge",
        "title": "Tokio worker busy ratio", "metrics": ["dynamo_tokio_worker_busy_ratio"],
        "split_by": None,
        "why": "Runtime saturation. A saturated runtime makes host-side work, not the "
               "GPU, the limiter.",
        "issues": ["PERF-gil", "PERF-tokio-starvation"], "caveat": None,
    },
    {
        "id": "fe_blocking_pool", "tab": "frontend", "unit": "threads", "kind": "gauge",
        "title": "Blocking-pool threads",
        "metrics": ["dynamo_tokio_blocking_threads", "dynamo_tokio_blocking_idle_threads"],
        "split_by": None,
        "why": "Idle threads falling to zero while work queues is the signature of a "
               "blocking pool that has become the bottleneck.",
        "issues": ["PERF-tokenize-ttft"], "caveat": None,
    },
    {
        "id": "fe_blocking_queue", "tab": "frontend", "unit": "items", "kind": "gauge",
        "title": "Blocking-pool queue depth", "metrics": ["dynamo_tokio_blocking_queue_depth"],
        "split_by": None,
        "why": "Work waiting for a blocking thread.",
        "issues": ["PERF-tokenize-ttft"],
        "caveat": "Constant 0 on healthy runs.",
    },

    # ------------------------------------------------------------------ router
    {
        "id": "ro_kv_hit", "tab": "router", "unit": "ratio", "kind": "hist_mean",
        "title": "Router KV hit rate", "metrics": ["dynamo_component_router_kv_hit_rate"],
        "split_by": None,
        "why": "Routing quality as the router believes it to be: the prefix-match "
               "estimate it scored candidate workers on.",
        "issues": ["PERF-cache-hit-drop", "PERF-router-belief"],
        "caveat": "A BELIEF, not a measurement, and it has been observed reporting "
                  "high reuse on traffic with none. Corroborate against the engine "
                  "hit rate -- but only on workers whose engine config sets "
                  "kv_cache_config.enable_block_reuse: true. Comparing it against a "
                  "worker with reuse disabled produces a guaranteed false alarm.",
    },
    {
        "id": "ro_queue_pending", "tab": "router", "unit": "requests", "kind": "gauge",
        "title": "Router queue pending requests",
        "metrics": ["dynamo_frontend_router_queue_pending_requests"], "split_by": None,
        "why": "Queue depth is the strongest single predictor of TTFT -- stronger than "
               "KV utilisation, which costs nothing while the queue is empty.",
        "issues": ["PERF-admission-queue"],
        "caveat": "Constant 0 on healthy runs.",
    },
    {
        "id": "ro_queue_isl", "tab": "router", "unit": "tokens", "kind": "gauge",
        "title": "Router queue pending input tokens",
        "metrics": ["dynamo_frontend_router_queue_pending_isl_tokens"], "split_by": None,
        "why": "Queued WORK, not queued request count -- a few very long prompts and "
               "many short ones queue identically by count.",
        "issues": ["PERF-admission-queue"], "caveat": "Constant 0 on healthy runs.",
    },
    {
        "id": "ro_backpressure", "tab": "router", "unit": "events/s", "kind": "counter_rate",
        "title": "Router backpressure",
        "metrics": ["dynamo_frontend_router_queue_backpressure_total"], "split_by": None,
        "why": "The router actively refusing work.",
        "issues": ["PERF-admission-queue"], "caveat": "Constant 0 on healthy runs.",
    },
    {
        "id": "ro_overhead_hash", "tab": "router", "unit": "ms", "kind": "hist_mean",
        "title": "Router block-hashing time",
        "metrics": ["dynamo_router_overhead_block_hashing_ms"], "split_by": None,
        "why": "Router compute per request; grows with prompt length.",
        "issues": ["PERF-router-overhead"], "caveat": None,
    },
    {
        "id": "ro_overhead_match", "tab": "router", "unit": "ms", "kind": "hist_mean",
        "title": "Router index match time",
        "metrics": ["dynamo_router_overhead_indexer_find_matches_ms"], "split_by": None,
        "why": "Prefix-index lookup cost, which scales with the indexed corpus.",
        "issues": ["PERF-router-overhead"], "caveat": None,
    },

    # ------------------------------------------------------------------ engine
    {
        "id": "en_kv_util", "tab": "engine", "unit": "ratio", "kind": "gauge",
        "title": "KV-cache utilisation", "metrics": ["trtllm_kv_cache_utilization"],
        "split_by": "worker_id",
        "why": "KV pressure per worker. High utilisation with an empty queue is not "
               "itself a problem; high utilisation WITH a queue is.",
        "issues": ["PERF-kv-pressure"],
        "caveat": "Split by worker_id, never dp_rank -- engine rank series are a "
                  "replicated broadcast and would render identically.",
    },
    {
        "id": "en_requests_running", "tab": "engine", "unit": "requests", "kind": "gauge",
        "title": "Requests running", "metrics": ["trtllm_num_requests_running"],
        "split_by": "worker_id",
        "why": "Engine occupancy against its configured batch ceiling.",
        "issues": ["PERF-batch-starvation"], "caveat": None,
    },
    {
        "id": "en_requests_waiting", "tab": "engine", "unit": "requests", "kind": "gauge",
        "title": "Requests waiting", "metrics": ["trtllm_num_requests_waiting"],
        "split_by": "worker_id",
        "why": "Work admitted to the engine but not yet scheduled.",
        "issues": ["PERF-batch-starvation"], "caveat": None,
    },
    {
        "id": "en_iter_latency", "tab": "engine", "unit": "s", "kind": "gauge",
        "title": "Iteration latency", "metrics": ["trtllm_iteration_latency_seconds"],
        "split_by": "worker_id",
        "why": "Per-step wall time. A spread of two orders of magnitude within one run "
               "means steps are stalling on something other than compute.",
        "issues": ["PERF-gil", "PERF-iteration-stall"], "caveat": None,
    },
    {
        "id": "en_completed_rate", "tab": "engine", "unit": "requests/s", "kind": "counter_rate",
        "title": "Requests completed per worker",
        "metrics": ["trtllm_num_requests_completed_total"], "split_by": "worker_id",
        "why": "Load balance across workers, measured as delivered work rather than as "
               "routing intent. This is where a routing imbalance becomes visible.",
        "issues": ["PERF-decode-imbalance", "PERF-session-affinity"], "caveat": None,
    },
    {
        "id": "en_gen_tokens", "tab": "engine", "unit": "tokens/s", "kind": "counter_rate",
        "title": "Generation throughput", "metrics": ["trtllm_generation_tokens_total"],
        "split_by": "worker_id",
        "why": "Delivered decode throughput per worker.",
        "issues": ["PERF-decode-imbalance"], "caveat": None,
    },
    {
        "id": "en_kv_hit", "tab": "engine", "unit": "ratio", "kind": "gauge",
        "title": "Engine KV-cache hit rate", "metrics": ["trtllm_kv_cache_hit_rate"],
        "split_by": "worker_id",
        "why": "Reuse as the ENGINE measured it. Disagreement with the router's hit "
               "rate localises the fault to the routing layer rather than the cache.",
        "issues": ["PERF-router-belief", "PERF-cache-hit-drop"],
        "caveat": "Reads a hard 0 on any worker whose engine config sets "
                  "kv_cache_config.enable_block_reuse: false -- commonly the decode "
                  "side of a disagg deployment, where reuse is off by design. Read "
                  "this against trtllm_config_<mode>.yaml in the bundle before "
                  "concluding the cache is broken.",
    },
    {
        "id": "en_queue_time", "tab": "engine", "unit": "s", "kind": "hist_mean",
        "title": "Engine queue time", "metrics": ["trtllm_request_queue_time_seconds"],
        "split_by": "worker_id",
        "why": "Time inside the engine before execution, distinct from router-side "
               "admission delay.",
        "issues": ["PERF-admission-queue"], "caveat": None,
    },
    {
        "id": "en_ttft", "tab": "engine", "unit": "s", "kind": "hist_mean",
        "title": "Engine-side TTFT", "metrics": ["trtllm_time_to_first_token_seconds"],
        "split_by": "worker_id",
        "why": "The engine's own first-token time. The gap to the frontend's TTFT is "
               "everything Dynamo adds.",
        "issues": ["PERF-frontend-tax"], "caveat": None,
    },
    {
        "id": "en_tpot", "tab": "engine", "unit": "s", "kind": "hist_mean",
        "title": "Time per output token", "metrics": ["trtllm_time_per_output_token_seconds"],
        "split_by": "worker_id",
        "why": "Steady-state decode speed, the SLO most sensitive to batch composition.",
        "issues": ["PERF-itl-regression"], "caveat": None,
    },
    {
        "id": "en_spec_accept", "tab": "engine", "unit": "tokens", "kind": "gauge",
        "title": "Speculative-decode acceptance length",
        "metrics": ["trtllm_spec_decode_acceptance_length"], "split_by": "worker_id",
        "why": "How much speculation is actually paying off; collapse here shows up as "
               "an ITL regression with no other cause.",
        "issues": ["PERF-itl-regression"], "caveat": None,
    },
    {
        "id": "en_success_rate", "tab": "engine", "unit": "requests/s", "kind": "counter_rate",
        "title": "Request completions by finish reason",
        "metrics": ["trtllm_request_success_total"], "split_by": "finished_reason",
        "why": "A shift in finish reason is how truncation or cancellation shows up; "
               "throughput can rise purely because responses got shorter.",
        "issues": ["PERF-truncation"], "caveat": None,
    },
]


def _series_key(labels: dict, split_by: str | None) -> str:
    """Series identity within a panel: the split label's value, else a single series."""
    if not split_by:
        return "all"
    return str((labels or {}).get(split_by, "unknown"))


def evaluate(scrapes, panels=None, max_points: int = 2000) -> dict:
    """Evaluate every panel over ``scrapes`` in one pass.

    ``scrapes`` is the parsed ``server_metrics_export.jsonl``: an ordered list of
    ``(timestamp_ns, {metric_name: [{"labels":…, "value":…}, …]})``.

    Returns ``{panel_id: {"spec": <row minus arithmetic>, "series": {key: [[t_s, v], …]}}}``
    with ``t_s`` relative to the first scrape. Panels whose metrics never appear are
    omitted entirely, which is what lets a tab drop cleanly rather than render empty.

    Long runs are decimated to ``max_points`` per series AFTER the arithmetic, so rates
    and interval means are computed on every sample and only the plotted set is thinned.
    """
    panels = PANELS if panels is None else panels
    if not scrapes:
        return {}
    t0 = scrapes[0][0]
    # panel id -> series key -> list of (t_s, raw components)
    acc: dict[str, dict[str, list]] = {p["id"]: {} for p in panels}
    prev: dict[str, dict[str, tuple]] = {p["id"]: {} for p in panels}

    for ts, metrics in scrapes:
        t_s = (ts - t0) / 1e9
        for panel in panels:
            kind, names, split = panel["kind"], panel["metrics"], panel["split_by"]
            if kind == "gauge":
                for entry in metrics.get(names[0], []) or []:
                    v = entry.get("value")
                    if isinstance(v, (int, float)):
                        acc[panel["id"]].setdefault(_series_key(entry.get("labels"), split), []).append([t_s, v])
                # A second gauge in `metrics` is an additional named series (e.g. idle
                # threads alongside total threads), not a second arithmetic input.
                for extra in names[1:]:
                    for entry in metrics.get(extra, []) or []:
                        v = entry.get("value")
                        if isinstance(v, (int, float)):
                            acc[panel["id"]].setdefault(extra.rsplit("_", 2)[-1], []).append([t_s, v])
            elif kind == "counter_rate":
                for entry in metrics.get(names[0], []) or []:
                    v = entry.get("value")
                    if not isinstance(v, (int, float)):
                        continue
                    key = _series_key(entry.get("labels"), split)
                    last = prev[panel["id"]].get(key)
                    prev[panel["id"]][key] = (t_s, v)
                    # A counter that went backwards means the process restarted. Emitting
                    # the negative delta would draw a throughput cliff that never happened.
                    if last and t_s > last[0] and v >= last[1]:
                        acc[panel["id"]].setdefault(key, []).append([t_s, (v - last[1]) / (t_s - last[0])])
            elif kind == "hist_mean":
                sums = metrics.get(names[0] + "_sum", []) or []
                counts = {(_series_key(e.get("labels"), split)): e.get("value")
                          for e in (metrics.get(names[0] + "_count", []) or [])}
                for entry in sums:
                    key = _series_key(entry.get("labels"), split)
                    s, c = entry.get("value"), counts.get(key)
                    if not isinstance(s, (int, float)) or not isinstance(c, (int, float)):
                        continue
                    last = prev[panel["id"]].get(key)
                    prev[panel["id"]][key] = (t_s, s, c)
                    # Interval mean, not cumulative: a cumulative mean over a long run
                    # flattens and hides the late-run degradation this is here to catch.
                    if last and c > last[2] and s >= last[1]:
                        acc[panel["id"]].setdefault(key, []).append([t_s, (s - last[1]) / (c - last[2])])
            elif kind == "ratio":
                a_entries = metrics.get(names[0], []) or []
                b_by_key = {(_series_key(e.get("labels"), split)): e.get("value")
                            for e in (metrics.get(names[1], []) or [])}
                for entry in a_entries:
                    key = _series_key(entry.get("labels"), split)
                    a, b = entry.get("value"), b_by_key.get(key)
                    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                        continue
                    last = prev[panel["id"]].get(key)
                    prev[panel["id"]][key] = (t_s, a, b)
                    if last:
                        da, db = a - last[1], b - last[2]
                        if da >= 0 and db >= 0 and (da + db) > 0:
                            acc[panel["id"]].setdefault(key, []).append([t_s, da / (da + db)])

    out: dict[str, dict] = {}
    for panel in panels:
        series = {k: v for k, v in acc[panel["id"]].items() if v}
        if not series:
            continue
        for key, pts in series.items():
            if len(pts) > max_points:
                stride = -(-len(pts) // max_points)
                series[key] = pts[::stride]
        out[panel["id"]] = {
            "tab": panel["tab"], "title": panel["title"], "unit": panel["unit"],
            "kind": panel["kind"], "why": panel["why"], "split_by": panel["split_by"],
            "source": panel["metrics"], "caveat": panel.get("caveat"),
            "issues": panel.get("issues", []),
            "series": series,
        }
    return out
