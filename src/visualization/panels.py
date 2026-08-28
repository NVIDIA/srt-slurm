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
        # One series PER WORKER THREAD. Collapsing them into a single unsplit series
        # interleaves every thread's samples at the same timestamps and draws a point
        # cloud, not a trend -- and a runtime is saturated when its BUSIEST threads
        # are, which an undifferentiated blob cannot show.
        "split_by": "worker",
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

    # ---------------------------------------------------------------------
    # Rows below are transcribed from PANEL_BACKLOG.md, itself derived from the
    # issue register; each carries the backlog id it came from. They are DATA --
    # the evaluator above is unchanged by their addition.
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------- frontend
    # F3
    {
        "id": "fe_tok_cache_token_hit", "tab": "frontend", "unit": "ratio", "kind": "ratio",
        "title": "Tokenizer cache hit ratio, token-weighted",
        "metrics": ["dynamo_frontend_tokenizer_cache_cached_tokens_total",
                    "dynamo_frontend_tokenizer_cache_uncached_tokens_total"],
        "split_by": None,
        "why": "Weights every cache decision by the tokens it saved or cost. A single miss on a "
               "very long prompt costs more than many hits on short ones, so a request-level hit "
               "rate can read healthy while the tokenizer re-encodes most of the token volume.",
        "issues": ["PERF-02", "PERF-01"],
        "caveat": "Read alongside the request-level hit ratio, never instead of it: the two answer "
                  "different questions and diverge whenever prompt lengths are skewed.",
    },
    # F5
    {
        "id": "fe_tokio_alive_tasks", "tab": "frontend", "unit": "count", "kind": "gauge",
        "title": "Alive runtime tasks", "metrics": ["dynamo_tokio_alive_tasks"],
        "split_by": None,
        "why": "Task population on the async runtime. Growth without a matching rise in in-flight "
               "requests means tasks are accumulating rather than completing.",
        "issues": ["PERF-18", "PERF-27"], "caveat": None,
    },
    # F5
    {
        "id": "fe_tokio_poll_time", "tab": "frontend", "unit": "ns", "kind": "gauge",
        "title": "Worker mean task poll time", "metrics": ["dynamo_tokio_worker_mean_poll_time_ns"],
        "split_by": None,
        "why": "How long a single poll occupies a runtime worker. Long polls are the mechanism by "
               "which one blocking call starves every other task on the same thread.",
        "issues": ["PERF-18", "PERF-27"],
        "caveat": "One series per runtime worker thread, so this family is a large share of the "
                  "series budget. Reduce across the worker label rather than drawing every thread.",
    },
    # F7
    {
        "id": "fe_stage_duration", "tab": "frontend", "unit": "s", "kind": "hist_mean",
        "title": "Pipeline stage duration", "metrics": ["dynamo_frontend_stage_duration_seconds"],
        "split_by": "stage",
        "why": "Where frontend time is spent before the request reaches a worker: preprocess, "
               "route, transport roundtrip. Localises added latency to a stage rather than to "
               "the frontend as a whole.",
        "issues": ["PERF-31", "PERF-20", "PERF-42"],
        "caveat": "A proxy for pre-route queue wait, not a measurement of it. The frontend log "
                  "records that measure the wait directly are not in the bundle.",
    },
    # F7
    {
        "id": "fe_stage_requests", "tab": "frontend", "unit": "requests", "kind": "gauge",
        "title": "Requests in pipeline stage", "metrics": ["dynamo_frontend_stage_requests"],
        "split_by": "stage",
        "why": "Instantaneous occupancy of each frontend stage. Occupancy piling up in one stage "
               "is where a frontend-side backlog first becomes visible.",
        "issues": ["PERF-31", "PERF-20"],
        "caveat": "An instantaneous gauge sampled at scrape rate, so short bursts are missed and "
                  "most samples read zero even when the stage is doing work.",
    },
    # F8
    {
        "id": "fe_requests_by_status", "tab": "frontend", "unit": "requests/s", "kind": "counter_rate",
        "title": "Request outcomes by status", "metrics": ["dynamo_frontend_requests_total"],
        "split_by": "status",
        "why": "Success versus error as delivered throughput. A throughput number that does not "
               "separate outcomes can rise purely because requests started failing faster.",
        "issues": ["PERF-25", "PERF-43", "PERF-11"],
        "caveat": "The family also carries an error_type label that this split collapses. Break "
                  "out by error_type before concluding anything about the kind of failure.",
    },
    # F8
    {
        "id": "fe_component_errors", "tab": "frontend", "unit": "events/s", "kind": "counter_rate",
        "title": "Work-handler errors by type", "metrics": ["dynamo_component_errors_total"],
        "split_by": "error_type",
        "why": "The only error counter that ever fires anywhere in the stack: the engine-side "
               "error family is declared but never sampled, so a purely engine-side failure has "
               "no counter at all.",
        "issues": ["PERF-25", "PERF-43", "PERF-11"],
        "caveat": "Some error_type values are benign publish-path noise that fires on nearly every "
                  "request. Aggregating across error_type reads as a near-total error rate on a "
                  "healthy run -- always keep the split.",
    },
    # F9
    {
        "id": "fe_disconnected_clients", "tab": "frontend", "unit": "count", "kind": "gauge",
        "title": "Disconnected clients", "metrics": ["dynamo_frontend_disconnected_clients"],
        "split_by": None,
        "why": "Clients that gave up before the response completed. Work already spent on them is "
               "throughput the server produced and nobody received.",
        "issues": ["PERF-26", "PERF-25"], "caveat": None,
    },
    # F9
    {
        "id": "fe_model_cancellations", "tab": "frontend", "unit": "events/s", "kind": "counter_rate",
        "title": "Model request cancellations",
        "metrics": ["dynamo_frontend_model_cancellation_total"], "split_by": None,
        "why": "Cancellations observed at the HTTP boundary, which is where a client-side timeout "
               "first becomes visible to the server.",
        "issues": ["PERF-26", "PERF-25"],
        "caveat": "Registered lazily, on the first cancellation. Absence early in a run means "
                  "nothing has been cancelled yet, not that the counter is broken.",
    },
    # F9
    {
        "id": "fe_component_cancellations", "tab": "frontend", "unit": "events/s", "kind": "counter_rate",
        "title": "Work-handler cancellations per worker",
        "metrics": ["dynamo_component_cancellation_total"], "split_by": "worker_id",
        "why": "Cancellations as the backend sees them. A gap against the frontend's cancellation "
               "count is work the engine kept doing after the caller was gone.",
        "issues": ["PERF-26", "PERF-25"],
        "caveat": "The family also spans the indexer-query endpoints, whose volume dwarfs the "
                  "generate endpoint. Filter on dynamo_endpoint before reading a rate here.",
    },
    # F10
    {
        "id": "fe_plane_queue", "tab": "frontend", "unit": "s", "kind": "hist_mean",
        "title": "Request-plane queue time", "metrics": ["dynamo_request_plane_queue_seconds"],
        "split_by": None,
        "why": "Time from handler entry to the request actually being sent. This is the frontend's "
               "own dispatch delay, upstream of any engine or router queue.",
        "issues": ["PERF-20", "PERF-21", "PERF-39"], "caveat": None,
    },
    # F10
    {
        "id": "fe_plane_send", "tab": "frontend", "unit": "s", "kind": "hist_mean",
        "title": "Request-plane send time", "metrics": ["dynamo_request_plane_send_seconds"],
        "split_by": None,
        "why": "Cost of the send call itself. Separating it from queue time distinguishes a slow "
               "transport from a busy dispatcher.",
        "issues": ["PERF-20", "PERF-21", "PERF-39"], "caveat": None,
    },
    # F10
    {
        "id": "fe_plane_roundtrip_ttft", "tab": "frontend", "unit": "s", "kind": "hist_mean",
        "title": "Request-plane roundtrip to first response",
        "metrics": ["dynamo_request_plane_roundtrip_ttft_seconds"], "split_by": None,
        "why": "Send to first response item. Everything outside this window in the client's "
               "first-token time is frontend-side work rather than serving.",
        "issues": ["PERF-20", "PERF-21", "PERF-39"], "caveat": None,
    },
    # F10
    {
        "id": "fe_plane_inflight", "tab": "frontend", "unit": "requests", "kind": "gauge",
        "title": "Request-plane in-flight requests",
        "metrics": ["dynamo_request_plane_inflight_requests"], "split_by": None,
        "why": "Concurrency at the dispatch router, between the HTTP handler and the workers. A "
               "gap against frontend in-flight requests is work stuck before dispatch.",
        "issues": ["PERF-20", "PERF-38"], "caveat": None,
    },
    # F10
    {
        "id": "fe_network_transit", "tab": "frontend", "unit": "s", "kind": "hist_mean",
        "title": "Frontend-to-backend network transit",
        "metrics": ["dynamo_work_handler_network_transit_seconds"], "split_by": "dynamo_component",
        "why": "Cross-process wire time to the worker. It is the term a request-plane or codec "
               "change moves, and the one that must be excluded before blaming the engine.",
        "issues": ["PERF-20", "PERF-21", "PERF-39"],
        "caveat": "Emitted with byte-identical labels by every worker endpoint, so it is only "
                  "separable because the capture injects worker_id and dynamo_component. A live "
                  "scrape of the same endpoints would overwrite these series.",
    },
    # F11
    {
        "id": "fe_uptime", "tab": "frontend", "unit": "s", "kind": "gauge",
        "title": "Component uptime", "metrics": ["dynamo_component_uptime_seconds"],
        "split_by": "worker_id",
        "why": "The liveness signal. Because every other family registers lazily, a missing metric "
               "means 'has not happened yet' -- only a reset here means a process died.",
        "issues": ["PERF-43", "PERF-40", "PERF-45"], "caveat": None,
    },
    # F12
    {
        "id": "fe_arrival_rate", "tab": "frontend", "unit": "requests/s", "kind": "counter_rate",
        "title": "Request arrival rate", "metrics": ["dynamo_frontend_requests_started_total"],
        "split_by": None,
        "why": "Offered load as the server received it. Read against in-flight requests, a low "
               "arrival rate with idle capacity is the load-generator-bottleneck signature.",
        "issues": ["PERF-38", "PERF-28", "PERF-11"], "caveat": None,
    },
    # R-Q8
    {
        "id": "fe_input_seq_tokens", "tab": "frontend", "unit": "tokens", "kind": "hist_mean",
        "title": "Input sequence length", "metrics": ["dynamo_frontend_input_sequence_tokens"],
        "split_by": None,
        "why": "Prompt size entering the server, which sets both the prefill cost and the request "
               "payload size carried on the request plane.",
        "issues": ["PERF-39", "PERF-20"],
        "caveat": "Payload headroom against a transport cap cannot be computed here: the cap lives "
                  "in the recipe, outside the bundle, and the rejection counter never fires.",
    },

    # ------------------------------------------------------------------ router
    # R2
    {
        "id": "ro_kv_events_applied", "tab": "router", "unit": "events/s", "kind": "counter_rate",
        "title": "KV-event apply outcomes",
        "metrics": ["dynamo_component_kv_cache_events_applied"], "split_by": "status",
        "why": "Whether the router's prefix index is actually absorbing what the engines publish. "
               "A rising non-ok share means the index is drifting from engine reality, which is "
               "how routing decisions silently start being made on stale state.",
        "issues": ["PERF-10", "PERF-09", "PERF-06"],
        "caveat": "Emitted by the frontend endpoint, because the router runs in-process there. The "
                  "worker_id on these series is the frontend's own -- do not split by it.",
    },
    # R2
    {
        "id": "ro_kv_event_warnings", "tab": "router", "unit": "events/s", "kind": "counter_rate",
        "title": "KV-event indexer warnings",
        "metrics": ["dynamo_component_kv_cache_event_warnings"], "split_by": "warning_kind",
        "why": "Suspicious events the indexer accepted but flagged. Non-zero here without a "
               "corresponding apply failure is index corruption that no error counter reports.",
        "issues": ["PERF-10", "PERF-09"], "caveat": None,
    },
    # R2
    {
        "id": "ro_kv_events_dropped", "tab": "router", "unit": "events/s", "kind": "counter_rate",
        "title": "KV events dropped before the publisher",
        "metrics": ["dynamo_component_kv_publisher_engines_dropped_events_total"],
        "split_by": "worker_id",
        "why": "Events the engine produced and lost before the router ever saw them, detected via "
               "identifier gaps. Any non-zero value is decisive: the index cannot be correct.",
        "issues": ["PERF-10", "PERF-09"],
        "caveat": "Reads a constant zero on a healthy run, which is a finding rather than a broken "
                  "panel -- but it also means its behaviour under loss is unvalidated here.",
    },
    # R3
    {
        "id": "ro_worker_prefill_tokens", "tab": "router", "unit": "tokens", "kind": "gauge",
        "title": "Active prefill tokens per worker",
        "metrics": ["dynamo_frontend_worker_active_prefill_tokens"], "split_by": "worker_id",
        "why": "The prefill load the router can actually see at decision time. Routing quality is "
               "bounded by this view, not by what the engine knows.",
        "issues": ["PERF-03", "PERF-05", "PERF-04"],
        "caveat": "Router-side worker identifiers, which do not join to the engine-side ones. Pair "
                  "with active decode blocks: prefill load rising while decode load reads zero is "
                  "the window in which the router piles on a worker that is already busy.",
    },
    # R3
    {
        "id": "ro_worker_decode_blocks", "tab": "router", "unit": "count", "kind": "gauge",
        "title": "Active decode blocks per worker",
        "metrics": ["dynamo_frontend_worker_active_decode_blocks"], "split_by": "worker_id",
        "why": "The decode-side occupancy the router scores candidates on, in cache blocks.",
        "issues": ["PERF-03", "PERF-05", "PERF-04"],
        "caveat": "Converting blocks to tokens requires the router's own block size, which "
                  "disagrees with the engine's tokens-per-block by a wide factor. Do not use the "
                  "two interchangeably.",
    },
    # R4
    {
        "id": "ro_overhead_seq_hash", "tab": "router", "unit": "ms", "kind": "hist_mean",
        "title": "Router sequence-hashing time",
        "metrics": ["dynamo_router_overhead_seq_hashing_ms"], "split_by": None,
        "why": "Sequence-hash cost per request, the second half of the router's hashing work.",
        "issues": ["PERF-23", "PERF-33", "PERF-01"], "caveat": None,
    },
    # R4
    {
        "id": "ro_overhead_scheduling", "tab": "router", "unit": "ms", "kind": "hist_mean",
        "title": "Router scheduling time",
        "metrics": ["dynamo_router_overhead_scheduling_ms"], "split_by": None,
        "why": "Time inside worker selection itself, separated from hashing and index lookup. It "
               "is the term that grows with the candidate set rather than with prompt length.",
        "issues": ["PERF-23", "PERF-33"],
        "caveat": "Never take a quantile of this family: its bucket ladder runs to hundreds of "
                  "seconds while all observed mass sits in the smallest few buckets. Use the mean.",
    },
    # R4
    {
        "id": "ro_overhead_total", "tab": "router", "unit": "ms", "kind": "hist_mean",
        "title": "Router total overhead per request",
        "metrics": ["dynamo_router_overhead_total_ms"], "split_by": None,
        "why": "The reference line for the routing-cost decomposition. Total minus the named "
               "components is routing work that no component metric accounts for.",
        "issues": ["PERF-23", "PERF-33", "PERF-01"],
        "caveat": "Never take a quantile of this family: its bucket ladder runs to hundreds of "
                  "seconds while all observed mass sits in the smallest few buckets. Use the mean.",
    },
    # R5
    {
        "id": "ro_queue_cached_tokens", "tab": "router", "unit": "tokens", "kind": "gauge",
        "title": "Router queue pending cached tokens",
        "metrics": ["dynamo_frontend_router_queue_pending_cached_tokens"], "split_by": "worker_type",
        "why": "The already-cached share of queued input tokens. Subtracted from pending input "
               "tokens it gives queued WORK -- the tokens that must actually be computed -- which "
               "a queued request count cannot express.",
        "issues": ["PERF-11", "PERF-03"],
        "caveat": "Constant zero on runs that never queued. That is a finding, not a broken panel, "
                  "but it also leaves the panel's behaviour under real queueing unvalidated.",
    },
    # R6
    {
        "id": "ro_worker_last_isl", "tab": "router", "unit": "tokens", "kind": "gauge",
        "title": "Last observed input sequence length per rank",
        "metrics": ["dynamo_frontend_worker_last_input_sequence_tokens"], "split_by": "dp_rank",
        "why": "Prompt size landing on each prefill rank. Rank latency spread must be read against "
               "this before it is called a scheduling fault -- unequal work is not imbalance.",
        "issues": ["PERF-05", "PERF-52", "PERF-03"],
        "caveat": "A last-observed gauge, not an aggregate: it samples the most recent request per "
                  "rank, so it is noisy and must not be averaged into a rank-utilisation number.",
    },
    # R6
    {
        "id": "ro_worker_last_ttft", "tab": "router", "unit": "s", "kind": "gauge",
        "title": "Last observed time to first token per rank",
        "metrics": ["dynamo_frontend_worker_last_time_to_first_token_seconds"], "split_by": "dp_rank",
        "why": "The only genuinely per-rank latency series in the capture. Every engine-side rank "
               "family is a replicated broadcast, so rank imbalance is visible here or nowhere.",
        "issues": ["PERF-05", "PERF-52", "PERF-03"],
        "caveat": "A last-observed gauge, not an aggregate: it samples the most recent request per "
                  "rank, so it is noisy and must not be averaged into a rank-utilisation number.",
    },
    # R6
    {
        "id": "ro_worker_last_itl", "tab": "router", "unit": "s", "kind": "gauge",
        "title": "Last observed inter-token latency per worker",
        "metrics": ["dynamo_frontend_worker_last_inter_token_latency_seconds"], "split_by": "worker_id",
        "why": "Decode pace as the router sees it per worker, which is the input a load-aware "
               "decode policy would react to.",
        "issues": ["PERF-05", "PERF-52"],
        "caveat": "Emitted on decode workers only, all at rank zero, so it is split by worker "
                  "rather than by rank. A last-observed gauge, not an aggregate.",
    },

    # ------------------------------------------------------------------ engine
    # E1
    {
        "id": "en_max_active_requests", "tab": "engine", "unit": "requests", "kind": "gauge",
        "title": "Maximum active request slots",
        "metrics": ["trtllm_max_num_active_requests"], "split_by": "worker_id",
        "why": "The ceiling that running and scheduled request counts must be read against. "
               "Occupancy without its ceiling cannot distinguish a starved engine from a full one.",
        "issues": ["PERF-03", "PERF-08", "PERF-14"],
        "caveat": "Use this as the ceiling, not the engine's batch-size families: those report a "
                  "constant zero on every sample and would make headroom look infinite.",
    },
    # E5
    {
        "id": "en_scheduled_requests", "tab": "engine", "unit": "requests", "kind": "gauge",
        "title": "Scheduled requests", "metrics": ["trtllm_num_scheduled_requests"],
        "split_by": "worker_id",
        "why": "Batch size the engine actually ran per iteration. Chronically small batches "
               "against a large ceiling is throughput left on the table, and it is a distribution "
               "claim -- the mean can look fine while most iterations run a single request.",
        "issues": ["PERF-14", "PERF-15", "PERF-03"],
        "caveat": "Sampled at scrape rate while the engine iterates far faster, so this is a "
                  "subsample of the batch-size distribution, not the distribution itself. It is "
                  "also exactly the sum of context and generation requests.",
    },
    # E5
    {
        "id": "en_context_requests", "tab": "engine", "unit": "requests", "kind": "gauge",
        "title": "Context requests in batch", "metrics": ["trtllm_num_context_requests"],
        "split_by": "worker_id",
        "why": "The prefill half of batch composition. Context requests crowding out generation "
               "requests is how a decode worker's output rate collapses with no latency signal.",
        "issues": ["PERF-14", "PERF-15", "PERF-03"], "caveat": None,
    },
    # E5
    {
        "id": "en_generation_requests", "tab": "engine", "unit": "requests", "kind": "gauge",
        "title": "Generation requests in batch", "metrics": ["trtllm_num_generation_requests"],
        "split_by": "worker_id",
        "why": "The decode half of batch composition, and the term that sets decode efficiency.",
        "issues": ["PERF-14", "PERF-15"], "caveat": None,
    },
    # E6
    {
        "id": "en_prefill_batch_tokens", "tab": "engine", "unit": "tokens", "kind": "hist_mean",
        "title": "Context tokens per prefill iteration",
        "metrics": ["trtllm_prefill_batch_tokens"], "split_by": "worker_id",
        "why": "Prefill work per iteration against the configured token budget. Sitting at the "
               "budget means token-budget-bound, a diagnosis that looks like an unremarkable "
               "number until it is divided by the configured ceiling.",
        "issues": ["PERF-15", "PERF-08", "PERF-14"],
        "caveat": "Emitted by prefill-role workers only. The ceiling it must be divided by "
                  "lives in the engine config file in the bundle, not in any metric. Read it "
                  "as a distribution and not only as this interval mean -- the share of "
                  "iterations that actually REACH the ceiling and the average fraction of it "
                  "are different quantities, and a run can look budget-bound on one while "
                  "being far from it on the other.",
    },
    # E6
    {
        "id": "en_context_tokens", "tab": "engine", "unit": "tokens", "kind": "gauge",
        "title": "Total context tokens", "metrics": ["trtllm_total_context_tokens"],
        "split_by": "worker_id",
        "why": "Instantaneous context-token occupancy. Pinned at the configured maximum is the "
               "token-budget saturation signature.",
        "issues": ["PERF-15", "PERF-08"], "caveat": None,
    },
    # E7
    {
        "id": "en_block_reuse", "tab": "engine", "unit": "ratio", "kind": "ratio",
        "title": "Cumulative KV block reuse",
        "metrics": ["trtllm_kv_cache_reused_blocks_total", "trtllm_kv_cache_missed_blocks_total"],
        "split_by": "worker_id",
        "why": "Reuse computed from block counts rather than read from an instantaneous gauge. "
               "This is the stable, aggregatable form of engine cache effectiveness.",
        "issues": ["PERF-07", "PERF-06", "PERF-08"],
        "caveat": "A worker with no line here has block reuse disabled in its engine config -- "
                  "commonly the decode side of a disagg deployment -- which is not the same as "
                  "zero reuse. Do not substitute the iteration-prefixed counter twins: they are "
                  "byte-identical cumulative duplicates despite the name.",
    },
    # E7
    {
        "id": "en_iter_reuse_rate", "tab": "engine", "unit": "ratio", "kind": "gauge",
        "title": "Per-iteration KV block reuse rate",
        "metrics": ["trtllm_kv_cache_iter_reuse_rate"], "split_by": "worker_id",
        "why": "The instantaneous companion to the cumulative reuse ratio. Divergence between the "
               "two localises a reuse collapse to a window rather than to the whole run.",
        "issues": ["PERF-07", "PERF-06"],
        "caveat": "Instantaneous and un-aggregatable: averaging it across workers is not a hit "
                  "rate. Reads zero on workers whose engine config disables block reuse.",
    },
    # E8
    {
        "id": "en_spec_drafted", "tab": "engine", "unit": "tokens/s", "kind": "counter_rate",
        "title": "Draft tokens by position",
        "metrics": ["trtllm_spec_decode_drafted_tokens_total"], "split_by": "token_position",
        "why": "Speculation offered, broken out by draft position. Read against accepted tokens "
               "at the same position, this shows which position is being rejected -- the "
               "actionable part of an acceptance collapse.",
        "issues": ["PERF-16", "PERF-14"],
        "caveat": "The acceptance rate is accepted divided by drafted, which is not the same as "
                  "the ratio arithmetic this layer supports; read the two rate panels together "
                  "rather than expecting a single ratio series.",
    },
    # E8
    {
        "id": "en_spec_accepted", "tab": "engine", "unit": "tokens/s", "kind": "counter_rate",
        "title": "Accepted tokens by position",
        "metrics": ["trtllm_spec_decode_accepted_tokens_total"], "split_by": "token_position",
        "why": "Speculation that paid off, broken out by draft position. Its collapse shows up as "
               "an inter-token latency regression with no other visible cause.",
        "issues": ["PERF-16", "PERF-14"],
        "caveat": "A forced-acceptance environment override makes this report policy rather than "
                  "model behaviour. That setting lives outside the bundle, so check the run "
                  "config before reading the value as a model property.",
    },
    # E9 (also E3, prefill-side delivered work)
    {
        "id": "en_prompt_tokens", "tab": "engine", "unit": "tokens/s", "kind": "counter_rate",
        "title": "Prompt token throughput", "metrics": ["trtllm_prompt_tokens_total"],
        "split_by": "worker_id",
        "why": "Prefill work delivered per worker. Read against generation throughput it answers "
               "whether the engine is mostly prefilling -- a diagnosis no single metric states, "
               "and the prefill-side counterpart to a completed-request count.",
        "issues": ["PERF-08", "PERF-15", "PERF-06"],
        "caveat": "Emitted by prefill-role workers only, so there is no decode line to compare "
                  "against within this panel.",
    },
    # E9
    {
        "id": "en_prompt_cached_tokens", "tab": "engine", "unit": "tokens/s", "kind": "counter_rate",
        "title": "Cached prompt tokens", "metrics": ["trtllm_prompt_cached_tokens_total"],
        "split_by": "worker_id",
        "why": "Prompt tokens served from cache. Subtracted from prompt throughput it gives the "
               "tokens actually recomputed, which is how a reuse regression is told apart from a "
               "genuinely lighter workload.",
        "issues": ["PERF-08", "PERF-06", "PERF-07"], "caveat": None,
    },
    # E10
    {
        "id": "en_kv_offload_bytes", "tab": "engine", "unit": "bytes/s", "kind": "counter_rate",
        "title": "KV offload throughput",
        "metrics": ["trtllm_kv_cache_offload_bytes_total"], "split_by": "worker_id",
        "why": "Bytes evicted from GPU to the host pool. Sustained offload is cache pressure "
               "being paid in interconnect bandwidth rather than in visible latency.",
        "issues": ["PERF-24", "PERF-07"],
        "caveat": "Host-pool occupancy cannot be read alongside this: the host utilisation gauge "
                  "is a constant zero on every sample.",
    },
    # E10
    {
        "id": "en_kv_onboard_bytes", "tab": "engine", "unit": "bytes/s", "kind": "counter_rate",
        "title": "KV onboard throughput",
        "metrics": ["trtllm_kv_cache_onboard_bytes_total"], "split_by": "worker_id",
        "why": "Bytes pulled back from host to GPU. This traffic is prefill work that the reuse "
               "metrics count as a hit, so a high hit rate can still be paying transfer cost.",
        "issues": ["PERF-24", "PERF-07"],
        "caveat": "Registered lazily, on the first host-to-device onboard. Its absence early in a "
                  "run is not a gap.",
    },
    # E10
    {
        "id": "en_kv_intra_device_bytes", "tab": "engine", "unit": "bytes/s", "kind": "counter_rate",
        "title": "Intra-device KV copy throughput",
        "metrics": ["trtllm_kv_cache_intra_device_copy_bytes_total"], "split_by": "worker_id",
        "why": "Block copies within the GPU. It is cache bookkeeping that consumes bandwidth "
               "without appearing in any latency or reuse number.",
        "issues": ["PERF-24", "PERF-07"], "caveat": None,
    },
    # E12
    {
        "id": "en_prefill_time", "tab": "engine", "unit": "s", "kind": "hist_mean",
        "title": "Prefill phase duration",
        "metrics": ["trtllm_request_prefill_time_seconds"], "split_by": "worker_id",
        "why": "Context-phase duration per worker, separated from queueing and from decode.",
        "issues": ["PERF-17", "PERF-13", "PERF-04"],
        "caveat": "On prefill-role workers this coincides exactly with the inference-time family "
                  "because the request ends at the first token there. That coincidence is the "
                  "structural signature of disaggregation, not a duplicate to remove.",
    },
    # E12
    {
        "id": "en_decode_time", "tab": "engine", "unit": "s", "kind": "hist_mean",
        "title": "Decode phase duration",
        "metrics": ["trtllm_request_decode_time_seconds"], "split_by": "worker_id",
        "why": "Generation-phase duration per worker: first token to last token, which is the "
               "span an inter-token latency regression lengthens.",
        "issues": ["PERF-17", "PERF-13", "PERF-04"],
        "caveat": "Decode-role only, and registered lazily on the first decode completion. Its "
                  "absence early in a run is not a gap.",
    },
    # E12
    {
        "id": "en_e2e_latency", "tab": "engine", "unit": "s", "kind": "hist_mean",
        "title": "End-to-end request latency",
        "metrics": ["trtllm_e2e_request_latency_seconds"], "split_by": "worker_id",
        "why": "Whole-request time as the engine measured it. Its divergence from engine-side "
               "first-token time on decode workers is where the decode span actually lives.",
        "issues": ["PERF-17", "PERF-13"],
        "caveat": "Its bucket ladder differs from the first-token families, so cross-family "
                  "quantile comparison is meaningless. Compare means.",
    },
    # E13
    {
        "id": "en_gpu_memory", "tab": "engine", "unit": "bytes", "kind": "gauge",
        "title": "GPU memory usage", "metrics": ["trtllm_gpu_memory_usage_bytes"],
        "split_by": "worker_id",
        "why": "The only GPU-side signal in the whole metric surface. A leak or an approach to "
               "the device limit is visible here and nowhere else.",
        "issues": ["PERF-36", "PERF-34"],
        "caveat": "A leak and out-of-memory canary, nothing more: there is no utilisation, "
                  "occupancy, power or temperature signal anywhere in the capture. Host and "
                  "pinned memory gauges exist but read a constant zero.",
    },

    # --- closing the remaining AVAILABLE issues from the coverage audit ---------
    {
        # PERF-21
        "id": "ro_work_ttfr", "tab": "router", "unit": "s", "kind": "hist_mean",
        "title": "Work-handler time to first response",
        "metrics": ["dynamo_work_handler_time_to_first_response_seconds"],
        "split_by": "dynamo_component",
        "why": "How long the request plane took to get anything back from a component. "
               "It separates a component that is slow to start responding from one that "
               "is slow to finish, which end-to-end latency alone cannot.",
        "issues": ["PERF-21"], "caveat": None,
    },
    {
        # PERF-22
        "id": "ro_component_requests", "tab": "router", "unit": "requests/s",
        "kind": "counter_rate", "title": "Component request rate",
        "metrics": ["dynamo_component_requests_total"], "split_by": "worker_id",
        "why": "Delivered request rate per worker as the request plane saw it. Read "
               "against the engine's own completion rate, a persistent gap means work "
               "is being accepted faster than it is finished.",
        "issues": ["PERF-22"], "caveat": None,
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
