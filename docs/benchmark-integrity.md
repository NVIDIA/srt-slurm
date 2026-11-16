# Benchmark Integrity

This page lists changes that raise a benchmark number without making serving
faster or better for a real user. A result that depends on one of them is not a
performance result. This is an initial version: add patterns you have seen,
with the metric they inflate and how to detect them.

The test for every entry: would a real client, sending real traffic, see the
same gain? If the gain exists only because of how the benchmark client builds
requests or computes a metric, it is gaming.

## How the serving metrics are computed

The entries below refer to how [AIPerf](benchmarks.md)-style clients compute
the metrics. Check the client version you run before relying on these
definitions.

| Metric | Per-request definition | Reported as |
|---|---|---|
| TTFT | Time from request start to the first stream chunk with non-empty content (text, reasoning or tool call). Role-only and usage-only chunks do not count. | A percentile, where higher is worse |
| Inter-token latency (ITL) | (stream end - first-token time) / (output tokens - 1) | A percentile, where higher is worse |
| Interactivity (tok/s/user) | 1 / ITL | Usually 1 / p90(ITL), i.e. the slow 10% of requests |
| Throughput per GPU | (input tokens + output tokens, cached input included) / (last request end - first request start) / serving GPUs | One number per run |

Two consequences follow:

- Time moved out of TTFT lands in ITL, because ITL starts at the first token.
- Interactivity is set by the slowest requests, so a stall on a few requests
  costs far more than its average suggests.

## Patterns that game the benchmark

### Time to first token

| Pattern | Why it is not a real gain | Detect it |
|---|---|---|
| Stream a first token before the request can keep generating, e.g. send the prefill's sampled token while the request still waits for a decode slot | The wait does not disappear; it moves between token 1 and token 2. TTFT drops, interactivity collapses, and end-to-end latency is unchanged. | A long gap after the first token: per-request ITL far above the decode step time on requests with high queue wait. |
| Send an early non-empty placeholder chunk (whitespace, a fixed preamble) | The user receives no useful output sooner. | First chunk content identical across requests, or not produced by the model. |
| Report only the warm part of the run, or start the timed window after a cache warm-up the real workload does not have | Cold-start cost is part of serving. | Compare the timed window with the full log; per-window cache hit and TTFT. |

### Interactivity and inter-token latency

| Pattern | Why it is not a real gain | Detect it |
|---|---|---|
| Fixed or forced speculative acceptance that skips work: skip target sampling, skip verification, or skip the draft or target compute whose result a simulated acceptance length would discard | A simulated acceptance length is a measurement aid. Skipping the work it makes unnecessary removes compute a real run must do. | Kernel profiles missing sampling or verify kernels; step time that does not change when the simulated acceptance length is turned off. |
| Ignore the request's sampling parameters, e.g. run greedy when the request asks for temperature and top-k | Changes outputs and removes sampling cost. | Compare served sampling settings with the request; output diversity. |
| Use fields that only a trace replay carries as an oracle, e.g. `max_tokens` set to the recorded output length, to order admission or route requests | Real clients set `max_tokens` as a loose cap (often thousands of tokens, or the context limit), not as the answer's length. | The scheduler or router reads `max_tokens`, trace order or replay ids. A real length predictor uses only data a live request has: prompt length and conversation history. |
| Force output lengths with `ignore_eos` or `min_tokens` when the benchmark does not | Changes the work per request. | Compare with the benchmark's request settings. |

### Throughput per GPU

| Pattern | Why it is not a real gain | Detect it |
|---|---|---|
| Leave GPUs out of the count: prefill, frontend, draft, or shared or idle GPUs | Every GPU the serving system holds is a cost. | Compare the GPU count in the result with the job allocation. |
| Drop, fail or time out hard requests and count only the completed ones | The remaining requests are an easier distribution. | Error and incomplete counts; completed requests compared with the requests sent. |
| Raise cache hit by exploiting the synthetic trace: pre-load the timed prompts, or cache content a real user would not repeat | The hit rate does not transfer to real traffic. | Cache hit above the trace's theoretical ceiling; hits on the first turn of new sessions. |

### Accuracy

| Pattern | Why it is not a real gain | Detect it |
|---|---|---|
| Run the accuracy check on a different image, flag set or topology than the performance run | The accuracy result does not cover the measured configuration. | Compare the image digest and server arguments of both runs. |
| Lower precision, or skip layers, experts or tokens, beyond what the benchmark allows | Changes the model; a small accuracy suite can miss the loss. | Diff the precision and the model math against the reference; check the benchmark's allowed changes. |
| Tune on the accuracy suite itself, or cut reasoning length to finish faster | Overfits or truncates the output the user asked for. | Output length against the reference; a held-out accuracy check. |

## Real gains that need a check

These are legitimate when they hold for real traffic. Check that they do.

| Change | Check |
|---|---|
| Larger prefix cache or better eviction | The hit rate stays at or below the trace's theoretical ceiling, and the gain comes from fewer evictions, not from pre-loading. |
| More decode seats or extra pre-allocated slots | Count every added GPU in throughput per GPU. |
| Admission or routing by predicted output length | The predictor uses only data a live request has (prompt, history), not replay-only fields. |
| Simulated acceptance length for speculative decoding | All sampling and verify work still runs; report the real acceptance length from an accuracy run alongside it. |

## Contributing

Add a row when you find a new pattern: what it is, which metric it inflates, why
a real user would not see the gain, and how to detect it in logs or results.
