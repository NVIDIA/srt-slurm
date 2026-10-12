# SGLang log metrics

This synthetic example contains prefill/decode batch snapshots and completed
request timing records. It needs no cluster, client export, or telemetry capture.
From the repository root:

```bash
uv run srtctl dsight build examples/dsight/sglang \
  --output /tmp/dsight-sglang --single-file --no-otel --iteration-timezone UTC
uv run srtctl dsight query /tmp/dsight-sglang metrics \
  --name log_sglang_request_queue_duration_ms --points
```

Open `/tmp/dsight-sglang/index.html` and select **Metrics**. All 33 supported
SGLang metric families appear in the catalog. Useful pins are
`log_sglang_request_cached_input_fraction`,
`log_sglang_request_queue_duration_ms`, and `log_sglang_running_requests`.

The three prefill requests complete at the same logged millisecond. Their queue
durations are 10, 10, and 40 ms: the chart displays a median of 10 ms and labels
the three events, while the query retains each value and its source line.
The second host for `decode-0` illustrates a distributed worker; its DP-rank
observations remain separate.

The unprefixed `Prefill batch` lines use stock second-resolution timestamps and
omit the optional batch counter. Their new-token counts are 64, 64, and 128 at
the same second; the chart labels their median and preserves all three raw
observations. Missing ranks remain unknown. Alternative request records expose
bootstrap-queue and preallocation-queue durations under their own metric names,
without inventing a completed bootstrap or allocation-wait duration.

These timestamps are synthetic UTC values. For real logs, pass their actual
timezone. Request timing points mark log emission after completion. Stage
durations can overlap, and a logged zero may indicate a missing timing endpoint.
See the [metric definitions and limitations](../../../docs/dsight-log-metrics.md#sglang-batch-metrics).
