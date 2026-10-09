# Benchmarks

The `benchmark:` block (which client runs against the endpoint) and `post_eval:`. Scoring details for the accuracy suites: [Accuracy Benchmarks](accuracy.md).

## benchmark

Benchmark configuration. The `type` field determines which benchmark runner is used and what additional fields are available.

Set `benchmark.stream_output: true` to mirror `benchmark.out` to the orchestrator's
stdout while the benchmark runs (default: `false`). This works with custom and
built-in benchmark clients and preserves the complete log file and client exit
status. For a one-off run, use `srtctl apply -f recipe.yaml --set benchmark.stream_output=true`.
The output appears in the Slurm job log; CI must follow that log to display it
live. `apply` still submits asynchronously. Streaming forwards bytes as they
reach the log, so clients must flush their output (for Python clients, set
`PYTHONUNBUFFERED: "1"` in their environment). Manual/serve-only jobs have no
benchmark client to stream, and separate post-eval logs are not included.

**Per-type fields.** Every type accepts the shared fields (`placement`, `colocate_with_frontend`, `stream_output`, `aiperf_package`, `aiperf_args`, and `concurrencies`, which power telemetry reads for its measurement windows whatever the type) plus the fields its runner reads. [Benchmark types](schema-reference.md#benchmark-types), generated from the runner registry, lists them per type.

A `schema: 2` recipe that sets a field its type does not use is rejected at load with the list of accepted fields. Before this, such a field was a silent no-op (`isl` on `gsm8k`, `num_shots` on `sa-bench`). Each runner declares its fields as `config_fields`; adding a field to a runner means adding it there.

Types and defaults: [BenchmarkConfig](schema-reference.md#benchmarkconfig). `placement.node` is described under [placement](topology.md#placement); `colocate_with_frontend: false` reserves separate dedicated nodes for the frontend and the client.

### Available Benchmark Types

The registered types, a one-line summary of each, and the keys each accepts are in [Benchmark types](schema-reference.md#benchmark-types). `manual` (the default) runs no client. MLPerf runs as a `custom` benchmark with a bundled script; see [mlperf](#mlperf).

### manual

No benchmark is run. Use for manual testing and debugging.

For a one-off serving run, `srtctl apply -f config.yaml --serve-only` provides the same behavior without changing the recipe's configured benchmark.

```yaml
benchmark:
  type: "manual"
```

### custom

Run an arbitrary command with `bash -lc`. The command is passed verbatim; srt-slurm does not expand `{placeholder}` expressions. Use environment variables for runtime-discovered values:

```yaml
benchmark:
  type: custom
  command: >-
    ./run-benchmark.sh "$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT"
  env:
    MY_BENCHMARK_OPTION: "value"
```

Every custom benchmark command receives frontend metadata plus mode-specific metadata for each logical worker leader:

| Variable                        | Format                         | Description |
| ------------------------------- | ------------------------------ | ----------- |
| `SRT_FRONTEND_HOST`             | IP                             | Frontend/orchestrator IP |
| `SRT_FRONTEND_PORT`             | port                           | Frontend public port |
| `SRT_PREFILL_IPS`               | comma-separated IPs            | Prefill worker leader IPs |
| `SRT_PREFILL_ENDPOINTS`         | comma-separated `IP:port`      | Prefill worker endpoints |
| `SRT_DECODE_IPS`                | comma-separated IPs            | Decode worker leader IPs |
| `SRT_DECODE_ENDPOINTS`          | comma-separated `IP:port`      | Decode worker endpoints |
| `SRT_AGG_IPS`                   | comma-separated IPs            | Aggregated worker leader IPs |
| `SRT_AGG_ENDPOINTS`             | comma-separated `IP:port`      | Aggregated worker endpoints |
| `AIPERF_SERVER_METRICS_URLS`    | comma-separated HTTP URLs      | AIPerf-compatible `/metrics` URLs for all logical workers (plus SGLang attention-DP followers) |
| `SRT_SERVICE_<NAME>_NODES`      | comma-separated hostnames      | Nodes each launched service runs on, in placement order; `<NAME>` is the service name upper-cased with non-alphanumerics as `_` |
| `SRT_SERVICE_<NAME>_IPS`        | comma-separated IPs            | The same nodes' fabric IPs (the first is the head of a `ray` service) |
| `SRT_SERVICE_<NAME>_NODE_COUNT` | int                            | How many nodes the service spans |
| `SRT_GPUS_PER_NODE`             | int                            | `resources.gpus_per_node` |
| `SRT_WORKER_NODES`              | comma-separated hostnames      | Every engine worker node (empty when the job has no engine roles) |

Only variables for roles present in the recipe are emitted. Entries follow logical topology order (prefill index, decode index, or aggregated index). Multi-node follower ranks are excluded because they do not own separate engines; co-located logical workers retain repeated IPs and distinct ports so list positions remain aligned. One exception applies to `AIPERF_SERVER_METRICS_URLS`: an SGLang role with `dp-size` (or `data-parallel-size`) greater than 1 also lists each follower's URL right after its leader, because every node of an attention-DP worker schedules its own DP ranks and exports their metrics. For such a role the metrics list no longer aligns position by position with `SRT_*_ENDPOINTS`. Dynamo sidecar workers keep logical leaders only. With a Dynamo frontend, endpoint and metrics URLs use each leader's `DYN_SYSTEM_PORT`; other frontends use the worker HTTP port. If KVBM metrics are configured, their URLs are appended to `AIPERF_SERVER_METRICS_URLS` after the logical worker URLs.

Two caveats for `AIPERF_SERVER_METRICS_URLS`:

- **Dynamo TRT-LLM worker URLs are advertised when engine metrics are enabled.** This is the default via `engine.publish_metrics: true` (`--publish-metrics`) when the combined setting is omitted; `engine.publish_events_and_metrics: true` also enables them. False or unset uses `engine.publish_metrics` to decide whether to publish metrics and advertise worker URLs. URLs are omitted when metrics-only is false and the legacy combined flag is false or unset, including with observability enabled. This applies to built-in AIPerf and custom benchmarks, excluding sidecars, whose behavior is unchanged. Runtime-only metrics may still exist but do not constitute an engine-metrics capture. With `frontend.type: trtllm_serve` the gate is the worker's own engine config instead: its `/prometheus/metrics` URL is advertised when that role's `args.return_perf_metrics` is true (the srtctl default for trtllm_serve recipes; an explicit `false` drops the URL). KVBM URLs are unaffected; KVBM serves its own endpoint regardless of the flag.
- **An explicit `AIPERF_SERVER_METRICS_URLS` in the recipe `environment:` wins.** Injection is skipped when the variable is already set, so a curated endpoint list is never clobbered.

Values in `benchmark.env` are applied last and can explicitly override any automatically injected variable.

#### AgentX with a custom benchmark

The repository's `benchmarks/` directory is mounted at `/benchmarks` in job containers. Add this
benchmark block to a serving recipe:

```yaml
benchmark:
  type: custom
  command: /benchmarks/agentx.sh
  env:
    MODEL: "<Hugging Face model ID>"
    MODEL_PREFIX: "<InferenceX model prefix>"
    FRAMEWORK: "<framework>"
    PRECISION: "<precision>"
    CONC: "8"
    RESULT_FILENAME: "agentx_c8"
    DURATION: "900"
```

The launcher checks these seven values before downloading anything. It clones InferenceX `main`
and its AIPerf submodule into temporary storage, runs InferenceX's `srt_agentic.sh` against the
ready srt-slurm frontend, and removes the checkout afterward. AIPerf artifacts and the aggregate
JSON are written under `/logs/agentic` by default. The benchmark container needs network access
to GitHub, package downloads, and the AgentX trace dataset. The launch log records the resolved
InferenceX and AIPerf commits.

Set `AGENTX_INFERENCEX_REF` in `benchmark.env` to pin an InferenceX commit for repeatable runs.
`CONC_LIST` can specify multiple space-separated concurrency values; `CONC` is still required.
The launcher supplies InferenceX's common AIPerf settings and accepts overrides through
`benchmark.env`. Power capture is off unless `ENABLE_AGENTX_POWER=1`; when enabled, configure
the power inputs required by the InferenceX harness. `RESULT_DIR` and `AGENTIC_OUTPUT_DIR` may be
overridden to other persistent container paths.

The service variables are how a custom command drives something the job brought up rather than an inference endpoint: a job with no engine roles (`frontend.type: none`, a service that owns the nodes through `services[].nodes`) still runs its benchmark step, and the command finds the service through `SRT_SERVICE_*`. Launchers and clients that are not core live in the repo-root `benchmarks/` folder, mounted in every job container at `/benchmarks` (like `configs/` at `/configs`); `benchmarks/rl/miles/launch.sh` starts a [Miles](miles.md) RL run against a `ray` service.

### sa-bench (Serving Accuracy)

Throughput and latency benchmark at various concurrency levels.

```yaml
benchmark:
  type: "sa-bench"
  isl: 1024                          # Required: Input sequence length
  osl: 1024                          # Required: Output sequence length
  concurrencies: [256, 512]          # Required: Concurrency levels to test
  req_rate: "inf"                    # Optional: Request rate (default: "inf")
  reuse_http_connections: false      # Optional: Reuse HTTP connections (default: false)
```

**Concurrencies format**: Can be a list `[128, 256, 512]` or x-separated string `"128x256x512"`.

When `reuse_http_connections` is enabled, each `benchmark_serving.py` process uses one keep-alive connection pool. Warmup and formal runs remain isolated in separate processes and therefore never share a pool. The option currently applies only to SA-Bench's Dynamo HTTP adapter.

### sglang-bench

SGLang `bench_serving` benchmark at various concurrency levels.

```yaml
benchmark:
  type: "sglang-bench"
  isl: 1024                          # Required: Input sequence length
  osl: 1024                          # Required: Output sequence length
  concurrencies: [256, 512]          # Required: Concurrency levels to test
  req_rate: "inf"                    # Optional: Request rate (default: "inf")
```

**Concurrencies format**: Can be a list `[128, 256, 512]` or x-separated string `"128x256x512"`.

### mmlu

MMLU accuracy evaluation using sglang.test.run_eval.

```yaml
benchmark:
  type: "mmlu"
  num_examples: 200                  # Optional: Number of examples
  max_tokens: 2048                   # Optional: Max tokens per response
  repeat: 8                          # Optional: Number of repeats
  num_threads: 512                   # Optional: Concurrent threads
```

Runner defaults when unset: `num_examples` 200, `max_tokens` 2048, `repeat` 8, `num_threads` 512.

### gpqa

Graduate-level science QA evaluation using sglang.test.run_eval.

```yaml
benchmark:
  type: "gpqa"
  num_examples: 198                  # Optional: Number of examples
  max_tokens: 32768                  # Optional: Max tokens per response
  repeat: 8                          # Optional: Number of repeats
  num_threads: 128                   # Optional: Concurrent threads
```

Runner defaults when unset: `num_examples` 198, `max_tokens` 32768, `repeat` 8, `num_threads` 128.

### longbenchv2

Long-context evaluation benchmark.

```yaml
benchmark:
  type: "longbenchv2"
  max_context_length: 128000         # Optional: Max context length
  num_threads: 16                    # Optional: Concurrent threads
  max_tokens: 16384                  # Optional: Max tokens
  num_examples: null                 # Optional: Number of examples (all if null)
  categories:                        # Optional: Task categories
    - "multi_doc_qa"
    - "single_doc_qa"
```

Runner defaults when unset: `max_context_length` 128000, `num_threads` 16, `max_tokens` 16384; every example and category.

### router

Router performance benchmark with prefix caching. **Requires `frontend.type: sglang-router`**.

```yaml
benchmark:
  type: "router"
  isl: 14000                         # Optional: Input sequence length
  osl: 200                           # Optional: Output sequence length
  num_requests: 200                  # Optional: Number of requests
  concurrency: 20                    # Optional: Concurrency level
  prefix_ratios: [0.1, 0.3, 0.5, 0.7, 0.9]  # Optional: Prefix ratios to test
```

Runner defaults when unset: `isl` 14000, `osl` 200, `num_requests` 200, `concurrency` 20, `prefix_ratios` "0.1 0.3 0.5 0.7 0.9".

### mooncake-router

KV-aware routing benchmark using Mooncake conversation trace.

```yaml
benchmark:
  type: "mooncake-router"
  mooncake_workload: "conversation"  # Optional: Trace type
  ttft_threshold_ms: 2000            # Optional: Goodput TTFT threshold
  itl_threshold_ms: 25               # Optional: Goodput ITL threshold
```

Runner defaults when unset: `mooncake_workload` "conversation", `ttft_threshold_ms` 2000, `itl_threshold_ms` 25.

**Workload options**: `"mooncake"`, `"conversation"`, `"synthetic"`, `"toolagent"`

Dataset characteristics (conversation trace):
- 12,031 requests over ~59 minutes (3.4 req/s)
- Avg input: 12,035 tokens, Avg output: 343 tokens
- 36.64% cache efficiency potential

### agentperf

Trajectory-replay benchmark using the standalone [agentperf-client](https://github.com/ArtificialAnalysis-External/agentperf-client), a deterministic agentic load generator with a Rust streaming core. The client checkout is mounted into the container (pin the commit for comparable runs); the workload definition (trajectory dataset, user-assignments file, `settling_time_seconds`, `phase_timeout_seconds`, stop criteria) lives in the client's own config YAML. srtctl injects the endpoint, model and concurrency at run time via the client's `--base-url` / `--model` / `--concurrencies` flags. Note the client validates the workload YAML *before* merging CLI overrides, so the YAML must still carry syntactically valid placeholder `base_url`, `model` and `concurrencies` values, and `phase_timeout_seconds` must satisfy the client's ramp-up bound for the *injected* concurrency (`phase_timeout_seconds >= (concurrency - 1) / user_spawn_rate + settling_time_seconds + min_measurement_seconds`).

```yaml
benchmark:
  type: "agentperf"
  agentperf_client_dir: "/agentperf-client"       # Container path to the client checkout
  agentperf_config: "/workloads/agentperf.yaml"   # Container path to the client's workload YAML
  concurrencies: [1010]                           # One benchmark phase per level
  env:
    AGENTPERF_EXTRA_ARGS: "--seed 100"            # Optional: appended to agentperf/run.py verbatim

extra_mount:
  - "/path/on/host/agentperf-client:/agentperf-client"
  - "/path/on/host/workloads:/workloads"
```

Required: `agentperf_client_dir`, `agentperf_config`, and one of `concurrency` / `concurrencies` (the string form is x-separated, `"64x1010"`, as for the other types).

Notes:
- The first run of a job builds an isolated client runtime under `/tmp/agentperf-<jobid>` (uv env, pinned Rust toolchain, `rustcore` extension, tokenizer cache) and stages the trajectory and user-assignments datasets from shared storage to node-local `/tmp`; this preflight needs network egress from the benchmark node and adds several minutes before the first phase.
- The user-assignments file referenced by the workload YAML must cover the highest concurrency level (`assign_trajectories` fails loudly otherwise).
- Results land under `<log_dir>/agentperf/` (per-phase `*__traj*.{jsonl,txt,json}`, `requests.jsonl`, `phase_manifest.jsonl`); `rollup.py` normalizes them into `benchmark-rollup.json`.
- Two runs must not share a results dir concurrently (the client resets `phase_manifest.jsonl` at start).
- `telemetry:` (DCGM power measurement windows) is not supported with agentperf; the schema rejects non-sa-bench benchmark types at config load. Tachometer (`observability.enabled`) works normally.

### mlperf

MLPerf runs as a **`custom` benchmark driving the MLPerf team's `inference-endpoint` client**, not as a benchmark type. srt-slurm carries no MLPerf-specific schema at all; the driver is a script at `/srtctl-benchmarks/mlperf/bench.sh`, mounted for every benchmark type.

```yaml
benchmark:
  type: custom
  command: bash /srtctl-benchmarks/mlperf/bench.sh
  env:
    MLPERF_CLIENT_CONFIG: /configs/dsr1-interactive-submission.yaml
    MLPERF_MODE: both            # both (default) | perf | acc

extra_mount:
  - "/path/to/client-configs:/configs"
```

**The client config is passed through, not re-modelled.** It carries ~60 nested settings (model params, two datasets with accuracy scoring, load pattern, a ZeroMQ transport block, drain/warmup/early-stopping) and its shape moves with the client version. Expressing any of it as srt-slurm settings would be a losing race and lossy: anything not modelled becomes unsettable. The script rewrites exactly two values, being the only two the config cannot know before the cluster exists:

| Rewritten | Why |
|---|---|
| `endpoint_config.endpoints` | frontend IPs are assigned by Slurm at run time |
| `report_dir` | so results land with the job's other logs and get collected |

Everything else is passed through untouched, including unresolved `${VAR}` placeholders that the client expands itself at load time. This mirrors the MLPerf team's own launcher (`endpoints-launch`, `NVIDIA/src/sflow/tools/generate_endpoint_yaml.py`), which rewrites one key and leaves the rest.

Start from a template in the client repo (`src/inference_endpoint/config/templates/submission_template.yaml`) or one of the ~45 point configs in `endpoints-launch` under `NVIDIA/src/configs/<system>/<model>/point_*/client.yaml`.

| Variable | Required | Default | Description |
| -------- | -------- | ------- | ----------- |
| `MLPERF_CLIENT_CONFIG` | Yes | - | Container path to the client config |
| `MLPERF_MODE` | No | `both` | `perf`, `acc`, or `both`. These are the client's own mode names; note they are *not* the `performance`/`accuracy` spellings used for dataset types inside the client config |
| `MLPERF_ENDPOINTS` | No | the injected frontend | Comma-separated list, for client-side load balancing |
| `MLPERF_CLIENT_BIN` | No | `inference-endpoint` | Client executable |

Notes:

- **Do not mount the client config at `/configs`.** srt-slurm mounts its own `configs/` there, holding the `nats-server` and `etcd` binaries the head node starts from; an `extra_mount` onto the same path shadows them and the job dies early with `NATS binary not found: /configs/nats-server`, which reads like a broken install rather than a mount collision. Use any other path.
- **Run it in the MLPerf endpoint client image** (`endpoint_client_*.sqsh`). The client ships pre-installed there, so there is nothing to build; the script checks it is on `PATH` and fails with that message if not.
- **The endpoint is injected, never defaulted.** srt-slurm sets `SRT_FRONTEND_HOST` / `SRT_FRONTEND_PORT` for every custom benchmark, and the script errors if they are absent rather than quietly benchmarking localhost.
- **`MLPERF_ENDPOINTS` is how you get more than one frontend.** The client load-balances across the list itself, which is how MLPerf gets past the roughly 28k-connection ceiling of a single `ip:port`; its own submission configs ask for 84,000. srt-slurm exposes a single frontend today, so at submission scale this override is currently the only route.
- The script writes `benchmark-rollup.json` itself, which is the artifact srt-slurm's postprocess already reads. Per-run metrics are deliberately absent: this client does not use LoadGen and writes its own report format, which has not been observed here yet, and a fabricated parser would be worse than an honest gap. The record points at `report_dir` and lists what landed there.

## post_eval

How the accuracy evaluation is dispatched when the job environment sets `RUN_EVAL=true` (run after the benchmark) or `EVAL_ONLY=true` (run instead of it). srtctl forwards a built-in list of workflow variables into the eval process (`RUN_EVAL`, `EVAL_ONLY`, `MODEL`, `ISL`, `OSL`, `PREFILL_TP`, ...); this block extends that list and can replace the command, so a runner sets config instead of patching srtctl's source.

```yaml
post_eval:
  passthrough_env:          # forwarded into the eval process when set in the job environment
    - EVAL_FRAMEWORK
    - EVAL_CONC
    - EVAL_LIMIT
    - EVAL_SUITE
  command:                  # optional; replaces the built-in lm-eval runner command
    - bash
    - /infmax-workspace/benchmarks/evals/run.sh
    - "{endpoint}"
```

Fields: [PostEvalConfig](schema-reference.md#postevalconfig). `command` may use `{endpoint}` (the frontend URL) and `{infmax_workspace}`; it is not shell-interpreted.

`MODEL_NAME` (the served model name) and `EVAL_CONC` are always set by srtctl. `srtctl dry-run` prints the effective dispatch.

Eval steps inherit the recipe's `srun_options`, including container flags such as `container-writable`.
