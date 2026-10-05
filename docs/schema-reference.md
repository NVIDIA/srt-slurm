# Schema Reference

<!-- GENERATED FILE. Do not edit by hand. Regenerate with `srtctl schema-docs`; CI fails when this file is stale. -->

Field-level reference for the recipe layout (`schema: 2`) and the cluster config `srtslurm.yaml` (`ClusterConfig`), generated from the dataclasses in `srtctl.core.schema` and `srtctl.backends`. Each table lists the YAML key, the type, the default (`required` when there is none), and a description taken from the class docstring or the comment on the field. Nested types link to their own table. The pre-2.0 (v1) layout no longer loads; its key-by-key mapping onto this layout is in [legacy-v1.md](legacy-v1.md) and `srtctl migrate` rewrites it. For prose, examples, and semantics see [config-reference.md](config-reference.md). The same data is available as JSON Schema from `srtctl schema` and per field from the MCP `explain_field` tool.

## Recipe

Top-level keys of a recipe YAML.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | required | Job name: the Slurm `--job-name` (unless RUNNER_NAME is set) and the run's label in results. |
| `model` | [ModelConfig](#modelconfig) | required | Model weights, container image, and precision. |
| `resources` | [ResourceConfig](#resourceconfig) | required | GPU type, GPUs per node, and allocation knobs. The worker topology is `roles`. |
| `schema` | one of `2` | required | Recipe schema version. Every recipe declares `schema: 2`; a recipe without it is the pre-2.0 layout and does not load (see [legacy-v1.md](legacy-v1.md) and `srtctl migrate`). |
| `slurm` | [SlurmConfig](#slurmconfig) | `SlurmConfig()` | Slurm account, partition, and time limit; unset values come from srtslurm.yaml. |
| `engine` | str \| mapping | optional when every role sets `engine` | The engine type (`atom`, `sglang`, `tilert`, `tokenspeed`, `trtllm`, `vllm`, `mocker`) as a string, or a mapping with `type` plus the engine-wide knobs listed under [Engine types](#engine-types). |
| `roles` | dict[str, [RoleConfig](#roleconfig)] | `{}` | One block per worker role (`prefill`, `decode`, `agg`): nodes, workers, GPUs, env, engine args. |
| `frontend` | [FrontendConfig](#frontendconfig) | `FrontendConfig()` | The HTTP entry point in front of the workers (Dynamo frontend, router, nginx) and where it runs. |
| `dynamo` | [DynamoConfig](#dynamoconfig) | `DynamoConfig()` | Which Dynamo to install, its request/event planes, and native sidecar mode. |
| `benchmark` | [BenchmarkConfig](#benchmarkconfig) | `BenchmarkConfig()` | The client run once the workers are ready; `type` selects the runner. |
| `profiling` | [ProfilingConfig](#profilingconfig) | `ProfilingConfig()` | Nsight Systems or PyTorch profiling of the workers. |
| `output` | [OutputConfig](#outputconfig) | `OutputConfig()` | Where the job writes logs and results. |
| `health_check` | [HealthCheckConfig](#healthcheckconfig) | `HealthCheckConfig()` | How long to poll the frontend for ready workers before failing the run. |
| `observability` | [ObservabilityConfig](#observabilityconfig) | `ObservabilityConfig()` | Engine metrics and traces, Tachometer collection, and automatic Nsight tracing. |
| `telemetry` | [TelemetryConfig](#telemetryconfig) | `TelemetryConfig()` | GPU (DCGM) and CPU power sampling over the benchmark measurement windows. |
| `environment` | dict[str, str] | `{}` | Environment variables for every worker; applied after `roles.<role>.env`, so a key set in both takes this value. Values may use `{node}` and `{node_id}`. |
| `container_mounts` | dict[[FormattablePath](#formattablepath), [FormattablePath](#formattablepath)] | `{}` | Host path -> container path mounts for every container; both sides are FormattablePaths. |
| `extra_mount` | tuple[str, ...] \| None | `None` | Extra mounts as `host:container[:ro]` strings; `$VARS` expand, `{placeholders}` do not. |
| `srun_options` | dict[str, str] | `{}` | Extra srun options (`key: value` -> `--key=value`; empty value -> `--key`) for every job step. |
| `sbatch_directives` | dict[str, str] | `{}` | Extra `#SBATCH --key=value` lines (empty value -> `--key`); wins over the cluster defaults. |
| `enable_config_dump` | bool | `True` | Accepted for compatibility; srtctl does not read it. Workers dump their config where the engine supports it. |
| `setup_script` | str \| None | `None` | Custom setup script (runs before dynamo install and worker startup) e.g. "custom-setup.sh" -> runs /configs/custom-setup.sh |
| `host_setup` | [HostSetupConfig](#hostsetupconfig) | `HostSetupConfig()` | Commands run on each node's bare host, outside the container, before any worker starts. Cluster-wide default lives in srtslurm.yaml as default_host_setup; a recipe that sets this block replaces that default. |
| `services` | list[[ServiceConfig](#serviceconfig)] | `[]` | Long-running processes launched next to the job: generic sidecars (an experimental router built from a PR) and typed ones (a standalone Mooncake store per worker node). See docs/services.md. |
| `post_eval` | [PostEvalConfig](#postevalconfig) | `PostEvalConfig()` | Post-benchmark / eval-only evaluation dispatch: extra env forwarded into the eval process and an optional command override. Replaces the downstream source patch that used to extend the passthrough list in do_sweep.py. |
| `identity` | [IdentityConfig](#identityconfig) | `IdentityConfig()` | Virtual identity — declares what *should* be running (verified against fingerprint) |
| `reporting` | [ReportingConfig](#reportingconfig) \| None | `None` | Reporting configuration (status API, future: logs to S3, etc.) |

## Authoring surface

Three vocabularies carry the engine, the topology, and the placement. `engine` and `roles` are fields of the recipe (the table above and [RoleConfig](#roleconfig) below); the engine dataclasses' per-mode fields are internal and receive each role's settings at load. The pre-2.0 layout no longer loads and is documented for migration in [legacy-v1.md](legacy-v1.md).

### engine

`engine: <type>` or `engine: {type: <type>, ...}`. `type` is one of `atom`, `sglang`, `tilert`, `tokenspeed`, `trtllm`, `vllm`, `mocker`; the remaining keys are that engine's knobs, listed under [Engine types](#engine-types).
Use either one top-level engine or an explicit engine on every role, never both. Role engines do not inherit options from each other.

### roles

`roles.<role>` for `prefill`, `decode`, `agg`. The `agg` role is the aggregated deployment; `prefill` and `decode` together are the disaggregated one. Each role is a [RoleConfig](#roleconfig): `nodes` (or `colocate` on decode, to share the prefill nodes), `workers`, `gpus`, `env`, `args`, and optionally its own `engine` and `container`.

### placement

| Key | Values | Default |
|---|---|---|
| `frontend.placement.node` | `head` \| `first_decode` \| `dedicated` | `head` |
| `benchmark.placement.node` | `head` \| `last_decode` \| `dedicated` | `head` |
| `services[].placement.node` | `head` \| `infra` \| `dedicated` \| `prefill` \| `decode` \| `agg` \| `workers` | type default |

`dedicated` reserves a node for that component. The discovery plane (etcd, NATS) is placed through its `services:` entries; see [services.md](services.md).

## Recipe sections

### ModelConfig

Model configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `path` | str | required | Model weights directory, or a `model_paths` alias from srtslurm.yaml. Mounted at /model. |
| `container` | str | required | Container image (`.sqsh` path or registry URI), or a `containers` alias from srtslurm.yaml. |
| `precision` | str | required | Weight precision (`fp4`, `fp8`, `fp16`, `bf16`). Recorded with results; engine flags set the actual dtype. |
| `stage_dir` | str \| None | `None` | Optional: stage the model from shared storage to this node-local dir before workers start (e.g. "/raid/scratch/models"). None = use path directly. |

### ResourceConfig

Cluster facts and allocation knobs; the worker topology is the `roles:` block.

| Key | Type | Default | Description |
|---|---|---|---|
| `gpu_type` | str \| None | `None` | GPU type (h100, gb200, ...). Cluster fact, not a topology choice. Optional: a recipe that omits it inherits `default_gpu_type` from srtslurm.yaml, and `gpus_per_node` inherits the cluster `gpus_per_node`. Both are still worth setting in a recipe so it is self-describing for result rollups. |
| `gpus_per_node` | int | `4` | GPUs on each node. Inherits the cluster `gpus_per_node` when omitted, else 4. |
| `spread_workers` | bool | `False` | If True, place each partial-node worker on its own node instead of packing multiple onto the same node. Caller must reserve enough nodes (e.g. give roles.decode as many nodes as workers when its gpus < gpus_per_node). |
| `het_jobs` | bool \| None | `None` | SLURM heterogeneous-job opt-in. Tri-state: None defers to the cluster default `use_het_jobs` on ClusterConfig; True/False overrides per recipe. When effectively True (and we are in disaggregated mode), the prefill and decode sides are submitted as two het components each with their own `--segment`. See HetComponent above and docs/slurm-faq.md. |

### SlurmConfig

SLURM job settings.

| Key | Type | Default | Description |
|---|---|---|---|
| `account` | str \| None | `None` | Slurm account. Unset uses `default_account` from srtslurm.yaml. |
| `partition` | str \| None | `None` | Slurm partition. Unset uses `default_partition` from srtslurm.yaml. |
| `time_limit` | str \| None | `None` | Job time limit (HH:MM:SS). Unset uses `default_time_limit` from srtslurm.yaml. |

### RoleConfig

One worker role of the recipe: `roles.prefill`, `roles.decode`, or `roles.agg`.

| Key | Type | Default | Description |
|---|---|---|---|
| `nodes` | int \| one of `'colocate'` \| None | `None` | Nodes reserved for this role. `colocate` (decode only) reserves none and packs the decode workers onto the prefill nodes' free GPUs; `gpus` is then required on both roles and the loader rejects a split that does not fit. |
| `workers` | int \| None | `None` | Number of workers of this role. |
| `gpus` | int \| None | `None` | GPUs per worker. Defaults to `nodes * gpus_per_node // workers`; required when decode colocates. |
| `srun_options` | dict[str, str] | `{}` | Merged over the recipe srun_options on this role's worker steps only (e.g. a per-step mem cap). |
| `env` | dict[str, str] | `{}` | Environment for every worker of this role. |
| `args` | dict[str, Any] | `{}` | The engine's own CLI flags for this role, as a mapping (`tensor-parallel-size: 4`). |
| `extra_args` | list[str] | `[]` | Raw extra CLI arguments (TRT-LLM only). |
| `engine` | str \| mapping | `None` | Engine type or mapping with engine options. Set on every role when no top-level `engine` is declared; the two forms cannot be mixed, and role engines do not inherit options from each other. |
| `container` | str \| None | `None` | Optional role image; accepts cluster container aliases. Defaults to `model.container`. |
| `kv_events` | bool \| dict[str, Any] \| None | `None` | `true` for the default ZMQ publisher, or a mapping with `publisher` / `topic`. |
| `sidecar` | bool \| None | `None` | Run the native engine with a Dynamo sidecar (turns on `dynamo.sidecar`); every role must agree. |
| `critical` | bool | `True` | A worker of this role exiting fails the run. `false` keeps the run alive for probes that kill workers. |
| `restart` | [RestartPolicy](#restartpolicy) | `RestartPolicy()` | Relaunch exited workers in place: `never`, `on-failure`, `always`, or a mapping with `policy`, `max_restarts`, `backoff_seconds`, and `max_backoff_seconds`. |

### FrontendConfig

Frontend/router configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | str | `'dynamo'` | Frontend type - "dynamo" (default); "sglang-router" (SGLang Model Gateway), "vllm-router", "smg" (Shepherd Model Gateway, any backend), "llm-d" (llm-d Endpoint Picker behind Envoy, vLLM), "atomesh", and "tilert-router" (static routers); "sglang", "vllm", and "trtllm_serve" (direct: the single aggregate worker binds the public port, no router process); "none" (services-only job: no router, no OpenAI endpoint, no worker-count health gate; requires no engine roles). Pre-2.0 recipes spelled the router "sglang"; ``srtctl migrate`` rewrites that to "sglang-router". |
| `enable_multiple_frontends` | bool | `True` | Scale with nginx + multiple routers. When ``True`` (default), srtctl stands up nginx and fans out to ``num_additional_frontends + 1`` router replicas. When ``False``, there is NO nginx proxy — the benchmark must target the single master router (or a worker) directly at ``http://localhost:<port>``. ``benchmark.command`` has no placeholder substitution, so write the URL out literally. |
| `num_additional_frontends` | int | `9` | Additional routers beyond master (default: 9) |
| `nginx_container` | str | `'nginx:1.27.4'` | Custom nginx container image (default: nginx:1.27.4) |
| `nginx_raise_ulimit` | bool | `False` | Raise nofile before nginx and set ``worker_rlimit_nofile`` in generated nginx.conf. Off by default; enable on clusters that allow it. Override per job or set ``nginx_raise_ulimit`` in srtslurm.yaml for the cluster. |
| `nginx_session_affinity` | bool | `False` | Consistently hash ``nginx_session_affinity_header`` to a frontend. Requests without that header use a generated request ID and stay distributed. |
| `nginx_session_affinity_header` | str | `'X-Dynamo-Session-ID'` | Header hashed when affinity is on (default ``X-Dynamo-Session-ID``). Set ``X-Correlation-ID`` for clients (e.g. aiperf) that carry the session id in that header instead. |
| `nginx_keepalive_timeout` | str | `'600s'` | Idle timeout for client and upstream keepalive connections in the generated nginx.conf (default "600s"). nginx's own default is 75s, which closes a session's connection during the long recorded think-time of an agentic replay; the client's next write on that pooled socket then fails with "broken pipe" / "server disconnected" and nothing is logged server-side. |
| `worker_selection` | dict[str, Any] \| None | `None` | Inline Dynamo worker-selection policy configuration. srtctl writes this mapping under the top-level ``worker_selection`` key in a generated router policy YAML and passes it to the Dynamo frontend via ``--router-policy-config``. |
| `epp_config` | dict[str, Any] \| None | `None` | llm-d Endpoint Picker configuration (``EndpointPickerConfig``: ``plugins``, ``schedulingProfiles``, ...) for ``frontend.type: llm-d``. srtctl writes it to a file with the ``file-discovery`` plugin and ``dataLayer.discovery`` added, so the EPP reads the job's workers from the endpoints file srtctl renders. Required for prefill/decode jobs (the scheduler's P/D profiles); omitted, an aggregate job runs the EPP's built-in default profile. |
| `args` | dict[str, Any] \| None | `None` | CLI arguments passed to the frontend/router process |
| `env` | dict[str, str] \| None | `None` | Environment variables for frontend processes |
| `container_image` | str \| None | `None` | Optional router-specific image. Static routers use the model/backend image when omitted. |
| `numa_bind` | bool | `False` | Prefix the frontend process command with ``numactl --cpunodebind=0 --membind=0``. Off by default. Has no effect on direct frontends (``sglang``, ``vllm``, aggregate ``trtllm_serve``) that launch no separate frontend process. |
| `ctx_router` | dict[str, Any] \| None | `None` | trtllm_serve orchestrator (ser.yaml) options; ignored by other frontends. |
| `gen_router` | dict[str, Any] \| None | `None` | generation_servers.router |
| `server_config_extra` | dict[str, Any] \| None | `None` | extra top-level ser.yaml keys |
| `placement` | [PlacementConfig](#placementconfig) | `PlacementConfig()` | Where the frontend (trtllm_serve: the disaggregated orchestrator) runs. placement.node is "head" (default: the first prefill/CTX node), "first_decode" (the first decode/GEN worker-leader node), or "dedicated" (a node reserved for the frontend: needs at least 2 nodes, not supported with resources.het_jobs: true). |

### DynamoConfig

Dynamo installation configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `install` | bool | `True` | Install Dynamo into the container before workers start; false when the image already has it. |
| `top_of_tree` | bool | `False` | Clone and build Dynamo at HEAD (unpinned). No `source` equivalent; prefer a commit in `source.rev`. |
| `source` | [DynamoSourceConfig](#dynamosourceconfig) \| None | `None` | Which Dynamo to install: exactly one of git+rev, pypi, or wheel. Unset, and not top_of_tree: the PyPI release DEFAULT_PYPI_VERSION. |
| `request_plane` | one of `'nats'`, `'tcp'`, `'http'` | `'tcp'` | Transport frontends use to send requests to workers. |
| `event_plane` | one of `'nats'`, `'zmq'` \| None | `None` | Sets DYN_EVENT_PLANE for KV and worker events; unset follows the Dynamo image's default. |
| `sidecar` | bool | `False` | Job-wide native sidecar mode; prefer `roles.<role>.sidecar: true`. |
| `sidecar_port` | int | `50051` | Base loopback gRPC port between engine and sidecar; co-located workers get deterministic offsets. |
| `sidecar_binary` | str \| None | `None` | Standalone sidecar executable; unset runs `python3 -m dynamo.<framework>.sidecar`. |
| `sidecar_startup_timeout` | int | `3600` | Seconds to wait for the native engine's gRPC endpoint. |
| `sidecar_context_length` | int \| None | `None` | Context length the sidecar advertises (TRT-LLM); unset reads it from the engine. |
| `sidecar_args` | list[str] | `[]` | Extra arguments appended to the sidecar command. |

### BenchmarkConfig

Benchmark configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | str | `'manual'` | Benchmark runner: `manual` (none) or a registered type; see Benchmark types for the keys each accepts. |
| `stream_output` | bool | `False` | Mirror benchmark.out to the orchestrator's stdout while the client runs; keep the log file. |
| `isl` | int \| None | `None` | Input sequence length in tokens for synthetic requests. |
| `osl` | int \| None | `None` | Output sequence length in tokens for synthetic requests. |
| `concurrencies` | list[int] \| str \| None | `None` | Concurrency levels, one benchmark phase each: a list or an `x`-separated string (`"4x8x16"`). Telemetry uses each phase as a measurement window. |
| `req_rate` | str \| int \| None | `'inf'` | Request arrival rate in requests/s; `inf` sends as fast as concurrency allows. |
| `placement` | [PlacementConfig](#placementconfig) | `PlacementConfig()` | Where the benchmark client runs. placement.node is "head" (default: the orchestrator's node), "last_decode" (the last decode/GEN worker-leader node, isolating the client off the CTX/orchestrator node; use the injected $SRT_FRONTEND_HOST env in the benchmark command's URL), or "dedicated" (a node reserved for the client: needs at least 2 nodes, not supported with resources.het_jobs: true). |
| `colocate_with_frontend` | bool | `True` | Governs how dedicated placements combine when more than one of the benchmark client, the frontend, and the etcd/nats services asks for placement.node: dedicated. If True (default), every requested role shares a single reserved node. If False, each requested role gets its own reserved node (requires enough total nodes: worker count + number of dedicated roles). |
| `sweep` | [SweepConfig](#sweepconfig) \| None | `None` | Accepted for compatibility; no runner reads it. Sweep a recipe with the top-level `sweep:` block. |
| `num_examples` | int \| None | `None` | Accuracy benchmark fields |
| `max_tokens` | int \| None | `None` | Maximum generated tokens per response |
| `repeat` | int \| None | `None` | Times each example is evaluated; scores are averaged |
| `num_threads` | int \| None | `None` | Concurrent evaluation requests |
| `max_context_length` | int \| None | `None` | LongBench v2: skip examples longer than this many tokens |
| `categories` | list[str] \| None | `None` | LongBench v2: task categories to run; unset runs all |
| `num_shots` | int \| None | `None` | GSM8K few-shot examples |
| `temperature` | float \| None | `None` | Sampling temperature; unset uses the runner's default |
| `top_p` | float \| None | `None` | Nucleus sampling threshold; unset uses the runner's default |
| `top_k` | int \| None | `None` | Top-k sampling cutoff; unset uses the runner's default |
| `num_requests` | int \| None | `None` | Router benchmark fields |
| `concurrency` | int \| None | `None` | Single concurrency level (router, agentperf) |
| `prefix_ratios` | list[float] \| str \| None | `None` | Router: shared-prefix ratios to test (list or space-separated string). |
| `mooncake_workload` | str \| None | `None` | Mooncake router benchmark fields (uses aiperf with mooncake_trace) |
| `ttft_threshold_ms` | int \| None | `None` | Goodput TTFT threshold in ms (default: 2000) |
| `itl_threshold_ms` | int \| None | `None` | Goodput ITL threshold in ms (default: 25) |
| `random_range_ratio` | float \| None | `None` | Random input/output length range ratio (default: 0.8) |
| `num_prompts_mult` | int \| None | `None` | Multiplier for num_prompts = concurrency * mult (default: 10) |
| `num_warmup_mult` | int \| None | `None` | Multiplier for warmup prompts = concurrency * mult (default: 2) |
| `dataset_name` | str \| None | `None` | Custom dataset fields (sa-bench) |
| `dataset_path` | str \| None | `None` | Container path to dataset file (mount via extra_mount) |
| `agentperf_client_dir` | str \| None | `None` | AgentPerf benchmark fields (agentperf-client trajectory replay) |
| `agentperf_config` | str \| None | `None` | Container path to the client's workload YAML (endpoint/model/concurrency injected) |
| `trace_file` | str \| None | `None` | Trace replay benchmark fields (uses aiperf with mooncake_trace dataset type) |
| `custom_tokenizer` | str \| None | `None` | Custom tokenizer class (e.g., "module.path.ClassName") |
| `use_chat_template` | bool | `True` | Pass --use-chat-template to benchmark (default: true) |
| `reuse_http_connections` | bool | `False` | SA-Bench Dynamo adapter: reuse a benchmark-scoped HTTP connection pool. Opt-in to preserve the historical per-request ClientSession behavior. |
| `command` | str \| None | `None` | Custom benchmark hook. ``command`` is passed to ``bash -lc`` verbatim; srtctl does NOT substitute placeholders like ``{nginx_url}`` or ``{slurm_job_id}``. Render any parameters when generating the recipe. See srtctl.benchmarks.custom.CustomBenchmarkRunner for details. |
| `container_image` | str \| None | `None` | Image the benchmark client runs in (custom, agentperf); unset uses `model.container`. |
| `env` | dict[str, str] | `{}` | Extra environment variables for the benchmark client (custom, agentperf). |
| `aiperf_package` | str \| None | `None` | aiperf pip install spec (e.g., "aiperf>=0.7.0", "aiperf @ git+https://...@commit") If set, runs pip install <spec> before benchmarking. Upgrades if already installed. |
| `aiperf_args` | dict[str, Any] | `{}` | Extra aiperf CLI flags passed through to bench.sh (e.g., benchmark-duration: 600, workers-max: 200) |
| `slow_down_sleep_time` | float \| None | `None` | SA-Bench: optional SGLang /slow_down on decode workers (sglang frontend only; see benchmark_stage) |
| `slow_down_wait_time` | float \| None | `None` | seconds until POST clears slow_down; unset = feature off |

#### Benchmark types

`benchmark.type` selects a runner. Every type accepts the shared keys `aiperf_args`, `aiperf_package`, `colocate_with_frontend`, `concurrencies`, `placement`, `stream_output`, `sweep`, `type` plus the keys in its row; a recipe that sets any other `benchmark` key fails to load. Generated from the benchmark registry.

| Type | Description | Keys beyond the shared ones |
|---|---|---|
| `manual` | No benchmark client runs; the job serves until it is cancelled or hits its time limit. | none |
| `agentperf` | Run the agentperf-client trajectory replay against the frontend. | `agentperf_client_dir`, `agentperf_config`, `concurrency`, `container_image`, `env`, `isl` |
| `custom` | Run an arbitrary benchmark command inside a container. | `command`, `container_image`, `env` |
| `gpqa` | GPQA (Graduate-level science QA) accuracy evaluation. | `max_tokens`, `num_examples`, `num_threads`, `repeat` |
| `gsm8k` | GSM8K (Grade School Math 8K) accuracy evaluation. | `max_tokens`, `num_examples`, `num_shots`, `num_threads`, `repeat`, `temperature`, `top_k`, `top_p` |
| `lm-eval` | lm-eval accuracy evaluation using InferenceX benchmark_lib. | none |
| `longbenchv2` | LongBench v2 long-context evaluation benchmark. | `categories`, `max_context_length`, `max_tokens`, `num_examples`, `num_threads` |
| `mmlu` | MMLU accuracy evaluation benchmark. | `max_tokens`, `num_examples`, `num_threads`, `repeat` |
| `mooncake-router` | Mooncake Router benchmark for testing KV-aware routing using aiperf. | `itl_threshold_ms`, `mooncake_workload`, `ttft_threshold_ms` |
| `router` | Router performance benchmark. | `concurrency`, `isl`, `num_requests`, `osl`, `prefix_ratios` |
| `sa-bench` | SA-Bench throughput and latency benchmark. | `custom_tokenizer`, `dataset_name`, `dataset_path`, `isl`, `num_prompts_mult`, `num_warmup_mult`, `osl`, `random_range_ratio`, `req_rate`, `reuse_http_connections`, `slow_down_sleep_time`, `slow_down_wait_time`, `use_chat_template` |
| `sglang-bench` | SGLang benchmark runner. | `isl`, `osl`, `req_rate` |
| `trace-replay` | Trace replay benchmark using aiperf with a user-provided dataset. | `itl_threshold_ms`, `trace_file`, `ttft_threshold_ms` |

### ProfilingConfig

Profiling configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | str | `'none'` | "none", "nsys", "nsys-time", or "torch" |
| `extra_nsys_args` | list[str] \| None | `None` | Extra arguments passed to nsys profile (appended before `-o`; see get_nsys_prefix) |
| `nsys_trace` | str | `'cuda,nvtx'` | Non-TRT-LLM Nsight activity domains. ``cuda-sw`` can be selected explicitly where software tracing is preferred over hardware tracing. |
| `trace_fork_before_exec` | bool \| None | `None` | None preserves the existing Dynamo-specific default. Set explicitly for worker launchers that require or cannot tolerate child-process injection. |
| `capture_range_end` | str | `'stop'` | Non-TRT-LLM behavior when cudaProfilerStop closes a capture range. |
| `nsys_library_paths` | list[str] \| None | `None` | Optional paths prepended to LD_LIBRARY_PATH for the Nsight wrapper and profiled worker, for containers that do not discover the host libcuda. |
| `prefill` | [ProfilingPhaseConfig](#profilingphaseconfig) \| None | `None` | Phase-specific profiling step configs (not used for nsys-time) |
| `decode` | [ProfilingPhaseConfig](#profilingphaseconfig) \| None | `None` | Step window for the decode role (disaggregated runs). |
| `aggregated` | [ProfilingPhaseConfig](#profilingphaseconfig) \| None | `None` | Step window for the `agg` role (aggregated runs). |
| `delay_secs` | int \| None | `None` | nsys-time fields: time-based capture window, same on all workers |
| `duration_secs` | int \| None | `None` | nsys --duration: seconds to capture after delay |
| `benchmark_duration_secs` | int | `300` | total traffic generation duration (must cover delay + duration) |

### OutputConfig

Output paths and optional reproducibility artifacts.

| Key | Type | Default | Description |
|---|---|---|---|
| `log_dir` | [FormattablePath](#formattablepath) | `<lambda>()` | Directory for job logs and results; a FormattablePath, so `{job_id}` and `$VARS` expand. |
| `record_launch_plan` | bool | `False` | Save the realized `srun` scripts and a manifest under `logs/launch-plan/`. |

### HealthCheckConfig

Health check configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `max_attempts` | int | `180` | Maximum readiness polls of the frontend before the run fails; 180 x 10 s = 30 minutes by default (large models take time to load). |
| `interval_seconds` | int | `10` | Seconds between readiness polls. |
| `fatal_log_markers` | bool | `True` | Fail the run as soon as a worker's log prints a line the engine names as fatal (for TRT-LLM, the launcher's ``Rank<N> Task exit code: <non-zero>`` and ``Failed to initialize executor``), even while its srun step is still running. Without it a worker whose engine died behind a live launcher is only noticed when this health window runs out. |
| `extra_fatal_log_patterns` | list[str] | `[]` | Additional regular expressions, matched against every new worker log line, that fail the run the same way. |

### ObservabilityConfig

Observability configuration for OTEL tracing.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `False` | Master analytics knob. Default: False. |
| `enable_otel` | bool | `False` | If True, inject OTEL environment variables into all workers and frontends. Requires otel_endpoint to be set. Default: False. |
| `otel_endpoint` | str \| None | `None` | OTEL collector endpoint (e.g. "http://10.0.0.1:4317"). Required when enable_otel is True. |
| `tachometer` | [TachometerConfig](#tachometerconfig) | `TachometerConfig()` | Native Tachometer capture configuration. Collects on every run, independent of ``enabled``, unless ``tachometer.enabled: false`` (see :class:`TachometerConfig`). |
| `nsys` | [NsysObservabilityConfig](#nsysobservabilityconfig) | `NsysObservabilityConfig()` | Automatic Nsight Systems capture, enabled with the master switch. |

### TelemetryConfig

DCGM power telemetry for benchmark measurement windows.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `False` | Collect GPU power over each benchmark concurrency window. |
| `dcgm_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `None` | GPU power exporter image, port, command, labels and metrics. When `enabled` with no exporter and no CPU leg, the cluster `default_gpu_exporter` is used. |
| `collect_interval_ms` | int | `1000` | Milliseconds between collector cycles. Replaces the retired ``default_frequency``, which despite its name was a period in seconds (1000ms == the old 1.0 default). |
| `storage_subdir` | str | `'power'` | Output directory below the run's log directory. |
| `required` | bool | `False` | Fail the benchmark when publishable DCGM power artifacts cannot be produced. CPU power stays best-effort. |
| `clock_sync_check` | bool | `True` | Before any server starts, verify every allocation node reports an NTP-synchronised system clock. Sample timestamps (orchestrator host) and window boundaries (benchmark client host) are compared directly, so an unsynchronised node silently misaligns the measurement. Fails the job when ``required`` is true; otherwise the run continues and the manifest records ``clock_sync_unverified`` with ``publication_valid: false``. Set false on clusters where timedatectl/chronyc/ntpq are unavailable to unprivileged users. |
| `startup_timeout_seconds` | float | `30.0` | Seconds to wait for the exporters to answer before giving up (DCGM and CPU legs). |
| `request_timeout_seconds` | float | `2.0` | Per-request exporter timeout in seconds (DCGM and CPU legs). |
| `collector_join_timeout_seconds` | float \| None | `None` | None derives a safe shutdown budget from request_timeout_seconds. |
| `cpu_power_exporter` | [CpuPowerExporterConfig](#cpupowerexporterconfig) \| None | `None` | Head-node scrape of a per-node CPU power exporter. Setting the block enables it. |
| `cpu_power` | [CpuPowerConfig](#cpupowerconfig) | `CpuPowerConfig()` | In-job CPU power collector on each node, independent of `cpu_power_exporter`. |

### FormattablePath

A path that may contain placeholders requiring formatting.

| Key | Type | Default | Description |
|---|---|---|---|
| `template` | str | required | The path text, with `{placeholders}` and `$VARS` expanded at runtime. |

### HostSetupConfig

Commands run on the bare host of each allocated node, outside the container.

| Key | Type | Default | Description |
|---|---|---|---|
| `commands` | list[str] | `[]` | Shell commands run in order on each node, joined with ``&&``. |
| `teardown` | list[str] | `[]` | Shell commands run on each node after workers stop. Runs even when the job fails, so state that outlives the allocation (locked clocks persist for the next tenant) gets reset. |
| `nodes` | one of `'all'`, `'workers'` | `'all'` | Which nodes to target. "all" covers head, infra, and workers; "workers" covers only the nodes running backend workers. |
| `ignore_failure` | bool | `False` | When True, a failing node logs a warning instead of failing the job. |
| `timeout_seconds` | int | `300` | Per-node wall-clock budget for commands and for teardown. |

### ServiceConfig

One entry of the top-level ``services:`` list.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | required | Unique label; names the log file (``service_<name>.out``) and the tracked process. |
| `type` | str | `'generic'` | Service kind. ``generic`` (default) launches exactly what you wrote; ``mooncake-store`` runs a standalone Mooncake Store wired to the managed master. See ``docs/services.md`` for the kinds. |
| `command` | list[str] \| None | `None` | Argv to launch (not shell-interpreted). Required for ``generic``; typed kinds supply a default. |
| `args` | list[str] | `[]` | Extra argv appended to ``command``. |
| `container` | str \| None | `None` | Container image or ``srtslurm.yaml`` alias. Defaults to the kind's fallback image, then the job container. |
| `env` | dict[str, str] | `{}` | Environment for the service process, on top of what the kind injects. |
| `source` | [SourceConfig](#sourceconfig) \| None | `None` | Optional git source to clone before ``build_command`` and ``command`` run. Single-node placements only. |
| `build_command` | list[str] \| None | `None` | Argv run once inside the service container, from the clone, before ``command`` starts. Only meaningful with ``source``. |
| `placement` | [ServicePlacementConfig](#serviceplacementconfig) \| None | `None` | Where the service runs. Defaults to the kind's placement (``head`` for generic services, ``infra`` for etcd/nats/mooncake-master, ``workers`` for the exporters). |
| `nodes` | int \| None | `None` | Whole nodes this service owns: its pool. Pools add to the allocation next to the engine roles' nodes and are carved after them in declaration order, so a Ray cluster, a sandbox fleet and an engine role can each have their own nodes in one recipe. An owner is placed on its own pool (``placement.node: workers``); other services join it with ``placement.pool: <name>``. |
| `start` | str \| None | `None` | ``after_frontend`` (default for ``generic``) or ``before_workers`` (default for ``mooncake-store``). |
| `readiness` | [ServiceReadinessConfig](#servicereadinessconfig) \| None | `None` | Optional TCP port gate; the job waits for it on every service node before continuing. |
| `inherit_discovery_env` | bool | `True` | Inject ``ETCD_ENDPOINTS`` / ``NATS_SERVER`` so the service can register with the job's Dynamo discovery plane. |
| `critical` | bool \| None | `None` | When true a crash fails the run, like a worker dying. Default false for ``generic`` (a dead sidecar costs its own log, not the run) and true for ``mooncake-store``. Set true for anything in the live request path. |
| `terminal` | bool | `False` | This service is the job's run: the job ends when every instance of every terminal service has exited, and the worst exit code becomes the job's. A recipe with a terminal service has no benchmark step (``benchmark.type`` stays ``manual``); a torchrun pool that trains to completion is the shape. |
| `preamble` | str \| None | `None` | Shell run inside the container before ``command`` (``ulimit`` and friends). |
| `cpus_per_task` | int \| None | `None` | Optional ``srun --cpus-per-task``. |
| `cpu_bind` | str \| None | `None` | Optional ``srun --cpu-bind``. |
| `srun_options` | dict[str, str] | `{}` | Extra srun options for this service only. |
| `build_timeout_seconds` | int | `1800` | Kill ``build_command`` after this many seconds. |
| `enabled` | bool | `True` | ``false`` drops the service, including an implicit one (``etcd`` / ``nats`` under the Dynamo frontend, the default exporters) declared here by name. |
| `external` | str \| None | `None` | For discovery-plane kinds (``etcd``, ``nats``, ``mooncake-master``): use this already-running endpoint and launch nothing; the URL is what the job's processes are pointed at. |
| `options` | dict[str, Any] | `{}` | Kind-specific settings (``nats``: ``max_payload_mb``; ``mooncake-master``: ``store_config`` and ``device_names_by_gpu`` for vLLM). Unknown keys are rejected by the kind. |
| `metrics` | list[[ServiceMetricsConfig](#servicemetricsconfig)] | `[]` | Prometheus endpoints this service serves: one mapping or a list of ``{port, path, nodes, name}`` (``path`` defaults to ``/metrics``, ``nodes`` to ``all``). Tachometer scrapes each on every node the service runs on, or on its first node with ``nodes: first``, as endpoint ``<name>_<node>`` where ``name`` defaults to the service name. The exporter kinds declare theirs; write it for a generic service that publishes metrics, or on a ``ray`` service whose head serves a trainer's collector and router. |

### PostEvalConfig

How the post-benchmark (or eval-only) accuracy evaluation is dispatched.

| Key | Type | Default | Description |
|---|---|---|---|
| `passthrough_env` | list[str] | `[]` | Extra environment variable names forwarded from the orchestrator's environment into the eval process when set (on top of the built-in list: RUN_EVAL, EVAL_ONLY, MODEL, ISL, OSL, ...). |
| `command` | list[str] \| None | `None` | Argv that replaces the built-in lm-eval runner command. May use the placeholders ``{endpoint}`` (the frontend URL) and ``{infmax_workspace}`` (the InferenceMAX workspace mount). Not shell-interpreted; wrap in ``bash -lc`` yourself if you need a shell. |

### IdentityConfig

Virtual identity for runtime verification and reproduction.

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | [IdentityModelConfig](#identitymodelconfig) | `IdentityModelConfig()` | Expected HuggingFace repo and revision, checked against the download metadata at runtime. |
| `container` | [IdentityContainerConfig](#identitycontainerconfig) | `IdentityContainerConfig()` | Container image URI, recorded for reproduction only. |
| `frameworks` | dict[str, str] | `{}` | Package -> expected version for dynamo and one engine, checked via importlib.metadata at runtime. |

### ReportingConfig

Reporting configuration for status updates, AI analysis, and log exports.

| Key | Type | Default | Description |
|---|---|---|---|
| `status` | [ReportingStatusConfig](#reportingstatusconfig) \| None | `None` | Status collector endpoints that receive job lifecycle events. Unset sends nothing. |
| `ai_analysis` | [AIAnalysisConfig](#aianalysisconfig) \| None | `None` | Failure analysis run after a failed job. Unset disables it. |
| `s3` | [S3Config](#s3config) \| None | `None` | Upload of the log directory to S3-compatible storage after the run. Unset disables it. |

### RestartPolicy

How the worker supervisor treats a worker of one role that exits mid-run.

| Key | Type | Default | Description |
|---|---|---|---|
| `policy` | one of `'never'`, `'on-failure'`, `'always'` | `'never'` | ``never`` leaves a worker exit to ``critical`` (the default, today's behavior). ``on-failure`` relaunches after a non-zero exit; ``always`` relaunches after any exit, including a clean one. |
| `max_restarts` | int | `3` | Relaunches allowed per endpoint over the whole job. |
| `backoff_seconds` | float | `10.0` | Delay before the first relaunch. Doubles on every further relaunch of the same endpoint (10 s, 20 s, 40 s, ...). |
| `max_backoff_seconds` | float | `300.0` | Cap on the doubled delay. |

### PlacementConfig

Where a component (the frontend or the benchmark client) runs.

| Key | Type | Default | Description |
|---|---|---|---|
| `node` | str | `'head'` | A location name resolved against the worker topology (``head``, or a role-relative name such as ``first_decode`` / ``last_decode``), or ``dedicated`` to reserve a node for the component. A dedicated node is always the head location, so the two never combine. |

### DynamoSourceConfig

Where Dynamo comes from. Exactly one of ``git``, ``pypi``, or ``wheel``.

| Key | Type | Default | Description |
|---|---|---|---|
| `git` | str \| None | `None` | Repository URL to build from (default upstream when ``rev`` is set without it). Builds ``ai-dynamo-runtime`` with maturin and installs ``ai-dynamo`` from the checkout; cached on ``/configs`` by commit. |
| `rev` | str \| None | `None` | Immutable ref in ``git``: commit SHA, tag, or ``refs/pull/<n>/head``. |
| `sha` | str \| None | `None` | The commit ``rev`` resolved to; filled in by ``srtctl apply``. |
| `patches` | list[str] \| None | `None` | Cargo dependency replacements applied tree-wide before the build. |
| `pypi` | str \| None | `None` | Release version from PyPI. |
| `wheel` | str \| None | `None` | Staged nightly ``ai-dynamo`` version. |

### SweepConfig

Configuration for benchmark parameter sweeps.

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | one of `'zip'`, `'grid'` | `'zip'` | `zip` pairs the i-th value of every list; `grid` takes the Cartesian product. |
| `parameters` | dict[str, list[Any]] | `{}` | Parameter name -> list of values to sweep over. |

### ProfilingPhaseConfig

Profiling config for a single phase (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `start_step` | int \| None | `None` | Step to start profiling |
| `stop_step` | int \| None | `None` | Step to stop profiling |
| `capture_scope` | one of `'selected'`, `'all'` | `'all'` | `all` profiles every process of the phase; `selected` only `worker_index` / `worker_rank`. |
| `worker_index` | int | `0` | Logical worker within the phase |
| `worker_rank` | int | `0` | Physical process rank within that worker |

### TachometerConfig

Native Tachometer collection for an observability-enabled run.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool \| None | `None` | None (default) collects on every run; false opts out. |
| `binary_path` | str | `'tachometer-scraper'` | Scraper command or path on the compute nodes. |
| `collect_interval_ms` | int | `1000` | Milliseconds between scrapes of every endpoint — the same unit and name as dcgm-exporter's --collect-interval. Replaces the retired Hz-based ``default_frequency`` (1000ms == the old 1.0 Hz default). |
| `sync_interval_secs` | int | `120` | Seconds between intermediate Parquet compactions; 0 disables them. |
| `shutdown_grace_secs` | float | `120.0` | How long the scraper gets after SIGTERM to flush + compact final.parquet before the SIGKILL escalation. Compaction time scales with the arrow WAL accumulated since the last periodic sync. |
| `compaction_threads` | int | `4` | Threads for Parquet compaction (POLARS_MAX_THREADS). |
| `storage_subdir` | str | `'tachometer'` | Output directory below the run's log directory. |
| `extra_metadata` | dict[str, str] | `{}` | Static key/value metadata attached to every scraped endpoint. |
| `default_exporters` | bool | `True` | Run the built-in DCGM, node, and process exporters when their blocks are unset; false disables them. |
| `default_gpu_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `<lambda>()` | Resolved from srtslurm.yaml at load time; never read global config here. |
| `dcgm_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `None` | GPU exporter; unset uses the cluster default (DCGM exporter on port 9401). |
| `node_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `None` | Host metrics exporter; unset uses node-exporter on port 9101. |
| `process_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `None` | Per-process /proc exporter; unset uses the host-native configs/process-exporter on port 9256. |

### NsysObservabilityConfig

Automatic NVTX tracing and CPU sampling of workers and Dynamo frontends.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `True` | Set false to keep other observability signals without launching nsys. |
| `capture_window` | one of `'measured_workload'`, `'including_startup'` | `'measured_workload'` | measured_workload excludes warmup; including_startup spans process launch through teardown. |
| `report_timeout_secs` | int | `1800` | Maximum wait for a control acknowledgment or a step's report finalization. |
| `nvtx_injection_path` | str \| None | `None` | Optional container path to libToolsInjection64.so for NVTX injection. |
| `cpu_sampling` | one of `'system-wide'`, `'process-tree'`, `'none'` | `'system-wide'` | CPU IP sampling and context-switch scope. process-tree fails on engines with many threads ("Not enough resources ... switch to system-wide"); system-wide samples every process on the node, none records NVTX only. |

### TelemetryExporterConfig

Configuration for a metrics exporter deployed on worker nodes.

| Key | Type | Default | Description |
|---|---|---|---|
| `container_image` | str | required | Exporter image (registry URI or `containers` alias); ignored, and may be `""`, when `binary` is set. |
| `port` | int | required | Port the exporter serves `/metrics` on, on every worker node. |
| `command` | str \| None | `None` | Command line replacing the image's default entrypoint arguments. |
| `binary` | str \| None | `None` | Host executable to run without a container; relative paths resolve against the srtctl checkout. |
| `kind` | one of `'dcgm'`, `'custom'` | `'dcgm'` | GPU exporter kind: `dcgm` (built-in DCGM command, labels, metrics and tachometer scrape) or `custom` (any other exporter; set `command`, `gpu_labels` and `gpu_metrics`, and tachometer keeps its rows as served). |
| `gpu_labels` | [GpuLabelsConfig](#gpulabelsconfig) \| None | `None` | GPU power telemetry: labels identifying a GPU in the scrape; unset means DCGM (`gpu`, `UUID`). |
| `gpu_metrics` | [GpuMetricsConfig](#gpumetricsconfig) \| None | `None` | GPU power telemetry: per-GPU metrics to record; unset means DCGM. |

### CpuPowerExporterConfig

Best-effort CPU power collection via the cpu-power-exporter binary.

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int | `9405` | Port the exporter listens on and the head-node collector scrapes. |
| `source` | one of `'auto'`, `'acpi'`, `'dcgm'` | `'auto'` | Power reading back-end passed through to the bundled Rust binary's own ``--source`` flag (``auto`` \| ``acpi`` \| ``dcgm``). ``auto`` tries DCGM first and falls back to ACPI when libdcgm.so is absent or reports no CPU entities. Has no effect when the Python stdlib fallback exporter is used instead of the binary -- that fallback is ACPI-only. |

### CpuPowerConfig

Host-side CPU power collection on every worker node.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `False` | Master switch for this leg. Default: False. |
| `source` | one of `'auto'`, `'acpi'`, `'dcgm'` | `'auto'` | ``auto`` tries ACPI then DCGM and is best-effort; naming ``acpi`` or ``dcgm`` explicitly makes that provider mandatory. |
| `sample_interval_seconds` | float | `0.1` | Read period on each node, in seconds. |
| `startup_timeout_seconds` | float | `30.0` | How long to wait for every node's collector to publish its ready marker before giving up on readiness. |
| `required` | bool | `False` | Fail the job when the leg does not become ready or does not produce a valid publication. |
| `storage_subdir` | str | `'cpu_power'` | Directory below the run log directory that holds the CPU samples and manifest. Must differ from ``telemetry.storage_subdir``. |

### SourceConfig

A git repository at an immutable ref.

| Key | Type | Default | Description |
|---|---|---|---|
| `git` | str | required | Repository URL to clone. |
| `rev` | str | required | Immutable ref: a commit SHA, a tag, or ``refs/pull/<n>/head`` for an unmerged PR. Branch names are rejected because they move. |
| `path` | str \| None | `None` | Optional subdirectory of the clone to build and run from. |
| `sha` | str \| None | `None` | The commit ``rev`` resolved to. Filled in by ``srtctl apply`` at submit time; write it yourself only to pin an exact commit while keeping the human-readable ``rev`` beside it. |

### ServicePlacementConfig

Where a service runs.

| Key | Type | Default | Description |
|---|---|---|---|
| `node` | str | `'head'` | ``head`` or ``infra`` (one instance), ``dedicated`` (reserve the infra node exclusively; infra-class kinds only), ``prefill`` / ``decode`` / ``agg`` (one instance per distinct physical node that role's workers use), ``workers`` (one instance per engine worker node; on a service that owns nodes, its own pool), ``compute`` (engine worker nodes plus every pool), or ``all`` (every node of the allocation). |
| `pool` | str \| None | `None` | Run on the nodes another service owns (``services[].nodes``), one instance per node of that pool. Replaces ``node``. |
| `per` | str | `'node'` | ``node`` (default): one instance per placed node. ``worker``: one instance per engine worker on each placed node, attached to that worker: it runs with the worker's ``CUDA_VISIBLE_DEVICES`` and sees ``{worker_role}``, ``{worker_index}``, ``{worker_node_rank}``, ``{worker_gpus}``, ``{worker_gpu_count}``. A sidecar in the Kubernetes sense (the GPU Memory Service next to each vLLM worker). Only with ``node`` in ``prefill``, ``decode``, ``agg``, ``workers``. |

### ServiceReadinessConfig

Readiness gate: the launch blocks until the probe passes on every service node.

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int \| None | `None` | Shorthand for ``tcp: {port: <port>}``. |
| `tcp` | [TcpProbe](#tcpprobe) \| None | `None` | TCP connect probe. |
| `http` | [HttpProbe](#httpprobe) \| None | `None` | HTTP GET probe. |
| `log` | [LogProbe](#logprobe) \| None | `None` | Log-pattern probe against ``service_<name>.out``. |
| `timeout_seconds` | int | `120` | How long to wait per node before failing the job. |
| `interval_seconds` | int | `2` | Seconds between probe attempts. |

### ServiceMetricsConfig

One Prometheus endpoint a service serves: ``port``, ``path`` (default ``/metrics``), ``nodes``, ``name``.

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int | required | Port the endpoint is served on. |
| `path` | str | `'/metrics'` | URL path of the endpoint. |
| `nodes` | str | `'all'` | `all` scrapes every node the service runs on; `first` only its first node. |
| `name` | str \| None | `None` | Endpoint name in the Tachometer parquet (`<name>_<node>`); defaults to the service name. |

### IdentityModelConfig

Virtual model identity for runtime verification.

| Key | Type | Default | Description |
|---|---|---|---|
| `repo` | str \| None | `None` | HuggingFace model ID, e.g. "nvidia/Kimi-K2.5-NVFP4" |
| `revision` | str \| None | `None` | HuggingFace git commit SHA |

### IdentityContainerConfig

Container identity for reproduction (not verified at runtime).

| Key | Type | Default | Description |
|---|---|---|---|
| `image` | str \| None | `None` | Docker URI, e.g. "gitlab-master:5005/.../trtllm-arm64" |

### ReportingStatusConfig

Status reporting configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `endpoint` | str \| None | `None` | Base URL of one status collector; srtctl POSTs job lifecycle events there (see status-api-spec.md). |
| `endpoints` | list[str] \| None | `None` | Several collectors, each sent every event; merged with `endpoint`, deduplicated, trailing slash dropped. |
| `token_env` | str \| None | `None` | Name of the environment variable holding the bearer token the reporter sends as ``Authorization: Bearer`` on every request (default SRTCTL_STATUS_TOKEN). Only the variable name belongs in a recipe: the resolved config is written to the lockfile and the log directory, so a literal token there would leak. |
| `logging-stream-interval` | float \| None | `None` | Seconds between uploads of raw logs and Tachometer captures to every endpoint. Unset disables streaming; lifecycle events are unaffected. |

### AIAnalysisConfig

AI-powered failure analysis configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `False` | Whether to run AI analysis on benchmark failures |
| `openrouter_api_key` | str \| None | `None` | OpenRouter API key (falls back to OPENROUTER_API_KEY env var) |
| `gh_token` | str \| None | `None` | GitHub token for gh CLI (falls back to GH_TOKEN env var) |
| `repos_to_search` | list[str] | `<lambda>()` | GitHub repos to search for related PRs |
| `pr_search_days` | int | `14` | Number of days to look back for PRs |
| `prompt` | str \| None | `None` | Custom prompt template (uses DEFAULT_AI_ANALYSIS_PROMPT if None) Available variables: {log_dir}, {repos}, {pr_days} |

### S3Config

S3 upload configuration for log artifacts.

| Key | Type | Default | Description |
|---|---|---|---|
| `bucket` | str | required | S3 bucket name |
| `prefix` | str \| None | `None` | Optional prefix/path within bucket (e.g., "srtslurm/logs") |
| `region` | str \| None | `None` | AWS region (e.g., "us-west-2") |
| `endpoint_url` | str \| None | `None` | Custom S3-compatible endpoint URL (optional) |
| `access_key_id` | str \| None | `None` | AWS access key ID (falls back to AWS_ACCESS_KEY_ID env var) |
| `secret_access_key` | str \| None | `None` | AWS secret access key (falls back to AWS_SECRET_ACCESS_KEY env var) |
| `exclude` | list[str] \| None | `None` | Patterns `aws s3 sync` skips, relative to the log directory (`*` matches across directories). Omit for the defaults: aiperf's per-interval metrics scrapes and `inputs.json` under `artifacts/*/` and `sa-bench_*/*/` (tachometer already stores that series as parquet). Set to `[]` to ship the whole directory. |
| `archive` | list[str] \| None | `None` | Patterns (Python glob, `**` allowed) packed into one `bundle.tar.zst` uploaded next to the loose files and left out of the plain sync. Omit for the default, aiperf's per-request `profile_export.jsonl`; set to `[]` for no archive. |

### GpuLabelsConfig

The labels that identify a GPU in every sample of a GPU exporter's scrape.

| Key | Type | Default | Description |
|---|---|---|---|
| `index` | str | required | Label carrying the node-local GPU index srt-slurm allocates by. |
| `identity` | str | required | Label that is stable for one physical GPU across the run; recorded as `gpu_uuid`. |
| `instance` | list[str] | `[]` | Labels marking logical sub-device samples (MIG instances, partitions); such samples are dropped. |

### GpuMetricsConfig

The per-GPU metrics GPU power telemetry records from a GPU exporter.

| Key | Type | Default | Description |
|---|---|---|---|
| `power` | [GpuPowerMetricConfig](#gpupowermetricconfig) | required | Power draw in watts. |
| `gpu_util` | [GpuMetricConfig](#gpumetricconfig) \| None | `None` | GPU utilization, percent. |
| `sm_active` | [GpuMetricConfig](#gpumetricconfig) \| None | `None` | DCGM's SM-active fraction, 0-1; map only a metric with that meaning and range. |
| `temperature` | [GpuMetricConfig](#gpumetricconfig) \| None | `None` | GPU temperature, Celsius. |

### TcpProbe

Ready when ``port`` accepts a TCP connection on the service node.

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int | required | Port probed on the service node. |

### HttpProbe

Ready when ``GET http://<node>:<port><path>`` returns ``status``.

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int | required | Port probed on the service node. |
| `path` | str | `'/health'` | URL path requested. |
| `status` | int | `200` | HTTP status that counts as ready. |

### LogProbe

Ready when the service's log file contains a line matching the regular expression ``pattern``.

| Key | Type | Default | Description |
|---|---|---|---|
| `pattern` | str | required | Regular expression searched for in the service log. |

### GpuPowerMetricConfig

The per-GPU power metric in a GPU exporter's scrape, in watts.

| Key | Type | Default | Description |
|---|---|---|---|
| `metric` | str | required | Prometheus metric name. |
| `scope` | str | required | What the watts measure, recorded in the power manifest as `power_scope`. |

### GpuMetricConfig

One per-GPU metric in a GPU exporter's scrape.

| Key | Type | Default | Description |
|---|---|---|---|
| `metric` | str | required | Prometheus metric name. |

## Engine types

`engine.type` selects one of the following; the remaining `engine` keys are that type's knobs.

### AtomBackend

`engine.type: atom`

Launch ``atom.entrypoints.openai_server`` on ROCm workers.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'atom'` | `'atom'` | Engine type discriminator. |
| `connector` | one of `'mooncake'` | `'mooncake'` | KV-transfer connector between prefill and decode workers. |
| `mooncake_protocol` | one of `'rdma'`, `'tcp'` \| None | `None` | Mooncake transport; unset lets ATOM choose. |

### SGLangBackend

`engine.type: sglang`

SGLang backend configuration and launch implementation.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'sglang'` | `'sglang'` | Engine type discriminator. |
| `gpu_type` | str \| None | `None` | Accepted for compatibility; srtctl does not read it. Set `resources.gpu_type` instead. |

### TileRTBackend

`engine.type: tilert`

Launch TileRT's decode server with recipe-owned model and transport settings.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'tilert'` | `'tilert'` | Engine type discriminator. |
| `served_model_name` | str \| None | `None` | Model name the server reports to clients; unset uses the default served name. |

### TokenSpeedBackend

`engine.type: tokenspeed`

Launch TokenSpeed workers behind the Dynamo frontend or a static router such as SMG.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'tokenspeed'` | `'tokenspeed'` | Engine type discriminator. |

### TRTLLMBackend

`engine.type: trtllm`

TRTLLM backend configuration and launch implementation.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'trtllm'` | `'trtllm'` | Engine type discriminator. |
| `served_model_name` | str \| None | `None` | The name clients must use in a request's "model" field. Defaults to the checkpoint directory name. engine: type: trtllm served_model_name: "deepseek-ai/deepseek-r1" Set it when the client cannot be told which name to ask for. agentperf takes the name as a flag, so it never needs this; the MLPerf harness has it fixed in the benchmark definition, so the server must match or every request 404s. Top-level rather than a roles.<role>.args key because a role's args are dumped straight into the engine's YAML file, and this is a launcher flag the engine does not recognise. |
| `publish_metrics` | bool | `True` | Publish TRT-LLM engine metrics without enabling KV-cache events. Requires a Dynamo build supporting --publish-metrics; set False to omit the flag for older builds. Native trtllm-serve and sidecars are unaffected. Iteration statistics stay off regardless: srtctl bakes enable_iter_perf_stats: false into every engine section unless the recipe or observability sets it (TRTLLM_ENGINE_DEFAULTS), so this flag costs the per-request perf metrics only. |
| `publish_events_and_metrics` | bool \| None | `None` | Legacy compatibility flag for Dynamo builds without --publish-metrics. True emits only --publish-events-and-metrics, regardless of publish_metrics. False or None uses publish_metrics instead. Observability does not enable it. |
| `sequential_node_start` | int | `0` | Controls batched startup of workers that share the same node. 0 = start all workers in parallel (no constraint). 1 = fully sequential: one worker at a time, each must be ready before the next. N > 1 = start N workers simultaneously per batch, wait for all to be ready, then next batch. For trtllm_serve: readiness is an HTTP 200 on the worker's http_port. For dynamo.trtllm: readiness is a TCP connection on the worker's sys_port. |
| `numa_memory_bind` | bool \| one of `'local'` \| None | `None` | Worker memory policy. None (default) uses `numactl -m 0,1` only for gb200/gb300/vrnvl72 prefill and decode workers (case-sensitive GPU type). True uses nodes 0,1 for any GPU type or mode; False leaves the policy unchanged. CPU binding does not change these policies. "local" requires numa_cpu_bind=True and strictly binds memory to the task GPU's NUMA node. Local mode fails startup if GPU NUMA affinity is unknown. Local memory exhaustion can fail allocations; existing/shared pages are not migrated. |
| `numa_cpu_bind` | bool | `False` | Optional stricter NUMA CPU affinity for the worker process, in addition to numa_memory_bind. A previous post-hoc `taskset -pc <cpuset> $PPID` approach (see bind-b300-prefill-cpus.sh) only pins the leader PID *after* launch, so secondary threads spawned by Python/UCX/MPI/TRT-LLM can still land cross-socket. When true, srtctl instead: 1. sets TLLM_NUMA_AWARE_WORKER_AFFINITY=0 (disables TRT-LLM's own internal NUMA thread-pinning, which fights with the OS-level mask) 2. wraps the worker command (prefill/decode/agg) in `taskset -c <cpu_list>`, applied *before* exec so every spawned thread inherits the mask. The CPU list is discovered at runtime (configs/numa_cpu_bind.sh) from the physical GPU this task owns, not a static SLURM_LOCALID table — a static table assumes SLURM_LOCALID is a node-wide GPU ordinal, which breaks when two endpoints share a node (each gets its own srun step, so LOCALID restarts at 0 for both). Set numa_memory_bind="local" to also bind memory to that same NUMA node. |

### VLLMBackend

`engine.type: vllm`

vLLM backend configuration and launch implementation.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'vllm'` | `'vllm'` | Engine type discriminator. |
| `set_visible_devices` | bool | `False` | Use an environment mask instead of the engine's --device-ids option. |
| `connector` | str \| None | `'nixl'` | Default KV connector: "nixl", "lmcache", "lmcache-mp", "kvbm", "moriio", or a raw JSON string for --kv-transfer-config. Can be overridden per role by setting "connector" in roles.<role>.args; connector_for_mode resolves it. "moriio" (ROCm MoRI-IO) registers workers with the vLLM Router and needs frontend.type: vllm-router. dynamo 1.0.0+: translated to --kv-transfer-config (--connector was removed). |
| `failover` | [VLLMFailoverConfig](#vllmfailoverconfig) \| None | `None` | Shadow engine recovery: when set, every worker runs shadow_engines standby engines on its GPUs next to an implied `gms` service that owns the weights. Dynamo frontend only. |
| `allow_prefill_decode_colocation` | bool | `False` | Allow prefill and decode workers to share one node when the combined GPU request fits within gpus_per_node. Defaults off to preserve existing P/D node separation. |
| `allow_prefill_decode_colocation_across_nodes` | bool | `False` | Extend P/D colocation to multi-node topologies. When enabled together with allow_prefill_decode_colocation, workers are packed contiguously across the minimum number of nodes instead of reserving separate P/D node pools. Defaults off to preserve the original one-node-only policy. |
| `dp_launch_mode` | one of `'per_gpu'`, `'per_node'` | `'per_node'` | DP process layout. Per-node lets vLLM manage the node-local portion of a DP x TP x PP topology in one CUDA namespace and derives cross-node TP/PP rendezvous when a replica is larger than the node-local GPU allocation. Per-GPU remains available as a deprecated compatibility layout. |
| `vllm_serve_binary` | str | `'vllm'` | Executable used by direct aggregate frontend.type=vllm jobs. This can be set to vllm-rs (or its absolute path) to use the Rust OpenAI frontend. |

### MockerBackend

`engine.type: mocker`

Dynamo Mocker backend configuration and launch implementation.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'mocker'` | `'mocker'` | Engine type discriminator. |
| `engine_type` | str | `'vllm'` | Engine whose scheduler and KV-cache behavior the mocker simulates (`vllm`, `sglang`, ...). |
| `speedup_ratio` | float | `100.0` | How much faster than real time the simulated engine runs |
| `decode_speedup_ratio` | float | `1.0` | Extra speedup applied to decode steps only |
| `num_gpu_blocks_override` | int | `16384` | KV-cache blocks the simulated engine has |
| `max_num_seqs` | int | `256` | Maximum sequences scheduled per step |
| `max_num_batched_tokens` | int | `8192` | Maximum tokens scheduled per step |
| `block_size` | int \| None | `None` | KV-cache block size in tokens; unset uses the mocker default |
| `data_parallel_size` | int | `1` | Simulated data-parallel ranks per worker |
| `num_workers` | int | `1` | Mocker engines per worker process |
| `startup_time` | float \| None | `None` | Simulated model-load delay in seconds |
| `kv_transfer_bandwidth` | float \| None | `None` | Simulated prefill->decode KV transfer bandwidth |
| `kv_cache_dtype` | str \| None | `None` | KV-cache dtype the simulation sizes blocks for |
| `enable_prefix_caching` | bool | `True` | Simulate prefix caching; false passes --no-enable-prefix-caching |
| `enable_chunked_prefill` | bool | `True` | Simulate chunked prefill; false passes --no-enable-chunked-prefill |
| `preemption_mode` | str \| None | `None` | Scheduler preemption policy; unset uses the mocker default |

### VLLMFailoverConfig

Shadow engine recovery for vLLM workers (Dynamo GPU Memory Service).

| Key | Type | Default | Description |
|---|---|---|---|
| `shadow_engines` | int | `1` | Standby engines per worker. |
| `shared_dir` | str | `'/dev/shm'` | Node-local host directory that every container on a node sees. The GMS sockets and the lock file of a worker live under ``<shared_dir>/srtctl-<job_id>/<role>_<index>``. enroot bind-mounts the host's ``/dev/shm`` into every container; ``/tmp`` is a fresh tmpfs per container and does not work. |

## Cluster config

Top-level keys of `srtslurm.yaml`. Recipes inherit these defaults and resolve aliases through them.

| Key | Type | Default | Description |
|---|---|---|---|
| `cluster` | str \| None | `None` | Cluster name for status reporting |
| `default_account` | str \| None | `None` | Slurm account for recipes that omit `slurm.account`. |
| `default_partition` | str \| None | `None` | Slurm partition for recipes that omit `slurm.partition`. |
| `default_time_limit` | str \| None | `None` | Job time limit (HH:MM:SS) for recipes that omit `slurm.time_limit`. |
| `gpus_per_node` | int \| None | `None` | GPUs per node for recipes that omit `resources.gpus_per_node`. |
| `default_gpu_type` | str \| None | `None` | Default for ``ResourceConfig.gpu_type`` when the recipe omits it. Lets one recipe move between clusters of different GPU types without an edit. |
| `network_interface` | str \| None | `None` | Interface whose IP address frontends use to reach workers (e.g. `ib0`). Unset resolves the hostname. |
| `visible_devices_env` | str | `'CUDA_VISIBLE_DEVICES'` | GPU-subset mask passed to workers; ROCm clusters use ROCR_VISIBLE_DEVICES. |
| `default_gpu_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `<lambda>()` | Recipe exporter settings win. Explicit null disables the GPU default only. |
| `use_gpus_per_node_directive` | bool | `True` | Emit `#SBATCH --gpus-per-node`. Set false on clusters that reject or ignore it. |
| `use_segment_sbatch_directive` | bool | `True` | Emit `#SBATCH --segment` so the allocation stays inside one topology segment (NVL72 domain). |
| `use_exclusive_sbatch_directive` | bool | `False` | Emit `#SBATCH --exclusive` to keep other jobs off the allocated nodes. |
| `use_het_jobs` | bool | `False` | Default for ``ResourceConfig.het_jobs`` when the recipe doesn't set it. When True (and recipe doesn't override), the prefill side and decode side are submitted as two SLURM heterogeneous-job components, each with its own ``--segment``. Lets asymmetric layouts (e.g. prefill 12 + decode 10 nodes on GB200/GB300) preserve NVL72 affinity per side. |
| `default_sbatch_directives` | dict[str, str] \| None | `None` | Extra `#SBATCH --key=value` lines added to every job; a recipe's `sbatch_directives` wins per key. |
| `default_health_check` | dict[str, int] \| None | `None` | `health_check` block (`max_attempts`, `interval_seconds`) used when a recipe has none. |
| `srtctl_root` | str \| None | `None` | srtctl checkout on the shared filesystem that compute nodes mount at /srtctl-src. Default: this checkout. |
| `output_dir` | str \| None | `None` | Custom output directory for job logs |
| `record_launch_plan` | bool | `False` | Cluster-wide default for recording exact realized srun commands. Recipes can opt in independently with output.record_launch_plan. |
| `model_paths` | dict[str, str] \| None | `None` | Alias -> path map; a recipe's `model.path` may name an alias instead of a path. |
| `containers` | dict[str, str] \| None | `None` | Alias -> image map, resolved for every container key in a recipe (`model.container`, `roles.<role>.container`, ...). |
| `cloud` | dict[str, str] \| None | `None` | Free-form cloud settings. Accepted for compatibility; srtctl does not read it. |
| `default_mounts` | dict[str, str] \| None | `None` | Cluster-level container mounts (host_path -> container_path) Applied to all jobs on this cluster, useful for cluster-specific paths |
| `default_bash_preamble` | str \| None | `None` | Shell snippet prepended to every container srun (after env exports, before the main command). Useful for cluster-wide ulimits, e.g. ``"ulimit -n 1048576 -s unlimited -u 1048576"``. Silently dropped for sruns that bypass the bash wrapper (distroless containers). |
| `default_host_setup` | [HostSetupConfig](#hostsetupconfig) \| None | `None` | Commands run on every allocated node's bare host, outside the container, before workers start. Recipes override with their own `host_setup:` block. |
| `reporting` | [ReportingConfig](#reportingconfig) \| None | `None` | Status collectors, S3 log upload, and failure analysis for every job on this cluster. |
| `telemetry` | dict \| None | `None` | opaque dict, parsed by try_start_snapshotter |
| `nginx_raise_ulimit` | bool \| None | `None` | When set, applied to job configs that omit ``frontend.nginx_raise_ulimit``. Clusters that disallow raising nofile for nginx containers should use false. |
| `git_http_version` | str \| None | `None` | Works around intermittent git smart-HTTP/HTTP2 failures cloning github.com (stalls, or truncated responses git misreports as "could not read Username" auth-prompt failures). See git_clone_command_prefix() in core/config.py -- applied to every git clone/fetch srtctl performs. |
| `preflight` | bool | `True` | Run the pre-submit model.path / model.container / telemetry filesystem checks on ``srtctl apply``. Set false on clusters whose model or image paths exist only on compute nodes (node-local NVMe such as /raid), where the login node cannot stat them; every apply then behaves as if --no-preflight had been passed. The framework still fails loudly at runtime if a path is genuinely missing on the compute node. |
