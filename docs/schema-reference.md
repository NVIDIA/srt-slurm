# Schema Reference

<!-- GENERATED FILE. Do not edit by hand. Regenerate with `srtctl schema-docs`; CI fails when this file is stale. -->

Field-level reference for recipe YAML (`SrtConfig`) and the cluster config `srtslurm.yaml` (`ClusterConfig`), generated from the dataclasses in `srtctl.core.schema` and `srtctl.backends`. Each table lists the YAML key, the type, the default (`required` when there is none), and a description taken from the class docstring or the comment on the field. Nested types link to their own table. For prose, examples, and semantics see [config-reference.md](config-reference.md).

## Recipe

Top-level keys of a recipe YAML.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | required |  |
| `model` | [ModelConfig](#modelconfig) | required |  |
| `resources` | [ResourceConfig](#resourceconfig) | required |  |
| `schema` | int | `1` | Recipe schema version (YAML key `schema`). Absent means 1, the pre-2.0 layout. `schema: 2` selects the 2.0 layout; `srtctl migrate` upgrades a recipe in place. Both versions load on main. |
| `slurm` | [SlurmConfig](#slurmconfig) | `SlurmConfig()` |  |
| `backend` | [SGLangProtocol](#sglangprotocol) \| [TRTLLMProtocol](#trtllmprotocol) \| [VLLMProtocol](#vllmprotocol) \| [MockerProtocol](#mockerprotocol) | `SGLangProtocol()` |  |
| `frontend` | [FrontendConfig](#frontendconfig) | `FrontendConfig()` |  |
| `dynamo` | [DynamoConfig](#dynamoconfig) | `DynamoConfig()` |  |
| `benchmark` | [BenchmarkConfig](#benchmarkconfig) | `BenchmarkConfig()` |  |
| `profiling` | [ProfilingConfig](#profilingconfig) | `ProfilingConfig()` |  |
| `output` | [OutputConfig](#outputconfig) | `OutputConfig()` |  |
| `health_check` | [HealthCheckConfig](#healthcheckconfig) | `HealthCheckConfig()` |  |
| `infra` | [InfraConfig](#infraconfig) | `InfraConfig()` |  |
| `observability` | [ObservabilityConfig](#observabilityconfig) | `ObservabilityConfig()` |  |
| `telemetry` | [TelemetryConfig](#telemetryconfig) | `TelemetryConfig()` |  |
| `environment` | dict[str, str] | `{}` |  |
| `container_mounts` | dict[[FormattablePath](#formattablepath), [FormattablePath](#formattablepath)] | `{}` |  |
| `extra_mount` | tuple[str, ...] \| None | `None` |  |
| `srun_options` | dict[str, str] | `{}` |  |
| `sbatch_directives` | dict[str, str] | `{}` |  |
| `enable_config_dump` | bool | `True` |  |
| `setup_script` | str \| None | `None` | Custom setup script (runs before dynamo install and worker startup) e.g. "custom-setup.sh" -> runs /configs/custom-setup.sh |
| `host_setup` | [HostSetupConfig](#hostsetupconfig) | `HostSetupConfig()` | Commands run on each node's bare host, outside the container, before any worker starts. Cluster-wide default lives in srtslurm.yaml as default_host_setup; a recipe that sets this block replaces that default. |
| `services` | list[[ServiceConfig](#serviceconfig)] | `[]` | Long-running processes launched next to the job: generic sidecars (an experimental router built from a PR) and typed ones (a standalone Mooncake store per worker node). See docs/services.md. |
| `post_eval` | [PostEvalConfig](#postevalconfig) | `PostEvalConfig()` | Post-benchmark / eval-only evaluation dispatch: extra env forwarded into the eval process and an optional command override. Replaces the downstream source patch that used to extend the passthrough list in do_sweep.py. |
| `identity` | [IdentityConfig](#identityconfig) | `IdentityConfig()` | Virtual identity — declares what *should* be running (verified against fingerprint) |
| `reporting` | [ReportingConfig](#reportingconfig) \| None | `None` | Reporting configuration (status API, future: logs to S3, etc.) |

## Recipe sections

### ModelConfig

Model configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `path` | str | required |  |
| `container` | str | required |  |
| `precision` | str | required |  |
| `stage_dir` | str \| None | `None` | Optional: stage the model from shared storage to this node-local dir before workers start (e.g. "/raid/scratch/models"). None = use path directly. |

### ResourceConfig

Resource allocation configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `gpu_type` | str \| None | `None` | GPU type (h100, gb200, ...). Cluster fact, not a topology choice. Optional: a recipe that omits it inherits `default_gpu_type` from srtslurm.yaml, and `gpus_per_node` inherits the cluster `gpus_per_node`. Both are still worth setting in a recipe so it is self-describing for result rollups. |
| `gpus_per_node` | int | `4` |  |
| `prefill_nodes` | int \| None | `None` | Disaggregated mode |
| `decode_nodes` | int \| None | `None` |  |
| `prefill_workers` | int \| None | `None` |  |
| `decode_workers` | int \| None | `None` |  |
| `agg_nodes` | int \| None | `None` | Aggregated mode |
| `agg_workers` | int \| None | `None` |  |
| `spread_workers` | bool | `False` | If True, place each partial-node worker on its own node instead of packing multiple onto the same node. Caller must reserve enough nodes (e.g. set decode_nodes=decode_workers when gpus_per_decode<gpus_per_node). |
| `het_jobs` | bool \| None | `None` | SLURM heterogeneous-job opt-in. Tri-state: None defers to the cluster default `use_het_jobs` on ClusterConfig; True/False overrides per recipe. When effectively True (and we are in disaggregated mode), the prefill and decode sides are submitted as two het components each with their own `--segment`. See HetComponent above and docs/slurm-faq.md. |
| `gpus_per_prefill` | int \| None | `None` | Explicit GPUs per worker (override computed values) Use data_key to map from YAML field names to internal attribute names |
| `gpus_per_decode` | int \| None | `None` |  |
| `gpus_per_agg` | int \| None | `None` |  |

### SlurmConfig

SLURM job settings.

| Key | Type | Default | Description |
|---|---|---|---|
| `account` | str \| None | `None` |  |
| `partition` | str \| None | `None` |  |
| `time_limit` | str \| None | `None` |  |

### FrontendConfig

Frontend/router configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | str | `'dynamo'` | Frontend type - "dynamo" (default), "sglang", "vllm-router", "trtllm_serve", or direct "vllm" |
| `enable_multiple_frontends` | bool | `True` | Scale with nginx + multiple routers. When ``True`` (default), srtctl stands up nginx and fans out to ``num_additional_frontends + 1`` router replicas. When ``False``, there is NO nginx proxy — the benchmark must target the single master router (or a worker) directly at ``http://localhost:<port>``. ``benchmark.command`` has no placeholder substitution, so write the URL out literally. |
| `num_additional_frontends` | int | `9` | Additional routers beyond master (default: 9) |
| `nginx_container` | str | `'nginx:1.27.4'` | Custom nginx container image (default: nginx:1.27.4) |
| `nginx_raise_ulimit` | bool | `False` | Raise nofile before nginx and set ``worker_rlimit_nofile`` in generated nginx.conf. Off by default; enable on clusters that allow it. Override per job or set ``nginx_raise_ulimit`` in srtslurm.yaml for the cluster. |
| `nginx_session_affinity` | bool | `False` | Consistently hash ``nginx_session_affinity_header`` to a frontend. Requests without that header use a generated request ID and stay distributed. |
| `nginx_session_affinity_header` | str | `'X-Dynamo-Session-ID'` | Header hashed when affinity is on (default ``X-Dynamo-Session-ID``). Set ``X-Correlation-ID`` for clients (e.g. aiperf) that carry the session id in that header instead. |
| `nginx_keepalive_timeout` | str | `'600s'` | Idle timeout for client and upstream keepalive connections in the generated nginx.conf (default "600s"). nginx's own default is 75s, which closes a session's connection during the long recorded think-time of an agentic replay; the client's next write on that pooled socket then fails with "broken pipe" / "server disconnected" and nothing is logged server-side. |
| `args` | dict[str, Any] \| None | `None` | CLI arguments passed to the frontend/router process |
| `env` | dict[str, str] \| None | `None` | Environment variables for frontend processes |
| `container_image` | str \| None | `None` | Optional router-specific image. Static routers use the model/backend image when omitted. |
| `ctx_router` | dict[str, Any] \| None | `None` | trtllm_serve orchestrator (ser.yaml) options; ignored by other frontends. |
| `gen_router` | dict[str, Any] \| None | `None` | generation_servers.router |
| `server_config_extra` | dict[str, Any] \| None | `None` | extra top-level ser.yaml keys |
| `orchestrator_placement` | str | `'head'` | trtllm_serve: which node runs the disaggregated orchestrator. "head" (default) -> nodes.head (first prefill/CTX node) "first_decode" -> first decode/GEN worker-leader node |
| `dedicated_node` | bool | `False` | If True, reserve a node exclusively for the frontend/orchestrator instead of running it on a worker node. Requires at least 2 nodes. Not supported together with resources.het_jobs: true. Default: False. |

### DynamoConfig

Dynamo installation configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `install` | bool | `True` |  |
| `version` | str \| None | `'0.8.0'` |  |
| `hash` | str \| None | `None` |  |
| `top_of_tree` | bool | `False` |  |
| `wheel` | str \| None | `None` |  |
| `source` | [DynamoSourceConfig](#dynamosourceconfig) \| None | `None` | The 2.0 way to say which Dynamo: one of git+rev, pypi, or wheel. Mapped onto the legacy fields above in __post_init__, so every consumer keeps reading hash / version / wheel / cargo_patches unchanged. |
| `request_plane` | str | `'tcp'` |  |
| `event_plane` | str \| None | `None` |  |
| `sidecar` | bool | `False` |  |
| `sidecar_port` | int | `50051` |  |
| `sidecar_binary` | str \| None | `None` |  |
| `sidecar_startup_timeout` | int | `1200` |  |
| `sidecar_context_length` | int \| None | `None` |  |
| `sidecar_args` | list[str] | `[]` |  |
| `cargo_patches` | list[str] \| None | `None` | Optional dependency-declaration overrides applied to the dynamo Cargo.toml tree before a source build (requires `hash`). Each entry is a full `<crate> = <spec>` TOML line, e.g. 'dynamo-tokenizers = { git = "https://github.com/ai-dynamo/frontend-crates", branch = "..." }' The crate's existing declaration is replaced tree-wide, letting a source build pull a crate from an unmerged branch without waiting for a crates.io release. |

### BenchmarkConfig

Benchmark configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | str | `'manual'` |  |
| `isl` | int \| None | `None` |  |
| `osl` | int \| None | `None` |  |
| `concurrencies` | list[int] \| str \| None | `None` |  |
| `req_rate` | str \| int \| None | `'inf'` |  |
| `client_placement` | str | `'head'` | Which node runs the benchmark client: "head" (default) -> nodes.head (co-located with orchestrator by default) "last_decode" -> last decode/GEN worker-leader node (isolate the client off the CTX/orchestrator node). When the client lands on a different node than the orchestrator, use the injected $SRT_FRONTEND_HOST env in the benchmark command's URL. |
| `client_dedicated_node` | bool | `False` | If True, reserve a node exclusively for the benchmark client instead of running it on a worker node. Requires at least 2 nodes. Not supported together with resources.het_jobs: true. Default: False. |
| `colocate_with_frontend` | bool | `True` | Governs how the dedicated-node flags combine when more than one of client_dedicated_node, frontend.dedicated_node, and infra.etcd_nats_dedicated_node is set. If True (default), every requested role shares a single reserved node. If False, each requested role gets its own reserved node (requires enough total nodes: worker count + number of dedicated roles). |
| `sweep` | [SweepConfig](#sweepconfig) \| None | `None` |  |
| `num_examples` | int \| None | `None` | Accuracy benchmark fields |
| `max_tokens` | int \| None | `None` |  |
| `repeat` | int \| None | `None` |  |
| `num_threads` | int \| None | `None` |  |
| `max_context_length` | int \| None | `None` |  |
| `categories` | list[str] \| None | `None` |  |
| `num_shots` | int \| None | `None` | GSM8K few-shot examples |
| `temperature` | float \| None | `None` |  |
| `top_p` | float \| None | `None` |  |
| `top_k` | int \| None | `None` |  |
| `num_requests` | int \| None | `None` | Router benchmark fields |
| `concurrency` | int \| None | `None` |  |
| `prefix_ratios` | list[float] \| str \| None | `None` |  |
| `mooncake_workload` | str \| None | `None` | Mooncake router benchmark fields (uses aiperf with mooncake_trace) |
| `ttft_threshold_ms` | int \| None | `None` | Goodput TTFT threshold in ms (default: 2000) |
| `itl_threshold_ms` | int \| None | `None` | Goodput ITL threshold in ms (default: 25) |
| `random_range_ratio` | float \| None | `None` | Random input/output length range ratio (default: 0.8) |
| `num_prompts_mult` | int \| None | `None` | Multiplier for num_prompts = concurrency * mult (default: 10) |
| `num_warmup_mult` | int \| None | `None` | Multiplier for warmup prompts = concurrency * mult (default: 2) |
| `dataset_name` | str \| None | `None` | Custom dataset fields (sa-bench) |
| `dataset_path` | str \| None | `None` | Container path to dataset file (mount via extra_mount) |
| `agentperf_client_dir` | str \| None | `None` | AgentPerf benchmark fields (agentperf-client trajectory replay) |
| `agentperf_config` | str \| None | `None` |  |
| `trace_file` | str \| None | `None` | Trace replay benchmark fields (uses aiperf with mooncake_trace dataset type) |
| `custom_tokenizer` | str \| None | `None` | Custom tokenizer class (e.g., "module.path.ClassName") |
| `use_chat_template` | bool | `True` | Pass --use-chat-template to benchmark (default: true) |
| `reuse_http_connections` | bool | `False` | SA-Bench Dynamo adapter: reuse a benchmark-scoped HTTP connection pool. Opt-in to preserve the historical per-request ClientSession behavior. |
| `command` | str \| None | `None` | Custom benchmark hook. ``command`` is passed to ``bash -lc`` verbatim; srtctl does NOT substitute placeholders like ``{nginx_url}`` or ``{slurm_job_id}``. Render any parameters when generating the recipe. See srtctl.benchmarks.custom.CustomBenchmarkRunner for details. |
| `container_image` | str \| None | `None` |  |
| `env` | dict[str, str] | `{}` |  |
| `aiperf_package` | str \| None | `None` | aiperf pip install spec (e.g., "aiperf>=0.7.0", "aiperf @ git+https://...@commit") If set, runs pip install <spec> before benchmarking. Upgrades if already installed. |
| `aiperf_args` | dict[str, Any] | `{}` | Extra aiperf CLI flags passed through to bench.sh (e.g., benchmark-duration: 600, workers-max: 200) |
| `slow_down_sleep_time` | float \| None | `None` | SA-Bench: optional SGLang /slow_down on decode workers (sglang frontend only; see benchmark_stage) |
| `slow_down_wait_time` | float \| None | `None` | seconds until POST clears slow_down; unset = feature off |

### ProfilingConfig

Profiling configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | str | `'none'` | "none", "nsys", "nsys-time", or "torch" |
| `extra_nsys_args` | list[str] \| None | `None` | Extra arguments passed to nsys profile (appended before `-o`; see get_nsys_prefix) |
| `prefill` | [ProfilingPhaseConfig](#profilingphaseconfig) \| None | `None` | Phase-specific profiling step configs (not used for nsys-time) |
| `decode` | [ProfilingPhaseConfig](#profilingphaseconfig) \| None | `None` |  |
| `aggregated` | [ProfilingPhaseConfig](#profilingphaseconfig) \| None | `None` |  |
| `delay_secs` | int \| None | `None` | nsys-time fields: time-based capture window, same on all workers |
| `duration_secs` | int \| None | `None` | nsys --duration: seconds to capture after delay |
| `benchmark_duration_secs` | int | `300` | total traffic generation duration (must cover delay + duration) |

### OutputConfig

Output configuration with formattable paths.

| Key | Type | Default | Description |
|---|---|---|---|
| `log_dir` | [FormattablePath](#formattablepath) | `<lambda>()` |  |

### HealthCheckConfig

Health check configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `max_attempts` | int | `180` | 30 minutes default (large models take time to load) |
| `interval_seconds` | int | `10` |  |

### InfraConfig

Infrastructure configuration for etcd/nats placement.

| Key | Type | Default | Description |
|---|---|---|---|
| `etcd_nats_dedicated_node` | bool | `False` | If True, run etcd and nats on a dedicated node instead of the head node. This reserves the first node exclusively for infrastructure services. Default: False. |
| `nats_max_payload_mb` | int \| None | `None` | Maximum NATS message payload in MB. Default: None (uses NATS default of 1MB). Set to 24+ for disaggregated serving with long ISL (e.g. 65K+ tokens where prompt data exceeds 1MB in NATS messages). |

### ObservabilityConfig

Observability configuration for OTEL tracing.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `False` | Master analytics knob. Default: False. |
| `enable_otel` | bool | `False` | If True, inject OTEL environment variables into all workers and frontends. Requires otel_endpoint to be set. Default: False. |
| `otel_endpoint` | str \| None | `None` | OTEL collector endpoint (e.g. "http://10.0.0.1:4317"). Required when enable_otel is True. |
| `tachometer` | [TachometerConfig](#tachometerconfig) | `TachometerConfig()` | Native Tachometer capture configuration. Follows ``enabled`` unless ``tachometer.enabled`` is set explicitly (see :class:`TachometerConfig`). |

### TelemetryConfig

DCGM power telemetry for benchmark measurement windows.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `False` |  |
| `dcgm_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `None` |  |
| `collect_interval_ms` | int | `1000` | Milliseconds between collector cycles. Replaces the retired ``default_frequency``, which despite its name was a period in seconds (1000ms == the old 1.0 default). |
| `storage_subdir` | str | `'power'` |  |
| `required` | bool | `False` |  |
| `startup_timeout_seconds` | float | `30.0` |  |
| `request_timeout_seconds` | float | `2.0` |  |
| `collector_join_timeout_seconds` | float \| None | `None` | None derives a safe shutdown budget from request_timeout_seconds. |

### FormattablePath

A path that may contain placeholders requiring formatting.

| Key | Type | Default | Description |
|---|---|---|---|
| `template` | str | required |  |

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
| `container` | str \| None | `None` | Container image or ``srtslurm.yaml`` alias. Defaults to the kind's fallback (Mooncake's ``mooncake_kv_store.container``), then the job container. |
| `env` | dict[str, str] | `{}` | Environment for the service process, on top of what the kind injects. |
| `source` | [SourceConfig](#sourceconfig) \| None | `None` | Optional git source to clone before ``build_command`` and ``command`` run. Single-node placements only. |
| `build_command` | list[str] \| None | `None` | Argv run once inside the service container, from the clone, before ``command`` starts. Only meaningful with ``source``. |
| `placement` | [ServicePlacementConfig](#serviceplacementconfig) \| None | `None` | Where the service runs. Defaults to the kind's placement (``head`` for generic services, ``infra`` for etcd/nats/mooncake-master, ``workers`` for the exporters). |
| `start` | str \| None | `None` | ``after_frontend`` (default for ``generic``) or ``before_workers`` (default for ``mooncake-store``). |
| `readiness` | [ServiceReadinessConfig](#servicereadinessconfig) \| None | `None` | Optional TCP port gate; the job waits for it on every service node before continuing. |
| `inherit_discovery_env` | bool | `True` | Inject ``ETCD_ENDPOINTS`` / ``NATS_SERVER`` so the service can register with the job's Dynamo discovery plane. |
| `critical` | bool \| None | `None` | When true a crash fails the run, like a worker dying. Default false for ``generic`` (a dead sidecar costs its own log, not the run) and true for ``mooncake-store``. Set true for anything in the live request path. |
| `preamble` | str \| None | `None` | Shell run inside the container before ``command`` (``ulimit`` and friends). |
| `cpus_per_task` | int \| None | `None` | Optional ``srun --cpus-per-task``. |
| `cpu_bind` | str \| None | `None` | Optional ``srun --cpu-bind``. |
| `srun_options` | dict[str, str] | `{}` | Extra srun options for this service only. |
| `build_timeout_seconds` | int | `1800` | Kill ``build_command`` after this many seconds. |
| `enabled` | bool | `True` | ``false`` drops the service, including an implicit one (``etcd`` / ``nats`` under the Dynamo frontend, the default exporters) declared here by name. |
| `external` | str \| None | `None` | For discovery-plane kinds (``etcd``, ``nats``, ``mooncake-master``): use this already-running endpoint and launch nothing; the URL is what the job's processes are pointed at. |
| `options` | dict[str, Any] | `{}` | Kind-specific settings (``nats``: ``max_payload_mb``; ``mooncake-master``: ``store_config`` for vLLM). Unknown keys are rejected by the kind. |

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
| `model` | [IdentityModelConfig](#identitymodelconfig) | `IdentityModelConfig()` |  |
| `container` | [IdentityContainerConfig](#identitycontainerconfig) | `IdentityContainerConfig()` |  |
| `frameworks` | dict[str, str] | `{}` |  |

### ReportingConfig

Reporting configuration for status updates, AI analysis, and log exports.

| Key | Type | Default | Description |
|---|---|---|---|
| `status` | [ReportingStatusConfig](#reportingstatusconfig) \| None | `None` |  |
| `ai_analysis` | [AIAnalysisConfig](#aianalysisconfig) \| None | `None` |  |
| `s3` | [S3Config](#s3config) \| None | `None` |  |

### DynamoSourceConfig

Where Dynamo comes from. Exactly one of ``git``, ``pypi``, or ``wheel``.

| Key | Type | Default | Description |
|---|---|---|---|
| `git` | str \| None | `None` | Repository URL to build from (default upstream when ``rev`` is set without it). Builds ``ai-dynamo-runtime`` with maturin and installs ``ai-dynamo`` from the checkout; cached on ``/configs`` by commit. |
| `rev` | str \| None | `None` | Immutable ref in ``git``: commit SHA, tag, or ``refs/pull/<n>/head``. |
| `sha` | str \| None | `None` | The commit ``rev`` resolved to; filled in by ``srtctl apply``. |
| `patches` | list[str] \| None | `None` | Cargo dependency replacements applied tree-wide before the build (the legacy ``cargo_patches``). |
| `pypi` | str \| None | `None` | Release version from PyPI (the legacy ``version``). |
| `wheel` | str \| None | `None` | Staged nightly ``ai-dynamo`` version (the legacy ``wheel``). |

### SweepConfig

Configuration for benchmark parameter sweeps.

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | one of `'zip'`, `'grid'` | `'zip'` |  |
| `parameters` | dict[str, list[Any]] | `{}` |  |

### ProfilingPhaseConfig

Profiling config for a single phase (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `start_step` | int \| None | `None` | Step to start profiling |
| `stop_step` | int \| None | `None` | Step to stop profiling |

### TachometerConfig

Native Tachometer collection for an observability-enabled run.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool \| None | `None` |  |
| `binary_path` | str | `'tachometer-scraper'` |  |
| `collect_interval_ms` | int | `1000` | Milliseconds between scrapes of every endpoint — the same unit and name as dcgm-exporter's --collect-interval. Replaces the retired Hz-based ``default_frequency`` (1000ms == the old 1.0 Hz default). |
| `sync_interval_secs` | int | `120` |  |
| `compaction_threads` | int | `4` |  |
| `storage_subdir` | str | `'tachometer'` |  |
| `extra_metadata` | dict[str, str] | `{}` |  |
| `default_exporters` | bool | `True` |  |
| `dcgm_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `None` |  |
| `node_exporter` | [TelemetryExporterConfig](#telemetryexporterconfig) \| None | `None` |  |

### TelemetryExporterConfig

Configuration for a metrics exporter deployed on worker nodes.

| Key | Type | Default | Description |
|---|---|---|---|
| `container_image` | str | required |  |
| `port` | int | required |  |
| `command` | str \| None | `None` |  |

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
| `node` | str | `'head'` | ``head`` or ``infra`` (one instance), ``dedicated`` (reserve the infra node exclusively; infra-class kinds only), ``prefill`` / ``decode`` / ``agg`` (one instance per distinct physical node that role's workers use), or ``workers`` (one instance per worker node). |

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
| `endpoint` | str \| None | `None` |  |
| `endpoints` | list[str] \| None | `None` |  |

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

### TcpProbe

Ready when ``port`` accepts a TCP connection on the service node.

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int | required |  |

### HttpProbe

Ready when ``GET http://<node>:<port><path>`` returns ``status``.

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int | required |  |
| `path` | str | `'/health'` |  |
| `status` | int | `200` |  |

### LogProbe

Ready when the service's log file contains a line matching the regular expression ``pattern``.

| Key | Type | Default | Description |
|---|---|---|---|
| `pattern` | str | required |  |

## Backend types

`backend.type` selects one of the following; the remaining `backend` keys are that type's fields.

### SGLangProtocol

`backend.type: sglang`

SGLang protocol - implements BackendProtocol.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'sglang'` | `'sglang'` |  |
| `gpu_type` | str \| None | `None` |  |
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `sglang_config` | [SGLangServerConfig](#sglangserverconfig) \| None | `None` | SGLang server CLI config per mode |
| `kv_events_config` | bool \| dict[str, Any] \| None | `None` | KV events config - enables --kv-events-config with auto-allocated ports Per-mode: {"prefill": true, "decode": {"publisher": "zmq", "topic": "custom"}} Or global: true (enables for prefill+decode with defaults) |
| `mooncake_kv_store` | [MooncakeKVStoreConfig](#mooncakekvstoreconfig) \| None | `None` | Mooncake KV store - launches mooncake_master on infra node and injects MOONCAKE_MASTER env var on all workers automatically |

### TRTLLMProtocol

`backend.type: trtllm`

TRTLLM protocol - implements BackendProtocol.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'trtllm'` | `'trtllm'` |  |
| `prefill_environment` | dict[str, str] | `{}` |  |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `prefill_extra_args` | list[str] | `[]` | Extra `trtllm-serve` CLI flags per mode, appended verbatim to the worker command (frontend.type: trtllm_serve only -- dynamo.trtllm takes a different CLI). `trtllm_config` already covers everything that belongs in the engine YAML, which is nearly everything: trtllm-serve merges that file into LlmArgs. But a few of its options configure the OpenAI SERVER layer rather than the engine and have no LlmArgs field, so no YAML key can reach them. The one that matters in practice is `--tool_parser` (a click.Choice consumed directly by the server constructor); note that its sibling `--reasoning_parser` IS forwarded into get_llm_args() and so remains settable from `trtllm_config`. backend: type: trtllm prefill_extra_args: ["--tool_parser", "glm47"] decode_extra_args: ["--tool_parser", "glm47"] |
| `decode_extra_args` | list[str] | `[]` |  |
| `aggregated_extra_args` | list[str] | `[]` |  |
| `trtllm_config` | [TRTLLMServerConfig](#trtllmserverconfig) \| None | `None` |  |
| `served_model_name` | str \| None | `None` | The name clients must use in a request's "model" field. Defaults to the checkpoint directory name. backend: type: trtllm served_model_name: "deepseek-ai/deepseek-r1" Set it when the client cannot be told which name to ask for. agentperf takes the name as a flag, so it never needs this; the MLPerf harness has it fixed in the benchmark definition, so the server must match or every request 404s. Top-level rather than a trtllm_config key because trtllm_config is dumped straight into the engine's YAML file, and this is a launcher flag the engine does not recognise. |
| `publish_events_and_metrics` | bool | `False` | Whether dynamo.trtllm workers pass `--publish-events-and-metrics`. Enables the worker to publish KV-cache events (add/evict) + metrics, which the dynamo frontend consumes for KV-cache-aware routing (router-mode: kv). This may impact performance so should be disabled if exact KV aware routing is not needed. |
| `sequential_node_start` | int | `0` | Controls batched startup of workers that share the same node. 0 = start all workers in parallel (no constraint). 1 = fully sequential: one worker at a time, each must be ready before the next. N > 1 = start N workers simultaneously per batch, wait for all to be ready, then next batch. For trtllm_serve: readiness is an HTTP 200 on the worker's http_port. For dynamo.trtllm: readiness is a TCP connection on the worker's sys_port. |
| `numa_memory_bind` | bool \| None | `None` | Whether to prefix the trtllm worker command with `numactl -m 0,1`. None (default) preserves the existing auto-detected behavior (enabled only for gb200/gb300). True/False forces numactl on/off regardless of gpu_type. |
| `numa_cpu_bind` | bool | `False` | Optional stricter NUMA CPU affinity for the worker process, in addition to numa_memory_bind. A previous post-hoc `taskset -pc <cpuset> $PPID` approach (see bind-b300-prefill-cpus.sh) only pins the leader PID *after* launch, so secondary threads spawned by Python/UCX/MPI/TRT-LLM can still land cross-socket. When true, srtctl instead: 1. sets TLLM_NUMA_AWARE_WORKER_AFFINITY=0 (disables TRT-LLM's own internal NUMA thread-pinning, which fights with the OS-level mask) 2. wraps the worker command (prefill/decode/agg) in `taskset -c <cpu_list>`, applied *before* exec so every spawned thread inherits the mask. The CPU list is discovered at runtime (configs/numa_cpu_bind.sh) from the physical GPU this task owns, not a static SLURM_LOCALID table — a static table assumes SLURM_LOCALID is a node-wide GPU ordinal, which breaks when two endpoints share a node (each gets its own srun step, so LOCALID restarts at 0 for both). |

### VLLMProtocol

`backend.type: vllm`

vLLM protocol - implements BackendProtocol.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'vllm'` | `'vllm'` |  |
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `vllm_config` | [VLLMServerConfig](#vllmserverconfig) \| None | `None` | vLLM server CLI config per mode |
| `set_cuda_visible_devices` | bool | `False` | Legacy device binding for vLLM builds without --device-ids. |
| `connector` | str \| None | `'nixl'` | Default KV connector: "nixl", "lmcache", or a raw JSON string for --kv-transfer-config. Can be overridden per mode by setting "connector" in vllm_config.prefill/decode/aggregated. dynamo 1.0.0+: translated to --kv-transfer-config (--connector was removed). |
| `mooncake_kv_store` | [VLLMMooncakeKVStoreConfig](#vllmmooncakekvstoreconfig) \| None | `None` | Mooncake KV store — when set, srtslurm launches mooncake_master on the infra node and auto-injects MOONCAKE_MASTER / MOONCAKE_TE_META_DATA_SERVER / MOONCAKE_LOCAL_HOSTNAME on every vLLM worker. |
| `kv_events_config` | bool \| dict[str, Any] \| None | `None` | KV events config - enables --kv-events-config with auto-allocated ports. Required for Dynamo's event-driven KV-aware routing. Global true enables defaults for prefill and decode workers. Per-mode: {"prefill": true, "decode": {"topic": "custom"}} |
| `allow_prefill_decode_colocation` | bool | `False` | Allow prefill and decode workers to share one node when the combined GPU request fits within gpus_per_node. Defaults off to preserve existing P/D node separation. |
| `allow_prefill_decode_colocation_across_nodes` | bool | `False` | Extend P/D colocation to multi-node topologies. When enabled together with allow_prefill_decode_colocation, workers are packed contiguously across the minimum number of nodes instead of reserving separate P/D node pools. Defaults off to preserve the original one-node-only policy. |
| `dp_launch_mode` | one of `'per_gpu'`, `'per_node'` | `'per_node'` | DP process layout. Per-node lets vLLM manage the node-local portion of a DP x TP x PP topology in one CUDA namespace and derives cross-node TP/PP rendezvous when a replica is larger than the node-local GPU allocation. Per-GPU remains available as a deprecated compatibility layout. |

### MockerProtocol

`backend.type: mocker`

Dynamo Mocker protocol - implements BackendProtocol.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | one of `'mocker'` | `'mocker'` |  |
| `engine_type` | str | `'vllm'` | Simulation parameters |
| `speedup_ratio` | float | `100.0` |  |
| `decode_speedup_ratio` | float | `1.0` |  |
| `num_gpu_blocks_override` | int | `16384` |  |
| `max_num_seqs` | int | `256` |  |
| `max_num_batched_tokens` | int | `8192` |  |
| `block_size` | int \| None | `None` |  |
| `data_parallel_size` | int | `1` |  |
| `num_workers` | int | `1` |  |
| `startup_time` | float \| None | `None` |  |
| `kv_transfer_bandwidth` | float \| None | `None` |  |
| `kv_cache_dtype` | str \| None | `None` |  |
| `enable_prefix_caching` | bool | `True` |  |
| `enable_chunked_prefill` | bool | `True` |  |
| `preemption_mode` | str \| None | `None` |  |
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `mocker_config` | [MockerServerConfig](#mockerserverconfig) \| None | `None` | Per-mode CLI overrides |

### SGLangServerConfig

SGLang server CLI configuration per mode (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill` | dict[str, Any] \| None | `None` |  |
| `decode` | dict[str, Any] \| None | `None` |  |
| `aggregated` | dict[str, Any] \| None | `None` |  |

### MooncakeKVStoreConfig

Mooncake KV store configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `container` | str \| None | `None` |  |
| `env` | dict[str, str] | `{}` |  |
| `master_extra_args` | list[str] | `[]` |  |

### TRTLLMServerConfig

SGLang server CLI configuration per mode (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill` | dict[str, Any] \| None | `None` |  |
| `decode` | dict[str, Any] \| None | `None` |  |
| `aggregated` | dict[str, Any] \| None | `None` |  |

### VLLMServerConfig

vLLM server CLI configuration per mode (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill` | dict[str, Any] \| None | `None` |  |
| `decode` | dict[str, Any] \| None | `None` |  |
| `aggregated` | dict[str, Any] \| None | `None` |  |

### VLLMMooncakeKVStoreConfig

Mooncake KV store config for the vLLM backend.

| Key | Type | Default | Description |
|---|---|---|---|
| `container` | str \| None | `None` |  |
| `env` | dict[str, str] | `{}` |  |
| `master_extra_args` | list[str] | `[]` |  |
| `store_config` | dict[str, Any] \| None | `None` | ``store_config`` values are JSON-serialized into MOONCAKE_CONFIG_PATH and parsed by vLLM's ``MooncakeStoreConfig`` dataclass — fields are a mix of str (e.g. ``protocol``), int (e.g. ``port``), and human-readable sizes (e.g. ``"4GB"``). Type as ``dict[str, Any]`` to avoid forcing users to quote numeric values. |

### MockerServerConfig

Mocker CLI configuration per mode (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill` | dict[str, Any] \| None | `None` |  |
| `decode` | dict[str, Any] \| None | `None` |  |
| `aggregated` | dict[str, Any] \| None | `None` |  |

## Cluster config

Top-level keys of `srtslurm.yaml`. Recipes inherit these defaults and resolve aliases through them.

| Key | Type | Default | Description |
|---|---|---|---|
| `cluster` | str \| None | `None` | Cluster name for status reporting |
| `default_account` | str \| None | `None` |  |
| `default_partition` | str \| None | `None` |  |
| `default_time_limit` | str \| None | `None` |  |
| `gpus_per_node` | int \| None | `None` |  |
| `default_gpu_type` | str \| None | `None` | Default for ``ResourceConfig.gpu_type`` when the recipe omits it. Lets one recipe move between clusters of different GPU types without an edit. |
| `network_interface` | str \| None | `None` |  |
| `use_gpus_per_node_directive` | bool | `True` |  |
| `use_segment_sbatch_directive` | bool | `True` |  |
| `use_exclusive_sbatch_directive` | bool | `False` |  |
| `use_het_jobs` | bool | `False` | Default for ``ResourceConfig.het_jobs`` when the recipe doesn't set it. When True (and recipe doesn't override), the prefill side and decode side are submitted as two SLURM heterogeneous-job components, each with its own ``--segment``. Lets asymmetric layouts (e.g. prefill 12 + decode 10 nodes on GB200/GB300) preserve NVL72 affinity per side. |
| `default_sbatch_directives` | dict[str, str] \| None | `None` |  |
| `default_health_check` | dict[str, int] \| None | `None` |  |
| `srtctl_root` | str \| None | `None` |  |
| `output_dir` | str \| None | `None` | Custom output directory for job logs |
| `model_paths` | dict[str, str] \| None | `None` |  |
| `containers` | dict[str, str] \| None | `None` |  |
| `cloud` | dict[str, str] \| None | `None` |  |
| `default_mounts` | dict[str, str] \| None | `None` | Cluster-level container mounts (host_path -> container_path) Applied to all jobs on this cluster, useful for cluster-specific paths |
| `default_bash_preamble` | str \| None | `None` | Shell snippet prepended to every container srun (after env exports, before the main command). Useful for cluster-wide ulimits, e.g. ``"ulimit -n 1048576 -s unlimited -u 1048576"``. Silently dropped for sruns that bypass the bash wrapper (distroless containers). |
| `default_host_setup` | [HostSetupConfig](#hostsetupconfig) \| None | `None` | Commands run on every allocated node's bare host, outside the container, before workers start. Recipes override with their own `host_setup:` block. |
| `reporting` | [ReportingConfig](#reportingconfig) \| None | `None` |  |
| `telemetry` | dict \| None | `None` | opaque dict, parsed by try_start_snapshotter |
| `nginx_raise_ulimit` | bool \| None | `None` | When set, applied to job configs that omit ``frontend.nginx_raise_ulimit``. Clusters that disallow raising nofile for nginx containers should use false. |
| `git_http_version` | str \| None | `None` | Works around intermittent git smart-HTTP/HTTP2 failures cloning github.com (stalls, or truncated responses git misreports as "could not read Username" auth-prompt failures). See git_clone_command_prefix() in core/config.py -- applied to every git clone/fetch srtctl performs. |
