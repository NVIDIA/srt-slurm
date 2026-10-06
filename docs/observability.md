# Observability and Telemetry

The `observability:` preset (nsys, OTEL, Tachometer) and the `telemetry:` providers (GPU and CPU power). Profiler settings: [Profiling](profiling.md).

## observability

Tachometer collection is **on by default for every run** (no configuration needed; `observability.tachometer.enabled: false` opts out). `observability.enabled` turns on the server metrics *content* (the TRT-LLM publish flag and engine statistics) and the trace surfaces:

```yaml
observability:
  enabled: true
```

The capture window aligns with the load, the same window the benchmark client's own `AIPERF_SERVER_METRICS_URLS` polling covers: on benchmark runs the scraper starts once the server passes the health gate (bring-up produces only dead-endpoint noise while workers load) and is stopped **gracefully** when the client exits, with a configurable grace period for compacting `final.parquet`. Runs without a discrete load window (serve-only, `manual`, eval-only) capture the whole serve session as before. Signal handlers and the critical-process monitor still use the process registry's existing teardown budget; the benchmark shutdown grace does not override those paths.

Tachometer scrapes all configured worker, frontend, DCGM, and node-exporter endpoints, independently of the benchmark client's `AIPERF_SERVER_METRICS_URLS` polling. This keeps the raw capture complete even when the client also collects metrics.

The legacy in-job Python RAW scraper is retired: a recipe still carrying `scrape_metrics`, `scrape_interval_seconds`, or `scrape_output` fails validation at submit time.

Fields: [ObservabilityConfig](schema-reference.md#observabilityconfig), [NsysObservabilityConfig](schema-reference.md#nsysobservabilityconfig). `tachometer.enabled: null` (the default) collects on every run, independent of `observability.enabled`; explicit `false` opts out. See [Profiling](profiling.md#observability-capture) for the nsys settings.

`observability.enabled: true` also enables nsys with NVTX tracing and CPU sampling
on every frontend and worker process/rank.
Its default `nsys.capture_window: measured_workload` starts collection after warmup and
stops it when the measured workload finishes. Supported bundled runners call
the boundary hooks automatically; custom/manual clients must call them at their
own phase boundaries. Set `nsys.capture_window: including_startup` to include startup and
warmup through teardown, or `nsys.enabled: false` to opt out. An enabled
top-level `profiling` mode takes precedence. The serving container must provide nsys and the required
NVTX support. See [Observability capture](profiling.md#observability-capture)
for timing, sampling, injection, and report-finalization settings.

SGLang workers always receive `--enable-metrics` unless the recipe sets it: native
`sglang.launch_server` serves `/metrics` only with the flag, and `dynamo.sglang`
merges the engine's `sglang:*` series into its system-port `/metrics` only when
SGLang was started with it.

Tachometer collects every worker rank, frontend, DCGM, node, and process metrics by default (minus the client-polled complement described above); the exporters launch from pinned multi-arch registry images with no configuration. Air-gapped clusters override the images via the `containers:` alias map in `srtslurm.yaml`; `default_exporters: false` disables the built-ins:

```yaml
observability:
  enabled: true
  tachometer:
    enabled: true
    collect_interval_ms: 1000
    sync_interval_secs: 120
    compaction_threads: 4
    storage_subdir: tachometer
    extra_metadata:
      cluster: production
    dcgm_exporter:
      container_image: /containers/dcgm-exporter.sqsh
      port: 9400
    node_exporter:
      container_image: /containers/node-exporter.sqsh
      port: 9100
    process_exporter:
      binary: /opt/srt/configs/process-exporter   # host-native (default mode); or set container_image instead
      container_image: ""
      port: 9256
```

Every Tachometer field: [TachometerConfig](schema-reference.md#tachometerconfig). The ones whose behavior needs more than a line:

| Tachometer field | Type | Default | Description |
| ---------------- | ---- | ------- | ----------- |
| `collect_interval_ms` | int | `1000` | Milliseconds between scrapes of every endpoint; the single cadence knob. It also drives the launched DCGM exporter's `--collect-interval` (an explicit `dcgm_exporter.command` wins) and the host sampler. Values below `1000` speed up DCGM NVML sampling and are warned about at launch: 100ms sampling measured ~2% decode ITL overhead on GB300. Replaces the retired Hz-based `default_frequency` |
| `shutdown_grace_secs` | float | `120.0` | Time the scraper gets after SIGTERM to flush and compact `final.parquet` before it is killed; compaction scales with the data accumulated since the last periodic sync |
| `default_exporters` | bool | `true` | Imply the built-in DCGM + node + process exporters when no explicit blocks are set. The exporters are [services](services.md#implicit-services) (`dcgm-exporter`, `node-exporter`, `process-exporter`); declaring one under `services:` by that name overrides it, and `srtctl dry-run` lists them |
| `dcgm_exporter` | object/null | built-in | Defaults to `nvcr.io#nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04` on port 9401; an explicit block overrides |
| `node_exporter` | object/null | built-in | Defaults to `quay.io#prometheus/node-exporter:v1.8.2` on port 9101 with the `cpu`, `infiniband`, `meminfo`, `processes`, `stat`, `vmstat`, `pressure` and `meminfo_numa` collectors on worker nodes. Includes major faults and page-reclaim counters; retains process-state and NUMA-node identity. An explicit block overrides |
| `process_exporter` | object/null | built-in | Defaults to the **host-native** `configs/process-exporter` binary (ncabatoff/process-exporter 0.8.7, installed by `make setup` for the compute arch, like `configs/nats-server` and `configs/etcd`) on port 9256, launched with plain `srun` (no container) on every allocated node (the `process-exporter` service, `placement.node: all`). Reads the host `/proc` and publishes per-process-group CPU seconds by mode, thread count, per-thread-name CPU and count (`-threads=true`), context switches, RSS and open fds. The passthrough filter retains metric names and labels, including summary sum/count suffixes, and attaches hostname and run metadata to raw rows. Groups (frontend, `dynamo_trtllm` / `dynamo_sglang` / `dynamo_vllm` handlers, `trtllm_engine` children, the SGLang engine processes `sglang_scheduler` / `sglang_dp_controller` / `sglang_detokenizer` matched on their retitled command lines, launcher, client, infra daemons) come from `<log_dir>/process-exporter.yml`, written at launch. If the binary is missing the leg is skipped with a warning (submit warns too). An explicit block may set `binary` (absolute, or relative to the srtctl checkout) or instead a `container_image` with `binary` unset to run it containerized; the upstream `FROM scratch` image is not used by default because pyxis/enroot on some clusters cannot start shell-less images |

Every exporter block ([TelemetryExporterConfig](schema-reference.md#telemetryexporterconfig)) accepts `container_image`, `port`, `command` and `binary`. `binary` selects host-native launch (the executable runs directly under `srun`, `container_image` is ignored and may be `""`); without it the exporter runs from `container_image`. One of the two must be set.

`make setup ARCH=<compute_arch>` downloads and checksum-verifies the matching Tachometer binary from the latest srt-slurm release and installs the process-exporter binary for the same arch. The scraper and the process exporter run as native `srun` processes; the DCGM and node exporters remain containerized on worker nodes. Run `make tachometer-scraper` to build the scraper from source instead. The process-exporter passthrough filter and node process-state/NUMA-label preservation require a scraper built from this revision or a release containing it; rebuild the scraper when using an older downloaded binary.

The pressure collector reports PSI only when the host exposes the corresponding `/proc/pressure` files; missing metrics indicate unavailable data. NUMA memory and allocation metrics retain the exported `node` label as `numa_node` in raw metric names, separately from host metadata. With `observability.enabled: true`, the existing local host sampler also records cumulative PSI stall totals in microseconds in its `psi` JSONL field. That optional sampler covers the sweep/orchestrator host only; it does not extend exporter placement to dedicated frontend or client nodes. Collector overhead has not been measured for this change.

Tachometer writes its Parquet stream under `<log_dir>/<storage_subdir>/raw/scrape/` (the leaf is created by the scraper itself; srtctl pre-creates only the parent, because the scraper refuses a pre-existing storage directory), compacting to `final.parquet` there on shutdown. Intermediate files remain in `<log_dir>/<storage_subdir>/local` until shutdown compaction completes. Rows carry an epoch `timestamp_ns` column, so they join directly with AIPerf records and Dynamo spans.

The scraper runs as a best-effort process: if it dies (or the binary is missing at runtime), the benchmark continues and the loss is visible in `tachometer.out` and the sweep log. `srtctl validate-setup` still fails fast at submit time when `bin/tachometer-scraper` is absent.

## telemetry

`telemetry` is reserved for DCGM power measurement. It can run alongside `observability.tachometer`; it does not start Tachometer itself.

When both are enabled, `telemetry.dcgm_exporter` is shared with Tachometer. Do not also configure `observability.tachometer.dcgm_exporter`; Tachometer can still launch an optional node exporter from its own block.

```yaml
telemetry:
  enabled: true
  collect_interval_ms: 1000
  storage_subdir: power
  required: true
  dcgm_exporter:
    container_image: /containers/dcgm-exporter.sqsh
    port: 9400
  cpu_power_exporter:
    port: 9405
    source: auto
```

Fields: [TelemetryConfig](schema-reference.md#telemetryconfig). `collect_interval_ms` is shared by the DCGM and CPU legs and must be at most `3000`.

`telemetry` requires a `benchmark.type` of `sa-bench`, `custom`, `agentic`, `agentx`, or `manual` (a `manual` job has no load window, so like serve-only it captures the whole serve session; use it when an external load generator drives the endpoint), the benchmark client on the head node (`benchmark.placement.node: head`, the default), and no dedicated node for the discovery plane (an `etcd`/`nats` service with `placement.node: dedicated` moves the head off the batch host the collector runs on).

### CPU power

`telemetry.cpu_power_exporter` is an independent, best-effort leg: its presence (not a separate `enabled` flag) turns CPU power collection on, and it can run with or without `dcgm_exporter` alongside it. On each worker node, srtctl launches a `cpu-power-exporter` process directly on the bare host (outside the model container, so it can read host power interfaces) and exposes it on `cpu_power_exporter.port`. It resolves the bundled Rust binary installed by `make setup` first, falling back to the ACPI-only Python stdlib exporter (`srtctl.core.cpu_power_exporter`) when that binary is absent. A head-node collector scrapes every worker's exporter on the shared `collect_interval_ms`/`request_timeout_seconds` cadence and writes per-sample rows plus a manifest under `<log_dir>/<storage_subdir>/cpu/` (`samples.csv`, `cpu_manifest.json`).

Fields: [CpuPowerExporterConfig](schema-reference.md#cpupowerexporterconfig). `source: auto` tries DCGM first and falls back to ACPI; it has no effect on the Python fallback exporter, which is ACPI-only.

`samples.csv` carries one row per sensor reading, with columns `schema_version, timestamp_unix, hostname, source, sensor, socket_id, power_w, total_power_w`. In ACPI mode, `total_power_w` is **not** a sum of the `cpu`- and `sysio`-kind rails; whenever a `grace`-kind channel exists for a socket, that channel alone is the node-level total (real hardware traces show `grace` at roughly 93-104W against `cpu`+`sysio` combined at roughly 53-58W for the same socket, i.e. `grace` measures the whole Grace SoC power boundary, not literally `cpu + sysio`). When no `grace` channel is present for a scrape, `total_power_w` is left blank for that row rather than guessed from the component rails; per-socket `power_w` values are always populated regardless. In DCGM mode, `total_power_w` is the single already-aggregate value DCGM reports per socket.

Because collection is always best-effort, there is no `required` knob for the CPU leg: a node that fails to expose its exporter (or fails to publish readings) simply produces gaps in `samples.csv`, and the run continues.
