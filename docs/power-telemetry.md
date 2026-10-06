# GPU Power Telemetry (`dcgm-power`)

The `dcgm-power` telemetry provider records raw per-GPU watts for every
allocated worker node, the topology needed to map each GPU to a `prefill`,
`decode`, or `agg` role, and the exact formal benchmark window for every
measured concurrency. It never integrates power into energy and never branches
on model, precision, or recipe; consumers integrate watts over the recorded
window themselves. The provider name is historical: the exporter that supplies
the watts is selected per cluster or recipe (see [GPU exporter labels and metrics](#gpu-exporter-labels-and-metrics)).

## How it works

- One GPU power exporter task runs on each allocated worker node, launched
  through the normal SLURM/process-registry path (one `srun` per heterogeneous
  group).
- A collector thread inside the orchestrator polls every exporter concurrently
  from the physical head node, so all sample timestamps and benchmark
  boundaries come from one clock.
- The profile's power metric (`DCGM_FI_DEV_POWER_USAGE` for DCGM) determines
  which GPUs have power readings; optional utilization and temperature
  readings accompany them. Device identity comes from the profile's index and identity labels
  (`gpu` and `UUID` for DCGM).
- **No in-tree benchmark stamps measurement windows yet**, so every run is
  currently unpublishable: it records `MEASUREMENT_WINDOW` reason codes, and
  `required: true` exits non-zero. The adapter belongs with the benchmark
  (for the current sa-bench path and its planned replacement alike): the
  benchmark child writes one window file per measured concurrency using the
  standalone `measurement_window.py` module and the windows directory passed
  in via `MEASUREMENT_WINDOW_DIR_ENV`.

## Configuration

```yaml
# NOTE: unsupported end-to-end until a benchmark adapter stamps windows —
# with this exact config every run is unpublishable and `required: true` fails.
benchmark:
  type: sa-bench          # future benchmark-side adapter must stamp the windows
  placement:
    node: head            # keeps sample and window clocks on one host
  isl: 8192
  osl: 1024
  concurrencies: [4]

telemetry:
  enabled: true
  collect_interval_ms: 1000         # milliseconds between collector cycles; must be <= 3000
  storage_subdir: power             # relative to the run log directory
  required: true                    # exit non-zero when artifacts are unpublishable
  startup_timeout_seconds: 30
  request_timeout_seconds: 2
  collector_join_timeout_seconds: 12
  dcgm_exporter:
    container_image: dcgm-exporter  # alias, path, or registry URI
    port: 9401
```

`dcgm-power` needs **only** `dcgm_exporter`. A recipe that sets
`telemetry.enabled: true` with no `dcgm_exporter` and no CPU leg inherits the
cluster's `default_gpu_exporter` block from `srtslurm.yaml`, so one recipe can
measure power on clusters with different GPUs. There is no `provider` key, and it does
not require the top-level `container_image` or a `node_exporter`, because the
collector runs inside srtctl. Config loading validates the block and rejects
inconsistent values with actionable messages; in particular
`collect_interval_ms` must not exceed the 3-second max sample gap the validator
accepts, or every window would fail `sample_gap_exceeded`. Telemetry stays
disabled by default.
The collector join timeout must exceed two complete request-cycle budgets
(`2 * (2 * request_timeout_seconds + 1 second)`), covering a scrape already in
flight when shutdown starts plus the final bracketing scrape.

## GPU exporter labels and metrics

The collector, parser, manifest and validator do not know which GPU vendor they
are measuring. The exporter config (`telemetry.dcgm_exporter`, or the cluster
`default_gpu_exporter` it inherits) states it explicitly:

- `kind: dcgm` (the default) is NVIDIA dcgm-exporter: the built-in 100 ms
  `dcgm-exporter` command, the DCGM labels and metrics below, and tachometer's
  `dcgm` filter with per-GPU worker labels (endpoint `dcgm_<node>`). `gpu_labels`
  and `gpu_metrics` may still override the DCGM ones (another DCGM power field,
  say); nothing is inferred from metric names.
- `kind: custom` is any other exporter. It must set `command`, `gpu_labels` and
  `gpu_metrics`, and tachometer keeps its rows as served (`passthrough` filter,
  endpoint `gpu-power_<node>`, no per-GPU worker labels).

Written out, the DCGM defaults and the AMD exporter have the same shape:

```yaml
# NVIDIA dcgm-exporter: the defaults, written out
gpu_labels:
  index: gpu
  identity: UUID
  instance: [GPU_I_ID, GPU_I_PROFILE]
gpu_metrics:
  power:
    metric: DCGM_FI_DEV_POWER_USAGE
    scope: gpu_device_board_as_reported_by_dcgm
  gpu_util:
    metric: DCGM_FI_DEV_GPU_UTIL
  sm_active:
    metric: DCGM_FI_PROF_SM_ACTIVE
  temperature:
    metric: DCGM_FI_DEV_GPU_TEMP
```

```yaml
# AMD rocm/device-metrics-exporter
kind: custom
gpu_labels:
  index: gpu_id
  identity: serial_number
gpu_metrics:
  power:
    metric: gpu_power_usage
    scope: amd_device_metrics_exporter_gpu_power_usage
  gpu_util:
    metric: gpu_gfx_activity
  temperature:
    metric: gpu_junction_temperature
```

- `gpu_labels.index` must carry the node-local GPU index srt-slurm allocates by;
  `identity` must be stable per physical GPU (it fills `gpu_uuid`); samples
  carrying an `instance` label (MIG instances, partitions) are dropped.
- `gpu_metrics.power` is required and `scope` is recorded as `power_scope`.
  `gpu_util` (percent), `sm_active` (0-1 fraction) and `temperature` (Celsius) are optional; their
  columns stay empty when unset. Units are fixed by the artifact, not the config.

The artifact layout is identical for every exporter. `manifest.json` records
`source_metric`, `power_scope`, `utilization_metrics` and `temperature_metric`, so a consumer can tell
the measurement boundaries apart without the config.

### Example: a `kind: custom` exporter

Any Prometheus exporter that reports per-GPU power can be used. This example
configures AMD's [rocm/device-metrics-exporter](https://github.com/ROCm/device-metrics-exporter)
`v1.5.2` at the cluster level, so recipes need not change:

```yaml
# srtslurm.yaml
visible_devices_env: ROCR_VISIBLE_DEVICES
default_gpu_exporter:
  container_image: "docker://rocm/device-metrics-exporter:v1.5.2"
  command: "/home/amd/tools/entrypoint.sh"
  port: 5000
  kind: custom
  gpu_labels:
    index: gpu_id
    identity: serial_number
  gpu_metrics:
    power:
      metric: gpu_power_usage
      scope: amd_device_metrics_exporter_gpu_power_usage
    gpu_util:
      metric: gpu_gfx_activity
    temperature:
      metric: gpu_junction_temperature
```

The command, port, label and metric names, and scope are that exporter's, at
that version, not srt-slurm's; check them against the exporter's documentation
for the image you pin. The same block works under `telemetry.dcgm_exporter` in a
recipe, as in the
[single-node example](https://github.com/NVIDIA/srt-slurm/blob/main/examples/features/amd-power-telemetry.yaml).
For any `kind: custom` exporter:

- Pyxis runs `command`, not the image's `ENTRYPOINT`, so `command` must start
  everything the entrypoint would.
- `port` must be the port the exporter serves on; srt-slurm passes it to the
  command only through a `{port}` placeholder.
- The exporter runs with the run's container mounts; mount any device nodes it
  needs.
- `gpu_labels.identity` must be unique per physical GPU. GPUs that share one
  (compute partitions, for example) fail device validation (`gpu_uuid_changed`),
  as MIG instances do on NVIDIA.

## Artifacts

```text
<log_dir>/<storage_subdir>/
├── manifest.json
├── samples.csv
└── windows/
    └── <benchmark-result-stem>.json
```

`samples.csv` has the exact header
`schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w,gpu_util_pct,sm_active,temperature_c` (version 3),
one row per observation, `(scrape_seq, hostname, gpu_index)` unique. Rows are
never interpolated, averaged, or role-attributed — role and heterogeneous
group live once in the manifest topology.

GPU temperature is optional Celsius from `gpu_metrics.temperature` (`DCGM_FI_DEV_GPU_TEMP`
for DCGM, `gpu_junction_temperature` on AMD MI3xx) in the same
exporter response as power; collection adds no request, process, or wait.
Temperature must match the power reading's GPU index and UUID. Missing,
duplicate, non-finite, instance-labelled, or DCGM blank/error values leave the temperature
cell empty without invalidating power. Older v1/v2 files remain readable;
v3 readers must be deployed before upgrading producers. Consumers must show
missing temperatures as unavailable, never zero. The exporter must expose the
temperature field; collection does not enable additional profiling counters.

`manifest.json` records producer identity (version, git commit, exporter image
and its SHA-256), the power metric and scope, the sample interval, expected and observed device sets, the
topology mapping, the expected window list, the SHA-256 of the finalized
`samples.csv` bytes, terminal status, per-window coverage validation, and
reason codes. `status` is the lifecycle outcome;
`publication_valid` is the separate publication gate. Reason codes are stable
machine-readable strings enumerated in `srtctl/core/power/contract.py`.
The digest is required for offline publication validation, so packages created
before `samples_sha256` was recorded cannot be certified by this validator.

A window file records the formal benchmark boundaries on the head-node Unix
clock plus a monotonic `duration`, and points at the SA-Bench result it
brackets; result and window are boundary-identical.

With `required: true`, all artifacts are written first and the job then exits
non-zero whenever the terminal manifest is not publishable. With
`required: false`, measurement invalidity leaves the benchmark exit code
unchanged; an operational failure — a collector that cannot be joined, a
benchmark child that cannot be reaped, or an internal error while finalizing
telemetry — fails the job in either mode.

On `SIGTERM`/`SIGINT` or a critical-process death, the shared process registry
tears processes down before the collector finalizes, so the final scrape sees
dead endpoints. The manifest fails closed (`exporter_exited` /
`collector_interrupted` force `publication_valid=false`); the cost is that a
job that was simply cancelled can record `exporter_exited`.

## Re-validating a retained run

The artifact package is self-describing. The manifest supplies producer
identity, expected topology, runtime-only failure history, and a stored
verdict; the validator does not trust that verdict on its own:

```bash
srtctl-validate-power \
  --power-dir outputs/12345/logs/power \
  --result-root outputs/12345/logs \
  --expect-role prefill=4 --expect-role decode=4 \
  --require-distinct-het-groups
```

It recomputes every disk-derived claim from `samples.csv`, the result files,
and `windows/`, then requires the stored disk-derived reason subset and
`publication_valid` verdict to agree. Runtime-only reasons such as HTTP,
exporter-process, and collector failures cannot be reconstructed after the
live job is gone, so they are checked for a known v1 enum value and lifecycle
consistency instead. Exit status is `0` only when the recomputed package is
publishable, the stored verdict is `true`, and the two agree; otherwise it is
`1` and every failure is printed. The `--expect-*` flags optionally assert an
expected job shape for hardware canaries.

### Cumulative sample coverage

Each expected GPU must also retain at least 95% of the expected sampling
intervals across its nearest bracketing samples. Expected intervals are
`floor((last_bracket - first_bracket) / sample_interval_seconds)`; observed
intervals are the number of distinct sample times in that same span minus one,
so a row written twice counts once. This avoids counting ordinary cadence
jitter as repeated loss or allowing warmup samples to hide missing
measurements. More than 5% missing intervals, more than one sample in twenty,
records `sample_loss_exceeded`, even when every individual gap is below 3
seconds. For example, sampling every 2 seconds with a recorded 1-second cadence
fails. A manifest without a finite positive `sample_interval_seconds` fails this
rule too, alongside the manifest field check.

The 3-second maximum gap and boundary checks still apply, but on short spans
the loss rule is the stricter one. At a 1-second cadence one dropped sample
passes only from 20 intervals, and one 3-second hole (two missing intervals)
only from 40; a shorter window rejects on that single loss by design, because
one lost second is a larger share of it. Formal sa-bench windows run for
minutes, where a single gap at the limit passes both rules. Session
finalization and offline validation use the recorded cadence and the same
coverage rule. Previously accepted sparse packages can fail revalidation; their
files are not rewritten. This limits sample loss, not the numerical error in
energy.

## Diagnosing slow scrapes

The collector writes best-effort `scrape-timings.jsonl` beside `samples.csv`.
Every line carries an `event`: `scrape` per settled endpoint request,
`cycle_write` per collection cycle, and one closing `diagnostic_summary`.
Join a `scrape` record to its GPU rows using `(hostname, scrape_seq)`.
Each records its start/end times, HTTP status or exception, request and parse
durations, sample timestamp, row count and reason codes. Failed HTTP requests
retain timing records, with null parse duration and sample timestamp, without
inventing power samples. Requests still unsettled when the cycle deadline
expires have no timing record.

Instants are unix timestamps; durations come from the monotonic clock.
`schedule_lag_seconds` measures request start against the background cycle's
scheduled slot, which `cycle_write` records as `scheduled_at_unix`; manual and
final bracketing scrapes use null for both. The collector writes a cycle's
endpoints together, so the `cycle_write` record keyed by `scrape_seq` carries
the batch's `writer_lock_wait_seconds`, `sample_write_seconds` and attempted
`row_count` once. Its `sample_write_completed` reports whether the batch was appended and
flushed; when it is false, `sample_write_error` names the exception class if
the append raised, or is null when the session was already finalizing and
refused the batch.

Only a daemon writer performs diagnostic file I/O, outside the sample writer
lock. Its queue holds at most 128 pending records; overflow drops diagnostics,
not power samples. A final `diagnostic_summary` reports `dropped_records`.
Missing summary means diagnostics may be incomplete. Shutdown waits only until
the existing collector deadline. This optional sidecar is not publication
validation evidence, and its absence or write failure does not invalidate power.
