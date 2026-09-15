# GPU Power Telemetry (`dcgm-power`)

The `dcgm-power` telemetry provider records raw per-GPU watts for every
allocated worker node, the topology needed to map each GPU to a `prefill`,
`decode`, or `agg` role, and the exact formal benchmark window for every
measured concurrency. It never integrates power into energy and never branches
on model, precision, or recipe; consumers integrate watts over the recorded
window themselves.

## How it works

- One DCGM exporter task runs on each allocated worker node, launched through
  the normal SLURM/process-registry path (one `srun` per heterogeneous group).
- A collector thread inside the orchestrator polls every exporter concurrently
  from the physical head node, so all sample timestamps and benchmark
  boundaries come from one clock.
- GPU board watts come from `DCGM_FI_DEV_POWER_USAGE`; optional GPU utilization
  and SM activity columns accompany each sample. Device identity comes from the
  `gpu` and `UUID` labels.
- SA-Bench stamps its formal measurement windows. A `benchmark.type: custom`
  command must publish its own windows through the contract below; launching a
  successful command alone does not make power artifacts publishable.

## Configuration

```yaml
benchmark:
  type: sa-bench
  placement:
    node: head  # keeps sample and window clocks on one host
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

GPU power collection needs **only** `dcgm_exporter`. It does not require a
scraper image or `node_exporter`, because the collector runs inside srtctl.
Config loading validates the block and rejects
inconsistent values with actionable messages; in particular
`collect_interval_ms` must not exceed the 3-second max sample gap the validator
accepts, or every window would fail `sample_gap_exceeded`. Telemetry stays
disabled by default. Tachometer is configured separately under `observability`.
The collector join timeout must exceed two complete request-cycle budgets
(`2 * (2 * request_timeout_seconds + 1 second)`), covering a scrape already in
flight when shutdown starts plus the final bracketing scrape.

## Custom benchmark window contract

For `benchmark.type: custom` with telemetry enabled, srtctl passes:

| Environment variable | Value |
| --- | --- |
| `SRT_MEASUREMENT_WINDOW_DIR` | `/logs/<storage_subdir>/windows` |
| `SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE` | `custom` |
| `SRT_MEASUREMENT_WINDOW_CONCURRENCIES` | Space-separated measured concurrencies, e.g. `4 8` |
| `SRT_MEASUREMENT_WINDOW_RESULT_ROOT` | `/logs` |

The custom command owns benchmark execution and timing. Publish one JSON window
per listed concurrency in the window directory, using atomic replacement. Its
`result_path` must be relative to the result root, remain beneath that root, and
have the same filename stem as the window. For example, `windows/load_4.json`
may refer to `agentx/load_4.json` under `/logs`.

A completed window looks like:

```json
{
  "schema_version": 1,
  "benchmark_type": "custom",
  "concurrency": 4,
  "result_path": "agentx/load_4.json",
  "benchmark_start_time_unix": 1000.0,
  "benchmark_end_time_unix": 1020.0,
  "duration": 20.0,
  "clock_source": "head_node_unix_clock",
  "status": "completed",
  "reason": null
}
```

The referenced result must contain identical start, end, and duration fields.
Write `status: running` at the formal window's start with null end/duration/reason,
then replace it after the result is durable. The completed window must bracket
only the measured workload, excluding warmup. Incomplete, missing, duplicate,
out-of-root, or timing-mismatched windows fail publication validation. The
benchmark's adapter remains responsible for workload-specific counts and metrics.

## Clock and dedicated infrastructure placement

With power telemetry enabled, `SLURMD_NODENAME` identifies the batch host that runs
the collector. srtctl keeps head and benchmark on that host, including when its
name is not first in Slurm's expanded nodelist. Missing or invalid batch-host
metadata fails before the workload starts.

In a schema 2 recipe, etcd/nats services with `placement.node: dedicated` reserve
the last non-head node. In heterogeneous jobs this node belongs to group 0; decode
nodes and the requested prefill/decode worker counts are preserved. This applies
to SA-Bench and custom benchmarks alike. A dedicated frontend reserves the batch
host; if frontend and infrastructure share a reservation, both use that host.
Otherwise infrastructure retains its separate reservation.

The benchmark client must use `placement.node: head` (the default); a dedicated
client is rejected because it runs away from the collector's clock. Free-form
`srun_options.nodelist`/`nodefile` overrides and recipe environment overrides of
Slurm allocation metadata or the window contract are also rejected.
`srtctl dry-run` displays the clock placement and custom window environment before submit.

## Artifacts

```text
<log_dir>/<storage_subdir>/
├── manifest.json
├── samples.csv
└── windows/
    └── <benchmark-result-stem>.json
```

`samples.csv` has the exact header
`schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w,gpu_util_pct,sm_active`,
one row per observation, `(scrape_seq, hostname, gpu_index)` unique. Rows are
never interpolated, averaged, or role-attributed — role and heterogeneous
group live once in the manifest topology.

`manifest.json` records producer identity (version, git commit, exporter image
and its SHA-256), the sample interval, expected and observed device sets, the
topology mapping, the expected window list, the SHA-256 of the finalized
`samples.csv` bytes, terminal status, per-window coverage validation, and
reason codes. `status` is the lifecycle outcome;
`publication_valid` is the separate publication gate. Reason codes are stable
machine-readable strings enumerated in `srtctl/core/power/contract.py`.
The digest is required for offline publication validation, so packages created
before `samples_sha256` was recorded cannot be certified by this validator.

A window file records the formal benchmark boundaries on the head-node Unix
clock plus a monotonic `duration`, and points at the benchmark result it
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
