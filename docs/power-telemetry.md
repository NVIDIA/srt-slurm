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
- Only `DCGM_FI_DEV_POWER_USAGE` is parsed. Device identity comes from the
  `gpu` and `UUID` labels.
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
  type: sa-bench          # window stamping ships with the benchmark adapter
  client_placement: head  # keeps sample and window clocks on one host
  isl: 8192
  osl: 1024
  concurrencies: [4]

telemetry:
  enabled: true
  provider: dcgm-power
  default_frequency: 1.0            # seconds between collector cycles; must be <= 3.0
  storage_subdir: power             # relative to the run log directory
  required: true                    # exit non-zero when artifacts are unpublishable
  startup_timeout_seconds: 30
  request_timeout_seconds: 2
  collector_join_timeout_seconds: 10
  dcgm_exporter:
    container_image: dcgm-exporter  # alias, path, or registry URI
    port: 9401
```

`dcgm-power` needs **only** `dcgm_exporter`. Unlike `provider: scraper` it does
not require the top-level `container_image` or a `node_exporter`, because the
collector runs inside srtctl. Config loading validates the block and rejects
inconsistent values with actionable messages; in particular
`default_frequency` must not exceed the 3-second max sample gap the validator
accepts, or every window would fail `sample_gap_exceeded`. Telemetry stays
disabled by default and existing `provider: scraper` recipes are unchanged.

## Artifacts

```text
<log_dir>/<storage_subdir>/
├── manifest.json
├── samples.csv
└── windows/
    └── <benchmark-result-stem>.json
```

`samples.csv` has the exact header
`schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w`,
one row per observation, `(scrape_seq, hostname, gpu_index)` unique. Rows are
never interpolated, averaged, or role-attributed — role and heterogeneous
group live once in the manifest topology.

`manifest.json` records producer identity (version, git commit, exporter image
and its SHA-256), the sample interval, expected and observed device sets, the
topology mapping, the expected window list, terminal status, per-window
coverage validation, and reason codes. `status` is the lifecycle outcome;
`publication_valid` is the separate publication gate. Reason codes are stable
machine-readable strings enumerated in `srtctl/core/power/contract.py`.

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

The artifact package is self-describing; the manifest supplies the expected
topology, never the verdict:

```bash
srtctl-validate-power \
  --power-dir outputs/12345/logs/power \
  --result-root outputs/12345/logs \
  --expect-role prefill=4 --expect-role decode=4 \
  --require-distinct-het-groups
```

It exits `0` when the package is publishable and `1` otherwise, printing each
failing reason code. The `--expect-*` flags optionally assert an expected job
shape for hardware canaries.
