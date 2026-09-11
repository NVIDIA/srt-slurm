# CPU Power Telemetry

Host-side CPU power collection for NVIDIA Grace nodes, run alongside GPU DCGM
power telemetry as an independent, best-effort leg.

## Table of Contents

- [Overview](#overview)
- [Enabling It](#enabling-it)
- [How It Starts](#how-it-starts)
- [Collection Sources](#collection-sources)
- [Output Format](#output-format)
- [Computing Total Energy Over a Run](#computing-total-energy-over-a-run)
- [Relationship to GPU Power Telemetry](#relationship-to-gpu-power-telemetry)
- [Alternative: Host-Side Python Collector](#alternative-host-side-python-collector)

---

## Overview

CPU power collection is a scrape-based, head-node-orchestrated design, not an
in-job daemon per node writing its own files. On each worker node, srtctl
launches a small HTTP exporter process directly on the bare host (outside the
model container, so it can read host power interfaces) that serves a
Prometheus `/metrics` endpoint. A single collector thread on the head node
(`CpuPowerCollector`, `src/srtctl/core/power/cpu_session.py`) polls every
worker's exporter on a fixed interval, parses the scrape body, and appends
rows to one shared `cpu/samples.csv` for the whole run.

The leg is fully decoupled from the GPU DCGM power pipeline's lifecycle and is
always best-effort: an unresolvable node, an unreachable exporter, a malformed
scrape, or a wedged collector thread is absorbed and logged, never raised into
the benchmark. There is no `required` flag for CPU power — gaps in
`samples.csv` are the visible cost of a failure, not a blocked run.

## Enabling It

Presence of `telemetry.cpu_power_exporter` (not a separate `enabled` flag)
turns CPU power collection on:

```yaml
telemetry:
  enabled: true                # master switch; also gates GPU DCGM power telemetry
  cpu_power_exporter:
    port: 9405                 # default; exporter listen port / collector scrape port
    source: auto                # "auto" | "acpi" | "dcgm", passed to the exporter binary
```

`telemetry.enabled: true` no longer requires `dcgm_exporter` — a recipe may
configure `cpu_power_exporter` alone with no DCGM leg at all. The sampling
cadence and timeouts (`collect_interval_ms`, `request_timeout_seconds`,
`startup_timeout_seconds`, `collector_join_timeout_seconds`) are shared with
the DCGM leg on `TelemetryConfig`; there is no separate CPU-specific set.

Config lives in `CpuPowerExporterConfig` (`src/srtctl/core/schema.py`), nested
under `TelemetryConfig.cpu_power_exporter`. Port-collision validation
(against `dcgm_exporter`, `observability.tachometer`'s exporters, and any
Dynamo system port) happens in `SrtConfig._validate_cpu_power_exporter`.

## How It Starts

`start_cpu_power_telemetry()` in `src/srtctl/cli/mixins/telemetry_stage.py` is
called from the sweep startup path alongside the tachometer and GPU DCGM
exporter. It resolves the exporter binary, then launches one `srun` task per
worker node (or per het-group chunk, `use_bash_wrapper=False` — bare host, no
container):

```bash
srun --nodes=<N> --ntasks=<N> --nodelist=<nodes> \
     --output=<log_dir>/telemetry_cpu_power_exporter.%N.out \
     [--het-group=<id>] \
     <cpu-power-exporter binary> --port 9405 --source auto
```

The binary is resolved via `_resolve_bundled_binary("cpu-power-exporter")` — a
Rust binary installed by `make setup`. When that binary is absent or not
executable, srtctl falls back to a Python stdlib exporter
(`python3 -m srtctl.core.cpu_power_exporter`), which is ACPI-only and has no
`--source` flag; a non-`auto` `source` request logs a warning in that case
instead of being silently dropped.

Once the exporter tasks are launched, `CpuPowerCollector.start()` resolves
each worker's IP (`get_hostname_ip`, respecting `runtime.network_interface`),
opens the `cpu/samples.csv` writer, and starts a background thread that polls
every endpoint's `/metrics` every `collect_interval_ms` and appends parsed rows.
Any launch failure for the exporter tasks themselves is caught and logged; the
collector object is still returned (with whatever endpoints did resolve) so
the caller doesn't have to special-case a partial launch.

At job teardown, `stop_and_finalize()` stops the collector thread, closes the
CSV writer, and writes `cpu_manifest.json` (non-authoritative: per-node
scrape/error counts and the resolved source mode, for debugging — the CSV is
the source of truth).

## Collection Sources

The exporter binary itself decides ACPI vs. DCGM per its own `--source` flag:

- **`acpi`** — reads Linux ACPI `power_meter` hwmon sysfs channels. Reports
  per-channel detail: `cpu_rail`, `soc`, `dram`, and (where firmware exposes
  it) a `total`-kind rail per socket. Domain names vary by platform (e.g.
  "Grace Power Socket 0" vs. a generic "Total Power socket 0", some suffixed
  with "in uW"); the exporter classifies all known variants into these kinds.
- **`dcgm`** — reads DCGM CPU entity power directly, one already-aggregated
  value per socket.
- **`auto`** (default) — tries DCGM first, falls back to ACPI when DCGM is
  unavailable or reports no CPU entities.

The exporter resolves this once at process startup and serves only one metric
family (`cpu_power_dcgm_watts` or `cpu_power_acpi_watts`) for its lifetime.
Client-side parsing (`src/srtctl/core/power/cpu_parser.py`) prefers ACPI
readings if a scrape body ever contained both, since ACPI carries more detail.

## Output Format

`samples.csv` under `<log_dir>/<telemetry.storage_subdir>/cpu/` has header:

```
schema_version, timestamp_unix, hostname, source, sensor, socket_id, power_w, total_power_w
```

- **`power_w`** — one sensor's power reading for that scrape. `sensor` names
  look like `CPU0:cpuPowerUsageW` (ACPI) or a DCGM field label; granularity is
  per-socket.
- **`total_power_w`** — the node-level total for that scrape, duplicated on
  every sensor row at the same `(hostname, timestamp_unix)`. In DCGM mode this
  is the sum of the per-socket DCGM values. In ACPI mode it is **not** a sum of
  the `cpu_rail`-, `soc`-, and `dram`-kind rails: whenever a `total`-kind
  channel exists for a socket, that channel alone is the total. Real hardware
  traces show the `total` rail at roughly 93-104W against `cpu_rail`+`soc`
  combined at roughly 53-58W for the same socket — `total` measures the whole
  Grace SoC power boundary, not literally `cpu_rail + soc`. **When no
  `total`-kind channel is present for a scrape, `total_power_w` is left
  blank** for every row from that scrape rather than guessed from the
  component rails; per-sensor `power_w` values are still populated. Consumers
  reading this CSV (e.g.
  `srtctl.analysis.power_energy_report.load_cpu_samples`) must skip blank
  `total_power_w` rows rather than treat them as `0`.

`cpu_manifest.json` alongside it is non-authoritative debugging metadata:
per-node scrape/error counts and the resolved source mode, plus start/stop
timestamps and the producer's git commit.

## Computing Total Energy Over a Run

The collector intentionally never integrates power into energy — same
philosophy as the GPU power artifact contract
(`src/srtctl/core/power/contract.py`: it never integrates power into energy;
that belongs to consumers of the artifact contract). To get run-total energy:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("samples.csv")
df = df[df["total_power_w"] != ""]  # skip scrapes with no total-kind channel

# total_power_w repeats across every sensor row for the same (hostname, timestamp);
# dedupe before integrating or sockets get double-counted.
per_node_ts = (
    df[["hostname", "timestamp_unix", "total_power_w"]]
    .drop_duplicates(subset=["hostname", "timestamp_unix"])
    .sort_values(["hostname", "timestamp_unix"])
)

def energy_joules(group: pd.DataFrame) -> float:
    return float(np.trapezoid(group["total_power_w"].astype(float), x=group["timestamp_unix"]))

energy_per_node_j = per_node_ts.groupby("hostname").apply(energy_joules)
run_total_wh = energy_per_node_j.sum() / 3600
```

Use trapezoidal integration (`np.trapezoid`; `np.trapz` was removed in numpy
2.0), not `mean(power) * duration` — the scrape loop is not perfectly uniform,
and scrape failures leave gaps. For per-sensor energy instead of per-node,
group by `(hostname, sensor)` (or `(hostname, socket_id)`) on `power_w`
instead of `total_power_w`.

## Relationship to GPU Power Telemetry

GPU power telemetry (`start_gpu_power_telemetry`, same mixin) works
similarly in shape — an exporter process per worker node scraped by a
head-node collector — but the exporter is a containerized DCGM exporter
sidecar (`telemetry.dcgm_exporter`, launched via `_start_exporter_container`)
rather than a bare-host process, and it is not best-effort by default:
`telemetry.required` (which applies to the DCGM leg) can fail the benchmark
stage if publishable GPU power artifacts can't be produced. CPU power has no
equivalent `required` semantics; it is always best-effort.

GPU power *limits* (apply/restore audited caps, `src/srtctl/core/gpu_power_limit.py`)
are a separate, unrelated top-level config (`gpu_power_limits`) — not part of
`telemetry.cpu_power_exporter`.

## Alternative: Host-Side Python Collector

`telemetry.cpu_power` is a second, independent CPU power leg that predates the
scraper design. Instead of an exporter plus a head-node poller, srtctl launches
`python3 -m srtctl.core.cpu_power` directly on the bare host of every worker
node. Each collector reads Linux ACPI `power_meter` hwmon channels (or DCGM CPU
entity field 1130) itself, writes its own per-node CSV under
`<storage_subdir>/nodes/`, and drops a ready marker. At teardown the head node
(`CpuPowerTelemetrySession`, `src/srtctl/core/cpu_power_session.py`) merges the
node CSVs into `<storage_subdir>/samples.csv` and writes `manifest.json`.

```yaml
telemetry:
  enabled: true
  cpu_power:
    enabled: true              # presence alone is not enough; this flag turns the leg on
    source: auto               # "auto" (ACPI then DCGM, best-effort) | "acpi" | "dcgm" (mandatory)
    sample_interval_seconds: 0.1
    startup_timeout_seconds: 30.0
    required: false            # true fails the job if the leg is not ready or not publishable
    storage_subdir: cpu_power  # must differ from telemetry.storage_subdir
```

Differences from `cpu_power_exporter`:

- **Fail-closed is available.** `required: true` blocks the formal benchmark
  when collectors do not become ready on every node, and turns an
  unpublishable result into a nonzero exit code. The scraper leg has no
  equivalent.
- **No network hop.** Readings never leave the node until aggregation, so
  there is no port to reserve and no exporter binary to install.
- **Separate artifacts.** Output lands in `cpu_power/` by default, with an
  extra `timestamp_local` column, not in `power/cpu/`. The energy report
  (`python -m srtctl.analysis.power_energy_report <log_dir>`) discovers either
  location; when a run has both, pass `--cpu-samples <path>` to pick one.
- **Per-socket utilization (DCGM source only).** Alongside power field 1130
  the DCGM reader watches CPU entity fields 1100-1104 and appends five
  columns to every sample row: `cpu_util_total`, `cpu_util_user`,
  `cpu_util_nice`, `cpu_util_sys`, `cpu_util_irq`, reported by DCGM as a
  fraction of the socket's CPU time. ACPI has no utilization, so those cells
  stay blank. The per-node `*.metadata.json` lists the field ids and unit.
  This bumped the samples schema to v3; v2 readers that select columns by
  name are unaffected.

The energy report summarizes utilization per concurrency window as a mean and
max of the samples inside the window, per socket and per node (and for the GPU
leg's `gpu_util_pct`/`sm_active`, per GPU, node, and role). It is reported next
to the joules, never integrated, and a window with no utilization samples is a
warning rather than an error. Note that `sm_active` is only populated when the
DCGM exporter is configured to emit `DCGM_FI_PROF_SM_ACTIVE`; the default
counter set does not include it.

Each window also carries **perf/W** and a **timing comparison**:

- `perf_per_watt`: output and total tokens/s over the computed window, GPU
  average watts and CPU+GPU combined average watts (joules / duration), and
  the four ratios tokens/s per watt. These are the reciprocal of the J/token
  figures and are computed from the same window. The combined variants are
  null, with a warning, unless both a CPU and a GPU leg produced samples, so
  they never silently equal the GPU-only number.
- `timing.computed`: the window the report integrates over (aiperf: earliest
  request start to latest request end across profiling records; sa-bench: its
  wall-clock start/end). `timing.reported`: the benchmark's own account, for
  comparison only, never validation -- sa-bench's `duration`, aiperf's
  aggregate `benchmark_duration`/`start_time`/`end_time`, or, failing that,
  the profiling-phase NOTICE lines in `benchmark.out` when exactly one phase
  ran (time-of-day only, so no absolute start/end). `timing.coverage`: the
  first and last power samples the trapezoid actually spanned. Every
  breakdown row records its own `sample_start_unix`/`sample_end_unix`/`samples`.
- **Power statistics per breakdown row** (socket, GPU, node, role):
  `avg_power_w` is time-weighted (joules / window duration) and therefore
  consistent with the energy figure even under uneven sampling; `mean_w` is
  the plain sample mean over the same samples, kept alongside so any gap
  between the two is visible. `min_w`, `p5_w`, `p50_w`, `p95_w`, `p99_w`, and
  `max_w` are sample-based percentiles (numpy linear interpolation) over the
  exact samples the trapezoid spanned; `samples` says how many points they
  rest on. Node and role rows are computed over the *summed* series, so a
  node p99 is the 99th percentile of the node's total power, not a sum of
  per-device percentiles. The text table shows p50/p95/p99/max; JSON has all.

The two legs may be enabled together. They share no ports or directories and
neither one's failure affects the other.
