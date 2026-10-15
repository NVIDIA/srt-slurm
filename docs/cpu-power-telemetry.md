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

- **`acpi`** — reads Linux ACPI `power_meter` hwmon sysfs channels. The driver
  name can be on the hwmon node or under `device/`: `acpi_power_meter` used
  `hwmon_device_register()` through Linux v6.14, which creates no `name` on the
  hwmon class node; v6.15 moved to `hwmon_device_register_with_info()`. Reports
  per-channel detail: `cpu_rail`, `soc`, `dram`, and (where firmware exposes
  it) a `total`-kind rail per socket. Domain names vary by platform (e.g.
  "Grace Power Socket 0" vs. a generic "Total Power socket 0", some suffixed
  with "in uW"); the exporter classifies all known variants into these kinds.
- **`dcgm`** — reads the DCGM CPU-entity power fields. DCGM has no CPU power
  backend of its own: its sysmon module reads the same ACPI hwmon channels by
  `power1_oem_info` label, so each field *is* one ACPI rail:

  | DCGM field | hwmon label it reads | srtctl rail | CSV column |
  | --- | --- | --- | --- |
  | 1130 `DCGM_FI_DEV_CPU_POWER_WATTS` | `CPU Power Socket N` | `cpu_rail` | `power_w` **and** `cpu_rail_w` |
  | 1132 `DCGM_FI_DEV_SYSIO_POWER_UTIL_CURRENT` | `SysIO Power Socket N` | `soc` | `soc_w` |
  | 1131 `DCGM_FI_DEV_CPU_POWER_LIMIT_WATTS` | `Grace Power Socket N` (**cap** only) | — | — |
  | 1133 `DCGM_FI_DEV_MODULE_POWER_UTIL_CURRENT` | `Module Power Socket N` | — (unverified; excluded) | — |

  No DCGM field reports the `Grace Power Socket N` envelope's draw. Field
  1133 does read a usage file, but what "Module" spans on GB200/GB300
  (Grace alone, or the Grace+Blackwell superchip) has not been measured on a
  live node; it is excluded until `dcgmi dmon -e 1130,1131,1132,1133 -i
  cpu:0` alongside `cat /sys/class/hwmon/*/device/power1_oem_info` settles
  it. So
  **DCGM-mode `power_w` is the CPU rail, roughly half of ACPI-mode `power_w`**
  (about 53 W vs 100 W per socket on GB200 reference runs). The energy report
  attaches a "CPU rail only" warning to DCGM-sourced runs; do not compare their
  CPU figures with ACPI-mode runs as like-for-like. The exporter publishes one
  `cpu_power_dcgm_watts{socket,field_id}` sample per field; the collector files
  1130 as `power_w` (as it always has) and also as `cpu_rail_w`, and 1132 as
  `soc_w`. Older exporter builds publish 1130 alone without a `field_id`
  label; the collector treats those samples as 1130.
  Source: NVIDIA/DCGM `modules/sysmon/DcgmSystemMonitor.cpp` (label → file
  map) and `modules/sysmon/DcgmModuleSysmon.cpp` (field id → getter).
- **`auto`** (default) — walks a fixed ladder at startup, most informative
  source first, and each source must prove itself before it is used. Every
  step down is logged at WARN in the exporter's `.out` file with the reason:

  1. **ACPI, live** — `power_meter` hwmon sensors were discovered *and* at
     least one **socket-total** channel (`Grace Power Socket N` / `Total
     Power socket N`) reads a positive value. Discovery alone is not enough:
     a node can expose channels that never report, and publishing them would
     integrate to 0 J and pass for a measurement. Only the total counts
     because it is what becomes `power_w`; a live component rail next to a
     dead envelope would otherwise vouch for a socket whose power can never
     be published. If no total reads positive, the probe is repeated once
     after 1 s (hwmon averages can read 0 on the first poll after boot)
     before ACPI is declared dead.
  2. **DCGM, fields 1130 + 1132** — CPU rail and SysIO.
  3. **DCGM, field 1130 alone** — when this libdcgm refuses 1132 for CPU
     entities (older release, or a sysmon without SysIO). 1130 alone is the
     floor every earlier exporter had; in `auto` DCGM is the last resort, so
     failing here would mean no CPU power at all.

     Either DCGM rung is accepted only if at least one CPU entity then
     returns a positive field 1130 on a first read — that is the value filed
     as `power_w`. SysIO alone does not make DCGM usable: a node where only
     1132 reads would publish `soc_w` for the whole run with no `power_w`
     and no socket rows. An all-`None` read usually means the embedded
     daemon lacks hardware access (running without root while a system
     dcgm-exporter holds the session).
  4. **Exit non-zero** — no source could be established.

  `--source acpi` and `--source dcgm` run only their own rungs (ACPI still
  requires the liveness probe to pass; DCGM still tries 1130+1132 then 1130)
  and exit non-zero instead of stepping down. The Python host collector
  (`srtctl.core.cpu_power`) implements the same ladder.

The exporter resolves this once at process startup and serves only one base
metric family (`cpu_power_dcgm_watts` or `cpu_power_acpi_watts`) for its
lifetime. Client-side parsing (`src/srtctl/core/power/cpu_parser.py`) prefers
ACPI readings if a scrape body ever contained both, since ACPI carries more
detail.

In both modes the exporter also serves superchip module power from NVML:

```
cpu_power_nvml_watts{type="module",socket="0",gpus="0,1",source="nvml"} 498.720000
```

The reading is NVML field 185 (`NVML_FI_DEV_POWER_AVERAGE`, a 1 s average)
at `NVML_POWER_SCOPE_MODULE`, the value `nvidia-smi` prints as
`module.power.draw.average`. The instant field is not read, so the series
never mixes averaging windows. NVIDIA's [Grace Performance Tuning Guide,
Power and Thermals](https://docs.nvidia.com/grace-perf-tuning-guide/power-thermals.html)
lists "Module Power" for the Grace Hopper and Grace Blackwell superchips as
"Total power of the CG1 module, including DRAM power and regulator loss. This
also includes GPU and GPU HBM Power", reported through `nvidia-smi` from
release 570. CG1 names the one-GPU Grace Hopper module. The page does not
say what the meter covers on a two-GPU GB200 module, so that part is
measured below. On GB200 each GPU of a module reports the whole module, not
its own share:

```
module_w = Grace + GPU_a + GPU_b + the rest of the module
```

The exporter therefore publishes the mean of a socket's GPU readings, one
sample per socket. To avoid hiding disagreement in that mean, it withholds
a socket when its GPU readings differ by more than 5% of their mean. It
warns once when that starts and logs recovery at INFO when they agree again.

NVML initialization and reads run in a background thread, so blocked NVML
calls do not delay the HTTP server. The module series is omitted until a
reading is ready; snapshots older than 3 s are withheld. An independent
timer warns once if initialization takes longer than 3 s or readings become
missing or stale, and logs recovery at INFO when fresh readings return.
These messages do not depend on scrapes.

Measured on a GB200 node (driver 580.126.20, kernel 6.8.0-1046-nvidia-64k)
with an FMA kernel drawing about 400 W per loaded GPU, sampling
`nvidia-smi --query-gpu=power.draw.average,module.power.draw.average` and
the ACPI `Grace Power Socket 0` meter together at 1 Hz, 30 to 35 samples per
row:

| Load         | GPU0 / GPU1 `power.draw` (W) | GPU0 / GPU1 `module.power.draw` (W) | Grace socket 0 (W) |
| ------------ | ---------------------------- | ----------------------------------- | ------------------ |
| idle         | 204 / 198                    | 539 / 534                           | 96                 |
| GPU0 only    | 390 / 202                    | 731 / 725                           | 98                 |
| GPU0 + GPU1  | 397 / 400                    | 938 / 933                           | 101                |

With only GPU0 loaded, GPU1's own module reading rose 191 W while its chip
reading did not move, and the gap between the two module readings stayed at
6 W, the same as at idle. A meter reporting each GPU's share would have left
GPU1's reading at idle. The two readings of one module differed by at most
1.1% in a steady phase and 1.9% while load fell. The residual
`module - Grace - GPU_a - GPU_b` held at 38 to 42 W per socket under every
load. The page's regulator-loss and HBM terms are candidates for it, but the
probe did not attribute it.

GPUs are mapped to sockets by the NUMA node of their PCI device, taken as the
ACPI socket number. On GB200 that was tested by loading the CPUs of one NUMA
node at a time: NUMA node 1 (CPUs 72 to 143) raised `CPU Power Socket 1`
from 41 W to 203 W and left socket 0 flat, and NUMA node 0 did the reverse.
GPUs 0 and 1 sit on NUMA node 0, GPUs 2 and 3 on node 1. GB300 is
unverified. The collector writes `module_w` only onto a socket row the CPU
family produced, so the exporter warns once at startup for a module socket
with no CPU socket.

The series is best-effort. When NVML is unavailable the exporter logs the
reason at INFO and omits the series; `module_w` stays blank. NVML is the only
source of `module_w`. A firmware `Module Power Socket N` ACPI meter stays
unclassified. A scrape that carries only the NVML series has no socket power
and writes no rows.

## Output Format

`samples.csv` under `<log_dir>/<telemetry.storage_subdir>/cpu/` holds one row
per socket per scrape. Schema v3 has header:

```
schema_version, timestamp_unix, hostname, source, sensor, socket_id, power_w, cpu_rail_w, soc_w, dram_w, module_w, total_power_w
```

- **`power_w`**: the socket primary reading. ACPI uses the `total` envelope; DCGM uses field 1130, the CPU rail. A socket without its primary reading gets no row.
- **`cpu_rail_w`, `soc_w`, `dram_w`**: component rails, blank when absent. ACPI fills those its firmware exposes; DCGM fills `cpu_rail_w` (= `power_w`, field 1130) and `soc_w` (field 1132). Zero or non-finite readings are missing, never interpreted as idle power. Component rails never substitute for `power_w` or add to `total_power_w`.
- **`total_power_w`**: the sum of `power_w` across the node's sockets, repeated on each socket row. ACPI totals measure the socket envelope; DCGM totals sum CPU rails and are not directly comparable. Hardware
  traces show the `total` rail at roughly 93-104W against `cpu_rail`+`soc`
  combined at roughly 53-58W for the same socket. `total` measures the whole
  Grace SoC power boundary, not literally `cpu_rail + soc`.
- **`module_w`**: superchip module power from NVML, a superset of `power_w`.
  It covers Grace, its GPUs and the rest of the module, and is never
  summed into `total_power_w`. Consumers that want the whole superchip use
  `module_w` and must not add GPU power on top, because the GPUs are already
  inside it. It is blank when no module reading arrived. The row's `source`
  and `sensor` columns describe `power_w`. `module_w` always comes from
  `cpu_power_nvml_watts`.

v1 wrote one row per rail with no rail columns. v2 pivoted to one row per
socket. v3 appended `module_w`. Readers accept all three.

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
df = df[df["total_power_w"] != ""]  # skip scrapes without a node total

# total_power_w repeats across every socket row for the same (hostname, timestamp);
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
and scrape failures leave gaps. For per-socket energy instead of per-node,
group by `(hostname, socket_id)` on `power_w` instead of `total_power_w`.
Superchip energy integrates `module_w` the same way. Do not add GPU energy to
it.

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
entity fields 1130 and 1132) itself, writes its own per-node CSV under
`<storage_subdir>/nodes/`, and drops a ready marker. Its `--source auto`
walks the same ladder as the exporter (ACPI only if some sensor reads
positive, then DCGM 1130+1132, then 1130 alone), logging each step down to
the collector's `.out` file; the per-node `*.metadata.json` records which
DCGM power fields were actually watched (`power_fields[].watched`). At teardown the head node
(`CpuPowerTelemetrySession`, `src/srtctl/core/cpu_power_session.py`) merges the
node CSVs into `<storage_subdir>/samples.csv` and writes `manifest.json`.

```yaml
telemetry:
  enabled: true
  cpu_power:
    enabled: true              # presence alone is not enough; this flag turns the leg on
    source: auto               # "auto" (live ACPI, then DCGM 1130+1132, then 1130) | "acpi" | "dcgm" (mandatory)
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
- **Per-socket utilization (DCGM source only).** Alongside the power fields
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
