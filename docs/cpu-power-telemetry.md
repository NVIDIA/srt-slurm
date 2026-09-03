# CPU Power Telemetry

Host-side per-socket CPU power collection for NVIDIA Grace nodes, added on the `kylliang/power_study_20260901` branch alongside GPU DCGM power telemetry and audited GPU power limits.

## Table of Contents

- [Overview](#overview)
- [Enabling It](#enabling-it)
- [How It Starts](#how-it-starts)
- [Collection Sources](#collection-sources)
- [DCGM Python Bindings](#dcgm-python-bindings)
- [Output Format](#output-format)
- [Computing Total Energy Over a Run](#computing-total-energy-over-a-run)
- [Relationship to GPU Power Telemetry](#relationship-to-gpu-power-telemetry)

---

## Overview

`srtctl.core.cpu_power` is a standalone collector script (not a persistent system daemon) that srtslurm launches via `srun`, one task per allocated worker node, for the lifetime of a job. It samples per-socket CPU power on the host (not inside the model container, since it needs sysfs and the host DCGM install) and writes CSV artifacts under the run's telemetry directory.

It follows BTK's CPU-power source ordering for Grace: Linux ACPI `power_meter` CPU rails first, falling back to DCGM CPU entity field `1130` (`CPU_POWER_FIELD_ID`).

## Enabling It

Two flags must both be `true` in the recipe YAML — `telemetry.cpu_power.enabled` is not implied by `telemetry.enabled` alone:

```yaml
telemetry:
  enabled: true              # master switch; also gates GPU DCGM power telemetry
  cpu_power:
    enabled: true              # must be explicit
    source: auto                 # "auto" | "acpi" | "dcgm"
    sample_interval_seconds: 0.1
    startup_timeout_seconds: 30.0
    required: false               # if true, failed CPU power readiness blocks the benchmark stage
```

Config lives in `CpuPowerConfig` (`src/srtctl/core/schema.py`), nested under `TelemetryConfig.cpu_power`.

## How It Starts

`start_cpu_power_telemetry()` in `src/srtctl/cli/mixins/telemetry_stage.py` is called from `do_sweep.py` during job startup (alongside the tachometer and GPU DCGM exporter). It builds one `srun` task per worker node (or per het-group chunk) running:

```bash
srun --nodes=<N> --ntasks=<N> --nodelist=<nodes> \
     --output=<log_dir>/telemetry_cpu_power.%N.out \
     [--het-group=<id>] \
     python -m srtctl.core.cpu_power \
       --output-dir <run_dir>/telemetry/power/cpu/nodes \
       --ready-dir <run_dir>/telemetry/power/cpu/ready \
       --source auto \
       --interval-seconds 0.1
```

`use_bash_wrapper=False` — it runs directly on the host, no container wrapper. There is no separate `srtctl` subcommand for this; it's wired into the sweep lifecycle, not user-invocable directly.

Each node's process writes a `.ready.json` (or `.error.json` on failure) once its reader is initialized. `CpuPowerTelemetrySession.wait_for_readiness()` blocks until all expected nodes report ready or `startup_timeout_seconds` elapses. At job teardown, processes receive `SIGTERM`, `stop_and_finalize()` merges all per-node CSVs into one `samples.csv`, and writes a `manifest.json` with status/reason codes.

## Collection Sources

- **`AcpiPowerMeterReader`** — reads Linux ACPI `power_meter` hwmon sysfs channels matching `CPU Power Socket N`. Pure file reads, no DCGM dependency.
- **`DcgmCpuPowerReader`** — reads DCGM field 1130 for `DCGM_FE_CPU` entities. Connects via `pydcgm.DcgmHandle(ipAddress=None)`, which starts DCGM in **embedded** mode (loads `libdcgm.so` in-process) rather than dialing a separate `nv-hostengine` daemon — no system daemon needs to be running or managed by srtslurm.

  Fixed in `c1c0028c` ("fix: collect Grace CPU power through watched DCGM fields"): DCGM's live-data flag does not implicitly install a watch, so an explicit `DcgmGroup` + `DcgmFieldGroup` watch on field 1130 is created (`WatchFields`, 100ms freq / 60s max age / 600 samples), followed by a forced `dcgmUpdateAllFields` so the first sample is real instead of stale/empty.

`source: auto` tries ACPI first, then DCGM.

## DCGM Python Bindings

DCGM's Python bindings (`dcgm_agent`, `dcgm_fields`, `dcgm_structs`, `pydcgm`) are **not pip-installable**. They ship as part of the DCGM system package (`datacenter-gpu-manager`), installed on disk at a fixed path — not in Python site-packages. `_add_standard_dcgm_binding_path()` in `cpu_power.py` searches known install locations:

- `/usr/share/datacenter-gpu-manager-4/bindings/python3`
- `/usr/local/dcgm/bindings/python3`

and inserts the first match containing `dcgm_agent.py` onto `sys.path` before importing. If DCGM isn't installed on the node, the import fails and `DcgmCpuPowerReader` raises `CpuPowerSourceUnavailable`, which `collect()` handles by writing a `.error.json` and letting `source: auto` fall through (or failing the session if `dcgm` was forced).

Binding layers:

| Module | Role |
|---|---|
| `dcgm_agent` | Thin ctypes wrappers around raw C API calls |
| `dcgm_fields` | Field ID / entity type constants (`CPU_POWER_FIELD_ID = 1130`, `DCGM_FE_CPU`, ...) |
| `dcgm_structs` | ctypes struct/enum definitions for marshaling |
| `pydcgm` | Higher-level OOP wrappers (`DcgmHandle`, `DcgmGroup`, `DcgmFieldGroup`) used by `cpu_power.py` |

## Output Format

`samples.csv` under `<run_dir>/<telemetry.storage_subdir>/cpu/` has header:

```
schema_version, timestamp_unix, timestamp_local, hostname, source, sensor, socket_id, power_w, total_power_w
```

- **`schema_version`** — `2` as of the addition of `timestamp_local` (previously `1`, an 8-column row without it).
- **`timestamp_local`** — ISO 8601 wall-clock time with UTC offset for the same sample, e.g. `2026-09-03T14:32:07.891234-07:00`, derived from `timestamp_unix` via `datetime.fromtimestamp(ts).astimezone()` (the collector process's local timezone). Exists so a consumer that only has a benchmark log's local `HH:MM:SS`-style timestamps (no timezone) can read the real UTC offset for that node straight from this column instead of guessing the cluster's timezone.
- **`power_w`** — one socket's power reading. `sensor` names look like `CPU0:cpuPowerUsageW` — granularity is per-socket, not per-core.
- **`total_power_w`** — sum of `power_w` across all sockets on that node, computed once per timestamp and duplicated on every sensor row at that timestamp. It is instantaneous power, not a running/cumulative energy total.

A `manifest.json` and per-node `*.metadata.json` (sensor provenance, driver/semantics info) accompany the samples.

## Computing Total Energy Over a Run

The collector intentionally never integrates power into energy — same philosophy as the GPU power artifact contract (`src/srtctl/core/power/__init__.py`: *"It never integrates power into energy; that belongs to consumers of the artifact contract."*). To get run-total energy:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("samples.csv")

# total_power_w repeats across every sensor row for the same (hostname, timestamp);
# dedupe before integrating or sockets get double-counted.
per_node_ts = (
    df[["hostname", "timestamp_unix", "total_power_w"]]
    .drop_duplicates(subset=["hostname", "timestamp_unix"])
    .sort_values(["hostname", "timestamp_unix"])
)

def energy_joules(group: pd.DataFrame) -> float:
    return float(np.trapezoid(group["total_power_w"], x=group["timestamp_unix"]))

energy_per_node_j = per_node_ts.groupby("hostname").apply(energy_joules)
run_total_wh = energy_per_node_j.sum() / 3600
```

Use trapezoidal integration (`np.trapezoid`; `np.trapz` was removed in numpy 2.0), not `mean(power) * duration` — the sample loop is not perfectly uniform (`time.sleep(max(0, next_sample - now))`), and read failures leave gaps. For per-socket energy instead of per-node, group by `(hostname, socket_id)` on `power_w` instead of `total_power_w`.

## Relationship to GPU Power Telemetry

GPU power telemetry (`start_gpu_power_telemetry`, same mixin) works differently: it launches an actual **DCGM exporter container** (`telemetry_dcgm_exporter`) as a sidecar via `_start_exporter_container`, i.e. a real DCGM Prometheus-exporter process scraped over its own protocol. CPU power skips the exporter and talks to DCGM directly in-process via the Python bindings — a lighter-weight, embedded approach rather than a scraped-service approach.

GPU power *limits* (apply/restore audited caps, `src/srtctl/core/gpu_power_limit.py`, added in `dc9a5fcb`) are a separate, unrelated top-level config (`gpu_power_limits`) — not part of `telemetry.cpu_power`.
