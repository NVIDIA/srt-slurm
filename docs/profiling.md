# Profiling

srtctl supports two profiling backends for performance analysis: **Torch Profiler** and **NVIDIA Nsight Systems (nsys)**.

## Table of Contents

- [Quick Start](#quick-start)
- [Profiling Modes](#profiling-modes)
- [Configuration Options](#configuration-options)
  - [Top-level profiling section](#top-level-profiling-section)
  - [Parameters](#parameters)
- [Constraints](#constraints)
- [How It Works](#how-it-works)
- [Example Configurations](#example-configurations)
- [Output Files](#output-files)
  - [Viewing Results](#viewing-results)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Add a `profiling` section to your job YAML:

```yaml
# For disaggregated mode (prefill_nodes + decode_nodes)
profiling:
  type: "torch" # or "nsys"
  prefill:
    start_step: 0
    stop_step: 50
  decode:
    start_step: 0
    stop_step: 50
# For aggregated mode (agg_nodes)
# profiling:
#   type: "torch"
#   aggregated:
#     start_step: 0
#     stop_step: 50
```

## Profiling Modes

| Mode    | Description                                                      | Output                                         |
| ------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| `none`  | Default. No profiling, uses `dynamo.sglang` for serving          | -                                              |
| `torch` | PyTorch Profiler. Good for Python-level and CUDA kernel analysis | `/logs/profiles/{mode}/` (Chrome trace format) |
| `nsys`  | NVIDIA Nsight Systems. Low-overhead GPU profiling                | `/logs/profiles/{mode}/` (`*.nsys-rep`)        |

## Configuration Options

### Top-level `profiling` section

```yaml
profiling:
  type: "torch" # Required: "none", "torch", or "nsys"

  # nsys / nsys-time: extra arguments for nsys profile (e.g. ["--stats=true"])
  extra_nsys_args: []  # Optional

  # Disaggregated mode: must set both prefill and decode sections
  prefill:
    start_step: 0 # Step to start profiling for prefill workers
    stop_step: 50 # Step to stop profiling for prefill workers
  decode:
    start_step: 0 # Step to start profiling for decode workers
    stop_step: 50 # Step to stop profiling for decode workers


  # Aggregated mode: must set aggregated section (and must NOT set prefill/decode)
  # aggregated:
  #   start_step: 0   # Step to start profiling for aggregated workers
  #   stop_step: 50   # Step to stop profiling for aggregated workers
```

### Parameters

| Parameter               | Description                                   | Default  |
| ----------------------- | --------------------------------------------- | -------- |
| `prefill.start_step`    | Step number to begin prefill profiling        | `0`      |
| `prefill.stop_step`     | Step number to end prefill profiling          | `50`     |
| `decode.start_step`     | Step number to begin decode profiling         | `0`      |
| `decode.stop_step`      | Step number to end decode profiling           | `50`     |
| `aggregated.start_step` | Step number to begin aggregated profiling     | `0`      |
| `aggregated.stop_step`  | Step number to end aggregated profiling       | `50`     |

## Constraints

Profiling has specific requirements:

1. **Disaggregated mode**: When profiling disaggregated workers, both `profiling.prefill` and `profiling.decode` must be set.

2. **Aggregated mode**: When profiling aggregated workers, `profiling.aggregated` must be set (and `profiling.prefill`/`profiling.decode` must not be set).

## How It Works

### Normal Mode (`type: none`)

- Uses `dynamo.sglang` module for serving
- Standard disaggregated inference path

### Profiling Mode (`type: torch` or `nsys`)

- Uses `sglang.launch_server` module instead
- The `--disaggregation-mode` flag is automatically skipped (not supported by launch_server)
- Profiling script (`/scripts/profiling/profile.sh`) runs on leader nodes
- Sends requests via `sglang.bench_serving` to generate profiling workload

### nsys-specific behavior

When using `nsys`, workers are wrapped with:

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  -c cudaProfilerApi --capture-range-end stop \
  [extra_nsys_args...] \
  -o /logs/profiles/{mode}/{name} \
  python3 -m sglang.launch_server ...
```

You can pass extra arguments via `profiling.extra_nsys_args` (e.g. `["--stats=true", "--trace=osrt"]`).

### nsight-slurm mode (`type: nsight-slurm`)

[Nsight Slurm](https://gitlab-master.nvidia.com/mhallock/nsight-cloud-slurm) ("Nsight Cloud for
Slurm") coordinates Nsight Systems across a whole Slurm job: one coordinator per job, a connector
as the task entrypoint on every rank, time-aligned per-rank reports uploaded into a shared report
workspace with a manifest and common collection ids. With `type: nsight-slurm` srtctl launches
every worker step through the wrapper instead of prefixing the command with `nsys profile`:

```yaml
profiling:
  type: nsight-slurm
  nsight_slurm_home: /lustre/.../tools/nsight-slurm   # uv tool install root (bin/nsight-slurm, bin/nsight-slurm-connector)
  nsight_slurm_tool_path: /usr/local/bin/nsys        # nsys inside the worker container (default)
  nsight_slurm_profiling_mode: cuda-api               # at-launch | manual | cuda-api (default)
  # nsight_slurm_tool_options: [...]                  # nsys profile options; default = the playbook set
  prefill: {start_step: 1200, stop_step: 1300}        # cuda-api window (TLLM_PROFILE_START_STOP), as for type: nsys
  decode:  {start_step: 6000, stop_step: 6600}
```

What srtctl does, once, before the first worker step (all through the wrapper's CLI, logged to
`<log_dir>/nsight-slurm.out`): `configure tool-path`, `configure tool-command profile`,
`configure profiling-mode`, `configure tool-options ...`, `configure report-output
<log_dir>/nsight-slurm-reports`, `disable pyxis`, `coordinator start`. Every worker `srun` then
becomes `<home>/bin/nsight-slurm srun <native srun options> -- bash -c ...`; the wrapper re-execs
`srun` with `nsight-slurm-connector` in front of the application and appends its own
`--export=ALL`, so srtctl passes the task-environment exports (`ENROOT_REMAP_ROOT`) through the
wrapper's process environment instead of `--export`. The coordinator is stopped in the sweep's
cleanup after the worker steps are gone. `SLURM_SUBMIT_DIR` is redirected to the run's log dir for
the wrapper invocations, so its job state lands in `<log_dir>/.nsight-slurm/jobs/<job>/`.

Two details srtctl handles itself instead of relying on the wrapper's defaults:

- **Container mounts.** srtctl does not use `nsight-slurm enable pyxis`. That mode appends a second
  `--container-mounts` flag, and pyxis applies only the last `--container-mounts` it receives (SPANK
  options reach `slurmstepd` as one environment variable per option), which silently drops the job's
  own `/model`, `/logs` and `/configs` mounts. Instead srtctl adds, inside its single
  `--container-mounts` value: the install root read-only (venv and managed Python the connector's
  shebang points at), the connector file onto `/usr/local/bin/nsight-slurm-connector` so the bare
  command name the wrapper prefixes resolves on the image's `PATH`, and the log dir at its host path
  (the wrapper's job config dir and report root live under it and are referenced by absolute path).
  The connector's runtime files go to the container-local `/tmp`.
- **Coordinator address.** The wrapper publishes `tcp://${SLURMD_NODENAME}:<port>`; broker and
  client run ZMQ with `IPV6=1`, and ZMQ connects only to the first address a name resolves to. A
  node's own hostname can resolve to an unconnectable link-local IPv6 first, so the ranks on the
  coordinator's node would time out. srtctl sets `SLURMD_NODENAME=<head node IPv4>` for the
  wrapper CLI calls, making the published address an IP literal.
- **Where the reports end up.** In cuda-api mode the connector keeps the nsys session open after a
  capture range (`--capture-range-end repeat`) and writes nsys output into its runtime workspace,
  copying it to the report root only when its state machine sees a collection stop. With the nsys
  2026.3 agent the range states (`RangeCollection`, `RangeGeneration`) are not recognised by
  connector 1.5.0, so nothing is copied and a container-local workspace would vanish with the step.
  srtctl therefore configures the runtime workspace under `<log_dir>/nsight-slurm-runtime` (shared
  filesystem), and at teardown `flush_nsight_slurm()` runs as a pre-cleanup hook of the process
  registry, before any path kills the worker steps: `nsight-slurm stop --job`, then rescue-copy of
  `*.nsys-rep` from the runtime workspaces into `<report_root>/rescued/`, then SIGTERM to the
  wrapper processes so nsys can finalise, then a bounded wait for the report files to settle.

Requirements: `nsight_slurm_home` must be on a filesystem the compute nodes see at the same path
and built for the compute architecture (the connector runs inside the container; install with the
tool's `INSTALLER` or `uv tool install` from a compute node of the target arch); `nsys` present in
the image at `nsight_slurm_tool_path`; TCP from the compute nodes to the orchestrator node's
dynamically chosen coordinator ports. `cuda-api` mode requires the TRT-LLM backend (the PyExecutor
iteration trigger); use `at-launch` or `manual` (`nsight-slurm start/stop --job <id>`) elsewhere.
Connector-owned flags (`-o`, `--force-overwrite`, capture range, `--start-later`) are rejected in
`nsight_slurm_tool_options`. The block is independent of `observability.enabled`.

## Example Configurations

### Torch Profiler (Recommended for Python analysis)

```yaml
name: "profiling-torch"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 1
  prefill_workers: 1
  decode_workers: 1
  gpus_per_node: 4

profiling:
  type: "torch"
  prefill:
    start_step: 0
    stop_step: 50
  decode:
    start_step: 0
    stop_step: 50

backend:
  sglang_config:
    prefill:
      kv-cache-dtype: "fp8_e4m3"
      tensor-parallel-size: 4
    decode:
      kv-cache-dtype: "fp8_e4m3"
      tensor-parallel-size: 4
```

### Nsight Systems (Recommended for GPU kernel analysis)

```yaml
profiling:
  type: "nsys"
  prefill:
    start_step: 10
    stop_step: 30
  decode:
    start_step: 10
    stop_step: 30
```

## Output Files

After profiling completes, find results in the job's log directory:

Torch profiler traces example:

```text
logs/{job_id}_{workers}_{timestamp}/
└── profiles/
    ├── prefill/
    │   └── *.json
    └── decode/
        └── *.json
```

Nsight Systems (nsys) reports example:

```text
logs/{job_id}_{workers}_{timestamp}/
├── profile_all.out         # Unified profiling script output
└── profiles/
    ├── prefill/            # Nsys reports (if type: nsys)
    │   └── *.nsys-rep
    └── decode/
        └── *.nsys-rep
```

### Viewing Results

**Torch Profiler traces:**

- Open in Chrome: `chrome://tracing`
- Or use TensorBoard: `tensorboard --logdir=logs/.../profiles/`

**Nsight Systems reports:**

- Open with NVIDIA Nsight Systems GUI
- Or CLI: `nsys stats logs/.../profiles/decode/<name>.nsys-rep`

## Troubleshooting

### Validation errors about profiling sections

- Disaggregated mode requires both `profiling.prefill` and `profiling.decode` to be set.
- Aggregated mode requires `profiling.aggregated` to be set (and `profiling.prefill`/`profiling.decode` must not be set).

### Empty profile output
Ensure the benchmark workload is generating requests during the profiling window.

### Profile too short/long

Adjust `start_step` and `stop_step` to capture the desired range. A typical profiling run uses 30-100 steps.
