# Profiling

srtctl supports two profiling backends for performance analysis: **Torch Profiler** and **NVIDIA Nsight Systems (nsys)**.

## Table of Contents

- [Observability capture](#observability-capture)
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

## Observability capture

To capture the serving timeline alongside metrics and request traces, add:

```yaml
observability:
  enabled: true
```

This starts Nsight Systems on **every launched worker process, every TRT-LLM MPI
rank, and every Dynamo frontend**. Every profiler session records both NVTX
ranges and CPU samples by default.
The preset does not collect CUDA API or GPU kernel events. Use the explicit
`profiling` modes below for those domains.

By default, capture starts **after warmup** and stops **when the measured
workload finishes**. There is no delay or fixed end time to estimate. The
benchmark waits for every frontend and worker rank to acknowledge start before
sending measured traffic, and waits for report export at stop. Applications
remain running between capture windows. Each concurrency in a sweep gets a
separate report.

Optional settings are:

```yaml
observability:
  enabled: true
  nsys:
    enabled: true
    capture_window: measured_workload  # default: after warmup through workload completion
    report_timeout_secs: 1800          # control/report-finalization budget, not capture length
    # nvtx_injection_path: /opt/nsys/target-linux-sbsa/libToolsInjection64.so
```

Set `capture_window: including_startup` to include initialization and warmup:
collection then begins at each process launch and stops at teardown.

**NVTX tracing and CPU sampling are separate.**
[NVTX ranges](https://nvidia.github.io/NVTX/python/annotation_types.html#ranges)
are named start/end annotations emitted by instrumented application code. They
show when an operation ran and its elapsed duration; that duration can include
waiting, so a long range alone does not prove the CPU was busy.
[CPU sampling](https://docs.nvidia.com/nsight-systems/UserGuide/#cpu-profiling-on-linux)
periodically records instruction pointers and call stacks, helping identify
functions that consume CPU time without requiring an NVTX range around each
function. Both appear in the Nsight report.

Every frontend and worker session uses **process-tree** CPU sampling: its
report includes the launched application and its child processes. Each MPI
rank has its own profiler session and collects CPU samples for that rank.
There is no separate CPU-sampling toggle in this preset.

**Benchmark boundaries.** SA-Bench starts capture after its warmup and initial
probe, then stops after the measured requests finish, outside its timing
measurement. SGLang-Bench runs with `--warmup-requests 0`, so the entire client
invocation is captured. Trace-replay and mooncake-router start capture after
their separate warmup command and stop when the measured AIPerf command exits;
client initialization and final artifact writing can be included. Additional
AIPerf warmup flags are rejected in `measured_workload` mode because that
warmup would otherwise happen inside the capture. Other bundled runners currently require
`capture_window: including_startup` or `nsys.enabled: false`.

**Custom benchmarks and manual serving.** srtctl cannot infer an external
client's internal warmup boundary. Custom clients receive the control script,
directory, and timeout through `SRT_NSYS_CONTROL_SCRIPT`,
`SRT_NSYS_CONTROL_DIR`, and `SRT_NSYS_CONTROL_TIMEOUT`. Invoke the hooks at the
actual phase boundaries (the script is a no-op when the preset is disabled):

```bash
run_warmup
python3 /srtctl-runtime/nsys_window.py start
run_measured_workload
python3 /srtctl-runtime/nsys_window.py stop
```

For manual serving, run these commands inside a container sharing the run's
`/logs` and `/srtctl-runtime` mounts and set
`SRT_NSYS_CONTROL_DIR=/logs/profiles/.control`. Repeat the pair for additional
windows. A custom benchmark that finishes without a completed window fails
validation. If it exits with an active window, srtctl attempts to export that
report and fails validation because the stop boundary was missing. Use
`capture_window: including_startup` when the external client cannot provide hooks.

An enabled top-level `profiling` mode (`torch`, `nsys`, or `nsys-time`) takes
precedence and disables this automatic preset, including automatic frontend
capture. `profiling.type: none` leaves the preset active. With
`observability.enabled: false`, `nsys.enabled` has no effect.

**Container requirements.** The serving image must include `nsys` (or mount it
and set `SRTCTL_NSYS_BIN` on the submitting/orchestrating host) and Python 3.
`measured_workload` uses Nsight's interactive `launch`/`start`/`stop`/`shutdown`
commands and a shared writable `/logs` mount. `including_startup` additionally
needs Bash, `setsid`, `pgrep`, `pkill`, and `timeout`. The preset sets
`DYN_ENABLE_RUST_NVTX=1`; Dynamo must have been built with NVTX support to emit
Rust ranges. On TRT-LLM workers it
also sets `TLLM_LLMAPI_ENABLE_NVTX=1` and `TLLM_PROFILE_LOG_RANKS=all`. Set
`nvtx_injection_path` only when the image needs an explicit NVTX injection
library; it must be an absolute **container** path compatible with that nsys
installation. Frontend and worker CPU sampling uses `--sample=process-tree`,
a 26,000,000 sampling period, and 32 samples per backtrace, and requires the
host's perf permissions. Run `nsys status --environment` in the serving
environment to verify CPU sampling is supported.

**Reports and shutdown.** Files for `measured_workload` are under the run's
`logs/profiles/` directory:

- `frontend/<node>_frontend_<index>_window001.nsys-rep`
- `<mode>/<node>_<mode>_w<index>_profile_rank<rank>_window001.nsys-rep` for MPI workers
- `<mode>/<node>_<mode>_w<index>_profile_gpu<devices>_window001.nsys-rep` for other workers

Later windows use `_window002`, etc. Shadow engines include their `_e<id>`
suffix after the worker index so reports do not overwrite each other.
`including_startup` omits the window suffix.
Shared control requests and acknowledgments are kept in `profiles/.control/`.
A start/stop failure or missing rank fails the benchmark, rather than allowing
an unprofiled workload to appear successful.

If teardown interrupts an active capture, each wrapper stops it and waits for
all reports in its MPI step before terminating applications. This also bounds
`including_startup` capture when no benchmark end hook is used. Leave room in
the job time limit for report export and application shutdown (up to the configured
report budget plus 150 seconds); allocation timeouts and forced cancellation
can interrupt export. Inspect the `.nsys-rep` contents before calling a profiling
run successful: an existing file alone does not prove NVTX ranges or CPU samples
were collected. `srtctl dry-run` shows the effective window, opt-out, or
explicit-profiling precedence.

## Quick Start

Add a `profiling` section to your job YAML:

```yaml
# For disaggregated mode (roles.prefill + roles.decode)
profiling:
  type: "torch" # or "nsys"
  prefill:
    start_step: 0
    stop_step: 50
  decode:
    start_step: 0
    stop_step: 50
# For aggregated mode (roles.agg)
# profiling:
#   type: "torch"
#   aggregated:
#     start_step: 0
#     stop_step: 50
```

## Profiling Modes

| Mode    | Description                                                      | Output                                         |
| ------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| `none`  | No explicit profiling; the observability preset may still apply          | -                                              |
| `torch` | PyTorch Profiler. Good for Python-level and CUDA kernel analysis | `/logs/profiles/{mode}/` (Chrome trace format) |
| `nsys`  | NVIDIA Nsight Systems. Low-overhead GPU profiling                | `/logs/profiles/{mode}/` (`*.nsys-rep`)        |

## Configuration Options

### Top-level `profiling` section

```yaml
profiling:
  type: "torch" # Required: "none", "torch", or "nsys"

  # nsys / nsys-time command settings
  nsys_trace: "cuda,nvtx"
  trace_fork_before_exec: true
  capture_range_end: "stop"
  nsys_library_paths: ["/usr/local/cuda/compat"]
  extra_nsys_args: []

  # Disaggregated mode: must set both prefill and decode sections
  prefill:
    start_step: 0 # Step to start profiling for prefill workers
    stop_step: 50 # Step to stop profiling for prefill workers
    capture_scope: selected # Opt in to targeting; "all" is the default
    worker_index: 0 # Logical prefill worker to capture
    worker_rank: 0 # Physical process rank within that worker
  decode:
    start_step: 0 # Step to start profiling for decode workers
    stop_step: 50 # Step to stop profiling for decode workers
    capture_scope: selected
    worker_index: 0
    worker_rank: 0


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
| `*.capture_scope`       | Capture one selected process or all physical processes | `all` |
| `*.worker_index`        | Logical worker selected for iteration-based nsys | `0`    |
| `*.worker_rank`         | Physical process rank selected within that worker | `0`  |
| `nsys_trace`            | Non-TRT-LLM Nsight activity domains          | `cuda,nvtx` |
| `trace_fork_before_exec` | Override non-TRT-LLM child-process tracing; unset uses the Dynamo default | unset |
| `capture_range_end`     | Non-TRT-LLM action after a CUDA profiler range ends | `stop` |
| `nsys_library_paths`    | Paths prepended to the worker `LD_LIBRARY_PATH` | unset |
| `extra_nsys_args`       | Additional `nsys profile` arguments          | unset    |

For non-TRT-LLM workers, set `nsys_trace` to choose trace domains. Do not
also pass `--trace` in `extra_nsys_args`: that emits duplicate options whose
precedence depends on Nsight's argument parsing.

## Constraints

Profiling has specific requirements:

1. **Disaggregated mode**: When profiling disaggregated workers, both `profiling.prefill` and `profiling.decode` must be set.

2. **Aggregated mode**: When profiling aggregated workers, `profiling.aggregated` must be set (and `profiling.prefill`/`profiling.decode` must not be set).

## How It Works

### Normal Mode (`type: none`)

- Uses `dynamo.sglang` module for serving
- Standard disaggregated inference path

### Profiling Mode (`type: torch` or `nsys`)

- Supported benchmark scripts receive the selected worker control endpoints
  needed to control iteration-triggered profiling.
- For non-TRT-LLM `nsys`, each phase defaults to `capture_scope: all`,
  preserving the all-worker wrapping behavior of existing recipes and sending
  every usable control endpoint to the benchmark. Set `capture_scope: selected`
  to opt in to targeting the process identified by `worker_index` and
  `worker_rank`.
- A Dynamo worker is controlled through its `DYN_SYSTEM_PORT`, not through the
  public OpenAI serving port.
- Direct SGLang uses its native `/start_profile` and `/stop_profile` endpoints.
  In releases whose profile request model has no `start_step` field, the
  capture starts when the request arrives and only `num_steps` is honored; a
  nonzero configured `start_step` changes the capture length but not its start
  time.
  Dynamo-hosted vLLM and SGLang use
  `/engine/control/start_profile` and `/engine/control/stop_profile` on each
  selected worker's `DYN_SYSTEM_PORT`.
- A native Dynamo sidecar exposes its system control server only on the
  endpoint leader. `capture_scope: all` still wraps all physical processes but
  sends control once to that leader; `capture_scope: selected` must select rank
  0 in sidecar mode. Direct vLLM has the same leader-only control constraint.
- TRT-LLM remains endpoint-wide: its executor consumes
  `TLLM_PROFILE_START_STOP` and calls `cudaProfilerStart`/`cudaProfilerStop`
  internally, without benchmark-side HTTP control.

`worker_rank` identifies a process in srtctl's physical topology. With vLLM
`backend.dp_launch_mode: per_gpu`, this is the DP rank. With `per_node`, one
wrapped process may own multiple local DP ranks; use `per_gpu` when a capture
must produce a separate report for every DP rank. `worker_index` and
`worker_rank` are ignored when `capture_scope: all`. For non-TRT-LLM `nsys`,
validation warns if either ignored selector is nonzero; set
`capture_scope: selected` to use those selectors. Default zero-valued selectors
do not warn, so existing all-worker recipes remain unchanged.

### nsys-specific behavior

When using `nsys`, workers are wrapped with:

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  -c cudaProfilerApi --capture-range-end stop \
  [extra_nsys_args...] \
  -o /logs/profiles/{mode}/{name} \
  python3 -m sglang.launch_server ...
```

You can configure the trace domains and range behavior directly, and pass any
remaining arguments via `profiling.extra_nsys_args`. Explicit extra arguments
are appended after the generated options and can therefore override an Nsight
option when needed.

For an Ubuntu image without the Nsight CLI, set:

```yaml
setup_script: install-nsys-cli.sh
```

The bundled setup script installs the current `nsight-systems-cli` package on
both x86_64 and Arm64 containers.

## Example Configurations

### Torch Profiler (Recommended for Python analysis)

```yaml
schema: 2
name: "profiling-torch"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  gpus_per_node: 4

profiling:
  type: "torch"
  prefill:
    start_step: 0
    stop_step: 50
  decode:
    start_step: 0
    stop_step: 50

engine: sglang
roles:
  prefill:
    nodes: 1
    workers: 1
    args:
      kv-cache-dtype: "fp8_e4m3"
      tensor-parallel-size: 4
  decode:
    nodes: 1
    workers: 1
    args:
      kv-cache-dtype: "fp8_e4m3"
      tensor-parallel-size: 4
```

### Nsight Systems (Recommended for GPU kernel analysis)

```yaml
profiling:
  type: "nsys"
  nsys_trace: "cuda,nvtx"
  trace_fork_before_exec: true
  capture_range_end: "stop"
  prefill:
    start_step: 10
    stop_step: 30
    capture_scope: all
  decode:
    start_step: 10
    stop_step: 30
    capture_scope: all
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
