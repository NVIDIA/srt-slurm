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
    capture_scope: selected # "selected" (default) or "all"
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
| `*.capture_scope`       | Capture one selected process or all physical processes | `selected` |
| `*.worker_index`        | Logical worker selected for iteration-based nsys | `0`    |
| `*.worker_rank`         | Physical process rank selected within that worker | `0`  |
| `nsys_trace`            | Non-TRT-LLM Nsight activity domains          | `cuda,nvtx` |
| `trace_fork_before_exec` | Override non-TRT-LLM child-process tracing; unset uses the Dynamo default | unset |
| `capture_range_end`     | Non-TRT-LLM action after a CUDA profiler range ends | `stop` |
| `nsys_library_paths`    | Paths prepended to the worker `LD_LIBRARY_PATH` | unset |
| `extra_nsys_args`       | Additional `nsys profile` arguments          | unset    |

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
- For non-TRT-LLM `nsys`, each phase defaults to `capture_scope: selected`,
  which wraps the process identified by `worker_index` and `worker_rank`.
  Set `capture_scope: all` to wrap every physical process in that phase and
  send every usable control endpoint to the benchmark.
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
`worker_rank` are ignored when `capture_scope: all`.

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
