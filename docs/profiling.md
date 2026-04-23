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
