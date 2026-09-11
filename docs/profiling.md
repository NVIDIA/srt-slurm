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

**TRT-LLM workers (`backend.type: trtllm`)** are wrapped with the capture recipe from the Dynamo
Benchmark Playbook §9.5.1.1 "Dynamo + TRTLLM", the flag set that produced every usable multi-node
capture on nsys 2026.3.x / VR200 disaggregated serving:

```bash
nsys profile --force-overwrite=true \
  -t cuda-sw,nvtx,python-gil --cuda-graph-trace=graph \
  --sample=none --cpuctxsw=none --python-sampling=false --python-sampling-frequency=1000 \
  --gpu-metrics-devices=none --flush-on-cudaprofilerstop=false --cuda-flush-interval=0 \
  -c cudaProfilerApi --capture-range-end=stop \
  [extra_nsys_args...] --kill none --wait all \
  -o /logs/profiles/{mode}/{leader}_{mode}_w{index}_profile_rank%q{SLURM_PROCID} \
  trtllm-llmapi-launch python3 -m dynamo.trtllm ...
```

One `nsys` runs per srun task, i.e. per engine rank; `%q{SLURM_PROCID}` is nsys's own env-var
substitution, expanded inside the launched process, so each rank gets its own `.nsys-rep`.
The capture window is the TRT-LLM iteration range from `prefill`/`decode` `start_step`/`stop_step`
(`TLLM_PROFILE_START_STOP`): the PyExecutor calls `cudaProfilerStart`/`cudaProfilerStop` on every rank
at those iterations and `--capture-range-end=stop` finalises the report the moment the range closes.
Reports appear under `<log_dir>/profiles/<mode>/` while the benchmark is still running.

Why each flag (from the playbook):

| Flag | Reason |
| ---- | ------ |
| `-t cuda-sw`, not `cuda` | on nsys 2026.x plain `cuda` selects the HES hardware trace, which SIGSEGVs the KV transceiver's device-to-device `cudaMemcpyAsync`; `cuda-sw` forces the software tracer |
| `--cuda-graph-trace=graph` | decode runs CUDA graphs; with `node` the launches are invisible (~0.3 % GPU busy reads as idle), with `graph` each iteration is one entry and utilisation is real |
| `--sample=none --cpuctxsw=none` | `--sample=process-tree` wedged workers at `cudaProfilerStop` across 8 runs. Cost: no scheduler/CPU-time data, every duration is wall-clock. `nsys_cpuctxsw: process-tree` alone (context switches, no IP sampling) is the safer experiment; both need `kernel.perf_event_paranoid <= 2` on the compute node |
| `-c cudaProfilerApi --capture-range-end=stop` | finalise-at-process-exit has never produced a report on this stack |
| `--flush-on-cudaprofilerstop=false --cuda-flush-interval=0` | matches the known-good captures; nsys 2026.4 flips the flush default, so it is set explicitly |
| `--gpu-metrics-devices=none` | a second collection path with its own failure modes |

Worker environment set alongside: `TLLM_PROFILE_START_STOP=<start>-<stop>`, `TLLM_LLMAPI_ENABLE_NVTX=1`,
`TLLM_PROFILE_LOG_RANKS=<log_ranks>` (default `all`), `DYN_ENABLE_RUST_NVTX=1` (Dynamo's Rust NVTX
ranges; effective only on a wheel built with the `nvtx` cargo feature) and, when
`nvtx_injection_path` is set, `NVTX_INJECTION64_PATH` pointing at nsys's injection library inside the
container (TRT-LLM containers ship it next to nsys, e.g.
`/usr/local/cuda-0.gpgpu/NsightSystems-cli-2026.3.0/target-linux-sbsa-armv8/libToolsInjection64.so`).

Knobs (all optional, defaults = the recipe above): `nsys_trace`, `nsys_cuda_graph_trace` (`graph`|`node`),
`nsys_sample` and `nsys_cpuctxsw` (`none`|`process-tree`|`system-wide`), `nsys_python_sampling`,
`nsys_python_sampling_frequency`, `nsys_gpu_metrics_devices`, `nvtx_injection_path`, `log_ranks`,
plus `extra_nsys_args` (appended before `-o`). `nsys-time` uses the same trace flags with
`--delay`/`--duration` instead of the cudaProfilerApi window. The `profiling:` block is independent of
`observability.enabled`, so the capture cost can be measured against a default-visibility run.

Sizing the window: `start_step` counts engine iterations, which begin with traffic, not with worker
start. On the 8-node AgentX recipe the decode worker ran ~20 iterations/s and each prefill worker
~2.7/s in steady state, so `decode: 6000-6600` and `prefill: 1200-1300` both capture about 30-40 s
roughly 12 minutes into the benchmark.

**SGLang / vLLM workers** keep the time-based or `--trace-fork-before-exec` prefixes described in the
examples below; `extra_nsys_args` applies to them as well.

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
