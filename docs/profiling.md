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
- [On-demand capture (`nsys-manual`)](#on-demand-capture-nsys-manual)
- [Time-based capture (`nsys-time`)](#time-based-capture-nsys-time)
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

| Mode          | Description                                                                 | Output                                         |
| ------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| `none`        | Default. No profiling, uses `dynamo.sglang` for serving                     | -                                              |
| `torch`       | PyTorch Profiler. Good for Python-level and CUDA kernel analysis            | `/logs/profiles/{mode}/` (Chrome trace format) |
| `nsys`        | NVIDIA Nsight Systems, capture bounded by engine steps                      | `/logs/profiles/{mode}/` (`*.nsys-rep`)        |
| `nsys-manual` | nsys, capture fired by hand mid-run — see [below](#on-demand-capture-nsys-manual) | `/logs/profiles/{mode}/` (`*.nsys-rep`)  |
| `nsys-time`   | nsys, capture bounded by wall clock — see [below](#time-based-capture-nsys-time)  | `/logs/profiles/{mode}/` (`*.nsys-rep`)  |

Pick the nsys variant by how you want to bound the capture window: by engine
iteration (`nsys`), by hand while watching the run (`nsys-manual`), or by a
fixed delay and duration in seconds (`nsys-time`).

## Configuration Options

### Top-level `profiling` section

```yaml
profiling:
  type: "torch" # Required: "none", "torch", "nsys", "nsys-manual", or "nsys-time"

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

Both rules apply to `torch`, `nsys` and `nsys-time`. `nsys-manual` uses
`phases`/`duration_secs` instead and rejects the per-phase sections entirely.

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

## On-demand capture (`nsys-manual`)

Use this when you cannot express the interesting window in engine steps — for
example when you want a few seconds of steady-state traffic on a vLLM + dynamo
disaggregated job, after the load has settled but before the run winds down.

The recipe only picks a phase and a window length; the nsys flags are fixed in
code (`NSYS_MANUAL_DEFAULT_ARGS`) and are not configurable:

```yaml
profiling:
  type: "nsys-manual"
  phases: prefill      # "prefill" or "decode". Required for disaggregated jobs,
                       # must be omitted for aggregated ones.
  duration_secs: 5     # Optional, defaults to 5.
```

Exactly one process is profiled: **endpoint 0, rank 0** of the chosen phase. Every
other worker runs completely unwrapped — no nsys, and no vLLM `--profiler-config`
either — so the rest of the deployment stays a clean baseline and the capture does
not distort the numbers you are measuring.

At startup srtctl writes a trigger script into the job directory:

```bash
bash outputs/<job_id>/nsys-manual.sh
```

It takes no arguments. It POSTs `/engine/start_profile` to the worker, sleeps
`duration_secs`, then POSTs `/engine/stop_profile`.

### Timing the capture

The window is only useful if it lands on real traffic, so fire the script against
the stage banners that sa-bench writes into the worker logs:

```bash
tail -f outputs/<job_id>/logs/*_prefill_w0.out
```

```text
======== [17:56:30] cc=1024 warmup begin ========
======== [17:59:39] cc=1024 warmup end ========
======== [17:59:39] cc=1024 benchmark begin ========
```

Fire it after `benchmark begin`, and give yourself room: a run with a small
`num_prompts_mult` can be over in under 20 seconds, which is easy to miss. The
`benchmark.out` log also prints the measured window as wall-clock time, which is
handy afterwards for checking where your capture actually landed:

```text
Benchmark measurement start (UTC):     2026-08-09T10:24:54.114949+00:00
```

### Confirming the capture happened

The trigger goes to the Dynamo system server on `DYN_SYSTEM_PORT`, not to the vLLM
HTTP frontend, and dynamo logs are quiet by default — so a successful capture
leaves no request log. Look for nsys's own markers in the worker log instead:

```text
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-987b.qdstrm'
Generated:
	/logs/profiles/prefill/<node>_prefill_w0_profile_gpu0.nsys-rep
```

Note that the report is written when the worker exits, not when the window
closes, and nsys buffers those messages — so they show up at teardown, below the
SLURM step-cancellation notice. That ordering is normal and does not mean the
capture failed.

A full working recipe lives in
[`examples/nsys-manual-qwen35-disagg.yaml`](../examples/nsys-manual-qwen35-disagg.yaml).

## Time-based capture (`nsys-time`)

Same idea as `nsys-manual`, but the window is fixed in advance relative to worker
launch instead of being fired by hand. Useful for unattended runs:

```yaml
profiling:
  type: "nsys-time"
  delay_secs: 120             # nsys --delay: wait this long after worker launch
  duration_secs: 30           # nsys --duration: then capture this long
  benchmark_duration_secs: 300  # traffic must cover delay + duration
```

There is no cudaProfilerApi trigger here, so nothing needs to call
`/start_profile`. Make sure the benchmark is still generating load when the window
opens — the delay is counted from worker launch, which includes model load time.

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
- For `nsys-manual`, disaggregated jobs require `profiling.phases`, aggregated jobs must omit it, and the per-phase sections are not allowed at all.

### Empty profile output
Ensure the benchmark workload is generating requests during the profiling window.

With `nsys-manual` this is the usual failure: a short benchmark can finish before
you fire the trigger, leaving a report full of an idle engine. The `.nsys-rep` will
still be tens of megabytes — nsys writes process and symbol metadata regardless —
so file size proves nothing. Watch for the `benchmark begin` banner in the worker
log and enlarge `benchmark.num_prompts_mult` if the run is too short to aim at.

### Profile too short/long

Adjust `start_step` and `stop_step` to capture the desired range. A typical profiling run uses 30-100 steps.
