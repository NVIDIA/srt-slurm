# Recipe Guide

How to write a job recipe in the 2.0 (`schema: 2`) layout: what each block means, how the blocks interact, and worked examples.

Field-level facts (every key, type, default, and allowed value) are generated from the code into [schema-reference.md](schema-reference.md) and checked in CI; if a prose page and the generated one disagree, the generated one is right. Agents and editors can read the same data as JSON Schema (`srtctl schema`, published at [schema/recipe.schema.json](schema/recipe.schema.json)) or per field from the MCP `explain_field` tool. The pre-2.0 layout (`backend:`, `resources.prefill_nodes`, `infra:`, ...) is in [legacy-v1.md](legacy-v1.md); `srtctl migrate -f recipe.yaml --in-place` rewrites it.

| Topic | Page | Recipe keys |
| --- | --- | --- |
| Cluster defaults and aliases | [Cluster Config](cluster-config.md) | `srtslurm.yaml` |
| Engine and engine-wide knobs | [Engines](engines.md) | `engine` |
| Nodes, GPUs, Slurm | [Topology and Placement](topology.md) | `roles`, `placement`, `resources`, `slurm`, `sbatch_directives`, `srun_options` |
| Router and Dynamo | [Frontends and Dynamo](frontends.md) | `frontend`, `dynamo` |
| Load and evals | [Benchmarks](benchmarks.md) | `benchmark`, `post_eval` |
| Profilers | [Profiling](profiling.md) | `profiling` |
| Metrics and power | [Observability and Telemetry](observability.md) | `observability`, `telemetry` |
| Container paths and env | [Paths, Mounts, and Environment](runtime-env.md) | `container_mounts`, `extra_mount`, `environment`, `setup_script`, `host_setup` |
| Sidecars and infrastructure | [Services](services.md) | `services` |
| Many jobs from one file | [Parameter Sweeps](sweeps.md), [Config Overrides](overrides.md) | `sweep`, `base` / `override_*` |

This page covers the remaining top-level keys (`schema`, `name`, `model`, `output`, `health_check`, `enable_config_dump`) and complete examples.

## schema

`schema: 2` is the layout this document describes and the only one that loads. A recipe without the key is the pre-2.0 layout in [legacy-v1.md](legacy-v1.md) and is rejected.

Put the key first in the file, beside `base:` in an override file. Upgrade a recipe with `srtctl migrate -f recipe.yaml --in-place`, which preserves comments and key order and folds the legacy layout into `engine:`, `roles:`, `placement:`, `services:`, and `dynamo.source` (a directory is walked recursively). A `schema: 2` recipe that still carries a pre-2.0 key (`backend:`, `infra:`, `resources.prefill_nodes`, `dynamo.hash`, ...) is rejected too; the error names the keys and `srtctl migrate` rewrites them.

```yaml
schema: 2
name: "deepseek-r1-benchmark"
```

Two rules worth knowing: a `benchmark:` field the selected type does not read is a load error (see [benchmark](benchmarks.md#benchmark)), and `roles.<role>.nodes: 0` is rejected in favor of the explicit `colocate` (see [roles](topology.md#roles)).

## name

Job name: the Slurm `--job-name` (unless `RUNNER_NAME` is set) and the run's label in results.

```yaml
name: "deepseek-r1-benchmark"
```

## model

Model and container configuration.

```yaml
model:
  path: "deepseek-r1"       # Alias from srtslurm.yaml or full path
  container: "sglang"       # Container alias from srtslurm.yaml
  precision: "fp8"          # fp8, fp4, bf16, etc.
```

Fields: [ModelConfig](schema-reference.md#modelconfig).

## output

Output paths and optional reproducibility artifacts.

```yaml
output:
  log_dir: "./outputs/{job_id}/logs"
  record_launch_plan: false    # Save exact realized srun scripts and a manifest
```

The `log_dir` supports FormattablePath templating. See [FormattablePath Template System](runtime-env.md#formattablepath-template-system).

When `record_launch_plan` is enabled, srtctl creates `logs/launch-plan/manifest.json` and one executable shell
script for every realized `srun` invocation. Recording happens after Slurm assigns nodes, ports, heterogeneous
groups, mounts, and container paths, so the scripts describe what was actually launched rather than a pre-submit
estimate. Secret-like environment values are never persisted; the scripts name the environment variables that
must be supplied for replay. The resolved recipe, outer `sbatch_script.sh`, lockfile, and resource snapshot remain
alongside this directory and are referenced by the manifest.

## health_check

Health check configuration for worker readiness, and the worker log watch that fails a run early.

```yaml
health_check:
  max_attempts: 180
  interval_seconds: 10
  fatal_log_markers: true
  extra_fatal_log_patterns: []
```

Fields: [HealthCheckConfig](schema-reference.md#healthcheckconfig).

**Notes**:

- Default of 180 attempts at 10 second intervals = 30 minutes total wait time.
- Large models (e.g., 70B+ parameters) may require the full 30 minutes to load.
- Reduce `max_attempts` for smaller models or faster testing.

**Worker log watch** (`fatal_log_markers`): the process monitor normally learns that a worker died
from its srun step exiting. A TRT-LLM worker step is one `trtllm-llmapi-launch` task per GPU and the
engine is a child of the rank-0 task only; when that child dies the launcher prints
`Rank0 Task exit code: <n>` and the other ranks stay blocked, so the step never exits and the run
would otherwise wait out the whole health window. With the watch on, the monitor scans each critical
worker's log for the lines its engine declares fatal (TRT-LLM: `Rank<N> Task exit code: <non-zero>`
and `Failed to initialize executor`; other engines declare none) and fails the run within one monitor
poll, printing the process name and the matching line. Bare words such as `Traceback` or `MPI_Abort`
are deliberately not markers: Dynamo logs a traceback for every request cancelled at EOS, and MPI
abort lines appear on normal teardown. Use `extra_fatal_log_patterns` to add a marker for one recipe
(for example `"CUDA error: out of memory"`), and `fatal_log_markers: false` to switch the watch off,
for probes that kill workers on purpose. TRT-LLM endpoint steps are also launched with
`srun --kill-on-bad-exit=1`, so a launcher task that does exit non-zero ends the whole step.

## enable_config_dump

Accepted for compatibility; srtctl does not read it.

```yaml
enable_config_dump: true
```

SGLang (except under the `sglang` direct frontend) and vLLM workers always get `--dump-config-to`, which writes the resolved engine configuration to a JSON file in the log directory.

## Complete Examples

Every example below loads with `srtctl dry-run`. Model and container names are `srtslurm.yaml` aliases.

### Disaggregated Mode with Dynamo

```yaml
schema: 2
name: "deepseek-r1-disagg"

model:
  path: "deepseek-r1"
  container: "sglang"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  gpus_per_node: 4

slurm:
  time_limit: "04:00:00"

dynamo:
  source:
    pypi: "1.4.2"

frontend:
  type: dynamo
  enable_multiple_frontends: true
  args:
    router-mode: "kv"

engine: sglang
roles:
  prefill:
    nodes: 2
    workers: 4
    gpus: 2
    kv_events: true
    env:
      TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
    args:
      tensor-parallel-size: 2
      mem-fraction-static: 0.84
      kv-cache-dtype: "fp8_e4m3"
  decode:
    nodes: 4
    workers: 2
    gpus: 8
    env:
      TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
    args:
      tensor-parallel-size: 8
      mem-fraction-static: 0.83
      data-parallel-size: 8

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [128, 256, 512]

health_check:
  max_attempts: 180
  interval_seconds: 10
```

### Aggregated Mode with SGLang Router

```yaml
schema: 2
name: "qwen-agg-router"

model:
  path: "qwen3-32b"
  container: "sglang"
  precision: "bf16"

resources:
  gpu_type: "h100"
  gpus_per_node: 8

slurm:
  time_limit: "02:00:00"

frontend:
  type: sglang-router
  enable_multiple_frontends: false
  args:
    policy: "cache_aware"

engine: sglang
roles:
  agg:
    nodes: 4
    workers: 8
    gpus: 4
    args:
      tensor-parallel-size: 4
      mem-fraction-static: 0.9
      enable-dp-attention: true

benchmark:
  type: "router"
  isl: 14000
  osl: 200
  num_requests: 200
  prefix_ratios: [0.1, 0.3, 0.5, 0.7, 0.9]
```

### Profiling Example

```yaml
schema: 2
name: "profile-decode"

model:
  path: "llama-70b"
  container: "sglang"
  precision: "fp8"

resources:
  gpu_type: "h100"
  gpus_per_node: 8

slurm:
  time_limit: "01:00:00"

profiling:
  type: "torch"
  prefill:
    start_step: 5
    stop_step: 15
  decode:
    start_step: 5
    stop_step: 15

engine: sglang
roles:
  prefill:
    nodes: 1
    workers: 1
    gpus: 8
    args:
      tensor-parallel-size: 8
  decode:
    nodes: 1
    workers: 1
    gpus: 8
    args:
      tensor-parallel-size: 8

benchmark:
  type: "sa-bench"
  isl: 2048
  osl: 256
  concurrencies: "32x64"
  req_rate: "inf"
```

### Parameter Sweep Example

```yaml
schema: 2
name: "sweep-throughput"

model:
  path: "deepseek-r1"
  container: "sglang"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  gpus_per_node: 4

engine: sglang
roles:
  prefill:
    nodes: 1
    workers: 2
    gpus: 2
    args:
      tensor-parallel-size: 2
  decode:
    nodes: 2
    workers: 4
    gpus: 2
    args:
      tensor-parallel-size: 2

benchmark:
  type: "sa-bench"
  isl: "{isl}"
  osl: "{osl}"
  concurrencies: [64, 128, 256]

sweep:
  isl: [512, 1024, 2048, 4096]
  osl: [128, 256, 512, 1024]
```

### Config Override Example

```yaml
schema: 2
base:
  name: "disagg-fp8-benchmark"

  model:
    path: "deepseek-r1"
    container: "sglang"
    precision: "fp8"

  resources:
    gpu_type: "h100"
    gpus_per_node: 8

  engine: sglang
  roles:
    prefill:
      nodes: 2
      workers: 2
      gpus: 8
      args:
        tp-size: 8
    decode:
      nodes: 8
      workers: 8
      gpus: 8
      args:
        tp-size: 8

  benchmark:
    type: "sa-bench"
    isl: 1024
    osl: 8192
    concurrencies: [8192, 10240]

# One TP=16 decode worker spanning two nodes, prefill unchanged
override_tp16:
  roles:
    decode:
      workers: 4
      gpus: 16
      args:
        tp-size: 16

# Smaller cluster with fewer decode nodes
override_small:
  roles:
    decode:
      nodes: 4
      workers: 4
  benchmark:
    concurrencies: [4096]
```

### Custom Mounts and Setup

```yaml
schema: 2
name: "custom-setup"

model:
  path: "$MODELS_DIR/my-model"
  container: "$CONTAINERS_DIR/custom.sqsh"
  precision: "fp8"

resources:
  gpu_type: "h100"
  gpus_per_node: 8

engine: sglang
roles:
  agg:
    nodes: 2
    workers: 4
    gpus: 4
    args:
      tensor-parallel-size: 4

setup_script: "install-custom-sglang.sh"

environment:
  CUSTOM_VAR: "value"
  NCCL_DEBUG: "INFO"

container_mounts:
  "$HOME/datasets": "/datasets"
  "$SCRATCH/cache": "/cache"

extra_mount:
  - "/shared/data:/data:ro"

sbatch_directives:
  mail-user: "user@example.com"
  mail-type: "END,FAIL"
  reservation: "gpu-cluster"

srun_options:
  cpu-bind: "none"

output:
  log_dir: "$HOME/experiments/{job_id}/logs"

health_check:
  max_attempts: 120
  interval_seconds: 15
```
