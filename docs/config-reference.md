# Configuration Reference

Complete reference for job configuration YAML files.

## Table of Contents

- [Overview](#overview)
- [Cluster Config Discovery](#cluster-config-discovery)
- [name](#name)
- [model](#model)
- [resources](#resources)
- [slurm](#slurm)
- [frontend](#frontend)
- [backend](#backend)
- [benchmark](#benchmark)
- [dynamo](#dynamo)
- [profiling](#profiling)
- [output](#output)
- [health_check](#health_check)
- [infra](#infra)
- [sweep](#sweep)
- [Config Overrides](#config-overrides)
- [FormattablePath Template System](#formattablepath-template-system)
- [container_mounts](#container_mounts)
- [environment](#environment)
- [extra_mount](#extra_mount)
- [sbatch_directives](#sbatch_directives)
- [srun_options](#srun_options)
- [setup_script](#setup_script)
- [enable_config_dump](#enable_config_dump)
- [Complete Examples](#complete-examples)

---

## Overview

```yaml
name: "my-benchmark"           # Required: job name

model:                         # Required: model settings
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:                     # Required: GPU allocation
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 2

slurm:                         # Optional: SLURM overrides
  time_limit: "02:00:00"

frontend:                      # Optional: router/frontend config
  type: dynamo

backend:                       # Optional: worker config
  type: sglang
  sglang_config:
    prefill: {}
    decode: {}

benchmark:                     # Optional: benchmark config
  type: "sa-bench"
  isl: 1024
  osl: 1024

dynamo:                        # Optional: dynamo version
  version: "0.8.0"

profiling:                     # Optional: profiling config
  type: "none"

output:                        # Optional: output paths
  log_dir: "./outputs/{job_id}/logs"

health_check:                  # Optional: health check settings
  max_attempts: 180
  interval_seconds: 10

setup_script: "my-setup.sh"    # Optional: custom setup script
```

---

## Cluster Config Discovery

srtctl looks for `srtslurm.yaml` (cluster-wide settings) in this order:

1. **`SRTSLURM_CONFIG` environment variable** (if set) - explicit path to config file
2. Current working directory
3. Parent directory (1 level up)
4. Grandparent directory (2 levels up)

For users working in deep directory structures (e.g., study directories), set `SRTSLURM_CONFIG` in your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
export SRTSLURM_CONFIG="/path/to/srt-slurm/srtslurm.yaml"
```

This allows you to run `srtctl apply -f config.yaml` from anywhere without needing `srtslurm.yaml` nearby.

### Cluster Config Fields

The `srtslurm.yaml` file can contain the following fields:

| Field                           | Type   | Description                                           |
| ------------------------------- | ------ | ----------------------------------------------------- |
| `default_account`               | string | Default SLURM account                                 |
| `default_partition`             | string | Default SLURM partition                               |
| `default_time_limit`            | string | Default job time limit                                |
| `gpus_per_node`                 | int    | Default GPUs per node                                 |
| `network_interface`             | string | Network interface for NCCL                            |
| `srtctl_root`                   | string | Root directory for srtctl                             |
| `output_dir`                    | string | Custom output directory (overrides srtctl_root/outputs) |
| `model_paths`                   | dict   | Model path aliases                                    |
| `containers`                    | dict   | Container image aliases                               |
| `default_mounts`                | dict   | Cluster-wide container mounts                         |

**output_dir**: When set, job logs are written to `output_dir/{job_id}/logs` instead of `srtctl_root/outputs/{job_id}/logs`. Useful for CI/CD and ephemeral environments.

---

## name

| Field  | Type   | Required | Description                                        |
| ------ | ------ | -------- | -------------------------------------------------- |
| `name` | string | Yes      | Job name, used for identification and log prefixes |

```yaml
name: "deepseek-r1-benchmark"
```

---

## model

Model and container configuration.

```yaml
model:
  path: "deepseek-r1"       # Alias from srtslurm.yaml or full path
  container: "latest"       # Container alias from srtslurm.yaml
  precision: "fp8"          # fp8, fp4, bf16, etc.
```

| Field       | Type   | Required | Description                                              |
| ----------- | ------ | -------- | -------------------------------------------------------- |
| `path`      | string | Yes      | Model path alias (from `srtslurm.yaml`) or absolute path |
| `container` | string | Yes      | Container alias (from `srtslurm.yaml`) or `.sqsh` path   |
| `precision` | string | Yes      | Model precision (informational: fp4, fp8, fp16, bf16)    |

---

## resources

GPU allocation and worker topology.

### Disaggregated Mode (prefill + decode)

```yaml
resources:
  gpu_type: "gb200"
  gpus_per_node: 4          # GPUs per node (default: from srtslurm.yaml)

  prefill_nodes: 2          # Nodes for prefill workers
  prefill_workers: 4        # Number of prefill workers

  decode_nodes: 4           # Nodes for decode workers
  decode_workers: 8         # Number of decode workers
```

### Aggregated Mode (single worker type)

```yaml
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 2              # Nodes for aggregated workers
  agg_workers: 4            # Number of aggregated workers
```

| Field             | Type   | Default            | Description                           |
| ----------------- | ------ | ------------------ | ------------------------------------- |
| `gpu_type`        | string | -                  | GPU type: "gb200", "gb300", or "h100" |
| `gpus_per_node`   | int    | 4                  | GPUs per node                         |
| `prefill_nodes`   | int    | null               | Nodes dedicated to prefill            |
| `decode_nodes`    | int    | null               | Nodes dedicated to decode             |
| `prefill_workers` | int    | null               | Number of prefill workers             |
| `decode_workers`  | int    | null               | Number of decode workers              |
| `agg_nodes`       | int    | null               | Nodes for aggregated mode             |
| `agg_workers`     | int    | null               | Number of aggregated workers          |
| `gpus_per_prefill`| int    | computed           | Explicit GPUs per prefill worker      |
| `gpus_per_decode` | int    | computed           | Explicit GPUs per decode worker       |
| `gpus_per_agg`    | int    | computed           | Explicit GPUs per aggregated worker   |

**Notes**:

- Set `decode_nodes: 0` to have decode workers share nodes with prefill workers.
- Either use disaggregated mode (prefill_nodes/decode_nodes) OR aggregated mode (agg_nodes), not both.
- GPUs per worker are computed automatically: `(nodes * gpus_per_node) / workers`
- Use `gpus_per_prefill`, `gpus_per_decode`, `gpus_per_agg` to explicitly override the computed values

### Computed Properties

The ResourceConfig provides several computed properties:

- `is_disaggregated`: True if using prefill/decode mode
- `total_nodes`: Total nodes allocated (prefill + decode or agg)
- `num_prefill`, `num_decode`, `num_agg`: Worker counts for each role
- `gpus_per_prefill`, `gpus_per_decode`, `gpus_per_agg`: GPUs allocated per worker
- `prefill_gpus`, `decode_gpus`: Total GPUs for each role

---

## slurm

SLURM job settings.

```yaml
slurm:
  time_limit: "04:00:00"    # Job time limit
  account: "my-account"     # SLURM account (overrides srtslurm.yaml)
  partition: "batch"        # SLURM partition (overrides srtslurm.yaml)
```

| Field        | Type   | Default            | Description               |
| ------------ | ------ | ------------------ | ------------------------- |
| `time_limit` | string | from srtslurm.yaml | Job time limit (HH:MM:SS) |
| `account`    | string | from srtslurm.yaml | SLURM account             |
| `partition`  | string | from srtslurm.yaml | SLURM partition           |

---

## frontend

Frontend/router configuration.

```yaml
frontend:
  # Frontend type: "dynamo" (default) or "sglang"
  type: dynamo

  # Scaling
  enable_multiple_frontends: true     # Enable nginx + multiple routers
  num_additional_frontends: 9         # Additional routers (total = 1 + this)

  # CLI args passed to the frontend/router
  args:
    router-mode: "kv"                 # dynamo: router-mode
    policy: "cache_aware"             # sglang: policy
    no-kv-events: true                # boolean flags

  # Environment variables for frontend processes
  env:
    MY_VAR: "value"
```

| Field                       | Type | Default       | Description                         |
| --------------------------- | ---- | ------------- | ----------------------------------- |
| `type`                      | str  | dynamo        | Frontend type: "dynamo" or "sglang" |
| `enable_multiple_frontends` | bool | true          | Scale with nginx + multiple routers |
| `num_additional_frontends`  | int  | 9             | Additional routers beyond master    |
| `nginx_container`           | str  | nginx:1.27.4  | Custom nginx container image        |
| `args`                      | dict | null          | CLI args for the frontend           |
| `env`                       | dict | null          | Env vars for frontend processes     |

See [SGLang Router](sglang-router.md) for detailed architecture.

---

## backend

Worker configuration and SGLang settings.

```yaml
backend:
  type: sglang                        # Backend type (currently only sglang)

  # Per-mode environment variables
  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  aggregated_environment: {}

  # SGLang CLI config per mode
  sglang_config:
    prefill:
      tensor-parallel-size: 4
      mem-fraction-static: 0.84
      kv-cache-dtype: "fp8_e4m3"
      disaggregation-mode: "prefill"
      # ... any sglang CLI flag
    decode:
      tensor-parallel-size: 8
      mem-fraction-static: 0.83
      data-parallel-size: 8
      enable-dp-attention: true
    aggregated:
      # ... for aggregated mode

  # KV events (for kv-aware routing)
  kv_events_config:
    prefill: true                     # Enable for prefill workers
    decode: true                      # Enable for decode workers
```

| Field                     | Type        | Default | Description                             |
| ------------------------- | ----------- | ------- | --------------------------------------- |
| `type`                    | string      | sglang  | Backend type: "sglang" or "trtllm"      |
| `gpu_type`                | string      | null    | GPU type override                       |
| `prefill_environment`     | dict        | {}      | Environment variables for prefill       |
| `decode_environment`      | dict        | {}      | Environment variables for decode        |
| `aggregated_environment`  | dict        | {}      | Environment variables for aggregated    |
| `sglang_config`           | object      | null    | SGLang CLI configuration per mode       |
| `kv_events_config`        | bool/dict   | null    | KV events configuration                 |

### sglang_config

Per-mode SGLang server configuration. Any SGLang CLI flag can be specified (use kebab-case or snake_case):

| Common Flags                      | Type    | Description                           |
| --------------------------------- | ------- | ------------------------------------- |
| `tensor-parallel-size`            | int     | Tensor parallelism degree             |
| `data-parallel-size`              | int     | Data parallelism degree               |
| `expert-parallel-size`            | int     | Expert parallelism (MoE models)       |
| `mem-fraction-static`             | float   | GPU memory fraction (0.0-1.0)         |
| `kv-cache-dtype`                  | string  | KV cache precision (fp8_e4m3, etc.)   |
| `context-length`                  | int     | Max context length                    |
| `chunked-prefill-size`            | int     | Chunked prefill batch size            |
| `enable-dp-attention`             | bool    | Enable DP attention                   |
| `disaggregation-mode`             | string  | "prefill" or "decode"                 |
| `disaggregation-transfer-backend` | string  | Transfer backend ("nixl" or other)    |
| `served-model-name`               | string  | Model name for API                    |
| `grpc-mode`                       | bool    | Enable gRPC mode                      |

### kv_events_config

**Note:** KV events is a Dynamo frontend feature for kv-aware routing. It allows workers to publish cache/scheduling information over ZMQ for the Dynamo router to make intelligent routing decisions.

Enables `--kv-events-config` for workers with auto-allocated ZMQ ports.

```yaml
# Enable with defaults
kv_events_config: true         # prefill+decode with publisher=zmq, topic=kv-events

# Per-mode control
kv_events_config:
  prefill: true
  decode: true
  aggregated: true              # Enable for aggregated workers

# Custom settings
kv_events_config:
  prefill:
    publisher: "zmq"
    topic: "prefill-events"
  decode:
    topic: "decode-events"     # publisher defaults to "zmq"
  aggregated: true             # Enable for aggregated mode
```

Each worker leader gets a globally unique port starting at 5550:

| Worker    | Port |
| --------- | ---- |
| prefill_0 | 5550 |
| prefill_1 | 5551 |
| decode_0  | 5552 |
| decode_1  | 5553 |

### TRTLLM Backend

When using `type: trtllm`, the backend uses TRTLLM with MPI-style launching:

```yaml
backend:
  type: trtllm

  # Per-mode environment variables
  prefill_environment:
    CUDA_LAUNCH_BLOCKING: "1"
  decode_environment:
    CUDA_LAUNCH_BLOCKING: "1"

  # TRTLLM CLI config per mode
  trtllm_config:
    prefill:
      mem-fraction-static: 0.8
      chunked-prefill-size: 8192
    decode:
      mem-fraction-static: 0.9
```

| Field                 | Type   | Default | Description                             |
| --------------------- | ------ | ------- | --------------------------------------- |
| `type`                | string | -       | Must be "trtllm"                        |
| `prefill_environment` | dict   | {}      | Environment variables for prefill       |
| `decode_environment`  | dict   | {}      | Environment variables for decode        |
| `trtllm_config`       | object | null    | TRTLLM CLI configuration per mode       |

**Key differences from SGLang backend**:
- No aggregated mode support (prefill/decode only)
- Uses MPI-style launching (one srun per endpoint with all nodes)
- Uses `trtllm-llmapi-launch` for distributed launching
- Automatically sets `TRTLLM_EPLB_SHM_NAME` with unique UUID per endpoint

---

## benchmark

Benchmark configuration. The `type` field determines which benchmark runner is used and what additional fields are available.

### Available Benchmark Types

| Type              | Description                                    |
| ----------------- | ---------------------------------------------- |
| `manual`          | No benchmark (default), manual testing mode    |
| `sa-bench`        | Throughput/latency serving benchmark           |
| `sglang-bench`    | SGLang bench_serving benchmark                 |
| `mmlu`            | MMLU accuracy evaluation                       |
| `longbenchv2`     | Long-context evaluation benchmark              |
| `router`          | Router performance with prefix caching         |
| `mooncake-router` | KV-aware routing with Mooncake trace           |

### manual

No benchmark is run. Use for manual testing and debugging.

```yaml
benchmark:
  type: "manual"
```

### sa-bench (Serving Accuracy)

Throughput and latency benchmark at various concurrency levels.

```yaml
benchmark:
  type: "sa-bench"
  isl: 1024                          # Required: Input sequence length
  osl: 1024                          # Required: Output sequence length
  concurrencies: [256, 512]          # Required: Concurrency levels to test
  req_rate: "inf"                    # Optional: Request rate (default: "inf")
```

| Field           | Type        | Required | Default | Description                                |
| --------------- | ----------- | -------- | ------- | ------------------------------------------ |
| `isl`           | int         | Yes      | -       | Input sequence length                      |
| `osl`           | int         | Yes      | -       | Output sequence length                     |
| `concurrencies` | list/string | Yes      | -       | Concurrency levels (list or "NxM" format)  |
| `req_rate`      | string/int  | No       | "inf"   | Request rate                               |

**Concurrencies format**: Can be a list `[128, 256, 512]` or x-separated string `"128x256x512"`.

### sglang-bench

SGLang `bench_serving` benchmark at various concurrency levels.

```yaml
benchmark:
  type: "sglang-bench"
  isl: 1024                          # Required: Input sequence length
  osl: 1024                          # Required: Output sequence length
  concurrencies: [256, 512]          # Required: Concurrency levels to test
  req_rate: "inf"                    # Optional: Request rate (default: "inf")
```

| Field           | Type        | Required | Default | Description                                |
| --------------- | ----------- | -------- | ------- | ------------------------------------------ |
| `isl`           | int         | Yes      | -       | Input sequence length                      |
| `osl`           | int         | Yes      | -       | Output sequence length                     |
| `concurrencies` | list/string | Yes      | -       | Concurrency levels (list or "NxM" format)  |
| `req_rate`      | string/int  | No       | "inf"   | Request rate                               |

**Concurrencies format**: Can be a list `[128, 256, 512]` or x-separated string `"128x256x512"`.

### mmlu

MMLU accuracy evaluation using sglang.test.run_eval.

```yaml
benchmark:
  type: "mmlu"
  num_examples: 200                  # Optional: Number of examples
  max_tokens: 2048                   # Optional: Max tokens per response
  repeat: 8                          # Optional: Number of repeats
  num_threads: 512                   # Optional: Concurrent threads
```

| Field          | Type | Required | Default | Description                  |
| -------------- | ---- | -------- | ------- | ---------------------------- |
| `num_examples` | int  | No       | 200     | Number of examples to run    |
| `max_tokens`   | int  | No       | 2048    | Max tokens per response      |
| `repeat`       | int  | No       | 8       | Number of repeats            |
| `num_threads`  | int  | No       | 512     | Concurrent threads           |

### longbenchv2

Long-context evaluation benchmark.

```yaml
benchmark:
  type: "longbenchv2"
  max_context_length: 128000         # Optional: Max context length
  num_threads: 16                    # Optional: Concurrent threads
  max_tokens: 16384                  # Optional: Max tokens
  num_examples: null                 # Optional: Number of examples (all if null)
  categories:                        # Optional: Task categories
    - "multi_doc_qa"
    - "single_doc_qa"
```

| Field                | Type      | Required | Default | Description                    |
| -------------------- | --------- | -------- | ------- | ------------------------------ |
| `max_context_length` | int       | No       | 128000  | Max context length             |
| `num_threads`        | int       | No       | 16      | Concurrent threads             |
| `max_tokens`         | int       | No       | 16384   | Max tokens                     |
| `num_examples`       | int       | No       | all     | Number of examples             |
| `categories`         | list[str] | No       | all     | Task categories to run         |

### router

Router performance benchmark with prefix caching. **Requires `frontend.type: sglang`**.

```yaml
benchmark:
  type: "router"
  isl: 14000                         # Optional: Input sequence length
  osl: 200                           # Optional: Output sequence length
  num_requests: 200                  # Optional: Number of requests
  concurrency: 20                    # Optional: Concurrency level
  prefix_ratios: [0.1, 0.3, 0.5, 0.7, 0.9]  # Optional: Prefix ratios to test
```

| Field           | Type        | Required | Default                   | Description                |
| --------------- | ----------- | -------- | ------------------------- | -------------------------- |
| `isl`           | int         | No       | 14000                     | Input sequence length      |
| `osl`           | int         | No       | 200                       | Output sequence length     |
| `num_requests`  | int         | No       | 200                       | Number of requests         |
| `concurrency`   | int         | No       | 20                        | Concurrency level          |
| `prefix_ratios` | list/string | No       | "0.1 0.3 0.5 0.7 0.9"     | Prefix ratios to test      |

### mooncake-router

KV-aware routing benchmark using Mooncake conversation trace.

```yaml
benchmark:
  type: "mooncake-router"
  mooncake_workload: "conversation"  # Optional: Trace type
  ttft_threshold_ms: 2000            # Optional: Goodput TTFT threshold
  itl_threshold_ms: 25               # Optional: Goodput ITL threshold
```

| Field               | Type   | Required | Default        | Description                               |
| ------------------- | ------ | -------- | -------------- | ----------------------------------------- |
| `mooncake_workload` | string | No       | "conversation" | Trace type (see options below)            |
| `ttft_threshold_ms` | int    | No       | 2000           | Goodput TTFT threshold in ms              |
| `itl_threshold_ms`  | int    | No       | 25             | Goodput ITL threshold in ms               |

**Workload options**: `"mooncake"`, `"conversation"`, `"synthetic"`, `"toolagent"`

Dataset characteristics (conversation trace):
- 12,031 requests over ~59 minutes (3.4 req/s)
- Avg input: 12,035 tokens, Avg output: 343 tokens
- 36.64% cache efficiency potential

---

## dynamo

Dynamo installation configuration.

```yaml
dynamo:
  version: "0.8.0"            # Install from PyPI
  # OR
  hash: "abc123"              # Install from git commit
  # OR
  top_of_tree: true           # Install from main branch
```

| Field         | Type   | Default | Description                                            |
| ------------- | ------ | ------- | ------------------------------------------------------ |
| `install`     | bool   | true    | Whether to install dynamo (set false if pre-installed) |
| `version`     | string | "0.8.0" | PyPI version                                           |
| `hash`        | string | null    | Git commit hash (source install)                       |
| `top_of_tree` | bool   | false   | Install from main branch                               |

**Notes**:

- Set `install: false` if your container already has dynamo pre-installed.
- Only one of `version`, `hash`, or `top_of_tree` should be specified.
- `hash` and `top_of_tree` are mutually exclusive.
- When `hash` or `top_of_tree` is set, `version` is automatically cleared.
- Source installs (`hash` or `top_of_tree`) clone the repo and build with maturin.

---

## profiling

Profiling configuration for nsys or torch profiler.

```yaml
profiling:
  type: "nsys"                       # "none", "nsys", or "torch"

  # Phase-specific profiling step configs
  prefill:
    start_step: 10                   # Step to start profiling
    stop_step: 20                    # Step to stop profiling
  decode:
    start_step: 10
    stop_step: 20
  # OR for aggregated mode:
  aggregated:
    start_step: 10
    stop_step: 20
```

| Field         | Type   | Required | Default | Description                              |
| ------------- | ------ | -------- | ------- | ---------------------------------------- |
| `type`        | string | No       | "none"  | Profiling type: "none", "nsys", "torch"  |
| `prefill`     | object | Disaggregated | null | Prefill phase config                   |
| `decode`      | object | Disaggregated | null | Decode phase config                    |
| `aggregated`  | object | Aggregated | null | Aggregated phase config                  |

### ProfilingPhaseConfig

Each phase config has:

| Field        | Type | Required | Default | Description                    |
| ------------ | ---- | -------- | ------- | ------------------------------ |
| `start_step` | int  | No       | null    | Step to start profiling        |
| `stop_step`  | int  | No       | null    | Step to stop profiling         |

### Profiling Modes

- **nsys**: NVIDIA Nsight Systems profiling. Wraps worker command with `nsys profile`.
- **torch**: PyTorch profiler. Sets `SGLANG_TORCH_PROFILER_DIR` environment variable.

### Validation Rules

1. Disaggregated mode requires both `prefill` and `decode` phase configs when profiling is enabled.
2. Aggregated mode requires `aggregated` phase config when profiling is enabled.

### Example: Torch Profiling (Disaggregated)

```yaml
resources:
  gpu_type: "h100"
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 1
  decode_workers: 1

profiling:
  type: "torch"
  prefill:
    start_step: 5
    stop_step: 15
  decode:
    start_step: 5
    stop_step: 15
```

### Example: Nsys Profiling (Aggregated)

```yaml
resources:
  gpu_type: "h100"
  agg_nodes: 1
  agg_workers: 1

profiling:
  type: "nsys"
  aggregated:
    start_step: 10
    stop_step: 25
```

---

## output

Output configuration with formattable paths.

```yaml
output:
  log_dir: "./outputs/{job_id}/logs"
```

| Field     | Type            | Default                      | Description              |
| --------- | --------------- | ---------------------------- | ------------------------ |
| `log_dir` | FormattablePath | "./outputs/{job_id}/logs"    | Directory for log files  |

The `log_dir` supports FormattablePath templating. See [FormattablePath Template System](#formattablepath-template-system).

---

## health_check

Health check configuration for worker readiness.

```yaml
health_check:
  max_attempts: 180
  interval_seconds: 10
```

| Field              | Type | Default | Description                                      |
| ------------------ | ---- | ------- | ------------------------------------------------ |
| `max_attempts`     | int  | 180     | Maximum health check attempts (180 = 30 minutes) |
| `interval_seconds` | int  | 10      | Seconds between health check attempts            |

**Notes**:

- Default of 180 attempts at 10 second intervals = 30 minutes total wait time.
- Large models (e.g., 70B+ parameters) may require the full 30 minutes to load.
- Reduce `max_attempts` for smaller models or faster testing.

---

## infra

Infrastructure configuration for etcd/nats placement.

```yaml
infra:
  etcd_nats_dedicated_node: true
```

| Field                    | Type | Default | Description                                        |
| ------------------------ | ---- | ------- | -------------------------------------------------- |
| `etcd_nats_dedicated_node` | bool | false   | Reserve first node for infrastructure services     |

**Notes**:

- When `etcd_nats_dedicated_node: true`, the first allocated node is reserved exclusively for etcd and nats services.
- This can improve stability for large-scale deployments by isolating infrastructure services.
- The reserved node is not used for worker processes.

---

## sweep

Parameter sweep configuration for running multiple benchmark variations.

```yaml
sweep:
  mode: "zip"                        # "zip" or "grid"
  parameters:
    isl: [512, 1024, 2048]
    osl: [128, 256, 512]
```

| Field        | Type   | Default | Description                              |
| ------------ | ------ | ------- | ---------------------------------------- |
| `mode`       | string | "zip"   | Sweep mode: "zip" or "grid"              |
| `parameters` | dict   | {}      | Parameter name to list of values mapping |

### Sweep Modes

- **zip**: Pairs up parameters at matching indices. Parameters must have equal lengths.
  - Example: `isl=[512, 1024], osl=[128, 256]` produces 2 combinations:
    - `{isl: 512, osl: 128}`
    - `{isl: 1024, osl: 256}`

- **grid**: Cartesian product of all parameter values.
  - Example: `isl=[512, 1024], osl=[128, 256]` produces 4 combinations:
    - `{isl: 512, osl: 128}`
    - `{isl: 512, osl: 256}`
    - `{isl: 1024, osl: 128}`
    - `{isl: 1024, osl: 256}`

### Using Sweep Parameters

Reference sweep parameters in your config using `{placeholder}` syntax:

```yaml
benchmark:
  type: "sa-bench"
  isl: "{isl}"                       # Replaced by sweep value
  osl: "{osl}"                       # Replaced by sweep value
  concurrencies: [128, 256]

sweep:
  mode: "grid"
  parameters:
    isl: [512, 1024, 2048, 4096]
    osl: [128, 256, 512]
```

---

## Config Overrides

Config overrides let you define a base config plus multiple variants in a single YAML file. Each variant deep-merges a small set of changes onto the base, and is submitted as an independent SLURM job. This eliminates the need to duplicate entire config files when testing different parameter combinations.

### YAML Structure

```yaml
base:
  name: "my-benchmark"
  resources:
    decode_nodes: 8
  backend:
    sglang_config:
      decode:
        tp-size: 32
  benchmark:
    concurrencies: [8192, 10240]

override_tp64:
  backend:
    sglang_config:
      decode:
        tp-size: 64

override_small:
  resources:
    decode_nodes: 4
  benchmark:
    concurrencies: [4096]
```

| Key | Description |
|-----|-------------|
| `base` | Required. A complete, valid config (same structure as a normal recipe). |
| `override_<suffix>` | Optional. Partial config merged onto base. `<suffix>` is appended to the job name. |

### Naming

Override job names are auto-generated: `{base.name}_{suffix}`.

The example above produces three jobs: `my-benchmark`, `my-benchmark_tp64`, and `my-benchmark_small`.

### Deep Merge Semantics

| Type | Behavior | Example |
|------|----------|---------|
| **Scalar** (str/int/bool) | Override replaces base | `tp-size: 32` → `tp-size: 64` |
| **Dict** | Recursive merge — only specified keys change | Override `sglang_config.decode.tp-size: 64` leaves other decode keys untouched |
| **List** | Full replacement (no append) | `concurrencies: [4096]` replaces `[8192, 10240]` |
| **New key** | Added to base | Override adds fields base doesn't have |
| **`null` value** | Deletes the key from base | `extra_mount: null` removes it |

### Combining with Sweeps

Overrides and sweeps can coexist in the same file. Override expansion happens first, then each variant with a `sweep:` section is expanded via Cartesian product.

```yaml
base:
  name: "combined"
  sweep:
    chunked_prefill_size: [4096, 8192]
  backend:
    sglang_config:
      prefill:
        chunked-prefill-size: "{chunked_prefill_size}"

override_big:
  resources:
    decode_nodes: 16
```

This produces **4 jobs**: base × 2 sweep + override_big × 2 sweep.

### Backward Compatibility

Files without a `base` top-level key are treated as normal configs — no behavior change.

---

## FormattablePath Template System

FormattablePath is a powerful templating system for paths that supports runtime placeholders and environment variable expansion.

### How It Works

FormattablePath ensures that configuration values with placeholders are always explicitly formatted before use, preventing accidental use of unformatted templates.

```yaml
# Example usage in config
output:
  log_dir: "$HOME/logs/{job_id}/{run_name}"

container_mounts:
  "$HOME/data": "/data"
  "$HOME/logs/{job_id}": "/logs"
```

### Available Placeholders

| Placeholder         | Type   | Description                          | Example                        |
| ------------------- | ------ | ------------------------------------ | ------------------------------ |
| `{job_id}`          | string | SLURM job ID                         | "12345"                        |
| `{run_name}`        | string | Job name + job ID                    | "my-benchmark_12345"           |
| `{head_node_ip}`    | string | IP address of head node              | "10.0.0.1"                     |
| `{log_dir}`         | string | Resolved log directory path          | "/home/user/outputs/12345/logs"|
| `{model_path}`      | string | Resolved model path                  | "/models/deepseek-r1"          |
| `{container_image}` | string | Resolved container image path        | "/containers/sglang.sqsh"      |
| `{gpus_per_node}`   | int    | GPUs per node                        | 8                              |

### Environment Variable Expansion

FormattablePath also expands environment variables using `$VAR` or `${VAR}` syntax:

```yaml
output:
  log_dir: "$HOME/outputs/{job_id}/logs"
  # Expands to: /home/username/outputs/12345/logs
```

Common environment variables:
- `$HOME` - User home directory
- `$USER` - Username
- `$SLURM_JOB_ID` - SLURM job ID (also available as `{job_id}`)

### Extra Placeholders

Some contexts support additional placeholders:

| Placeholder       | Context           | Description                     |
| ----------------- | ----------------- | ------------------------------- |
| `{nginx_url}`     | Frontend config   | Nginx URL for load balancing    |
| `{frontend_url}`  | Frontend config   | Frontend/router URL             |
| `{index}`         | Worker config     | Worker index                    |
| `{host}`          | Worker config     | Worker host                     |
| `{port}`          | Worker config     | Worker port                     |

### Examples

```yaml
# Log directory with job ID
output:
  log_dir: "./outputs/{job_id}/logs"

# Mount user data into container
container_mounts:
  "$HOME/datasets": "/datasets"
  "./outputs/{job_id}": "/outputs"

# Custom paths with environment variables
extra_mount:
  - "$SCRATCH/cache:/cache"
  - "${DATA_DIR}/models:/models:ro"
```

---

## container_mounts

Custom container mount mappings with FormattablePath support.

```yaml
container_mounts:
  "$HOME/datasets": "/datasets"
  "$HOME/outputs/{job_id}": "/outputs"
  "/shared/cache": "/cache"
```

| Key (Host Path)     | Value (Container Path) | Description                       |
| ------------------- | ---------------------- | --------------------------------- |
| FormattablePath     | FormattablePath        | Host path -> Container mount path |

Both keys and values support FormattablePath templating with placeholders and environment variables.

### Default Mounts

The following mounts are always added automatically:

| Host Path              | Container Path       | Description                  |
| ---------------------- | -------------------- | ---------------------------- |
| Model path             | `/model`             | Resolved model directory     |
| Log directory          | `/logs`              | Log output directory         |
| `configs/` directory   | `/configs`           | NATS, etcd binaries          |
| Benchmark scripts      | `/srtctl-benchmarks` | Bundled benchmark scripts    |

### Cluster-Level Mounts

You can also define cluster-wide mounts in `srtslurm.yaml` using the `default_mounts` field. These are applied to all jobs on the cluster, after the built-in defaults but before job-level mounts.

```yaml
# In srtslurm.yaml
default_mounts:
  "/cluster/special/libs": "/opt/libs"
  "$SCRATCH": "/scratch"
```

Environment variables (e.g., `$SCRATCH`, `$HOME`) are expanded. This is useful for mounting cluster-specific paths that are required by certain images without adding them to every job config.

### Mount Priority

Mounts have the following priority (highest to lowest):

1. **Job-level `container_mounts`** - FormattablePath dict (highest priority)
2. **Job-level `extra_mount`** - simple `host:container` strings
3. **Cluster-level** - `default_mounts` from `srtslurm.yaml`
4. **Built-in defaults** - model, logs, configs, benchmark scripts (lowest priority)

Job-level mounts always take precedence over cluster-level and built-in defaults.

---

## environment

Global environment variables for all worker processes.

```yaml
environment:
  MY_VAR: "value"
  CUDA_LAUNCH_BLOCKING: "1"
  NCCL_DEBUG: "INFO"
```

| Key    | Value  | Description                      |
| ------ | ------ | -------------------------------- |
| string | string | Environment variable name=value  |

### Per-Worker Template Variables

Environment variable values support per-worker templating with these placeholders:

| Placeholder | Description                                    | Example      |
| ----------- | ---------------------------------------------- | ------------ |
| `{node}`    | Hostname of the node where the worker runs     | `"gpu-01"`   |
| `{node_id}` | Numeric index of the node in worker list (0-based) | `0`, `1`, `2` |

**Note**: For per-worker-mode environment variables, use `backend.prefill_environment`, `backend.decode_environment`, or `backend.aggregated_environment`.

---

## extra_mount

Additional container mounts as a list of mount specifications.

```yaml
extra_mount:
  - "/local/path:/container/path"
  - "/data:/data:ro"
  - "$HOME/cache:/cache"
```

| Format                        | Description                          |
| ----------------------------- | ------------------------------------ |
| `host_path:container_path`    | Read-write mount                     |
| `host_path:container_path:ro` | Read-only mount                      |

**Note**: Unlike `container_mounts`, `extra_mount` uses simple string format, not FormattablePath. Environment variables are still expanded.

---

## sbatch_directives

Additional SLURM sbatch directives.

```yaml
sbatch_directives:
  mail-user: "user@example.com"
  mail-type: "END,FAIL"
  comment: "Benchmark run for paper"
  reservation: "my-reservation"
  constraint: "volta"
  exclusive: ""                       # Flag without value
  gres: "gpu:8"
```

| Directive     | Example Value           | Description                           |
| ------------- | ----------------------- | ------------------------------------- |
| `mail-user`   | "user@example.com"      | Email for notifications               |
| `mail-type`   | "END,FAIL"              | When to send email (BEGIN,END,FAIL)   |
| `comment`     | "My job description"    | Job comment for tracking              |
| `reservation` | "my-reservation"        | Use a specific reservation            |
| `constraint`  | "volta"                 | Node feature constraint               |
| `exclusive`   | ""                      | Exclusive node access (flag)          |
| `gres`        | "gpu:8"                 | Generic resource specification        |
| `dependency`  | "afterok:12345"         | Job dependency                        |
| `qos`         | "high"                  | Quality of service                    |

**Format**: Each directive becomes `#SBATCH --{key}={value}` or `#SBATCH --{key}` if value is empty.

---

## srun_options

Additional srun options for worker processes.

```yaml
srun_options:
  cpu-bind: "none"
  mpi: "pmix"
  overlap: ""                         # Flag without value
  ntasks-per-node: "1"
```

| Option            | Example Value | Description                              |
| ----------------- | ------------- | ---------------------------------------- |
| `cpu-bind`        | "none"        | CPU binding mode (none, cores, sockets)  |
| `mpi`             | "pmix"        | MPI implementation                       |
| `overlap`         | ""            | Allow step overlap (flag)                |
| `ntasks-per-node` | "1"           | Tasks per node                           |
| `gpus-per-task`   | "1"           | GPUs per task                            |
| `mem`             | "0"           | Memory per node                          |

**Format**: Each option becomes `--{key}={value}` or `--{key}` if value is empty.

---

## setup_script

Run a custom script before dynamo install and worker startup.

```yaml
setup_script: "install-custom-deps.sh"
```

| Field          | Type   | Default | Description                              |
| -------------- | ------ | ------- | ---------------------------------------- |
| `setup_script` | string | null    | Script filename (must be in `configs/`)  |

**Notes**:

- Script must be located in the `configs/` directory.
- Script runs inside the container before dynamo installation.
- Useful for installing custom SGLang versions, additional dependencies, or patches.

**Example setup script** (`configs/install-sglang-main.sh`):

```bash
#!/bin/bash
pip install --quiet git+https://github.com/sgl-project/sglang.git
```

---

## enable_config_dump

Enable dumping worker configuration to JSON for debugging.

```yaml
enable_config_dump: true
```

| Field               | Type | Default | Description                          |
| ------------------- | ---- | ------- | ------------------------------------ |
| `enable_config_dump`| bool | true    | Dump config JSON for debugging       |

When enabled, worker startup commands include `--dump-config-to` which writes the resolved configuration to a JSON file.

---

## Complete Examples

### Disaggregated Mode with Dynamo

```yaml
name: "deepseek-r1-disagg"

model:
  path: "deepseek-r1"
  container: "0.5.6"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  gpus_per_node: 4
  prefill_nodes: 2
  prefill_workers: 4
  decode_nodes: 4
  decode_workers: 8

slurm:
  time_limit: "04:00:00"

frontend:
  type: dynamo
  enable_multiple_frontends: true
  args:
    router-mode: "kv"

backend:
  type: sglang

  kv_events_config:
    prefill: true

  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"

  sglang_config:
    prefill:
      tensor-parallel-size: 4
      mem-fraction-static: 0.84
      kv-cache-dtype: "fp8_e4m3"
    decode:
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

dynamo:
  version: "0.8.0"
```

### Aggregated Mode with SGLang Router

```yaml
name: "qwen-agg-router"

model:
  path: "qwen3-32b"
  container: "latest"
  precision: "bf16"

resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 4
  agg_workers: 8

slurm:
  time_limit: "02:00:00"

frontend:
  type: sglang
  enable_multiple_frontends: false
  args:
    policy: "cache_aware"

backend:
  type: sglang
  sglang_config:
    aggregated:
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
name: "profile-decode"

model:
  path: "llama-70b"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "h100"
  gpus_per_node: 8
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 1
  decode_workers: 1

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

backend:
  type: sglang
  sglang_config:
    prefill:
      tensor-parallel-size: 8
    decode:
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
name: "sweep-throughput"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  gpus_per_node: 4
  prefill_nodes: 1
  prefill_workers: 2
  decode_nodes: 2
  decode_workers: 4

benchmark:
  type: "sa-bench"
  isl: "{isl}"
  osl: "{osl}"
  concurrencies: [64, 128, 256]

sweep:
  mode: "grid"
  parameters:
    isl: [512, 1024, 2048, 4096]
    osl: [128, 256, 512, 1024]
```

### Config Override Example

```yaml
base:
  name: "disagg-fp8-benchmark"

  model:
    path: "deepseek-r1"
    container: "latest"
    precision: "fp8"

  resources:
    gpu_type: "h100"
    gpus_per_node: 8
    prefill_nodes: 2
    prefill_workers: 2
    decode_nodes: 8
    decode_workers: 8

  backend:
    sglang_config:
      prefill:
        tp-size: 8
      decode:
        tp-size: 8

  benchmark:
    type: "sa-bench"
    isl: 1024
    osl: 8192
    concurrencies: [8192, 10240]

# Use TP=64 for both prefill and decode
override_tp64:
  backend:
    sglang_config:
      prefill:
        tp-size: 64
      decode:
        tp-size: 64

# Smaller cluster with fewer decode nodes
override_small:
  resources:
    decode_nodes: 4
    decode_workers: 4
  benchmark:
    concurrencies: [4096]
```

### Custom Mounts and Setup

```yaml
name: "custom-setup"

model:
  path: "$MODELS_DIR/my-model"
  container: "$CONTAINERS_DIR/custom.sqsh"
  precision: "fp8"

resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 2
  agg_workers: 4

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
