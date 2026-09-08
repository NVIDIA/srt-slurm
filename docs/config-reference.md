# Configuration Reference

Complete reference for job configuration YAML files.

This page is the prose guide: what each block means, how the pieces interact, and worked examples. The authoritative field-by-field list (every key, type, and default) is generated from the code in [schema-reference.md](schema-reference.md) and checked in CI, so if this page and that one disagree, the generated one is right.

## Table of Contents

- [Overview](#overview)
- [Cluster Config Discovery](#cluster-config-discovery)
- [name](#name)
- [model](#model)
- [engine](#engine)
- [roles](#roles)
- [placement](#placement)
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
- [telemetry](#telemetry)
- [sweep](#sweep)
- [Config Overrides](#config-overrides)
- [FormattablePath Template System](#formattablepath-template-system)
- [container_mounts](#container_mounts)
- [environment](#environment)
- [extra_mount](#extra_mount)
- [sbatch_directives](#sbatch_directives)
- [srun_options](#srun_options)
- [setup_script](#setup_script)
- [host_setup](#host_setup)
- [post_eval](#post_eval)
- [services](#services)
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
| `gpus_per_node`                 | int    | Default GPUs per node (applied to recipes that omit `resources.gpus_per_node`) |
| `default_gpu_type`              | string | Default `resources.gpu_type` for recipes that omit it |
| `network_interface`             | string | Network interface for NCCL                            |
| `srtctl_root`                   | string | Root directory for srtctl                             |
| `output_dir`                    | string | Custom output directory (overrides srtctl_root/outputs) |
| `model_paths`                   | dict   | Model path aliases                                    |
| `containers`                    | dict   | Container image aliases, resolved for every image key in a recipe (see below) |
| `default_mounts`                | dict   | Cluster-wide container mounts                         |
| `default_bash_preamble`         | string | Shell snippet prepended to every container srun       |
| `default_host_setup`            | object | Commands run on every node's bare host, outside the container |
| `nginx_raise_ulimit`          | bool   | Optional default for `frontend.nginx_raise_ulimit`  |

**output_dir**: When set, job logs are written to `output_dir/{job_id}/logs` instead of `srtctl_root/outputs/{job_id}/logs`. Useful for CI/CD and ephemeral environments.

**containers**: A map from alias to image path or registry URI. One resolver walks the whole recipe and replaces any string under a `container`, `container_image`, `image`, or `nginx_container` key that matches an alias: `model.container`, `frontend.container_image`, `frontend.nginx_container`, `benchmark.container_image`, the Tachometer and power exporter images, `backend.mooncake_kv_store.container`, and any future block that names an image. Literal paths and registry URIs pass through untouched. Free-form maps (`environment`, `*_environment`, `env`, `args`, engine config blocks, `container_mounts`) and the `identity` block are never rewritten.

**default_bash_preamble**: A shell snippet (e.g. `"ulimit -n 1048576 -s unlimited -u 1048576"`) prepended to every container srun launched by srtctl — workers, frontends, telemetry, benchmark, postprocess. Runs before per-call `bash_preamble` and the main command, so cluster-wide ulimits apply to everything downstream. Silently dropped for distroless containers (e.g. `prom/node-exporter`) that bypass the bash wrapper; a WARNING log is emitted in that case.

**default_host_setup**: A [`host_setup`](#host_setup) block applied to every job on the cluster — for node state that has to be set outside the container, such as locking GPU clocks. A recipe that sets its own `host_setup:` block replaces it entirely; `host_setup: {commands: []}` opts a single run out.

**nginx_raise_ulimit**: When set to `true` or `false`, this value is applied to jobs that omit `frontend.nginx_raise_ulimit` in the recipe. Use `true` on clusters where raising the nginx container’s open-file limit is allowed; leave unset if each job should rely on the frontend default (`false`). A recipe that sets `frontend.nginx_raise_ulimit` always wins.

### Running without `srtslurm.yaml`

`srtslurm.yaml` is optional. A recipe can be fully self-sustaining as long as it supplies everything the cluster yaml would otherwise provide:

- Set `slurm.account`, `slurm.partition`, and `slurm.time_limit` directly in the recipe (no `default_*` fallback).
- Use absolute paths for `model.path`, `model.container`, and any other container fields — alias resolution is a no-op without the yaml's `containers:` / `model_paths:` maps.
- List every cluster-side mount the job needs in `extra_mount` (e.g. the lustre share that holds your model weights and `.sqsh` files). `default_mounts` is the only `srtslurm.yaml` field with no recipe-level equivalent until you spell mounts out yourself.
- Set `resources.gpus_per_node` explicitly.
- Status reporting and S3 log upload are skipped (their config lives under `reporting:` in the cluster yaml).

Workers' nats and etcd come from the dynamo/sglang container, not the yaml, so disagg/agg topologies still work end-to-end. `srtctl_root` falls back to the package install path automatically.

This is useful for portable recipes that you want to share across clusters or hand to a teammate without dragging cluster config along.

---

## schema

| Field    | Type    | Required | Description                                                                 |
| -------- | ------- | -------- | --------------------------------------------------------------------------- |
| `schema` | integer | No       | Recipe schema version. Absent means `1` (the pre-2.0 layout); `2` is current. |

Every supported version loads. Put the key first in the file, beside `base:` in an override file. Upgrade a recipe with `srtctl migrate -f recipe.yaml --in-place`, which preserves comments and key order and folds the legacy layout into `roles:`, `placement:`, and `dynamo.source` (a directory is walked recursively). `srtctl migrate --verify -f <path>` migrates in memory and checks that the v1 and v2 documents resolve to the same config; CI runs it over the examples and the historical recipe corpus (golden equality).

```yaml
schema: 2
name: "deepseek-r1-benchmark"
```

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

## engine

`engine:` names the inference engine that builds every worker role's command. A bare string is the common form; a mapping carries engine-wide knobs, the fields that are not per role:

```yaml
engine: sglang
```

```yaml
engine:
  type: vllm
  connector: nixl               # vLLM KV connector for disaggregation
```

```yaml
engine:
  type: trtllm
  served_model_name: "Qwen/Qwen3-0.6B"
```

```yaml
engine:
  type: mocker
  engine_type: vllm
  speedup_ratio: 100
```

Valid types are `sglang`, `vllm`, `trtllm`, and `mocker`. `engine` is normalized into the internal `backend` block before validation (`engine: sglang` is `backend: {type: sglang}`), so the per-engine field tables under [backend](#backend) still describe the engine-wide knobs; only the per-role parts (`<mode>_environment`, `<engine>_config.<mode>`, `<mode>_extra_args`, `kv_events_config`) have moved into [roles](#roles). A v2 recipe needs no `backend:` block. `backend:` still loads as the v1 spelling and `srtctl migrate` rewrites it.

---

## roles

`roles:` is the 2.0 way to describe a worker role. It groups everything about a role in one place instead of spreading it across `resources`, `backend.*_environment`, and `backend.<engine>_config.*`:

```yaml
roles:
  prefill:
    nodes: 2          # -> resources.prefill_nodes
    workers: 6        # -> resources.prefill_workers
    gpus: 2           # -> resources.gpus_per_prefill
    env:              # -> backend.prefill_environment
      PYTHONUNBUFFERED: "1"
    args:             # -> backend.<engine>_config.prefill (engine from backend.type)
      tensor-parallel-size: 2
      disaggregation-mode: prefill
  decode:
    nodes: 0          # 0 shares the prefill node's spare GPUs
    workers: 2
    gpus: 2
    env:
      PYTHONUNBUFFERED: "1"
    args:
      tensor-parallel-size: 2
      disaggregation-mode: decode
```

`env` and `args` are ordinary YAML mappings, written exactly as `backend.prefill_environment` and `backend.sglang_config.prefill` were. Nothing needs JSON or inline `{}` syntax.

Role names are `prefill`, `decode`, and `agg`. The aggregated role is `agg` (matching `resources.agg_*`); its `env` and `args` map to `backend.aggregated_environment` and `backend.<engine>_config.aggregated`. Per-role `extra_args` maps to `backend.<mode>_extra_args` (TRT-LLM). `roles:` is normalized into those fields before validation, so it is exactly equivalent to writing them directly; you cannot set both for the same role.

Two more per-role keys replace job-wide knobs:

| Key | Maps to | Notes |
| --- | --- | --- |
| `kv_events` | `backend.kv_events_config.<mode>` | `true` for the default ZMQ publisher, or a mapping with `publisher` / `topic`; set per role instead of one job-wide flag |
| `sidecar` | `dynamo.sidecar` | `true` runs the native engine with a Dynamo sidecar; every role must agree because the mode is job-wide, the sidecar knobs (`sidecar_port`, ...) stay under `dynamo` |
| `engine` | `backend.type` | Optional; must equal the top-level [engine](#engine) when both are given |

The legacy fields (`resources.prefill_workers`, `backend.prefill_environment`, `backend.sglang_config.prefill`, ...) still load unchanged, so v1 recipes keep working, and both forms are valid v2. `srtctl migrate -f recipe.yaml --in-place` rewrites the legacy layout into `roles:` (and `placement:` / `dynamo.source`), preserving comments and key order; `srtctl migrate --verify -f <dir>` proves that every recipe under a directory resolves to the same config before and after. The `examples/` are written with `roles:` (except `features/override.yaml`, kept legacy to show that the v1 layout still loads).

---

## placement

`placement:` is one vocabulary for where the frontend and the benchmark client run, replacing the per-block placement knobs:

```yaml
frontend:
  placement:
    node: head          # head | first_decode | dedicated
benchmark:
  placement:
    node: last_decode   # head | last_decode | dedicated
```

The discovery plane (etcd, NATS) is placed through its services: an `etcd` or `nats` entry under [`services`](#services) with `placement.node: dedicated`. See [Implicit Services](services.md#implicit-services).

`node: dedicated` reserves a node for that component (and implies the head location, which the legacy validation already required). Any other value is a location string.

| Block | `node: dedicated` sets | `node: <location>` sets |
| --- | --- | --- |
| `frontend` | `frontend.dedicated_node: true` + `orchestrator_placement: head` | `frontend.orchestrator_placement: <location>` |
| `benchmark` | `benchmark.client_dedicated_node: true` + `client_placement: head` | `benchmark.client_placement: <location>` |

Like `roles:`, this is normalized into the existing fields before validation, so it is exactly equivalent to writing them, cannot be combined with them for the same block, and the legacy fields still load.

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
| `gpu_type`        | string | `default_gpu_type` | GPU type, e.g. "gb200", "gb300", "h100". Optional; inherits `default_gpu_type` from `srtslurm.yaml` when omitted |
| `gpus_per_node`   | int    | cluster / 4        | GPUs per node; inherits the cluster `gpus_per_node` when omitted, else 4 |
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

### CPU allocation visibility

srtctl records both the requested GPU topology and the effective CPU allocation. At runtime it:

- logs `SLURM_JOB_CPUS_PER_NODE`, `SLURM_CPUS_ON_NODE`, and process CPU affinity;
- writes `logs/resource_snapshot.json` with per-node/total CPUs, backend/configured GPUs, the warning threshold, and verdict;
- adds the snapshot to `lock.resource_snapshot` in `recipe.lock.yaml`;
- adds CPU allocation and warning state to the job metadata used by `srtctl monitor`; and
- records CPU model, logical CPU count, affinity, and SLURM CPU variables in each worker fingerprint beside GPU details.

The warning uses a fixed, conservative baseline of one effective CPU per backend GPU. For example, a four-GPU backend that receives only two CPUs produces a prominent `CPU ALLOCATION WARNING` before services start. Increase the request with the appropriate cluster policy, such as `cpus-per-task`, `cpus-per-gpu`, or an exclusive-node directive.

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
  # Frontend type: "dynamo" (default), "sglang", "vllm-router", "trtllm_serve", or "vllm"
  type: dynamo

  # Scaling
  enable_multiple_frontends: true     # Enable nginx + multiple routers
  num_additional_frontends: 9         # Additional routers (total = 1 + this)

  # Optional: raise nofile for nginx (shell ulimit + worker_rlimit_nofile in nginx.conf).
  # Default false. Set true on clusters that allow it; can also set nginx_raise_ulimit in srtslurm.yaml.
  # nginx_raise_ulimit: true

  # CLI args passed to the frontend/router
  args:
    router-mode: "kv"                 # dynamo: router-mode
    policy: "cache_aware"             # sglang: policy
    no-kv-events: true                # boolean flags

  # Environment variables for frontend processes
  env:
    MY_VAR: "value"

  # Optional static-router image; defaults to model.container
  # container_image: vllm-router
```

| Field                       | Type | Default       | Description                         |
| --------------------------- | ---- | ------------- | ----------------------------------- |
| `type`                      | str  | dynamo        | Frontend type: "dynamo", "sglang", "vllm-router", "trtllm_serve", or "vllm" |
| `enable_multiple_frontends` | bool | true          | Scale with nginx + multiple routers |
| `num_additional_frontends`  | int  | 9             | Additional routers beyond master    |
| `nginx_container`           | str  | nginx:1.27.4  | Custom nginx container image        |
| `nginx_raise_ulimit`      | bool | false         | When true with nginx in use, run `ulimit -n 1048576` before nginx and emit `worker_rlimit_nofile 1048576` in generated `nginx.conf`. Off by default so restrictive clusters do not fail. Cluster `srtslurm.yaml` may set `nginx_raise_ulimit` for jobs that omit this field. |
| `args`                      | dict | null          | CLI args for the frontend           |
| `env`                       | dict | null          | Env vars for frontend processes     |
| `container_image`           | str  | null          | Static-router image; falls back to `model.container` |

See [SGLang Router](sglang-router.md) for detailed architecture.

### trtllm_serve frontend

`type: trtllm_serve` runs the `trtllm-serve disaggregated` orchestrator as the
router (for `backend.type: trtllm`). Instead of the dynamo request plane, srtctl
collects the prefill/decode worker addresses and writes a static `ser.yaml`
(`context_servers` = prefill, `generation_servers` = decode), then launches the
orchestrator on the head node. The trtllm workers are started as `trtllm-serve`
OpenAI servers rather than `dynamo.trtllm`.

Because the orchestrator is a single process, set
`enable_multiple_frontends: false` (the nginx + multi-router path is not
supported). A configuration can be switched between the two TRT-LLM serving stacks by
changing only `frontend.type` between `dynamo` and `trtllm_serve`; start from the
`examples/trtllm/dynamo-disagg.yaml` and `examples/trtllm/trtllm-serve-disagg.yaml` examples.

### vllm frontend

`type: vllm` runs aggregate vLLM jobs **without Dynamo**. The OpenAI-compatible
HTTP server is the aggregate `vllm serve` worker itself — there is no separate
router/frontend process, and srtctl skips NATS/etcd startup.

Use this for aggregate throughput benchmarks where Dynamo orchestration is not
needed. Disaggregated prefill/decode layouts still require a real router such as
Dynamo (`frontend.type: dynamo`).

**Requirements**

| Constraint | Value |
| ---------- | ----- |
| `backend.type` | `vllm` |
| Job layout | Aggregate only; no prefill/decode workers |
| `agg_workers` | Exactly `1` — scale across nodes with `agg_nodes`, not with replicas |
| `enable_multiple_frontends` | `false` (nginx + multi-router path is unsupported) |

Nothing load-balances between aggregate endpoints here, so `agg_workers: 2` is
rejected at load time: the extra replica would either idle behind the single
public address or collide on the port. Use `frontend.type: dynamo` when you want
several aggregate replicas behind one endpoint.

**Single-node example**

```yaml
frontend:
  type: vllm
  enable_multiple_frontends: false

resources:
  agg_nodes: 1
  agg_workers: 1
  gpus_per_node: 8

backend:
  type: vllm
  vllm_config:
    aggregated:
      tensor-parallel-size: 8
```

**Multi-node example (TP/PP across nodes)**

```yaml
frontend:
  type: vllm
  enable_multiple_frontends: false

resources:
  agg_nodes: 2
  agg_workers: 1
  gpus_per_node: 8

backend:
  type: vllm
  vllm_config:
    aggregated:
      tensor-parallel-size: 8
      pipeline-parallel-size: 2
```

srtctl launches one `vllm serve` process per node. The endpoint leader
(`node_rank=0`) binds the public OpenAI port; follower ranks run headless engine
workers. Multi-node coordination flags (`--master-addr`, `--nnodes`,
`--node-rank`, `--headless`) are derived from the allocated topology — **do not
set them in the recipe**.

`master-port` / `master_port` remains an optional recipe override and is passed
to every node rank. Set it when jobs may share a leader node and need distinct
vLLM rendezvous ports; otherwise vLLM's default is used.

**Topology-managed `vllm_config` keys**

The following keys are owned by srtctl and are stripped at runtime if present in
`vllm_config.{aggregated,prefill,decode}`:

- `headless`
- `host`, `port`
- `master-addr` / `master_addr`
- `nnodes`
- `node-rank` / `node_rank`

Existing recipes that still contain these keys generally continue to work
because the values are ignored. One exception is `headless` combined with the
default `dp_launch_mode: per_node` and `data-parallel-size`: backend validation
rejects that combination before direct-vLLM command construction, so remove
`headless` from such recipes. `srtctl dry-run` emits a **WARNING** for each
accepted key so operators can clean up recipes over time.

Health checks, benchmark clients, and `SRT_FRONTEND_HOST` target the **aggregate
endpoint leader** (the node running the public `vllm serve`), not necessarily the
Slurm head node.

Compare with `frontend.type: dynamo` + `backend.type: vllm`, which keeps Dynamo as
the request router and uses `python3 -m dynamo.vllm` workers with NATS/etcd.

### vllm-router frontend

`type: vllm-router` launches the official vLLM Router in front of direct
`vllm serve` workers. It supports aggregate replicas and disaggregated P/D
topologies without Dynamo or NATS/etcd. See [vLLM Router](vllm-router.md) for
complete topology examples and the division of responsibility between the
upstream vLLM backend topology and Router adapter.

---

## backend

Worker configuration and SGLang settings.

**v1 spelling.** In 2.0 the engine type and engine-wide knobs live under [engine](#engine) and the per-mode fields under [roles](#roles); `backend:` is still accepted so v1 recipes load unchanged, and `srtctl migrate` rewrites it. The field tables below remain the reference for each engine's knobs.

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

### vLLM DP launch mode

vLLM data-parallel endpoints use one process per node by default. srtslurm
derives whether each TP/PP replica is node-local or spans multiple nodes:

```yaml
backend:
  type: vllm
  vllm_config:
    prefill:
      data-parallel-size: 8
    decode:
      data-parallel-size: 16
```

| Value      | Process layout                                                               |
| ---------- | ---------------------------------------------------------------------------- |
| `per_node` | One process per node (default); supports node-local or distributed TP/PP      |
| `per_gpu`  | One process per DP rank (TP×PP GPUs each; deprecated compatibility mode)     |

Set `backend.dp_launch_mode: per_gpu` only when temporarily preserving the
legacy process layout. srtslurm emits a configuration-time deprecation warning
for Dynamo-backed DP configurations that select it. `per_gpu` will be removed
in a future release.

When `TP x PP` fits on one node, srtslurm derives
`--data-parallel-size-local` and `--data-parallel-start-rank`, then enables
`--data-parallel-hybrid-lb` so every node-local process registers with the
Dynamo frontend. When `TP x PP` is larger than the node-local GPU allocation,
srtslurm instead derives the multi-node rendezvous arguments and makes every
process except the global leader headless. For example, both DP4 x TP4 and
DP2 x TP8 are selected automatically on four-GPU nodes.

Do not set `data-parallel-size-local`, `data-parallel-start-rank`,
`data-parallel-hybrid-lb`, or `headless` manually; srtslurm owns those values.
The allocation must be regular: `DP x TP x PP` must match the endpoint GPU
count, and a TP/PP replica must divide evenly within or across nodes.

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

**Per-type fields (schema 2).** Every type accepts the shared fields (`client_placement`, `client_dedicated_node`, `colocate_with_frontend`, `sweep`, `aiperf_package`, `aiperf_args`, `export_node_metrics`) plus the fields its runner reads:

| `type` | Fields |
| --- | --- |
| `sa-bench` | `isl`, `osl`, `concurrencies`, `req_rate`, `random_range_ratio`, `num_prompts_mult`, `num_warmup_mult`, `dataset_name`, `dataset_path`, `custom_tokenizer`, `use_chat_template`, `reuse_http_connections`, `slow_down_sleep_time`, `slow_down_wait_time` |
| `sglang-bench` | `isl`, `osl`, `concurrencies`, `req_rate` |
| `gsm8k` | `num_examples`, `max_tokens`, `repeat`, `num_threads`, `num_shots`, `temperature`, `top_p`, `top_k` |
| `mmlu`, `gpqa` | `num_examples`, `max_tokens`, `repeat`, `num_threads` |
| `longbenchv2` | `num_examples`, `max_tokens`, `num_threads`, `max_context_length`, `categories` |
| `router` | `isl`, `osl`, `num_requests`, `concurrency`, `prefix_ratios` |
| `mooncake-router` | `mooncake_workload`, `ttft_threshold_ms`, `itl_threshold_ms` |
| `trace-replay` | `concurrencies`, `ttft_threshold_ms`, `itl_threshold_ms`, `trace_file` |
| `agentperf` | `isl`, `concurrencies`, `concurrency`, `agentperf_client_dir`, `agentperf_config`, `container_image`, `env` |
| `custom` | `command`, `container_image`, `env` |
| `lm-eval`, `manual` | shared fields only |

A `schema: 2` recipe that sets a field its type does not use is rejected at load with the list of accepted fields. A schema 1 recipe gets a warning and keeps loading. Before this, such a field was a silent no-op (`isl` on `gsm8k`, `num_shots` on `sa-bench`). Each runner declares its fields as `config_fields`; adding a field to a runner means adding it there.

### Post-process: node metrics CSV

When `export_node_metrics` is `true`, after the benchmark finishes srtctl prepends
`srtctl_root` to `sys.path` and calls `analysis.srtlog.export_node_metrics.export_node_metrics`
in-process on the job output directory. That writes per-node batch CSVs and `gen_throughput.csv`
under `logs/node_metrics/` (next to worker logs).

- Set **`srtctl_root`** in `srtslurm.yaml` to the srt-slurm repository root (the directory that contains `analysis/srtlog/`). This path is inserted at the front of `sys.path` for the import.
- The export process needs **`pandas`** and **`pyarrow`** (same as the analysis dashboard).

```yaml
benchmark:
  type: "sa-bench"
  export_node_metrics: true   # default: false
  # ... other benchmark fields
```

| Field                  | Type | Default | Description                                      |
| ---------------------- | ---- | ------- | ------------------------------------------------ |
| `export_node_metrics`  | bool | `false` | Export node batch CSVs + gen throughput summary |

### Available Benchmark Types

| Type              | Description                                    |
| ----------------- | ---------------------------------------------- |
| `manual`          | No benchmark (default), manual testing mode    |
| `custom`          | Arbitrary command with runtime endpoint metadata |
| `sa-bench`        | Throughput/latency serving benchmark           |
| `sglang-bench`    | SGLang bench_serving benchmark                 |
| `mmlu`            | MMLU accuracy evaluation                       |
| `gpqa`            | GPQA (Graduate-level science QA) evaluation    |
| `longbenchv2`     | Long-context evaluation benchmark              |
| `router`          | Router performance with prefix caching         |
| `mooncake-router` | KV-aware routing with Mooncake trace           |
| `agentperf`       | AgentPerf trajectory replay (agentperf-client) |

### manual

No benchmark is run. Use for manual testing and debugging.

For a one-off serving run, `srtctl apply -f config.yaml --serve-only` provides the same behavior without changing
the recipe's configured benchmark.

```yaml
benchmark:
  type: "manual"
```

### custom

Run an arbitrary command with `bash -lc`. The command is passed verbatim; srt-slurm does not
expand `{placeholder}` expressions. Use environment variables for runtime-discovered values:

```yaml
benchmark:
  type: custom
  command: >-
    ./run-benchmark.sh "$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT"
  env:
    MY_BENCHMARK_OPTION: "value"
```

Every custom benchmark command receives frontend metadata plus mode-specific metadata for each
logical worker leader:

| Variable                        | Format                         | Description |
| ------------------------------- | ------------------------------ | ----------- |
| `SRT_FRONTEND_HOST`             | IP                             | Frontend/orchestrator IP |
| `SRT_FRONTEND_PORT`             | port                           | Frontend public port |
| `SRT_PREFILL_IPS`               | comma-separated IPs            | Prefill worker leader IPs |
| `SRT_PREFILL_ENDPOINTS`         | comma-separated `IP:port`      | Prefill worker endpoints |
| `SRT_DECODE_IPS`                | comma-separated IPs            | Decode worker leader IPs |
| `SRT_DECODE_ENDPOINTS`          | comma-separated `IP:port`      | Decode worker endpoints |
| `SRT_AGG_IPS`                   | comma-separated IPs            | Aggregated worker leader IPs |
| `SRT_AGG_ENDPOINTS`             | comma-separated `IP:port`      | Aggregated worker endpoints |
| `AIPERF_SERVER_METRICS_URLS`    | comma-separated HTTP URLs      | AIPerf-compatible `/metrics` URLs for all logical workers |

Only variables for modes present in the recipe are emitted. Entries follow logical topology order
(prefill index, decode index, or aggregated index). Multi-node follower ranks are excluded because
they do not own separate engines; co-located logical workers retain repeated IPs and distinct ports
so list positions remain aligned. With a Dynamo frontend, endpoint and metrics URLs use each
leader's `DYN_SYSTEM_PORT`; other frontends use the worker HTTP port. If KVBM metrics are configured,
their URLs are appended to `AIPERF_SERVER_METRICS_URLS` after the logical worker URLs.

Two caveats for `AIPERF_SERVER_METRICS_URLS`:

- **TRT-LLM worker URLs are omitted when the workers publish no metrics.** A TRT-LLM worker
  launched without `--publish-events-and-metrics` (the default; `observability.enabled` turns it
  on) serves nothing on its sys-port `/metrics`, so those URLs are not advertised. KVBM URLs are
  unaffected — KVBM serves its own endpoint regardless of the flag.
- **An explicit `AIPERF_SERVER_METRICS_URLS` in the recipe `environment:` wins.** Injection is
  skipped when the variable is already set, so a curated endpoint list is never clobbered.

Values in `benchmark.env` are applied last and can explicitly override any automatically injected
variable.

### sa-bench (Serving Accuracy)

Throughput and latency benchmark at various concurrency levels.

```yaml
benchmark:
  type: "sa-bench"
  isl: 1024                          # Required: Input sequence length
  osl: 1024                          # Required: Output sequence length
  concurrencies: [256, 512]          # Required: Concurrency levels to test
  req_rate: "inf"                    # Optional: Request rate (default: "inf")
  reuse_http_connections: false      # Optional: Reuse HTTP connections (default: false)
```

| Field                    | Type        | Required | Default | Description                                                   |
| ------------------------ | ----------- | -------- | ------- | ------------------------------------------------------------- |
| `isl`                    | int         | Yes      | -       | Input sequence length                                         |
| `osl`                    | int         | Yes      | -       | Output sequence length                                        |
| `concurrencies`          | list/string | Yes      | -       | Concurrency levels (list or "NxM" format)                     |
| `req_rate`               | string/int  | No       | "inf"   | Request rate                                                  |
| `reuse_http_connections` | bool        | No       | `false` | Reuse a process-scoped HTTP pool for the SA-Bench Dynamo adapter |

**Concurrencies format**: Can be a list `[128, 256, 512]` or x-separated string `"128x256x512"`.

When `reuse_http_connections` is enabled, each `benchmark_serving.py` process
uses one keep-alive connection pool. Warmup and formal runs remain isolated in
separate processes and therefore never share a pool. The option currently
applies only to SA-Bench's Dynamo HTTP adapter.

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

### gpqa

Graduate-level science QA evaluation using sglang.test.run_eval.

```yaml
benchmark:
  type: "gpqa"
  num_examples: 198                  # Optional: Number of examples
  max_tokens: 32768                  # Optional: Max tokens per response
  repeat: 8                          # Optional: Number of repeats
  num_threads: 128                   # Optional: Concurrent threads
```

| Field          | Type | Required | Default | Description                  |
| -------------- | ---- | -------- | ------- | ---------------------------- |
| `num_examples` | int  | No       | 198     | Number of examples to run    |
| `max_tokens`   | int  | No       | 32768   | Max tokens per response      |
| `repeat`       | int  | No       | 8       | Number of repeats            |
| `num_threads`  | int  | No       | 128     | Concurrent threads           |

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

### agentperf

Trajectory-replay benchmark using the standalone
[agentperf-client](https://github.com/ArtificialAnalysis-External/agentperf-client) — a deterministic
agentic load generator with a Rust streaming core. The client checkout is mounted into the container
(pin the commit for comparable runs); the workload definition (trajectory dataset, user-assignments
file, `settling_time_seconds`, `phase_timeout_seconds`, stop criteria) lives in the client's own
config YAML. srtctl injects the endpoint, model and concurrency at run time via the client's
`--base-url` / `--model` / `--concurrencies` flags. Note the client validates the workload YAML
*before* merging CLI overrides, so the YAML must still carry syntactically valid placeholder
`base_url`, `model` and `concurrencies` values — and `phase_timeout_seconds` must satisfy the
client's ramp-up bound for the *injected* concurrency
(`phase_timeout_seconds >= (concurrency - 1) / user_spawn_rate + settling_time_seconds +
min_measurement_seconds`).

```yaml
benchmark:
  type: "agentperf"
  agentperf_client_dir: "/agentperf-client"       # Container path to the client checkout
  agentperf_config: "/workloads/agentperf.yaml"   # Container path to the client's workload YAML
  concurrencies: [1010]                           # One benchmark phase per level
  env:
    AGENTPERF_EXTRA_ARGS: "--seed 100"            # Optional: appended to agentperf/run.py verbatim

extra_mount:
  - "/path/on/host/agentperf-client:/agentperf-client"
  - "/path/on/host/workloads:/workloads"
```

| Field                  | Type        | Required | Default | Description                                            |
| ---------------------- | ----------- | -------- | ------- | ------------------------------------------------------ |
| `agentperf_client_dir` | string      | Yes      | —       | Container path to an agentperf-client checkout         |
| `agentperf_config`     | string      | Yes      | —       | Container path to the client's workload YAML           |
| `concurrencies`        | list/string | Yes*     | —       | Levels, one client phase each; string form is x-separated (`"64x1010"`), matching other benchmark types |
| `concurrency`          | int         | Yes*     | —       | Single level (alternative to `concurrencies`)          |

*One of `concurrency` / `concurrencies` is required.

Notes:
- The first run of a job builds an isolated client runtime under `/tmp/agentperf-<jobid>`
  (uv env, pinned Rust toolchain, `rustcore` extension, tokenizer cache) and stages the trajectory
  and user-assignments datasets from shared storage to node-local `/tmp` — this preflight needs
  network egress from the benchmark node and adds several minutes before the first phase.
- The user-assignments file referenced by the workload YAML must cover the highest concurrency
  level (`assign_trajectories` fails loudly otherwise).
- Results land under `<log_dir>/agentperf/` (per-phase `*__traj*.{jsonl,txt,json}`,
  `requests.jsonl`, `phase_manifest.jsonl`); `rollup.py` normalizes them into
  `benchmark-rollup.json`.
- Two runs must not share a results dir concurrently (the client resets `phase_manifest.jsonl`
  at start).
- `telemetry:` (DCGM power measurement windows) is not supported with agentperf — the schema
  rejects non-sa-bench benchmark types at config load. Tachometer
  (`observability.enabled`) works normally.

---

## dynamo

Dynamo installation configuration.

```yaml
dynamo:
  source:                     # 2.0: one block for where Dynamo comes from
    git: https://github.com/ai-dynamo/dynamo
    rev: refs/pull/14000/head # a commit, a tag, or a PR head; never a branch name
    # sha: <filled in by srtctl apply>
  sidecar: false              # Use native engines with Dynamo sidecars
```

```yaml
dynamo:
  source:
    pypi: "1.4.2"             # a release from PyPI
```

```yaml
dynamo:
  source:
    wheel: "1.5.0.dev20260901" # a staged nightly wheel
```

| Field                    | Type         | Default | Description                                            |
| ------------------------ | ------------ | ------- | ------------------------------------------------------ |
| `install`                | bool         | true    | Whether to install dynamo (set false if pre-installed) |
| `source`                 | object       | null    | Exactly one of `git` + `rev` (optionally `patches`, `sha`), `pypi`, or `wheel`; see below |
| `version`                | string       | "0.8.0" | Legacy: PyPI version (same as `source.pypi`)           |
| `hash`                   | string       | null    | Legacy: git commit hash (same as `source.git` + `rev`) |
| `top_of_tree`            | bool         | false   | Legacy: install from main branch                       |
| `wheel`                  | string       | null    | Legacy: exact `ai-dynamo` nightly version (same as `source.wheel`) |
| `cargo_patches`          | list[string] | null    | Legacy: Cargo dependency replacements (same as `source.patches`) |
| `sidecar`                | bool         | false   | Replace legacy Python workers with native engines and Dynamo sidecars |
| `sidecar_port`           | int          | 50051   | Base loopback gRPC port; co-located workers receive deterministic offsets |
| `sidecar_binary`         | string/null  | null    | Optional standalone executable; null uses `python3 -m dynamo.<framework>.sidecar` |
| `sidecar_args`           | list[string] | []      | Extra arguments passed to the sidecar launcher         |
| `sidecar_startup_timeout` | int         | 1200    | Seconds to wait for the native gRPC endpoint            |
| `sidecar_context_length` | int/null     | null    | TRT-LLM context length override                         |

**Notes**:

- Set `install: false` if your container already has dynamo pre-installed.
- `source` is the same shape `services[].source` uses. `git` defaults to the upstream repository when only `rev` is given, so a fork is `git: https://github.com/<you>/dynamo`.
- `rev` must be immutable: a commit SHA, a tag such as `v1.4.2`, or `refs/pull/<n>/head` for an unmerged PR. `main`, `master`, and `HEAD` are rejected; use `top_of_tree: true` if you really want a moving target.
- `srtctl apply` resolves a non-commit `rev` with `git ls-remote`, writes the commit as `source.sha` into the submitted `config.yaml` (comments preserved, the recipe on disk is untouched), and echoes it as `pinned_sources` in `--json` output. The job builds that commit and the `/configs/dynamo-wheels` cache is keyed by it, so two runs of one recipe cannot silently build different code because the PR moved. If the login node cannot reach the remote, the submit continues with a warning and the compute node fetches the ref by name.
- `source` cannot be combined with `hash`, `top_of_tree`, `wheel`, or `cargo_patches`. The legacy fields keep working unchanged; `source` maps onto them at load, so nothing downstream changes.
- Source installs (`source.git`, `hash`, or `top_of_tree`) clone the repo and build with maturin; `patches` / `cargo_patches` replace Cargo dependency declarations tree-wide before the build.
- `srtctl dry-run` prints the resolved Dynamo source.

### Native sidecar mode

Set `dynamo.sidecar: true` to run the framework's native engine process beside a CPU-only Dynamo sidecar instead of launching `python3 -m dynamo.<framework>`. The engine and sidecar share one Slurm step and have a coupled lifecycle: if either exits, srtctl terminates the other and marks the worker failed.

By default, srtctl launches `python3 -m dynamo.<framework>.sidecar`. The `ai-dynamo` package supplies this module and pins the matching `ai-dynamo-runtime` wheel, which embeds the native Rust sidecar. The configured Dynamo version, wheel, source hash, or preinstalled container runtime must include the selected framework's launcher. No separate Cargo build is performed at job startup.

Nightly deployments should select an exact `dynamo.wheel` version so srtctl stages and installs the matching `ai-dynamo` and `ai-dynamo-runtime` artifacts on every worker. Set `dynamo.sidecar_binary` only to launch a compatible standalone executable already present in the container or a bind mount.

```yaml
frontend:
  type: dynamo

backend:
  type: vllm  # sglang, vllm, or trtllm

dynamo:
  wheel: "<nightly-with-sidecars>"
  sidecar: true
  sidecar_port: 50051
  sidecar_args:
    - --grpc-connections
    - "4"
```

The default sidecar commands are `python3 -m dynamo.sglang.sidecar`, `python3 -m dynamo.vllm.sidecar`, and `python3 -m dynamo.trtllm.sidecar`. All three use the shared `--grpc-endpoint` flag.

SGLang exposes gRPC and starts the sidecar only on an endpoint leader; distributed followers are engine-only. vLLM automatically uses one managed process per node for data-parallel endpoints and exposes the complete DP group through the leader's sidecar. Multi-node tensor-parallel vLLM endpoints remain rejected until their `vllm-rs` launch path is validated. TensorRT-LLM supports sidecars for aggregated workers only and runs the sidecar on MPI rank zero. `dynamo.sidecar_context_length` can override the TRT-LLM context length inferred from `trtllm_config.aggregated.max_seq_len`.

vLLM sidecar mode sets `VLLM_PLUGINS` to an empty value by default. This prevents image-installed plugins from replacing native engine output types that must match the fixed `vllm-rs` MessagePack contract. A recipe can explicitly set `VLLM_PLUGINS` in `prefill_environment`, `decode_environment`, or `aggregated_environment` when every selected plugin is compatible with the sidecar protocol.

---

## profiling

Profiling configuration for nsys or torch profiler.

```yaml
profiling:
  type: "nsys"                       # "none", "nsys", or "torch"

  # Extra arguments for nsys profile (when type is nsys or nsys-time)
  extra_nsys_args: ["--stats=true"]       # Optional: list of strings

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
| `extra_nsys_args` | list[string] | No | null | Extra args for nsys profile (when type is `nsys` or `nsys-time`) |
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
  extra_nsys_args: ["--stats=true", "--trace=osrt"]
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

The v1 spelling for where the discovery plane (etcd, NATS) runs. In 2.0 etcd and NATS are [services](#services), implied by the Dynamo frontend and taken over by declaring them:

```yaml
services:
  - name: etcd
    type: etcd
    placement:
      node: dedicated
  - name: nats
    type: nats
    placement:
      node: dedicated
    options:
      max_payload_mb: 24
```

The v1 block still loads and means exactly that (`srtctl migrate` rewrites it):

```yaml
infra:
  etcd_nats_dedicated_node: true
  nats_max_payload_mb: 24
```

| Field                    | Type | Default | Description                                        |
| ------------------------ | ---- | ------- | -------------------------------------------------- |
| `etcd_nats_dedicated_node` | bool | false   | Reserve the first allocated node for etcd and NATS; no workers run there. Isolates the discovery plane on large jobs. |
| `nats_max_payload_mb` | int | none | Raise the NATS message size limit (long prompts on the NATS request plane). |

A recipe cannot say both: declared `etcd`/`nats` services and `infra.etcd_nats_dedicated_node` must agree.

---

## observability

Tachometer collection is **on by default for every run** (no configuration needed; `observability.tachometer.enabled: false` opts out). `observability.enabled` turns on the server metrics *content* (the TRT-LLM publish flag and engine statistics) and the trace surfaces:

```yaml
observability:
  enabled: true
```

Tachometer scrapes the **complement** of what the benchmark client polls: worker endpoints that appear in `AIPERF_SERVER_METRICS_URLS` are left to the client (a worker endpoint is never double-polled — the extra scrape load has previously made a submission irreproducible), while the frontend, DCGM, and node-exporter endpoints are always Tachometer's. On runs whose benchmark has no aiperf client (sa-bench, lm-eval, serve-only, manual), the complement expands to every endpoint.

The legacy in-job Python RAW scraper is retired: a recipe still carrying `scrape_metrics`, `scrape_interval_seconds`, or `scrape_output` fails validation at submit time. Historical `raw_prometheus.jsonl` artifacts remain readable by the post-processing ingest.

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `enabled` | bool | `false` | Enable server-side metrics/traces, Tachometer collection, and host sampling |
| `enable_otel` | bool | `false` | Inject OTEL tracing environment variables |
| `otel_endpoint` | string/null | `null` | OTEL collector endpoint |
| `tachometer` | object | `enabled: null` | Native Tachometer collection settings; `enabled: null` follows `observability.enabled`, explicit `false` opts out |

The component perf dashboard is **not** configured here. It is built in post-processing on every run; `enabled` decides which capture legs exist and therefore which tabs the page carries. See [Component Performance Dashboard](component-dashboard.md).

Tachometer collects every worker rank, frontend, DCGM, and node metrics by default (minus the client-polled complement described above) — the exporters launch from pinned multi-arch registry images with no configuration. Air-gapped clusters override the images via the `containers:` alias map in `srtslurm.yaml`; `default_exporters: false` disables the built-ins:

```yaml
observability:
  enabled: true
  tachometer:
    enabled: true
    collect_interval_ms: 1000
    sync_interval_secs: 120
    compaction_threads: 4
    storage_subdir: tachometer
    extra_metadata:
      cluster: production
    dcgm_exporter:
      container_image: /containers/dcgm-exporter.sqsh
      port: 9400
    node_exporter:
      container_image: /containers/node-exporter.sqsh
      port: 9100
```

| Tachometer field | Type | Default | Description |
| ---------------- | ---- | ------- | ----------- |
| `enabled` | bool/null | `null` | `null` means ON for every run (decoupled from `observability.enabled`); explicit `false` opts out |
| `binary_path` | string | `tachometer-scraper` | Scraper command or path on the compute nodes |
| `collect_interval_ms` | int | `1000` | Milliseconds between scrapes of every endpoint; the single cadence knob — it also drives the launched DCGM exporter's `--collect-interval` (an explicit `dcgm_exporter.command` wins) and the host sampler. Values below `1000` speed up DCGM NVML sampling and are warned about at launch: 100ms sampling measured ~2% decode ITL overhead on GB300. Replaces the retired Hz-based `default_frequency` |
| `sync_interval_secs` | int | `120` | Interval for intermediate Parquet compaction; `0` disables it |
| `compaction_threads` | int | `4` | Value passed as `POLARS_MAX_THREADS` |
| `storage_subdir` | string | `tachometer` | Output directory below the run log directory |
| `extra_metadata` | dict | `{}` | Static string metadata added to every endpoint |
| `default_exporters` | bool | `true` | Launch the built-in DCGM + node exporters when no explicit blocks are set (sweep path only) |
| `dcgm_exporter` | object/null | built-in | Defaults to `nvcr.io#nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04` on port 9401; an explicit block overrides |
| `node_exporter` | object/null | built-in | Defaults to `quay.io#prometheus/node-exporter:v1.8.2` on port 9101; an explicit block overrides |

`make setup ARCH=<compute_arch>` downloads and checksum-verifies the matching Tachometer binary from the latest srt-slurm release. The scraper runs as a native `srun` process on the head node; configured exporters remain containerized on worker nodes. Run `make tachometer-scraper` to build from source instead.

Tachometer writes its Parquet stream under `<log_dir>/<storage_subdir>/raw/scrape/` (the leaf is created by the scraper itself — srtctl pre-creates only the parent, because the scraper refuses a pre-existing storage directory), compacting to `final.parquet` there on shutdown. Intermediate files remain in `<log_dir>/<storage_subdir>/local` until shutdown compaction completes. Rows carry an epoch `timestamp_ns` column, so they join directly with AIPerf records and Dynamo spans; the post-processing ingest converts the Parquet into the dashboard's `server_metrics_export.jsonl`.

The scraper runs as a best-effort process: if it dies (or the binary is missing at runtime), the benchmark continues and the loss is visible in `tachometer.out` and the sweep log. `srtctl validate-setup` still fails fast at submit time when `bin/tachometer-scraper` is absent.

---

## telemetry

`telemetry` is reserved for DCGM power measurement. It can run alongside `observability.tachometer`; it does not start Tachometer itself.

When both are enabled, `telemetry.dcgm_exporter` is shared with Tachometer. Do not also configure `observability.tachometer.dcgm_exporter`; Tachometer can still launch an optional node exporter from its own block.

```yaml
telemetry:
  enabled: true
  collect_interval_ms: 1000
  storage_subdir: power
  required: true
  dcgm_exporter:
    container_image: /containers/dcgm-exporter.sqsh
    port: 9400
```

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `enabled` | bool | `false` | Enable DCGM power collection |
| `dcgm_exporter` | object/null | `null` | DCGM exporter image, port, and optional command; required when enabled |
| `collect_interval_ms` | int | `1000` | Milliseconds between collector cycles; must be at most `3000` (replaces the retired `default_frequency`, which was seconds despite its name) |
| `storage_subdir` | string | `power` | Output directory below the run log directory |
| `required` | bool | `false` | Fail the benchmark when publishable power artifacts cannot be produced |
| `startup_timeout_seconds` | float | `30.0` | Exporter readiness timeout |
| `request_timeout_seconds` | float | `2.0` | Per-request exporter timeout |
| `collector_join_timeout_seconds` | float/null | `null` | Shutdown join timeout; defaults from `request_timeout_seconds` |

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

## host_setup

Commands run on each allocated node's **bare host, outside the container**, before any worker starts.

This is the counterpart to [`setup_script`](#setup_script), which runs *inside* the container. Use `host_setup` for node state the container cannot reach: locking GPU clocks, loading a kernel module, dropping caches.

```yaml
host_setup:
  commands:
    - "sudo -n nvidia-smi -lmc <min>,<max>"
  teardown:
    - "sudo -n nvidia-smi -rmc"
  nodes: all
  ignore_failure: false
  timeout_seconds: 300
```

| Field             | Type            | Default | Description                                                              |
| ----------------- | --------------- | ------- | ------------------------------------------------------------------------ |
| `commands`        | list[string]    | `[]`    | Shell commands run in order on each node, joined with `&&`                |
| `teardown`        | list[string]    | `[]`    | Commands run on each node after workers stop, on success and failure alike |
| `nodes`           | `all`/`workers` | `all`   | `all` covers head, infra, and workers; `workers` only the worker nodes    |
| `ignore_failure`  | bool            | `false` | Log a warning instead of failing the job when a node's commands fail      |
| `timeout_seconds` | int             | `300`   | Per-node wall-clock budget, for `commands` and `teardown` alike           |

**How it runs**: the orchestrator itself runs on the host (not in a container), so it fans these out as one container-less `srun` per node, in parallel. Output lands in `<log_dir>/host_setup_<node>.out` and `<log_dir>/host_teardown_<node>.out`.

**Notes**:

- **Commands run as you, not as root.** Anything privileged needs passwordless sudo (`sudo -n ...`). A `sudo` that prompts for a password will hang until `timeout_seconds` and then fail the job — verify first with `srun --jobid <job> --overlap -w <node> sudo -n true`. If sudo prompts, no recipe change helps; the cluster's SLURM `Prolog=` (which runs as root) is the only route.
- **Prefer setting `teardown` whenever `commands` changes persistent node state.** `nvidia-smi -lmc` outlives the allocation, so without a matching `-rmc` the next job on that node inherits your locked clocks. `srtctl dry-run` warns when `commands` is set without `teardown`.
- `teardown` runs from the job's cleanup path, so it fires on failure and cancellation too, and never changes the job's exit code.
- Set cluster-wide via `default_host_setup` in `srtslurm.yaml` — that's the right home when *the cluster's machines* need this, rather than one recipe. See [Cluster Config Fields](#cluster-config-fields).
- `srtctl dry-run -f config.yaml` renders the commands, their scope, and which file they came from.

---

## post_eval

How the accuracy evaluation is dispatched when the job environment sets `RUN_EVAL=true` (run after the benchmark) or `EVAL_ONLY=true` (run instead of it). srtctl forwards a built-in list of workflow variables into the eval process (`RUN_EVAL`, `EVAL_ONLY`, `MODEL`, `ISL`, `OSL`, `PREFILL_TP`, ...); this block extends that list and can replace the command, so a runner sets config instead of patching srtctl's source.

```yaml
post_eval:
  passthrough_env:          # forwarded into the eval process when set in the job environment
    - EVAL_FRAMEWORK
    - EVAL_CONC
    - EVAL_LIMIT
    - EVAL_SUITE
  command:                  # optional; replaces the built-in lm-eval runner command
    - bash
    - /infmax-workspace/benchmarks/evals/run.sh
    - "{endpoint}"
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `passthrough_env` | list[string] | `[]` | Extra environment variable names copied from the orchestrator's environment into the eval process when set |
| `command` | list[string] | none | Argv replacing the lm-eval runner. Placeholders: `{endpoint}` (frontend URL), `{infmax_workspace}`. Not shell-interpreted |

`MODEL_NAME` (the served model name) and `EVAL_CONC` are always set by srtctl. `srtctl dry-run` prints the effective dispatch.

---

## services

Long-running processes srtctl launches and tracks next to the workers, frontend, and benchmark client. One list covers the built-in infrastructure (etcd and NATS under the Dynamo frontend, the Mooncake master, the DCGM and node exporters tachometer scrapes: implied by the rest of the recipe, declared only to change something), generic sidecars (an experimental router built from a PR), and typed services (a standalone Mooncake store per worker node). Full reference: [services.md](services.md), in particular [Implicit Services](services.md#implicit-services).

```yaml
services:
  - name: etcd
    type: etcd                   # implied by frontend.type: dynamo; declared here to move it
    placement:
      node: dedicated
  - name: my-sidecar
    type: generic                # generic (default) | etcd | nats | mooncake-master | dcgm-exporter | node-exporter | mooncake-store
    command:
      - python3
      - -m
      - my_package.my_sidecar
    args:
      - --port
      - "9000"
    container: my-image          # alias or path; default: job container
    env:
      MY_FLAG: "1"
    placement:
      node: head                 # head | infra | dedicated | prefill | decode | agg | workers
    start: after_frontend        # infra | before_workers | after_frontend
    readiness:
      port: 9000
      timeout_seconds: 120
    inherit_discovery_env: true  # ETCD_ENDPOINTS / NATS_SERVER
    critical: false
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | required | Unique; names `service_<name>.out` and the tracked process |
| `type` | string | `generic` | Registered service kind; supplies defaults, injected env, and for the typed kinds the command |
| `enabled` | bool | `true` | `false` drops the service; how an implied one is switched off |
| `external` | string | none | `etcd`, `nats`, `mooncake-master`: address of an already-running instance; nothing launches |
| `options` | dict | `{}` | Kind-specific knobs (`nats.max_payload_mb`, exporter `port` / `collect_interval_ms`, `mooncake-master.store_config`) |
| `command` | list[string] | type default | Argv, not shell-interpreted; required for `generic` |
| `args` | list[string] | `[]` | Appended to `command` |
| `container` | string | type fallback, then job container | Image or `srtslurm.yaml` alias |
| `env` | dict | `{}` | Service environment; placeholders like `{node_ip}` are substituted |
| `placement.node` | string | type default | One instance for `head`/`infra`/`dedicated` (`dedicated` reserves a node); one per node for `prefill`/`decode`/`agg`/`workers` |
| `start` | string | type default | `infra` (etcd, nats), `before_workers` (mooncake-master, mooncake-store), `after_frontend` (generic, exporters) |
| `readiness` | object | type default | One probe (`port`/`tcp`, `http`, or `log`) plus `timeout_seconds` and `interval_seconds`; the job waits for it on every service node. Typed kinds gate on their well-known ports by default |
| `inherit_discovery_env` | bool | `true` | Inject the Dynamo discovery env |
| `critical` | bool | type default | A crash fails the run when true |
| `source`, `build_command` | object, list[string] | none | Clone an immutable git rev and build once before launch; single-node placements only |
| `build_timeout_seconds` | int | `1800` | `build_command` is killed when this runs out so a hung build cannot hold the allocation |
| `preamble`, `cpus_per_task`, `cpu_bind`, `srun_options` | | none | Pass-through launch knobs for this service |

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
