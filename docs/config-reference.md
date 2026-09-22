# Configuration Reference

Complete reference for job configuration YAML files in the 2.0 (`schema: 2`) layout.

This page is the prose guide: what each block means, how the pieces interact, and worked examples. The authoritative field-by-field list (every key, type, and default) is generated from the code in [schema-reference.md](schema-reference.md) and checked in CI, so if this page and that one disagree, the generated one is right. The pre-2.0 layout (`backend:`, `resources.prefill_nodes`, `infra:`, `dynamo.version`, ...) is documented in one place, [legacy-v1.md](legacy-v1.md); `srtctl migrate -f recipe.yaml --in-place` rewrites a v1 recipe into the layout described here.

## Table of Contents

- [Overview](#overview)
- [Cluster Config Discovery](#cluster-config-discovery)
- [schema](#schema)
- [name](#name)
- [model](#model)
- [engine](#engine)
- [roles](#roles)
- [placement](#placement)
- [resources](#resources)
- [slurm](#slurm)
- [frontend](#frontend)
- [benchmark](#benchmark)
- [dynamo](#dynamo)
- [profiling](#profiling)
- [output](#output)
- [health_check](#health_check)
- [observability](#observability)
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
schema: 2                      # Required: recipe layout version
name: "my-benchmark"           # Required: job name

model:                         # Required: model settings
  path: "deepseek-r1"
  container: "sglang"
  precision: "fp8"

resources:                     # Cluster facts: GPU type and GPUs per node
  gpu_type: "gb200"
  gpus_per_node: 4

slurm:                         # Optional: SLURM overrides
  time_limit: "02:00:00"

frontend:                      # Optional: router/frontend config
  type: dynamo

engine: sglang                 # Required: sglang | vllm | trtllm | mocker
roles:                         # Required: one block per worker role
  prefill:
    nodes: 1
    workers: 2
    gpus: 2
    args:
      tensor-parallel-size: 2
  decode:
    nodes: 2
    workers: 2
    gpus: 4
    args:
      tensor-parallel-size: 4

benchmark:                     # Optional: benchmark config
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512]

dynamo:                        # Optional: where Dynamo comes from
  source:
    pypi: "1.4.2"

profiling:                     # Optional: profiling config
  type: "none"

output:                        # Optional: output paths
  log_dir: "./outputs/{job_id}/logs"

health_check:                  # Optional: health check settings
  max_attempts: 180
  interval_seconds: 10

setup_script: "my-setup.sh"    # Optional: custom setup script
```

The v1 spelling of this layout (`backend:`, `resources.prefill_nodes` and friends, `infra:`, `dynamo.version`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

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
| `visible_devices_env`           | string | Worker GPU-subset mask; defaults to `CUDA_VISIBLE_DEVICES` |
| `default_gpu_exporter`          | dict/null | Cluster GPU exporter; defaults to DCGM, explicit null disables it |
| `srtctl_root`                   | string | Root directory for srtctl                             |
| `output_dir`                    | string | Custom output directory (overrides srtctl_root/outputs) |
| `model_paths`                   | dict   | Model path aliases                                    |
| `containers`                    | dict   | Container image aliases, resolved for every image key in a recipe (see below) |
| `default_mounts`                | dict   | Cluster-wide container mounts                         |
| `default_bash_preamble`         | string | Shell snippet prepended to every container srun       |
| `default_host_setup`            | object | Commands run on every node's bare host, outside the container |
| `nginx_raise_ulimit`          | bool   | Optional default for `frontend.nginx_raise_ulimit`  |
| `preflight`                     | bool   | `false` skips the pre-submit path checks on every `apply` (default `true`) |
| `reporting`                     | object | Status collector (`status`), log upload (`s3`) and failure analysis (`ai_analysis`); see below and [status-api-spec.md](status-api-spec.md) |

**reporting.s3**: After a run, a small container on the head node uploads the log directory to `s3://<bucket>/<prefix>/<YYYY-MM-DD>/<job_id>/` (`endpoint_url` for MinIO or another S3-compatible store; credentials only through `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the submit shell, since the literal fields would land in the lockfile). Not everything is worth shipping: a benchmark run's directory is 250 MB to 2 GB on lustre, over 95% of it aiperf's per-interval scrape of the worker and DCGM `/metrics` endpoints, the same series tachometer already stores as parquet, stored twice (raw per concurrency, and reshaped again in `perf_dashboard_bundle/`). The upload therefore follows a policy:

| Shipped as-is | Packed into `bundle.tar.zst` (`archive`) | Skipped (`exclude`) |
|---|---|---|
| config, lockfile, job JSON, sbatch script, git state, fingerprints, resource snapshot; sweep, worker, frontend, service and benchmark logs; results JSON, rollup, `profile_export_aiperf.*`; `perf_dashboard.html`; `tachometer/` parquet | `artifacts/**/profile_export.jsonl`, `sa-bench_*/**/profile_export.jsonl` (aiperf's per-request records, 13 to 40 MB raw, under 1 MB compressed) | `server_metrics_export.jsonl`, `server_metrics_export.json`, `gpu_telemetry_export.jsonl` and `inputs.json` under `artifacts/*/` and `sa-bench_*/*/` (the aiperf artifact roots; a same-named file from another benchmark type is not touched), `perf_dashboard_bundle/*`, `perf_dashboard.json` |

`exclude` uses `aws s3 sync` pattern rules (relative to the log directory, `*` matches across directories); `archive` uses Python glob rules with `**`. Either list replaces its default when set; `exclude: []` ships the whole directory, `archive: []` makes no archive. The archive is built under `/tmp` in the container, so nothing is added to the log directory on the cluster. One caveat: with tachometer disabled, dropping the aiperf scrape leaves no engine-metrics record outside `perf_dashboard.html`; enable tachometer, or override `exclude`.

```yaml
reporting:
  s3:
    bucket: "srt-logs"
    prefix: "sa-b200"
    endpoint_url: "https://minio.example.com"   # omit for AWS
    # exclude: []                                # ship everything
    # archive: ["artifacts/**/profile_export.jsonl", "*.out"]   # also pack the worker logs
```

**output_dir**: When set, job logs are written to `output_dir/{job_id}/logs` instead of `srtctl_root/outputs/{job_id}/logs`. Useful for CI/CD and ephemeral environments.

**containers**: A map from alias to image path or registry URI. One resolver walks the whole recipe and replaces any string under a `container`, `container_image`, `image`, or `nginx_container` key that matches an alias: `model.container`, `frontend.container_image`, `frontend.nginx_container`, `benchmark.container_image`, the Tachometer and power exporter images, `services[].container`, and any future block that names an image. Literal paths and registry URIs pass through untouched. Free-form maps (`environment`, `roles.<role>.env`, `roles.<role>.args`, `services[].env`, `container_mounts`) and the `identity` block are never rewritten.

**default_bash_preamble**: A shell snippet (e.g. `"ulimit -n 1048576 -s unlimited -u 1048576"`) prepended to every container srun launched by srtctl: workers, frontends, telemetry, benchmark, postprocess. Runs before per-call `bash_preamble` and the main command, so cluster-wide ulimits apply to everything downstream. Silently dropped for distroless containers (e.g. `prom/node-exporter`) that bypass the bash wrapper; a WARNING log is emitted in that case.

**default_host_setup**: A [`host_setup`](#host_setup) block applied to every job on the cluster, for node state that has to be set outside the container, such as locking GPU clocks. A recipe that sets its own `host_setup:` block replaces it entirely; `host_setup: {commands: []}` opts a single run out.

**preflight**: `srtctl apply` normally stats `model.path`, `model.container` and the telemetry images on the submitting node before calling sbatch. On clusters where those live only on compute nodes (node-local NVMe such as `/raid/models`), that check can never pass from the login node; set `preflight: false` and every `apply` behaves as if `--no-preflight` had been passed, with an INFO line saying so. Paths are still resolved at runtime and the framework fails loudly on the compute node if one is genuinely missing.

**nginx_raise_ulimit**: When set to `true` or `false`, this value is applied to jobs that omit `frontend.nginx_raise_ulimit` in the recipe. Use `true` on clusters where raising the nginx container's open-file limit is allowed; leave unset if each job should rely on the frontend default (`false`). A recipe that sets `frontend.nginx_raise_ulimit` always wins.

### Running without `srtslurm.yaml`

`srtslurm.yaml` is optional. A recipe can be fully self-sustaining as long as it supplies everything the cluster yaml would otherwise provide:

- Set `slurm.account`, `slurm.partition`, and `slurm.time_limit` directly in the recipe (no `default_*` fallback).
- Use absolute paths for `model.path`, `model.container`, and any other container fields; alias resolution is a no-op without the yaml's `containers:` / `model_paths:` maps.
- List every cluster-side mount the job needs in `extra_mount` (e.g. the lustre share that holds your model weights and `.sqsh` files). `default_mounts` is the only `srtslurm.yaml` field with no recipe-level equivalent until you spell mounts out yourself.
- Set `resources.gpus_per_node` explicitly.
- Status reporting and S3 log upload are skipped (their config lives under `reporting:` in the cluster yaml).

Workers' nats and etcd come from the dynamo/sglang container, not the yaml, so disagg/agg topologies still work end-to-end. `srtctl_root` falls back to the package install path automatically.

This is useful for portable recipes that you want to share across clusters or hand to a teammate without dragging cluster config along.

---

## schema

| Field    | Type    | Required | Description                                                                 |
| -------- | ------- | -------- | --------------------------------------------------------------------------- |
| `schema` | integer | Yes      | Recipe layout version. `2` is the layout this document describes. Absent means `1`, the layout in [legacy-v1.md](legacy-v1.md). |

Put the key first in the file, beside `base:` in an override file. Upgrade a recipe with `srtctl migrate -f recipe.yaml --in-place`, which preserves comments and key order and folds the legacy layout into `engine:`, `roles:`, `placement:`, `services:`, and `dynamo.source` (a directory is walked recursively). `srtctl migrate --verify -f <path>` migrates in memory and checks that the v1 and v2 documents resolve to the same config; CI runs it over the examples and the historical recipe corpus (golden equality).

```yaml
schema: 2
name: "deepseek-r1-benchmark"
```

Schema 2 is stricter than schema 1 in two places: a `benchmark:` field the selected type does not read is a load error rather than a silent no-op (see [benchmark](#benchmark)), and `roles.<role>.nodes: 0` is rejected in favor of the explicit `colocate` (see [roles](#roles)).

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
  container: "sglang"       # Container alias from srtslurm.yaml
  precision: "fp8"          # fp8, fp4, bf16, etc.
```

| Field       | Type   | Required | Description                                              |
| ----------- | ------ | -------- | -------------------------------------------------------- |
| `path`      | string | Yes      | Model path alias (from `srtslurm.yaml`) or absolute path |
| `container` | string | Yes      | Container alias (from `srtslurm.yaml`) or `.sqsh` path   |
| `precision` | string | Yes      | Model precision (informational: fp4, fp8, fp16, bf16)    |

---

## engine

GPU scheduling uses upstream's existing cluster settings. For eight-GPU
allocations on GRES-only clusters, set `use_gpus_per_node_directive: false`
and `default_sbatch_directives: {gres: "gpu:8"}`.

### GPU visibility on AMD

Set `visible_devices_env: ROCR_VISIBLE_DEVICES` in the cluster profile for ROCm
workers. GPU subsets then use only that mask, without applying a second mask to
already-renumbered devices. Set `default_gpu_exporter: null` to disable the
NVIDIA GPU exporter, or configure an exporter image, port, and command once for
the cluster. Other telemetry is unchanged; an explicit recipe exporter wins.

For vLLM builds without `--device-ids`, set `engine.set_visible_devices: true`.
This is one explicit boolean, not automatic vLLM version detection. The default
is false: vLLM binds devices with `--device-ids`. There is no CUDA-named alias.

`engine:` names the inference engine that builds every worker role's command. A bare string is the common form; a mapping carries the engine-wide knobs, the fields that are not per role:

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

For `trtllm_serve`, an explicit `served_model_name` is passed to the worker's
`--served_model_name` option so the server and benchmark/eval clients use the same
API model name. Leave it unset to retain the server's default. This is a server
option, not a key in `roles.*.args` (the engine YAML).

```yaml
engine:
  type: mocker
  engine_type: vllm
  speedup_ratio: 100
```

Valid types are `sglang`, `vllm`, `trtllm`, and `mocker`. Everything that is per role (the role's environment, its engine CLI flags, `extra_args`, `kv_events`) lives under [roles](#roles); everything else about the engine lives here. The generated tables under [Backend types](schema-reference.md#backend-types) list every engine-wide knob per engine; the ones worth knowing are:

| Engine | Engine-wide knobs |
| --- | --- |
| `sglang-router` | none beyond `type` |
| `vllm` | `connector` (default `nixl`), `dp_launch_mode`, `vllm_serve_binary`, `set_visible_devices`, `allow_prefill_decode_colocation`, `allow_prefill_decode_colocation_across_nodes` |
| `trtllm` | `served_model_name`, `publish_metrics`, `publish_events_and_metrics`, `sequential_node_start`, `numa_memory_bind`, `numa_cpu_bind` |
| `mocker` | the simulation parameters: `engine_type`, `speedup_ratio`, `decode_speedup_ratio`, `num_gpu_blocks_override`, `max_num_seqs`, `max_num_batched_tokens`, `block_size`, `data_parallel_size`, ... |

The v1 spelling of this (`backend.type` plus the engine-wide keys under `backend:`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

### vLLM DP launch mode

vLLM data-parallel endpoints use one process per node by default. srtslurm derives whether each TP/PP replica is node-local or spans multiple nodes:

```yaml
engine: vllm
roles:
  prefill:
    args:
      data-parallel-size: 8
  decode:
    args:
      data-parallel-size: 16
```

| Value      | Process layout                                                               |
| ---------- | ---------------------------------------------------------------------------- |
| `per_node` | One process per node (default); supports node-local or distributed TP/PP      |
| `per_gpu`  | One process per DP rank (TP x PP GPUs each; deprecated compatibility mode)    |

Set `engine.dp_launch_mode: per_gpu` only when temporarily preserving the legacy process layout. srtslurm emits a configuration-time deprecation warning for Dynamo-backed DP configurations that select it. `per_gpu` will be removed in a future release.

When `TP x PP` fits on one node, srtslurm derives `--data-parallel-size-local` and `--data-parallel-start-rank`, then enables `--data-parallel-hybrid-lb` so every node-local process registers with the Dynamo frontend. When `TP x PP` is larger than the node-local GPU allocation, srtslurm instead derives the multi-node rendezvous arguments and makes every process except the global leader headless. For example, both DP4 x TP4 and DP2 x TP8 are selected automatically on four-GPU nodes.

Do not set `data-parallel-size-local`, `data-parallel-start-rank`, `data-parallel-hybrid-lb`, or `headless` manually; srtslurm owns those values. The allocation must be regular: `DP x TP x PP` must match the endpoint GPU count, and a TP/PP replica must divide evenly within or across nodes.

### TRT-LLM metrics publication

With `frontend.type: dynamo`, prefill, decode, and aggregated TRT-LLM workers publish engine metrics by default using `--publish-metrics`, regardless of whether observability is enabled. Without observability, this does not enable KV events. `observability.enabled: true` retains its existing superset behavior: it additionally enables `--publish-events-and-metrics` when the combined setting is omitted or null. Explicitly requesting the combined flag also works without observability.

```yaml
engine:
  type: trtllm
  publish_metrics: true               # default
  publish_events_and_metrics: null    # unset: inherit the defaults below
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `publish_metrics` | bool | true | Pass `--publish-metrics` to Dynamo TRT-LLM workers; does not enable KV events |
| `publish_events_and_metrics` | bool or null | unset | `false`: disable both publication flags; `true`: enable the combined flag; unset/null: inherit defaults |

**An explicit `engine.publish_events_and_metrics: false` is a master opt-out:** neither publication flag is passed, even if `publish_metrics` is true or observability is enabled. This differs from omitting the combined setting, which keeps metrics on by default. Unset values remain null through config serialization so a saved config does not acquire an opt-out.

| `publish_events_and_metrics` | Observability | Default publication flags |
| --- | --- | --- |
| omitted / null | disabled / omitted | `--publish-metrics` |
| omitted / null | enabled | `--publish-metrics --publish-events-and-metrics` |
| `false` | either | none |
| `true` | either | `--publish-metrics --publish-events-and-metrics` |

**Compatibility:** the metrics-only flag requires a Dynamo build containing [ai-dynamo/dynamo#12162](https://github.com/ai-dynamo/dynamo/pull/12162) or equivalent support. Older builds (including Dynamo v1.4.2) reject the flag. Set `engine.publish_metrics: false` to omit only the new flag, including when observability is enabled; this does not disable a combined flag enabled by observability or the recipe. Set `engine.publish_events_and_metrics: false` to omit **both** flags. srt-slurm does not substitute the combined flag as an automatic compatibility fallback, because that would enable KV events. Omitting the flag does not override metrics-related environment variables supplied by the user. Metrics collection adds engine telemetry work; metrics-only does not mean zero overhead, but with the iteration-statistics default below the remaining cost is the per-request perf metrics.

**Iteration statistics default.** srtctl bakes `enable_iter_perf_stats: false` into every TRT-LLM engine section a recipe uses (prefill and decode, or aggregated), under both `frontend.type: dynamo` and `trtllm_serve`, creating the section when the recipe has none. This is a setdefault: an explicit `enable_iter_perf_stats: true` in the recipe wins, and `observability.enabled: true` keeps its own `true` because its expansion runs first. The default exists because `dynamo.trtllm` turns `--publish-metrics` into `enable_iter_perf_stats: true` in the engine arguments, and the engine YAML is merged over those arguments and wins on conflicts; without the explicit key every Dynamo worker collects TensorRT-LLM's per-iteration statistics (KV-cache stats and CUDA-event step timing on every executor loop). The request-level `trtllm_*` series (request latency, TTFT, TPOT, queue/prefill/decode time, token counters) do not need the key: they come from the per-request perf metrics, which `--publish-metrics` sets on the Dynamo path and `return_perf_metrics: true` sets for trtllm-serve. What the default drops is the iteration-level `trtllm_*` gauges (`trtllm_kv_cache_*`, running/waiting requests, iteration latency) and, on Dynamo, the `dynamo_component_kvstats_*` gauges, the router worker-load sample and the Planner's forward-pass metrics; set the key to `true` or enable `observability` to get them back. One visible effect to expect on a default run: the component dashboard's engine-tab KV-cache utilisation and hit-rate panels have no data, and the Dynamo bench dashboard's KV-utilisation series sits at the gauge's seeded 0 %, because both read gauges that only iteration statistics update. Engine sections whose `backend` is the legacy `tensorrt` engine are left alone: its `LlmArgs` rejects the key on containers older than TensorRT-LLM v1.3.0rc21, and that backend always collected the statistics anyway.

```yaml
roles:
  decode:
    args:
      enable_iter_perf_stats: true   # opt back in for one role
```

Saved and locked recipes carry the resolved key, so a later `observability.enabled: true` on such a file meets an explicit `false` rather than an omission; srtctl warns at load time and `srtctl dry-run` shows the value, and removing the line restores the default.

These options do not change native `trtllm_serve` or sidecar worker commands. `srtctl dry-run` shows the publication flag selected for Dynamo TRT-LLM workers and, for every TRT-LLM backend, the per-role `enable_iter_perf_stats` / `return_perf_metrics` values the engine YAML will carry.

TRT-LLM workers can span partially occupied nodes. For example, on four-GPU nodes,
two DEP6 prefill workers need three nodes:

```yaml
roles:
  prefill:
    nodes: 3
    workers: 2
    gpus: 6
    args:
      tensor_parallel_size: 6
      moe_expert_parallel_size: 6
      pipeline_parallel_size: 1
      enable_attention_dp: true
```

The workers use `A[0,1,2,3] + B[0,1]` and `C[0,1,2,3] + B[2,3]`.
Each endpoint launches exactly six MPI ranks and gets its own per-node
`CUDA_VISIBLE_DEVICES`. Full nodes lead the rank order to keep TRT-LLM's local
device mapping consistent. Layouts incompatible with that mapping (for example,
seven ranks split 4+3) are rejected before launch. Backend-specific communication
requirements still apply; this does not enable arbitrary uneven layouts in every
TRT-LLM communication backend.

TRT-LLM endpoints also set `MASTER_ADDR` to the rank-zero node and use a distinct
`MASTER_PORT` per endpoint. This overrides container hooks that infer rank zero
from Slurm's sorted node list. Explicit recipe environment values take precedence.

**Other TRT-LLM launch facts**: TRT-LLM supports prefill, decode, and aggregated roles, uses MPI-style launching (one srun per endpoint with all of its nodes) through `trtllm-llmapi-launch`, and sets `TRTLLM_EPLB_SHM_NAME` to a unique UUID per endpoint.

---

## roles

`roles:` describes the worker roles. It groups everything about a role in one place: how many nodes and workers it gets, how many GPUs each worker takes, its environment, and the engine's own CLI flags.

```yaml
engine: sglang
roles:
  prefill:
    nodes: 2          # nodes reserved for this role
    workers: 6        # number of workers
    gpus: 2           # GPUs per worker
    env:              # environment for every process of this role
      PYTHONUNBUFFERED: "1"
    args:             # the engine's CLI flags, as a mapping
      tensor-parallel-size: 2
  decode:
    nodes: colocate   # share the prefill nodes' spare GPUs
    workers: 2
    gpus: 2
    env:
      PYTHONUNBUFFERED: "1"
    args:
      tensor-parallel-size: 2
```

Role names are `prefill`, `decode`, and `agg`. A recipe is disaggregated (prefill and decode) or aggregated (agg only), never both. Every role a recipe launches is declared here; the `roles.<role>.engine` key is optional and, when given, must equal the top-level [engine](#engine).

| Key | Type | Description |
| --- | --- | --- |
| `nodes` | int, or `colocate` | Nodes reserved for the role. `colocate` is valid on `decode` only and packs the decode workers onto the prefill nodes' free GPUs. `0` is rejected: say `colocate`. |
| `workers` | int | Number of workers in the role |
| `gpus` | int | GPUs per worker. Computed as `nodes * gpus_per_node / workers` when omitted; required on both roles when decode is colocated |
| `env` | dict | Environment variables for every process of this role. Values support the per-worker `{node}` / `{node_id}` placeholders described under [environment](#environment) |
| `args` | dict | The engine's CLI flags for this role (`sglang` and `vllm` flags, the `trtllm` engine YAML, the mocker overrides). Any flag the engine accepts, kebab-case or snake_case, written as an ordinary YAML mapping. srtctl adds the topology flags itself (`disaggregation-mode`, ports, hosts, rank arguments); see [frontend](#frontend) for the keys each frontend owns |
| `extra_args` | list[string] | TRT-LLM only: extra `trtllm-serve` CLI flags appended verbatim to the worker command (`frontend.type: trtllm_serve`). For the few options that configure the OpenAI server layer and have no engine YAML key, such as `--tool_parser` |
| `kv_events` | bool or dict | Publish KV cache events for the Dynamo router; see below |
| `sidecar` | bool | Run the native engine with a Dynamo sidecar; see [Native sidecar mode](#native-sidecar-mode) |
| `critical` | bool | Whether a worker of this role exiting fails the run (default `true`); see [critical](#critical) |
| `engine` | string | Optional; must equal the top-level `engine` when both are given |

`env` and `args` are ordinary YAML mappings. Nothing needs JSON or inline `{}` syntax. Boolean flags are `flag-name: true`.

**GPUs per worker**: `gpus` is `(nodes * gpus_per_node) / workers` when omitted. Set it explicitly when a role should not fill its nodes, when several workers share a node, or whenever it makes the recipe self-describing. `resources.spread_workers: true` puts each partial-node worker on its own node instead of packing them.

### Colocating decode on the prefill nodes

`decode.nodes: colocate` reserves no nodes for decode and places the decode workers on whatever GPUs the prefill workers leave free on the prefill nodes. `gpus` must be given on both roles (the per-node formula cannot derive a split), and loading fails if the split does not fit, using the engine's real packing, so an oversubscribed layout is caught by `srtctl dry-run` instead of by the job.

```yaml
resources:
  gpu_type: "h100"
  gpus_per_node: 8

engine: sglang
roles:
  prefill:
    nodes: 1
    workers: 2
    gpus: 2          # 4 of the node's 8 GPUs
    args:
      tensor-parallel-size: 2
  decode:
    nodes: colocate
    workers: 1
    gpus: 4          # the remaining 4 GPUs
    args:
      tensor-parallel-size: 4
```

### kv_events

KV events are a Dynamo frontend feature for kv-aware routing (`frontend.args.router-mode: kv`): workers publish cache/scheduling information over ZMQ and the Dynamo router uses it to place requests. Setting `kv_events` on a role passes `--kv-events-config` to that role's workers with auto-allocated ZMQ ports.

```yaml
roles:
  prefill:
    kv_events: true              # publisher=zmq, topic=kv-events
  decode:
    kv_events:
      publisher: "zmq"
      topic: "decode-events"     # publisher defaults to "zmq"
```

Each worker leader gets a globally unique port starting at 5550:

| Worker    | Port |
| --------- | ---- |
| prefill_0 | 5550 |
| prefill_1 | 5551 |
| decode_0  | 5552 |
| decode_1  | 5553 |

### critical

srtctl treats every worker as critical: when one exits, the process monitor fails the run and tears the job down. A workload that kills workers on purpose, such as a migration probe or a fault-tolerance test, needs the survivors to keep serving, so set `critical: false` on the role whose workers it kills:

```yaml
roles:
  decode:
    workers: 4
    critical: false            # a decode worker exiting does not end the run
```

The flag is per role and defaults to `true`. It changes only how a worker exit is treated; the health gate before the benchmark still requires every worker to come up.

The v1 spelling of this section (`resources.prefill_nodes`, `resources.prefill_workers`, `resources.gpus_per_prefill`, `resources.prefill_critical`, `resources.decode_nodes: 0`, `backend.prefill_environment`, `backend.sglang_config.prefill`, `backend.prefill_extra_args`, `backend.kv_events_config`, and the `decode` and `aggregated` counterparts) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

---

## placement

`placement:` is one vocabulary for where the frontend and the benchmark client run:

```yaml
frontend:
  placement:
    node: head          # head | first_decode | dedicated
benchmark:
  placement:
    node: last_decode   # head | last_decode | dedicated
```

`node: dedicated` reserves a node for that component: the job asks Slurm for one more node and nothing else runs there. Any other value names an existing node: `head` is the first allocated node (where the orchestrator runs), `first_decode` and `last_decode` are the first and last node of the decode role. The default for both blocks is `head`. `telemetry` requires the benchmark client on `head`.

The discovery plane (etcd, and NATS when a plane uses it) is placed through its services: an `etcd` or `nats` entry under [`services`](#services) with `placement.node: dedicated`. See [Implicit Services](services.md#implicit-services).

The v1 spelling of this (`frontend.orchestrator_placement`, `frontend.dedicated_node`, `benchmark.client_placement`, `benchmark.client_dedicated_node`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

---

## resources

Cluster facts about the GPUs the job runs on. The worker topology (nodes, workers, GPUs per worker) lives under [roles](#roles).

```yaml
resources:
  gpu_type: "gb200"
  gpus_per_node: 4          # GPUs per node (default: from srtslurm.yaml)
  spread_workers: false     # one partial-node worker per node instead of packing
  het_jobs: null            # SLURM heterogeneous job for prefill and decode; null: cluster default
```

| Field             | Type   | Default            | Description                           |
| ----------------- | ------ | ------------------ | ------------------------------------- |
| `gpu_type`        | string | `default_gpu_type` | GPU type, e.g. "gb200", "gb300", "h100". Optional; inherits `default_gpu_type` from `srtslurm.yaml` when omitted. Still worth setting so the recipe is self-describing for result rollups |
| `gpus_per_node`   | int    | cluster / 4        | GPUs per node; inherits the cluster `gpus_per_node` when omitted, else 4 |
| `spread_workers`  | bool   | false              | Place each partial-node worker on its own node instead of packing several onto one node. The recipe must reserve enough nodes (e.g. `roles.decode.nodes` equal to `roles.decode.workers` when `gpus` is below `gpus_per_node`) |
| `het_jobs`        | bool or null | null         | Submit prefill and decode as two SLURM heterogeneous-job components, each with its own `--segment`. `null` defers to the cluster's `use_het_jobs`; see [slurm-faq.md](slurm-faq.md) |

The total node count is the sum of every role's `nodes`, every service's `nodes` (a pool of whole nodes the service owns, see [pools.md](pools.md)), plus one for each `placement.node: dedicated` (frontend, benchmark client, the discovery plane through its services). Pools are carved after the engine roles' nodes, in declaration order. `srtctl dry-run` prints the resulting sbatch request and the node map.

A services-only job has no engine roles at all: the services that own nodes declare `nodes` (see [pools.md](pools.md)), `frontend.type: none` skips the frontend layer and the worker-count health gate, and the `services:` readiness probes are the only gate before the benchmark step runs. This is the shape of a Ray cluster driving an RL trainer, or a client run against an endpoint the job does not own.

The v1 spelling of the worker topology (`resources.prefill_nodes`, `prefill_workers`, `gpus_per_prefill`, `decode_nodes`, `decode_workers`, `gpus_per_decode`, `agg_nodes`, `agg_workers`, `gpus_per_agg`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it into `roles:`.

### CPU allocation visibility

srtctl records both the requested GPU topology and the effective CPU allocation. At runtime it:

- logs `SLURM_JOB_CPUS_PER_NODE`, `SLURM_CPUS_ON_NODE`, and process CPU affinity;
- writes `logs/resource_snapshot.json` with per-node/total CPUs, backend/configured GPUs, the warning threshold, and verdict;
- adds the snapshot to `lock.resource_snapshot` in `recipe.lock.yaml`;
- adds CPU allocation and warning state to the job metadata used by `srtctl monitor`; and
- records CPU model, logical CPU count, affinity, and SLURM CPU variables in each worker fingerprint beside GPU details.

The warning uses a fixed, conservative baseline of one effective CPU per backend GPU. For example, a four-GPU backend that receives only two CPUs produces a prominent `CPU ALLOCATION WARNING` before services start. Increase the request with the appropriate cluster policy, such as `cpus-per-task`, `cpus-per-gpu`, or an exclusive-node directive.

### Computed Properties

Internally the resolved topology exposes several computed properties, visible in `recipe.lock.yaml` and `srtctl dry-run`:

- `is_disaggregated`: True if the recipe has prefill and decode roles
- `total_nodes`: Total nodes allocated (prefill + decode or agg, plus dedicated nodes)
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
  # Frontend type: "dynamo" (default), "sglang-router", "vllm-router", direct "sglang", "vllm", "trtllm_serve",
  # or "none" for a services-only job (no router, no OpenAI endpoint, no worker-count health gate; only
  # valid without engine roles, see services[].nodes)
  type: dynamo

  # Where it runs; see placement
  placement:
    node: head

  # Scaling
  enable_multiple_frontends: true     # Enable nginx + multiple routers
  num_additional_frontends: 9         # Additional routers (total = 1 + this)

  # Optional: raise nofile for nginx (shell ulimit + worker_rlimit_nofile in nginx.conf).
  # Default false. Set true on clusters that allow it; can also set nginx_raise_ulimit in srtslurm.yaml.
  # nginx_raise_ulimit: true

  # CLI args passed to the frontend/router
  args:
    router-mode: "kv"                 # dynamo: router-mode
    policy: "cache_aware"             # sglang-router: policy
    no-kv-events: true                # boolean flags

  # Dynamo only: inline worker-selection policy config. srtctl writes the
  # router policy YAML and passes --router-policy-config automatically.
  worker_selection:
    prefill: max-kv-overlap
    decode: default
    instances:
      - name: max-kv-overlap
        type: dynamo-two-tier-cost-fn
        parameters:
          cache_threshold: 0.0
          balance_abs_threshold: 1000000000
          balance_rel_threshold: 1000000000.0

  # Environment variables for frontend processes
  env:
    MY_VAR: "value"

  # Optional static-router image; defaults to model.container
  # container_image: vllm-router
```

| Field                       | Type | Default       | Description                         |
| --------------------------- | ---- | ------------- | ----------------------------------- |
| `type`                      | str  | dynamo        | `dynamo`; static routers `sglang-router`, `vllm-router`; direct (one aggregate worker binds the public port, no router process) `sglang`, `vllm`, `trtllm_serve` |
| `placement.node`            | str  | head          | `head`, `first_decode`, or `dedicated`; see [placement](#placement) |
| `enable_multiple_frontends` | bool | true          | Scale with nginx + multiple routers |
| `num_additional_frontends`  | int  | 9             | Additional routers beyond master    |
| `nginx_container`           | str  | nginx:1.27.4  | Custom nginx container image        |
| `nginx_raise_ulimit`      | bool | false         | When true with nginx in use, run `ulimit -n 1048576` before nginx and emit `worker_rlimit_nofile 1048576` in generated `nginx.conf`. Off by default so restrictive clusters do not fail. Cluster `srtslurm.yaml` may set `nginx_raise_ulimit` for jobs that omit this field. |
| `worker_selection`          | dict | null          | Dynamo worker-selection mapping. Written as `/logs/router_policy_config.yaml` and passed with `--router-policy-config`. Cannot be combined with a manually supplied policy-config argument or environment variable. |
| `args`                      | dict | null          | CLI args for the frontend           |
| `env`                       | dict | null          | Env vars for frontend processes     |
| `container_image`           | str  | null          | Static-router image; falls back to `model.container` |

See [SGLang Router](sglang-router.md) for detailed architecture.

### trtllm_serve frontend

`type: trtllm_serve` runs the `trtllm-serve disaggregated` orchestrator as the router (for `engine: trtllm`). Instead of the dynamo request plane, srtctl collects the prefill/decode worker addresses and writes a static `ser.yaml` (`context_servers` = prefill, `generation_servers` = decode), then launches the orchestrator on the head node. The trtllm workers are started as `trtllm-serve` OpenAI servers rather than `dynamo.trtllm`.

Because the orchestrator is a single process, set `enable_multiple_frontends: false` (the nginx + multi-router path is not supported). A configuration can be switched between the two TRT-LLM serving stacks by changing only `frontend.type` between `dynamo` and `trtllm_serve`; start from the `examples/trtllm/dynamo-disagg.yaml` and `examples/trtllm/trtllm-serve-disagg.yaml` examples.

**Worker metrics default.** srtctl sets `return_perf_metrics: true` in the `args` of every role a `trtllm_serve` recipe uses (prefill and decode, or agg), creating the mapping when the recipe has none. This is a setdefault: an explicit `return_perf_metrics: false` in the recipe wins and is warned about. trtllm-serve mounts a worker's `/prometheus/metrics` route only when the engine runs with that flag, and TensorRT-LLM's own default is `false`, so without it Tachometer's `backend_*` endpoints answer HTTP 404 and the capture has no worker-level data. The route carries the per-request series (request latency, TTFT, TPOT, queue/prefill/decode time, token counters); it applies independently of `observability.enabled`, which keeps its own expansion. The same load step also sets `enable_iter_perf_stats: false` unless the recipe or observability says otherwise (see [Iteration statistics default](#trt-llm-metrics-publication)), so the route carries the per-request series only and the workers skip TensorRT-LLM's per-iteration statistics.

### vllm frontend

`type: vllm` runs aggregate vLLM jobs **without Dynamo**. The OpenAI-compatible HTTP server is the aggregate `vllm serve` worker itself: there is no separate router/frontend process, and srtctl skips NATS/etcd startup.

Use this for aggregate throughput benchmarks where Dynamo orchestration is not needed. Disaggregated prefill/decode layouts still require a real router such as Dynamo (`frontend.type: dynamo`).

**Requirements**

| Constraint | Value |
| ---------- | ----- |
| `engine` | `vllm` |
| Job layout | Aggregate only (`roles.agg`); no prefill/decode roles |
| `roles.agg.workers` | Exactly `1`; scale across nodes with `roles.agg.nodes`, not with replicas |
| `enable_multiple_frontends` | `false` (nginx + multi-router path is unsupported) |

Nothing load-balances between aggregate endpoints here, so `roles.agg.workers: 2` is rejected at load time: the extra replica would either idle behind the single public address or collide on the port. Use `frontend.type: dynamo` when you want several aggregate replicas behind one endpoint.

**Single-node example**

```yaml
frontend:
  type: vllm
  enable_multiple_frontends: false

resources:
  gpus_per_node: 8

engine: vllm
roles:
  agg:
    nodes: 1
    workers: 1
    args:
      tensor-parallel-size: 8
```

**Multi-node example (TP/PP across nodes)**

```yaml
frontend:
  type: vllm
  enable_multiple_frontends: false

resources:
  gpus_per_node: 8

engine: vllm
roles:
  agg:
    nodes: 2
    workers: 1
    args:
      tensor-parallel-size: 8
      pipeline-parallel-size: 2
```

srtctl launches one `vllm serve` process per node. The endpoint leader (`node_rank=0`) binds the public OpenAI port; follower ranks run headless engine workers. Multi-node coordination flags (`--master-addr`, `--nnodes`, `--node-rank`, `--headless`) are derived from the allocated topology: **do not set them in the recipe**.

`master-port` / `master_port` remains an optional recipe override and is passed to every node rank. Set it when jobs may share a leader node and need distinct vLLM rendezvous ports; otherwise vLLM's default is used.

**Topology-managed `args` keys**

The following keys are owned by srtctl and are stripped at runtime if present in `roles.<role>.args` for a vLLM role:

- `headless`
- `host`, `port`
- `master-addr` / `master_addr`
- `nnodes`
- `node-rank` / `node_rank`

Existing recipes that still contain these keys generally continue to work because the values are ignored. One exception is `headless` combined with the default `dp_launch_mode: per_node` and `data-parallel-size`: engine validation rejects that combination before direct-vLLM command construction, so remove `headless` from such recipes. `srtctl dry-run` emits a **WARNING** for each accepted key so operators can clean up recipes over time.

Health checks, benchmark clients, and `SRT_FRONTEND_HOST` target the **aggregate endpoint leader** (the node running the public `vllm serve`), not necessarily the Slurm head node.

To use vLLM's Rust OpenAI frontend in managed-engine mode, set `engine.vllm_serve_binary` to `vllm-rs`. An absolute path is also accepted when the executable is installed in the container but is not on `PATH`:

```yaml
frontend:
  type: vllm
  enable_multiple_frontends: false

engine:
  type: vllm
  vllm_serve_binary: /usr/local/lib/python3.12/dist-packages/vllm/vllm-rs
roles:
  agg:
    nodes: 1
    workers: 1
    args:
      tensor-parallel-size: 4
      tokenizer-mode: hf
      reasoning-parser: auto
      tool-call-parser: auto
```

The default remains `vllm`, so existing recipes continue to use the Python frontend. This setting only changes direct `frontend.type: vllm` jobs; Dynamo, sidecar, and `vllm-router` launch paths are unchanged.

Compare with `frontend.type: dynamo` + `engine: vllm`, which keeps Dynamo as the request router and uses `python3 -m dynamo.vllm` workers discovered through etcd.

### vllm-router frontend

`type: vllm-router` launches the official vLLM Router in front of direct `vllm serve` workers. It supports aggregate replicas and disaggregated P/D topologies without Dynamo or NATS/etcd. See [vLLM Router](vllm-router.md) for complete topology examples and the division of responsibility between the upstream vLLM backend topology and Router adapter.

`engine.connector: moriio` (AMD MoRI-IO on ROCm) switches the same frontend to the Router's discovery mode. srtctl launches one Router on the head node with `--kv-connector moriio --vllm-discovery-address 0.0.0.0:36367` and no worker URLs, gives every prefill and decode worker a role-aware `MoRIIOConnector` `--kv-transfer-config` that carries the Router's address, the worker's own routable IP and HTTP port, and the handshake and notify listeners the port allocator reserved for it, and waits on the Router's `/health`, which answers 503 until a prefill and a decode have registered. Both roles must run the connector, the topology must be prefill/decode, and `frontend.enable_multiple_frontends` must be `false`. See [MoRI-IO discovery](vllm-router.md#mori-io-discovery).

---

## benchmark

Benchmark configuration. The `type` field determines which benchmark runner is used and what additional fields are available.

Set `benchmark.stream_output: true` to mirror `benchmark.out` to the orchestrator's
stdout while the benchmark runs (default: `false`). This works with custom and
built-in benchmark clients and preserves the complete log file and client exit
status. For a one-off run, use `srtctl apply -f recipe.yaml --set benchmark.stream_output=true`.
The output appears in the Slurm job log; CI must follow that log to display it
live. `apply` still submits asynchronously. Streaming forwards bytes as they
reach the log, so clients must flush their output (for Python clients, set
`PYTHONUNBUFFERED: "1"` in their environment). Manual/serve-only jobs have no
benchmark client to stream, and separate post-eval logs are not included.

**Per-type fields.** Every type accepts the shared fields (`placement`, `colocate_with_frontend`, `stream_output`, `aiperf_package`, `aiperf_args`, and `concurrencies`, which power telemetry reads for its measurement windows whatever the type) plus the fields its runner reads:

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

A `schema: 2` recipe that sets a field its type does not use is rejected at load with the list of accepted fields. Before this, such a field was a silent no-op (`isl` on `gsm8k`, `num_shots` on `sa-bench`). Each runner declares its fields as `config_fields`; adding a field to a runner means adding it there.

| Shared field | Type | Default | Description |
| --- | --- | --- | --- |
| `placement.node` | string | `head` | `head`, `last_decode`, or `dedicated`; see [placement](#placement) |
| `colocate_with_frontend` | bool | `true` | When both the frontend and the client ask for a dedicated node, share one reserved node; `false` reserves one node each |
| `aiperf_package`, `aiperf_args` | string, list | none | AIPerf install spec and extra client flags for the AIPerf-backed runners |
| `concurrencies` | list or `"NxM"` string | none | Concurrency levels; also the measurement windows telemetry reads |

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
| `mlperf`          | MLPerf Inference LoadGen (mlcommons/inference)  |

### manual

No benchmark is run. Use for manual testing and debugging.

For a one-off serving run, `srtctl apply -f config.yaml --serve-only` provides the same behavior without changing the recipe's configured benchmark.

```yaml
benchmark:
  type: "manual"
```

### custom

Run an arbitrary command with `bash -lc`. The command is passed verbatim; srt-slurm does not expand `{placeholder}` expressions. Use environment variables for runtime-discovered values:

```yaml
benchmark:
  type: custom
  command: >-
    ./run-benchmark.sh "$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT"
  env:
    MY_BENCHMARK_OPTION: "value"
```

Every custom benchmark command receives frontend metadata plus mode-specific metadata for each logical worker leader:

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
| `SRT_SERVICE_<NAME>_NODES`      | comma-separated hostnames      | Nodes each launched service runs on, in placement order; `<NAME>` is the service name upper-cased with non-alphanumerics as `_` |
| `SRT_SERVICE_<NAME>_IPS`        | comma-separated IPs            | The same nodes' fabric IPs (the first is the head of a `ray` service) |
| `SRT_SERVICE_<NAME>_NODE_COUNT` | int                            | How many nodes the service spans |
| `SRT_GPUS_PER_NODE`             | int                            | `resources.gpus_per_node` |
| `SRT_WORKER_NODES`              | comma-separated hostnames      | Every engine worker node (empty when the job has no engine roles) |

Only variables for roles present in the recipe are emitted. Entries follow logical topology order (prefill index, decode index, or aggregated index). Multi-node follower ranks are excluded because they do not own separate engines; co-located logical workers retain repeated IPs and distinct ports so list positions remain aligned. With a Dynamo frontend, endpoint and metrics URLs use each leader's `DYN_SYSTEM_PORT`; other frontends use the worker HTTP port. If KVBM metrics are configured, their URLs are appended to `AIPERF_SERVER_METRICS_URLS` after the logical worker URLs.

Two caveats for `AIPERF_SERVER_METRICS_URLS`:

- **Dynamo TRT-LLM worker URLs are advertised when engine metrics are enabled.** This is the default via `engine.publish_metrics: true` (`--publish-metrics`) when the combined setting is omitted; `engine.publish_events_and_metrics: true` also enables them. Explicit `engine.publish_events_and_metrics: false` suppresses both publication flags and worker URLs, regardless of the metrics-only setting. URLs are also omitted when no flag is enabled (for example, metrics-only false and the combined setting omitted without observability). This applies to built-in AIPerf and custom benchmarks, excluding sidecars, whose behavior is unchanged. Runtime-only metrics may still exist but do not constitute an engine-metrics capture. With `frontend.type: trtllm_serve` the gate is the worker's own engine config instead: its `/prometheus/metrics` URL is advertised when that role's `args.return_perf_metrics` is true (the srtctl default for trtllm_serve recipes; an explicit `false` drops the URL). KVBM URLs are unaffected; KVBM serves its own endpoint regardless of the flag.
- **An explicit `AIPERF_SERVER_METRICS_URLS` in the recipe `environment:` wins.** Injection is skipped when the variable is already set, so a curated endpoint list is never clobbered.

Values in `benchmark.env` are applied last and can explicitly override any automatically injected variable.

The service variables are how a custom command drives something the job brought up rather than an inference endpoint: a job with no engine roles (`frontend.type: none`, a service that owns the nodes through `services[].nodes`) still runs its benchmark step, and the command finds the service through `SRT_SERVICE_*`. Launchers and clients that are not core live in the repo-root `benchmarks/` folder, mounted in every job container at `/benchmarks` (like `configs/` at `/configs`); `benchmarks/rl/miles/launch.sh` starts a [Miles](miles.md) RL run against a `ray` service.

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

When `reuse_http_connections` is enabled, each `benchmark_serving.py` process uses one keep-alive connection pool. Warmup and formal runs remain isolated in separate processes and therefore never share a pool. The option currently applies only to SA-Bench's Dynamo HTTP adapter.

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

Router performance benchmark with prefix caching. **Requires `frontend.type: sglang-router`**.

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

Trajectory-replay benchmark using the standalone [agentperf-client](https://github.com/ArtificialAnalysis-External/agentperf-client), a deterministic agentic load generator with a Rust streaming core. The client checkout is mounted into the container (pin the commit for comparable runs); the workload definition (trajectory dataset, user-assignments file, `settling_time_seconds`, `phase_timeout_seconds`, stop criteria) lives in the client's own config YAML. srtctl injects the endpoint, model and concurrency at run time via the client's `--base-url` / `--model` / `--concurrencies` flags. Note the client validates the workload YAML *before* merging CLI overrides, so the YAML must still carry syntactically valid placeholder `base_url`, `model` and `concurrencies` values, and `phase_timeout_seconds` must satisfy the client's ramp-up bound for the *injected* concurrency (`phase_timeout_seconds >= (concurrency - 1) / user_spawn_rate + settling_time_seconds + min_measurement_seconds`).

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
| `agentperf_client_dir` | string      | Yes      | -       | Container path to an agentperf-client checkout         |
| `agentperf_config`     | string      | Yes      | -       | Container path to the client's workload YAML           |
| `concurrencies`        | list/string | Yes*     | -       | Levels, one client phase each; string form is x-separated (`"64x1010"`), matching other benchmark types |
| `concurrency`          | int         | Yes*     | -       | Single level (alternative to `concurrencies`)          |

*One of `concurrency` / `concurrencies` is required.

Notes:
- The first run of a job builds an isolated client runtime under `/tmp/agentperf-<jobid>` (uv env, pinned Rust toolchain, `rustcore` extension, tokenizer cache) and stages the trajectory and user-assignments datasets from shared storage to node-local `/tmp`; this preflight needs network egress from the benchmark node and adds several minutes before the first phase.
- The user-assignments file referenced by the workload YAML must cover the highest concurrency level (`assign_trajectories` fails loudly otherwise).
- Results land under `<log_dir>/agentperf/` (per-phase `*__traj*.{jsonl,txt,json}`, `requests.jsonl`, `phase_manifest.jsonl`); `rollup.py` normalizes them into `benchmark-rollup.json`.
- Two runs must not share a results dir concurrently (the client resets `phase_manifest.jsonl` at start).
- `telemetry:` (DCGM power measurement windows) is not supported with agentperf; the schema rejects non-sa-bench benchmark types at config load. Tachometer (`observability.enabled`) works normally.

### mlperf

MLPerf runs as a **`custom` benchmark driving the MLPerf team's `inference-endpoint` client**, not as a benchmark type. srt-slurm carries no MLPerf-specific schema at all; the driver is a script at `/srtctl-benchmarks/mlperf/bench.sh`, mounted for every benchmark type.

```yaml
benchmark:
  type: custom
  command: bash /srtctl-benchmarks/mlperf/bench.sh
  env:
    MLPERF_CLIENT_CONFIG: /configs/dsr1-interactive-submission.yaml
    MLPERF_MODE: both            # both (default) | perf | acc

extra_mount:
  - "/path/to/client-configs:/configs"
```

**The client config is passed through, not re-modelled.** It carries ~60 nested settings (model params, two datasets with accuracy scoring, load pattern, a ZeroMQ transport block, drain/warmup/early-stopping) and its shape moves with the client version. Expressing any of it as srt-slurm settings would be a losing race and lossy: anything not modelled becomes unsettable. The script rewrites exactly two values, being the only two the config cannot know before the cluster exists:

| Rewritten | Why |
|---|---|
| `endpoint_config.endpoints` | frontend IPs are assigned by Slurm at run time |
| `report_dir` | so results land with the job's other logs and get collected |

Everything else is passed through untouched, including unresolved `${VAR}` placeholders that the client expands itself at load time. This mirrors the MLPerf team's own launcher (`endpoints-launch`, `NVIDIA/src/sflow/tools/generate_endpoint_yaml.py`), which rewrites one key and leaves the rest.

Start from a template in the client repo (`src/inference_endpoint/config/templates/submission_template.yaml`) or one of the ~45 point configs in `endpoints-launch` under `NVIDIA/src/configs/<system>/<model>/point_*/client.yaml`.

| Variable | Required | Default | Description |
| -------- | -------- | ------- | ----------- |
| `MLPERF_CLIENT_CONFIG` | Yes | - | Container path to the client config |
| `MLPERF_MODE` | No | `both` | `perf`, `acc`, or `both`. These are the client's own mode names; note they are *not* the `performance`/`accuracy` spellings used for dataset types inside the client config |
| `MLPERF_ENDPOINTS` | No | the injected frontend | Comma-separated list, for client-side load balancing |
| `MLPERF_CLIENT_BIN` | No | `inference-endpoint` | Client executable |

Notes:

- **Do not mount the client config at `/configs`.** srt-slurm mounts its own `configs/` there, holding the `nats-server` and `etcd` binaries the head node starts from; an `extra_mount` onto the same path shadows them and the job dies early with `NATS binary not found: /configs/nats-server`, which reads like a broken install rather than a mount collision. Use any other path.
- **Run it in the MLPerf endpoint client image** (`endpoint_client_*.sqsh`). The client ships pre-installed there, so there is nothing to build; the script checks it is on `PATH` and fails with that message if not.
- **The endpoint is injected, never defaulted.** srt-slurm sets `SRT_FRONTEND_HOST` / `SRT_FRONTEND_PORT` for every custom benchmark, and the script errors if they are absent rather than quietly benchmarking localhost.
- **`MLPERF_ENDPOINTS` is how you get more than one frontend.** The client load-balances across the list itself, which is how MLPerf gets past the roughly 28k-connection ceiling of a single `ip:port`; its own submission configs ask for 84,000. srt-slurm exposes a single frontend today, so at submission scale this override is currently the only route.
- The script writes `benchmark-rollup.json` itself, which is the artifact srt-slurm's postprocess already reads. Per-run metrics are deliberately absent: this client does not use LoadGen and writes its own report format, which has not been observed here yet, and a fabricated parser would be worse than an honest gap. The record points at `report_dir` and lists what landed there.

---

## dynamo

Dynamo installation configuration. `source` says where Dynamo comes from; exactly one of `pypi`, `wheel`, or `git` + `rev`.

```yaml
dynamo:
  source:
    git: https://github.com/ai-dynamo/dynamo
    rev: refs/pull/14000/head # a commit, a tag, or a PR head; never a branch name
    # sha: <filled in by srtctl apply>
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
| `sidecar`                | bool         | false   | Job-wide sidecar mode; prefer `roles.<role>.sidecar: true` |
| `sidecar_port`           | int          | 50051   | Base loopback gRPC port; co-located workers receive deterministic offsets |
| `sidecar_binary`         | string/null  | null    | Optional standalone executable; null uses `python3 -m dynamo.<framework>.sidecar` |
| `sidecar_args`           | list[string] | []      | Extra arguments passed to the sidecar launcher         |
| `sidecar_startup_timeout` | int         | 3600    | Seconds to wait for the native gRPC endpoint (1 hour)   |
| `sidecar_context_length` | int/null     | null    | TRT-LLM context length override                         |

| `source` key | Meaning |
| --- | --- |
| `pypi` | An `ai-dynamo` release from PyPI, e.g. `"1.4.2"` |
| `wheel` | An exact `ai-dynamo` nightly version installed from staged wheels; the matching `ai-dynamo-runtime` wheel is installed automatically |
| `git` + `rev` | Clone and build with maturin. `git` defaults to the upstream repository when only `rev` is given, so a fork is `git: https://github.com/<you>/dynamo` |
| `patches` | With `git`: Cargo dependency replacements applied tree-wide before the build. Each entry is a full `<crate> = <spec>` TOML line |
| `sha` | Written by `srtctl apply`; the commit `rev` resolved to |

**Notes**:

- Set `install: false` if your container already has dynamo pre-installed.
- `source` is the same shape `services[].source` uses.
- `rev` must be immutable: a commit SHA, a tag such as `v1.4.2`, or `refs/pull/<n>/head` for an unmerged PR. `main`, `master`, and `HEAD` are rejected; pin the commit you mean.
- `srtctl apply` resolves a non-commit `rev` with `git ls-remote`, writes the commit as `source.sha` into the submitted `config.yaml` (comments preserved, the recipe on disk is untouched), and echoes it as `pinned_sources` in `--json` output. The job builds that commit and the `/configs/dynamo-wheels` cache is keyed by it, so two runs of one recipe cannot silently build different code because the PR moved. If the login node cannot reach the remote, the submit continues with a warning and the compute node fetches the ref by name.
- `srtctl dry-run` prints the resolved Dynamo source.

The v1 spelling of this (`dynamo.version`, `dynamo.hash`, `dynamo.top_of_tree`, `dynamo.wheel`, `dynamo.cargo_patches`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

### Native sidecar mode

Set `sidecar: true` on every role to run the framework's native engine process beside a CPU-only Dynamo sidecar instead of launching `python3 -m dynamo.<framework>`. The mode is job-wide, so every role must agree; the sidecar knobs (`sidecar_port`, `sidecar_args`, ...) stay under `dynamo`. `dynamo.sidecar: true` is the equivalent job-wide spelling. The engine and sidecar share one Slurm step and have a coupled lifecycle: if either exits, srtctl terminates the other and marks the worker failed.

For SGLang, srtctl also adds `--incremental-streaming-output` to the engine (the sidecar consumes deltas) and sets `SGLANG_RUST_BUILD_MODE=never` in the worker environment so the native gRPC extension is loaded from the image instead of being rebuilt with cargo, which images that run SGLang from a source checkout cannot do. Set either one in the role's `args` or `env` to override.

By default, srtctl launches `python3 -m dynamo.<framework>.sidecar`. The `ai-dynamo` package supplies this module and pins the matching `ai-dynamo-runtime` wheel, which embeds the native Rust sidecar. The configured Dynamo source or preinstalled container runtime must include the selected framework's launcher. No separate Cargo build is performed at job startup.

Nightly deployments should select an exact `dynamo.source.wheel` version so srtctl stages and installs the matching `ai-dynamo` and `ai-dynamo-runtime` artifacts on every worker. Set `dynamo.sidecar_binary` only to launch a compatible standalone executable already present in the container or a bind mount.

```yaml
frontend:
  type: dynamo

engine: vllm  # sglang, vllm, or trtllm
roles:
  agg:
    nodes: 1
    workers: 1
    sidecar: true
    args:
      tensor-parallel-size: 8

dynamo:
  source:
    wheel: "<nightly-with-sidecars>"
  sidecar_port: 50051
  sidecar_args:
    - --grpc-connections
    - "4"
```

The default sidecar commands are `python3 -m dynamo.sglang.sidecar`, `python3 -m dynamo.vllm.sidecar`, and `python3 -m dynamo.trtllm.sidecar`. All three use the shared `--grpc-endpoint` flag.

SGLang exposes gRPC and starts the sidecar only on an endpoint leader; distributed followers are engine-only. srtctl also adds `--incremental-streaming-output` to every SGLang sidecar engine (and logs that it did): the sidecar treats each gRPC chunk as a delta, and without the flag current SGLang builds stream the cumulative text per chunk, which shows up as repeated prefixes in responses and inflated token counts. Set `incremental-streaming-output` in the role's `args` yourself to override. TensorRT-LLM supports sidecars for aggregated workers only and runs the sidecar on MPI rank zero. `dynamo.sidecar_context_length` can override the TRT-LLM context length inferred from `roles.agg.args.max_seq_len`.

For multi-node vLLM data parallelism, srtctl launches one Rust gRPC frontend and one Dynamo sidecar on **every node** using hybrid load balancing. All nodes receive the global `--data-parallel-size`, while `--data-parallel-size-local` and `--data-parallel-start-rank` restrict each frontend to its colocated engines. For example, DP12 on three four-GPU nodes launches local DP4 with starting ranks 0, 4, and 8. Dynamo registers three endpoints, each covering its four local ranks. No node runs `--headless`, and readiness requires all three sidecars. vLLM sidecars require `engine.dp_launch_mode: per_node` (the default; `backend.dp_launch_mode` in schema 1); `per_gpu` is rejected during configuration validation, including for single-rank jobs.

The multi-node DP command uses Python-supervised Rust frontends (`python3 -m vllm.entrypoints.cli.main serve` with `VLLM_USE_RUST_FRONTEND=1`). Python coordinates the shared DP rendezvous and starts `vllm-rs frontend` with each node's local engine count and starting rank; requests are handled by Rust. This is a current launcher limitation: `vllm-rs serve` expects to own the complete DP group and does not implement hybrid startup. Python supervision is not a fundamental requirement of the sidecar architecture. Single-node sidecars continue to use `vllm-rs serve`.

Each Rust frontend has one API server; do not set `api-server-count` to another value or enable `grpc` (which selects the separate Python gRPC server) or `data-parallel-external-lb`.

For a TP/PP replica spanning nodes (including TP+EP with `data-parallel-size: 1`),
srtctl starts the Rust gRPC frontend and Dynamo sidecar **only on the leader**.
Followers run native `vllm serve --headless`, without an API server or sidecar.
All nodes share the leader address on the configured network interface and a
per-endpoint rendezvous port, with derived `--nnodes`, `--node-rank`, and
`--distributed-executor-backend mp`. TP × PP × PCP must equal the total allocated
GPUs, distributed evenly across nodes. For example, TP8 on two four-GPU nodes is
one Dynamo registration, not two. Recipe rendezvous overrides are replaced by
these allocation-derived values; `api-server-count` is omitted on followers.
With the default role setting `critical: true`, an unexpected exit on any node
fails the job and stops all workers through the job-wide process monitor. Multiple cross-node DP replicas within one endpoint
remain unsupported; use separate endpoints with DP=1 instead.

This path requires the same Python-supervised Rust frontend `--grpc-port`
integration described below. Native vLLM headless support alone is insufficient
to provide the leader's sidecar transport.

Hybrid sidecars require compatible changes in **both** projects: vLLM's Python `serve` command must support `--grpc-port` for its Rust frontend, the Rust Control service must report `ParallelismInfo.data_parallel_size_local`, and Dynamo's sidecar must register that local range using the reported global start rank. Stock vLLM 0.29.0 does not supply these interfaces. Pin a compatible image or source build; merely upgrading the Dynamo wheel is insufficient. To use a specific Rust binary in the multi-node path, set `VLLM_RUST_FRONTEND_PATH` in the recipe's worker environment. srtctl preserves that path and supplies the Rust frontend selection automatically.

vLLM sidecar mode sets `VLLM_PLUGINS` to an empty value by default. This prevents image-installed plugins from replacing native engine output types that must match the fixed `vllm-rs` MessagePack contract. A recipe can explicitly set `VLLM_PLUGINS` in a role's `env` when every selected plugin is compatible with the sidecar protocol.

---

## profiling

Profiling configuration for nsys or torch profiler.

```yaml
profiling:
  type: "nsys"                       # "none", "nsys", or "torch"

  # Nsight command settings (when type is nsys or nsys-time)
  nsys_trace: "cuda,nvtx"
  trace_fork_before_exec: true        # Optional; unset keeps the Dynamo default
  capture_range_end: "stop"
  nsys_library_paths: ["/usr/local/cuda/compat"]
  extra_nsys_args: ["--stats=true"]

  # Phase-specific profiling step configs
  prefill:
    start_step: 10                   # Step to start profiling
    stop_step: 20                    # Step to stop profiling
    capture_scope: "selected"        # Opt in to targeting; "all" is the default
    worker_index: 0                  # Logical worker to profile
    worker_rank: 0                   # Physical process rank within the worker
  decode:
    start_step: 10
    stop_step: 20
  # OR for aggregated mode:
  aggregated:
    start_step: 10
    stop_step: 20
```

| Field | Type | Required | Default | Description |
| ----- | ---- | -------- | ------- | ----------- |
| `type` | string | No | "none" | Profiling type: "none", "nsys", "nsys-time", or "torch" |
| `nsys_trace` | string | No | "cuda,nvtx" | Nsight activity domains for non-TRT-LLM workers |
| `trace_fork_before_exec` | bool | No | null | Override non-TRT-LLM child-process tracing; null enables it for Dynamo only |
| `capture_range_end` | string | No | "stop" | Non-TRT-LLM Nsight behavior when a CUDA profiler range ends |
| `nsys_library_paths` | list[string] | No | null | Paths prepended to the worker `LD_LIBRARY_PATH` |
| `extra_nsys_args` | list[string] | No | null | Extra args for `nsys profile` |
| `prefill` | object | Disaggregated | null | Prefill phase config |
| `decode` | object | Disaggregated | null | Decode phase config |
| `aggregated` | object | Aggregated | null | Aggregated phase config (the `agg` role) |

### ProfilingPhaseConfig

Each phase config has:

| Field | Type | Required | Default | Description |
| ----- | ---- | -------- | ------- | ----------- |
| `start_step` | int | No | null | Step to start profiling |
| `stop_step` | int | No | null | Step to stop profiling |
| `capture_scope` | string | No | "all" | Capture all physical processes or opt in to "selected" |
| `worker_index` | int | No | 0 | Logical worker selected for iteration-based Nsight |
| `worker_rank` | int | No | 0 | Physical process rank selected within the worker |

### Profiling Modes

- **nsys**: NVIDIA Nsight Systems profiling. For vLLM and SGLang, each phase
  captures all physical processes by default and sends every usable control
  endpoint to the benchmark. Set `capture_scope: selected` to opt in to one
  worker/rank. A Dynamo control endpoint uses the worker's `DYN_SYSTEM_PORT`.
  Native Dynamo sidecars and direct vLLM expose control only on the endpoint
  leader: all-process capture sends one control request to rank 0, while
  selected capture must explicitly target rank 0.
- **torch**: PyTorch profiler. Sets `SGLANG_TORCH_PROFILER_DIR` environment variable.

TRT-LLM does not use the HTTP profiling helper. Its executor uses
`TLLM_PROFILE_START_STOP` to trigger the CUDA profiler, so its Nsight wrapper
continues to cover the complete MPI endpoint. With vLLM `dp_launch_mode:
per_node`, a physical process can own multiple local DP ranks; use `per_gpu`
when each DP rank needs a separate report. `worker_index` and `worker_rank`
are ignored when `capture_scope: all`.

### Validation Rules

1. Disaggregated mode requires both `prefill` and `decode` phase configs when profiling is enabled.
2. Aggregated mode requires `aggregated` phase config when profiling is enabled.

### Example: Torch Profiling (Disaggregated)

```yaml
resources:
  gpu_type: "h100"

engine: sglang
roles:
  prefill:
    nodes: 1
    workers: 1
  decode:
    nodes: 1
    workers: 1

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

engine: sglang
roles:
  agg:
    nodes: 1
    workers: 1

profiling:
  type: "nsys"
  nsys_trace: "cuda,nvtx,osrt"
  extra_nsys_args: ["--stats=true"]
  aggregated:
    start_step: 10
    stop_step: 25
    capture_scope: all
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

## observability

Tachometer collection is **on by default for every run** (no configuration needed; `observability.tachometer.enabled: false` opts out). `observability.enabled` turns on the server metrics *content* (the TRT-LLM publish flag and engine statistics) and the trace surfaces:

```yaml
observability:
  enabled: true
```

The capture window aligns with the load, the same window the benchmark client's own `AIPERF_SERVER_METRICS_URLS` polling covers: on benchmark runs the scraper starts once the server passes the health gate (bring-up produces only dead-endpoint noise while workers load) and is stopped **gracefully** when the client exits, with a configurable grace period for compacting `final.parquet` before post-processing reads it. Runs without a discrete load window (serve-only, `manual`, eval-only) capture the whole serve session as before. Signal handlers and the critical-process monitor still use the process registry's existing teardown budget; the benchmark shutdown grace does not override those paths.

Tachometer scrapes all configured worker, frontend, DCGM, and node-exporter endpoints, independently of the benchmark client's `AIPERF_SERVER_METRICS_URLS` polling. This keeps the raw capture complete even when the client also collects metrics.

The legacy in-job Python RAW scraper is retired: a recipe still carrying `scrape_metrics`, `scrape_interval_seconds`, or `scrape_output` fails validation at submit time. Historical `raw_prometheus.jsonl` artifacts remain readable by the post-processing ingest.

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `enabled` | bool | `false` | Enable server-side metrics/traces, Nsight Systems capture, and host sampling |
| `nsys` | object | `enabled: true` | NVTX tracing and CPU sampling of all worker processes/ranks and Dynamo frontends when observability is enabled; see [Profiling](profiling.md#observability-capture) |
| `enable_otel` | bool | `false` | Inject OTEL tracing environment variables |
| `otel_endpoint` | string/null | `null` | OTEL collector endpoint |
| `tachometer` | object | `enabled: null` | Native Tachometer collection settings; `enabled: null` follows `observability.enabled`, explicit `false` opts out |

`observability.enabled: true` also enables nsys with NVTX tracing and CPU sampling
on every frontend and worker process/rank.
Its default `nsys.capture_window: measured_workload` starts collection after warmup and
stops it when the measured workload finishes. Supported bundled runners call
the boundary hooks automatically; custom/manual clients must call them at their
own phase boundaries. Set `nsys.capture_window: including_startup` to include startup and
warmup through teardown, or `nsys.enabled: false` to opt out. An enabled
top-level `profiling` mode takes precedence. The serving container must provide nsys and the required
NVTX support. See [Observability capture](profiling.md#observability-capture)
for timing, sampling, injection, and report-finalization settings.

The component perf dashboard is **not** configured here. It is built in post-processing on every run; `enabled` decides which capture legs exist and therefore which tabs the page carries. See [Component Performance Dashboard](component-dashboard.md).

Tachometer collects every worker rank, frontend, DCGM, node, and process metrics by default (minus the client-polled complement described above); the exporters launch from pinned multi-arch registry images with no configuration. Air-gapped clusters override the images via the `containers:` alias map in `srtslurm.yaml`; `default_exporters: false` disables the built-ins:

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
    process_exporter:
      binary: /opt/srt/configs/process-exporter   # host-native (default mode); or set container_image instead
      container_image: ""
      port: 9256
```

| Tachometer field | Type | Default | Description |
| ---------------- | ---- | ------- | ----------- |
| `enabled` | bool/null | `null` | `null` means ON for every run (decoupled from `observability.enabled`); explicit `false` opts out |
| `binary_path` | string | `tachometer-scraper` | Scraper command or path on the compute nodes |
| `collect_interval_ms` | int | `1000` | Milliseconds between scrapes of every endpoint; the single cadence knob. It also drives the launched DCGM exporter's `--collect-interval` (an explicit `dcgm_exporter.command` wins) and the host sampler. Values below `1000` speed up DCGM NVML sampling and are warned about at launch: 100ms sampling measured ~2% decode ITL overhead on GB300. Replaces the retired Hz-based `default_frequency` |
| `sync_interval_secs` | int | `120` | Interval for intermediate Parquet compaction; `0` disables it |
| `shutdown_grace_secs` | float | `120.0` | Time the scraper gets after SIGTERM to flush and compact `final.parquet` before it is killed; compaction scales with the data accumulated since the last periodic sync |
| `compaction_threads` | int | `4` | Value passed as `POLARS_MAX_THREADS` |
| `storage_subdir` | string | `tachometer` | Output directory below the run log directory |
| `extra_metadata` | dict | `{}` | Static string metadata added to every endpoint |
| `default_exporters` | bool | `true` | Imply the built-in DCGM + node + process exporters when no explicit blocks are set. The exporters are [services](services.md#implicit-services) (`dcgm-exporter`, `node-exporter`, `process-exporter`); declaring one under `services:` by that name overrides it, and `srtctl dry-run` lists them |
| `dcgm_exporter` | object/null | built-in | Defaults to `nvcr.io#nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04` on port 9401; an explicit block overrides |
| `node_exporter` | object/null | built-in | Defaults to `quay.io#prometheus/node-exporter:v1.8.2` on port 9101 with the `cpu`, `infiniband`, `meminfo`, `processes`, `stat`, `vmstat`, `pressure` and `meminfo_numa` collectors on worker nodes. Includes major faults and page-reclaim counters; retains process-state and NUMA-node identity. An explicit block overrides |
| `process_exporter` | object/null | built-in | Defaults to the **host-native** `configs/process-exporter` binary (ncabatoff/process-exporter 0.8.7, installed by `make setup` for the compute arch, like `configs/nats-server` and `configs/etcd`) on port 9256, launched with plain `srun` (no container) on every allocated node (the `process-exporter` service, `placement.node: all`). Reads the host `/proc` and publishes per-process-group CPU seconds by mode, thread count, per-thread-name CPU and count (`-threads=true`), context switches, RSS and open fds. The passthrough filter retains metric names and labels, including summary sum/count suffixes, and attaches hostname and run metadata to raw rows. Groups (frontend, `dynamo_trtllm` / `dynamo_sglang` / `dynamo_vllm` handlers, `trtllm_engine` children, launcher, client, infra daemons) come from `<log_dir>/process-exporter.yml`, written at launch. If the binary is missing the leg is skipped with a warning (submit warns too). An explicit block may set `binary` (absolute, or relative to the srtctl checkout) or instead a `container_image` with `binary` unset to run it containerized; the upstream `FROM scratch` image is not used by default because pyxis/enroot on some clusters cannot start shell-less images |

Every exporter block accepts `container_image`, `port`, `command` and `binary`. `binary` selects host-native launch (the executable runs directly under `srun`, `container_image` is ignored and may be `""`); without it the exporter runs from `container_image`. One of the two must be set.

`make setup ARCH=<compute_arch>` downloads and checksum-verifies the matching Tachometer binary from the latest srt-slurm release and installs the process-exporter binary for the same arch. The scraper and the process exporter run as native `srun` processes; the DCGM and node exporters remain containerized on worker nodes. Run `make tachometer-scraper` to build the scraper from source instead. The process-exporter passthrough filter and node process-state/NUMA-label preservation require a scraper built from this revision or a release containing it; rebuild the scraper when using an older downloaded binary.

The pressure collector reports PSI only when the host exposes the corresponding `/proc/pressure` files; missing metrics indicate unavailable data. NUMA memory and allocation metrics retain the exported `node` label as `numa_node` in raw metric names, separately from host metadata. With `observability.enabled: true`, the existing local host sampler also records cumulative PSI stall totals in microseconds in its `psi` JSONL field. That optional sampler covers the sweep/orchestrator host only; it does not extend exporter placement to dedicated frontend or client nodes. Collector overhead has not been measured for this change.

Tachometer writes its Parquet stream under `<log_dir>/<storage_subdir>/raw/scrape/` (the leaf is created by the scraper itself; srtctl pre-creates only the parent, because the scraper refuses a pre-existing storage directory), compacting to `final.parquet` there on shutdown. Intermediate files remain in `<log_dir>/<storage_subdir>/local` until shutdown compaction completes. Rows carry an epoch `timestamp_ns` column, so they join directly with AIPerf records and Dynamo spans; the post-processing ingest converts the Parquet into the dashboard's `server_metrics_export.jsonl`.

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
  cpu_power_exporter:
    port: 9405
    source: auto
```

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `enabled` | bool | `false` | Enable DCGM power collection |
| `dcgm_exporter` | object/null | `null` | DCGM exporter image, port, and optional command; required when enabled |
| `collect_interval_ms` | int | `1000` | Milliseconds between collector cycles (shared by the DCGM and CPU legs); must be at most `3000` (replaces the retired `default_frequency`, which was seconds despite its name) |
| `storage_subdir` | string | `power` | Output directory below the run log directory |
| `required` | bool | `false` | Fail the benchmark when publishable DCGM power artifacts cannot be produced (CPU power is always best-effort; see below) |
| `startup_timeout_seconds` | float | `30.0` | Exporter readiness timeout (shared by the DCGM and CPU legs) |
| `request_timeout_seconds` | float | `2.0` | Per-request exporter timeout (shared by the DCGM and CPU legs) |
| `collector_join_timeout_seconds` | float/null | `null` | Shutdown join timeout; defaults from `request_timeout_seconds` |
| `cpu_power_exporter` | object/null | `null` | Enables the independent CPU power leg; see below |

`telemetry` requires a `benchmark.type` of `sa-bench`, `custom`, `agentic`, `agentx`, or `manual` (a `manual` job has no load window, so like serve-only it captures the whole serve session; use it when an external load generator drives the endpoint), the benchmark client on the head node (`benchmark.placement.node: head`, the default), and no dedicated node for the discovery plane (an `etcd`/`nats` service with `placement.node: dedicated` moves the head off the batch host the collector runs on).

### CPU power

`telemetry.cpu_power_exporter` is an independent, best-effort leg: its presence (not a separate `enabled` flag) turns CPU power collection on, and it can run with or without `dcgm_exporter` alongside it. On each worker node, srtctl launches a `cpu-power-exporter` process directly on the bare host (outside the model container, so it can read host power interfaces) and exposes it on `cpu_power_exporter.port`. It resolves the bundled Rust binary installed by `make setup` first, falling back to the ACPI-only Python stdlib exporter (`srtctl.core.cpu_power_exporter`) when that binary is absent. A head-node collector scrapes every worker's exporter on the shared `collect_interval_ms`/`request_timeout_seconds` cadence and writes per-sample rows plus a manifest under `<log_dir>/<storage_subdir>/cpu/` (`samples.csv`, `cpu_manifest.json`).

| CPU power exporter field | Type | Default | Description |
| ------------------------- | ---- | ------- | ----------- |
| `port` | int | `9405` | Port the exporter listens on and the head-node collector scrapes |
| `source` | `auto`/`acpi`/`dcgm` | `auto` | Passed through to the bundled binary's own `--source` flag; `auto` tries DCGM first and falls back to ACPI. Has no effect on the Python fallback exporter, which is ACPI-only |

`samples.csv` carries one row per sensor reading, with columns `schema_version, timestamp_unix, hostname, source, sensor, socket_id, power_w, total_power_w`. In ACPI mode, `total_power_w` is **not** a sum of the `cpu`- and `sysio`-kind rails; whenever a `grace`-kind channel exists for a socket, that channel alone is the node-level total (real hardware traces show `grace` at roughly 93-104W against `cpu`+`sysio` combined at roughly 53-58W for the same socket, i.e. `grace` measures the whole Grace SoC power boundary, not literally `cpu + sysio`). When no `grace` channel is present for a scrape, `total_power_w` is left blank for that row rather than guessed from the component rails; per-socket `power_w` values are always populated regardless. In DCGM mode, `total_power_w` is the single already-aggregate value DCGM reports per socket.

Because collection is always best-effort, there is no `required` knob for the CPU leg: a node that fails to expose its exporter (or fails to publish readings) simply produces gaps in `samples.csv`, and the run continues.

---

## sweep

A top-level `sweep:` block turns one recipe into several jobs. It is a flat mapping from parameter name to a list of values; `srtctl apply` expands the Cartesian product of every list, substitutes each combination into the recipe, and submits one job per combination.

```yaml
sweep:
  isl: [512, 1024, 2048]
  osl: [128, 256, 512]
```

| Key | Type | Description |
| --- | --- | --- |
| `<parameter>` | list | One entry per sweep parameter; every combination of values becomes a job |

Reference sweep parameters anywhere in the recipe with `{placeholder}` syntax. A string that is exactly one placeholder takes the value with its original type (an integer stays an integer); a placeholder inside a longer string is interpolated as text. Each job is named `<name>_<param><value>_<param><value>...` and is otherwise the recipe with `sweep:` removed.

```yaml
name: "qwen3-0.6b-sweep"

engine: sglang
roles:
  agg:
    nodes: 1
    workers: 2
    gpus: 1
    args:
      tensor-parallel-size: 1
      max-running-requests: "{max_running_requests}"

benchmark:
  type: "sa-bench"
  isl: 128
  osl: 128
  concurrencies: "{concurrency}"

sweep:
  concurrency: [4, 8]
  max_running_requests: [16, 64]
```

This produces four jobs. `examples/features/sweep.yaml` is a runnable version. `srtctl dry-run` on a sweep recipe renders every expanded job.

---

## Config Overrides

Config overrides let you define a base config plus multiple variants in a single YAML file. Each variant deep-merges a small set of changes onto the base, and is submitted as an independent SLURM job. This eliminates the need to duplicate entire config files when testing different parameter combinations.

### YAML Structure

```yaml
schema: 2
base:
  name: "my-benchmark"
  engine: sglang
  roles:
    decode:
      nodes: 8
      args:
        tp-size: 32
  benchmark:
    concurrencies: [8192, 10240]

override_tp64:
  roles:
    decode:
      args:
        tp-size: 64

override_small:
  roles:
    decode:
      nodes: 4
  benchmark:
    concurrencies: [4096]
```

| Key | Description |
| --- | --- |
| `schema` | `2`, beside `base:` at the top of the file |
| `base` | Required. A complete, valid config (same structure as a normal recipe). |
| `override_<suffix>` | Optional. Partial config merged onto base. `<suffix>` is appended to the job name. |

### Naming

Override job names are auto-generated: `{base.name}_{suffix}`.

The example above produces three jobs: `my-benchmark`, `my-benchmark_tp64`, and `my-benchmark_small`.

### Deep Merge Semantics

| Type | Behavior | Example |
| --- | --- | --- |
| **Scalar** (str/int/bool) | Override replaces base | `tp-size: 32` becomes `tp-size: 64` |
| **Dict** | Recursive merge; only specified keys change | Override `roles.decode.args.tp-size: 64` leaves other decode keys untouched |
| **List** | Full replacement (no append) | `concurrencies: [4096]` replaces `[8192, 10240]` |
| **New key** | Added to base | Override adds fields base doesn't have |
| **`null` value** | Deletes the key from base | `extra_mount: null` removes it |

Because `roles.<role>` is a mapping, an override can change one role's `nodes`, `workers`, `gpus`, one `env` entry, or one `args` flag without restating the rest of the role.

### Combining with Sweeps

Overrides and sweeps can coexist in the same file. Override expansion happens first, then each variant with a `sweep:` section is expanded via Cartesian product.

```yaml
schema: 2
base:
  name: "combined"
  sweep:
    chunked_prefill_size: [4096, 8192]
  engine: sglang
  roles:
    prefill:
      args:
        chunked-prefill-size: "{chunked_prefill_size}"

override_big:
  roles:
    decode:
      nodes: 16
```

This produces **4 jobs**: base x 2 sweep + override_big x 2 sweep. `srtctl apply` expands each variant's sweep at submit time; `srtctl dry-run` renders the resulting jobs.

Files without a `base` top-level key are ordinary recipes.

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

**Note**: For per-role environment variables, use `roles.prefill.env`, `roles.decode.env`, or `roles.agg.env` (see [roles](#roles)). The role's `env` is applied first and the global `environment` after it, so a key set in both takes the global value.

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

- **Commands run as you, not as root.** Anything privileged needs passwordless sudo (`sudo -n ...`). A `sudo` that prompts for a password will hang until `timeout_seconds` and then fail the job; verify first with `srun --jobid <job> --overlap -w <node> sudo -n true`. If sudo prompts, no recipe change helps; the cluster's SLURM `Prolog=` (which runs as root) is the only route.
- **Prefer setting `teardown` whenever `commands` changes persistent node state.** `nvidia-smi -lmc` outlives the allocation, so without a matching `-rmc` the next job on that node inherits your locked clocks. `srtctl dry-run` warns when `commands` is set without `teardown`.
- `teardown` runs from the job's cleanup path, so it fires on failure and cancellation too, and never changes the job's exit code.
- Set cluster-wide via `default_host_setup` in `srtslurm.yaml`; that's the right home when *the cluster's machines* need this, rather than one recipe. See [Cluster Config Fields](#cluster-config-fields).
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
  - name: nats
    type: nats
    placement:
      node: dedicated            # etcd and nats share the infra node: both or neither
    options:
      max_payload_mb: 24         # raise the NATS message size limit
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
| `placement.node` | string | type default | One instance for `head`/`infra`/`dedicated` (`dedicated` reserves a node); one per node for `prefill`/`decode`/`agg`/`workers`; every allocated node for `all` |
| `start` | string | type default | `infra` (etcd, nats), `before_workers` (mooncake-master, mooncake-store), `after_frontend` (generic, exporters) |
| `readiness` | object | type default | One probe (`port`/`tcp`, `http`, or `log`) plus `timeout_seconds` and `interval_seconds`; the job waits for it on every service node. Typed kinds gate on their well-known ports by default |
| `inherit_discovery_env` | bool | `true` | Inject the Dynamo discovery env |
| `critical` | bool | type default | A crash fails the run when true |
| `source`, `build_command` | object, list[string] | none | Clone an immutable git rev and build once before launch; single-node placements only |
| `build_timeout_seconds` | int | `1800` | `build_command` is killed when this runs out so a hung build cannot hold the allocation |
| `preamble`, `cpus_per_task`, `cpu_bind`, `srun_options` | | none | Pass-through launch knobs for this service |

The Mooncake KV store is a `mooncake-master` service plus Mooncake env on the roles; see [mooncake-kv-store.md](mooncake-kv-store.md). The v1 spelling of the discovery plane (`infra.etcd_nats_dedicated_node`, `infra.nats_max_payload_mb`) and of the Mooncake master (`backend.mooncake_kv_store`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

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
