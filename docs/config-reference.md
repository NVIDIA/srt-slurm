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
| GPU power control (EPD) | [power_control](#power_control) on this page | `power_control` |
| Many jobs from one file | [Parameter Sweeps](sweeps.md), [Config Overrides](overrides.md) | `sweep`, `base` / `override_*` |

This page covers the remaining top-level keys (`schema`, `name`, `model`, `output`, `health_check`, `enable_config_dump`, `power_control`) and complete examples.

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

Output configuration with formattable paths.

```yaml
output:
  log_dir: "./outputs/{job_id}/logs"
```

The `log_dir` supports FormattablePath templating. See [FormattablePath Template System](runtime-env.md#formattablepath-template-system).

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

## power_control

Closed-loop GPU power control over the job's workers with NVIDIA's EPD controller (the `epd` package: NUPD redistributes a fixed budget by measured draw, UBPC lowers each node's cap toward measured draw × a buffer). EPD never touches an engine or a GPU itself: it senses NVML energy counters, reads the figure of merit (FoM) you choose, and writes its decisions to a file; a separate host process applies them.

```yaml
power_control:
  enabled: true
  mode: nupd                    # nupd | ubpc | both
  pools: shared                 # shared | per-role | explicit list (below)
  budget_w_per_gpu: 1500        # required: the per-GPU allowance
  fom: tokens_per_gpu           # required: the objective EPD protects
  interval_s: 30
  perf_threshold: 0.03
  ubpc_buffer: {prefill: 2.0, decode: 1.1}
  min_tgp_w: null
  warmup_policy: adaptive
  applier: sudo-smi             # sudo-smi | dcgmi
  epd_source: /lustre/share/nvidia-epd/src/epd
  dynamo_power_source: /lustre/share/dynamo/components/src/dynamo/power
  critical: true
```

Fields: [PowerControlConfig](schema-reference.md#powercontrolconfig) and, for explicit pools, [PowerPoolConfig](schema-reference.md#powerpoolconfig). `srtctl dry-run` prints the resolved pools, the overlay mounts and the sudo requirement; [examples/features/power-control.yaml](https://github.com/NVIDIA/srt-slurm/blob/main/examples/features/power-control.yaml) is a runnable recipe.

**What runs on each worker node.** Three processes, none inside another, started once the workers pass the health gate and before the benchmark client starts:

| Step | Where | Command | Role |
| --- | --- | --- | --- |
| worker step | the job container (one srun per endpoint, as always) | `python3 -m dynamo.<engine> ...` | serves; untouched by power control |
| `power_node_agent_<pool>_<node>` | a second container from the worker image on the same node (`srun --overlap`, the recipe's `srun_options`, the job mounts plus `epd_source` read-only at `<site_packages>/epd` and, when set, `dynamo_power_source` at `<site_packages>/dynamo/power`) | `python3 -m dynamo.power.node_agent --pool-name <pool> --rank <k> --nodelist <pool nodes> --log-dir /logs/power_control/<pool> --gpu-indices <this node's GPUs in the pool> --mode ... --budget-w-per-gpu ... --fom ... --interval-s ... --perf-threshold ... --ubpc-buffer ... [--max-tgp-w] --warmup-policy ... --metrics-url frontend=<frontend /metrics>,worker=<pool worker system /metrics>,... --gpus-per-node <N> --pool-gpu-count <GPUs in the pool>` | senses this node's GPUs (NVML, read-only) and writes EPD's energy snapshots; rank 0, the pool's first node, also runs the controller and the FoM provider and writes `epd/actuation/power_decision.json` (`{"per_node_gpu_power_limit_w": {<node>: <W>}}`) through EPD's file actuator |
| `power_applier_<node>` | the bare host (no container), the orchestrator's interpreter | `python -m dynamo.power.applier --decision-file <log_dir>/power_control/<pool>/epd/actuation/power_decision.json --node <node> --gpu-indices ... --mode sudo-smi\|dcgmi --budget-w-per-gpu ... --stale-after-s 90 --poll-s 5 --restore-default-on-exit --trace-csv <log_dir>/power_control/<pool>/applier_<node>.csv [--min-w-per-gpu <min_tgp_w>]` | polls the decision file and applies this node's entry per GPU index (`sudo nvidia-smi -pl <W> -i <N>`, or a DCGM group of those GPUs plus `dcgmi config --set -P <W>`), verifies through read-only NVML, writes the trace CSV (`ts, gpu, requested, setpoint, enforced, status` with status `ok`, `not_landed`, `clamped` or `unverified`); when the file is older than `stale_after_s` or disappears, it sets every GPU of the group to the uniform budget; on exit it restores each GPU's hardware default limit |

The node_agent container runs the same setup script and Dynamo install preamble as the workers and receives the recipe `environment:` (with the workers' `{node}` / `{node_id}` templating) plus EPD's own variables (`LOG_DIR`/`EPD_LOG_DIR`, `EPD_ACTUATOR=file`, `EPD_NODELIST`, `EPD_NUM_NODES`, `GPUS_PER_NODE`, `EPD_RUN_TRACE`, `EPD_SUMMARY_STDOUT`), so `python3 -m epd doctor` works in that container. A group that covers the whole node sees every GPU; a group that is only part of the node (a TEP6 tail) gets `NVIDIA_VISIBLE_DEVICES` set to its indices, because EPD's stock NVML monitor senses every GPU it can see and would otherwise average idle GPUs into the node's draw and earn the active ones a lower cap. `--gpu-indices` are host NVML indices and the node_agent verifies at start that the visible GPUs are exactly the group; `--pool-gpu-count` carries the pool's exact GPU total for per-GPU FoM normalisation. `min_tgp_w` is never passed to the node_agent (EPD's NUPD cap law has no per-node floor); the applier enforces it as `--min-w-per-gpu`, raising any requested cap below it and reporting the row as `clamped`. The decision file is the path EPD's `DecisionFilePublisher` writes, `<LOG_DIR>/epd/actuation/power_decision.json`: the node_agent hands `--log-dir` to EPD as `LOG_DIR` unchanged, the applier polls exactly that file, and teardown logs an error for a pool whose file never appeared (EPD publishes its starting caps the moment it starts, so a missing file means the appliers held the uniform budget throughout: a static-cap run, not an EPD run). `--metrics-url` names the frontend listeners and the pool ranks that run a Dynamo system server (every rank of a per-node launch; only the MPI leader of a TRT-LLM step), each node's IP resolved once; an unreachable URL is a missing sample for that interval, never a failure. The applier imports `dynamo.power` from `dynamo_power_source`'s grandparent directory (put on its `PYTHONPATH`); without `dynamo_power_source`, the orchestrator's interpreter must already import `dynamo.power`. It verifies every cap through `pynvml` (`nvidia-ml-py`, a declared srtctl dependency); `srtctl dry-run` warns when the interpreter cannot import it, since every trace row would then be `unverified`.

**Pools and the unit of control.** EPD sets one cap per *unit*; here the unit is a GPU group, the GPUs on one node that belong to one pool, so idle GPUs on a partially used node are neither sensed nor capped and the budget counts GPUs in use. `pools: shared` is one controller over every role with one budget (cap headroom moves between prefill and decode as the load pulls it); `per-role` is one controller per role, each with its own budget, FoM and buffer and no cross-role shift; an explicit list writes the pools out:

```yaml
power_control:
  enabled: true
  budget_w_per_gpu: 1300
  fom: gen_tokens
  epd_source: /lustre/share/nvidia-epd/src/epd
  pools:
    - { name: ctx, roles: [prefill], budget_w_per_gpu: 1800, fom: ctx_tokens, ubpc_buffer: 2.0 }
    - { name: gen, roles: [decode] }          # inherits the block's budget, FoM and buffer
```

Each entry names `roles` or `nodes` (hostnames, for allocations with a reservation); a role or node may be in one pool only, and knobs left unset come from the block. Nodes in no pool run uncontrolled. The stage derives the groups from the actual GPU map (`Endpoint.gpus_on_node`), so a node shared by two workers of one role (TEP6 prefill packed 4+2 / 2+4 on 4-GPU nodes) is one group, and it refuses a layout that puts one node in two pools (`per-role` over `roles.decode.nodes: colocate`, or an explicit list that cuts a role mid-node), an explicit node pool naming a node that runs no worker, or an empty pool. That check runs at job start, before host setup and the workers, so the refusal costs seconds rather than a model load, and again when the node_agents launch; `srtctl dry-run` already rejects the colocated case and a recipe with `frontend.type: none` (no `/metrics` to read the FoM from). Match pools to electrical units: a budget guarantees the sum of caps over the pool's nodes, so a pool spanning two PDU feeds can move watts from one feed to the other.

**FoM.** `fom` is required and has no default: it is the objective EPD protects. Forms: a preset (`output_tokens`, `tokens_per_gpu`, `per_user_speed_p25`, `tokens_per_joule`, `ctx_tokens`, `gen_tokens`, `inv_ttft_p50`), `prom:<metric>[:<label>=<value>,...]:<counter|gauge>:<sum|avg>` scraped from every `--metrics-url` (the frontend and the pool's workers), or `python:<module>:<Class>` exposing `get_perf() -> float`. The recipe loader checks the syntax; the node_agent checks the signal contract at start (higher is better, a per-interval rate, computed over the whole pool, `0.0` meaning no reading) and refuses to start EPD on a spec that fails it. A per-watt or per-joule FoM rewards cutting power, so pair it with `min_tgp_w` or a per-user-speed guard when latency matters.

**Budget and knobs.** `nupd` acts only when `budget_w_per_gpu` is below the hardware maximum with the recipe's clock locks in place; otherwise it moves caps that never throttle. With `max_tgp_w` set, the loader refuses `mode: nupd` unless the block's budget and every pool's budget are below it (EPD refuses to start a NUPD-only controller that has nothing to redistribute, and with `critical: true` that would tear the job down after the model loaded); without `max_tgp_w`, `srtctl dry-run` prints the reminder. `ubpc` works at any budget; the buffer sets how far caps fall, and two caveats apply: the headroom it frees disappears the moment the FoM dips (EPD lifts the caps back to the budget), and it has no load-aware release, so a cap set during quiet traffic stays until the FoM drops far enough to trigger a revert. `perf_threshold` 0.03-0.05 suits serving load; `interval_s` below 5-10 s does not let caps settle before the next read. Changing `fom` or `mode` is a new run, not a hot swap.

**Lifecycle and teardown.** Both steps are managed processes: with `critical: true` (default) a node_agent or applier exiting fails the run like a worker would. Fail-safe is to the budget, not to the hardware default: a controller that dies leaves a decision file that goes stale, and the applier then holds the pool's uniform budget. On the cleanup path (success, failure, cancel) the orchestrator stops the node_agents first (EPD finishes an in-flight cycle and writes `epd/run_summary.json`; their SIGTERM budget is `interval_s` + 15 s, at least 30 s), then the appliers (which restore the hardware default limits), then runs one container-less reset per node (`power_reset_<node>`, with the recipe's `srun_options`) that sets each controlled GPU back to its `power.default_limit` through the same privileged path, so no cap outlives the allocation (DCGM caps persist as hostengine target config, which is why the final reset is not optional). The reset uses `sudo -n`, so a missing sudoers rule fails at once instead of hanging on a password prompt, and its exit status covers every GPU of the node: a non-zero status is logged as an error naming the node that may still be capped.

**Privileges.** With `applier: sudo-smi` the applier and the final reset run as you on the bare host and need a passwordless sudoers rule for `nvidia-smi -pl <W> -i <N>` in that exact argument order (the form in-container `nvidia-smi` cannot use: it exits 4 without root). Check before spending an allocation: `srun --jobid <job> --overlap -w <node> sudo -n -l`. With `applier: dcgmi` the node's DCGM hostengine applies the caps and no privilege is needed. Use either `power_control` or a static cap set through `host_setup`, never both: two writers of the same limit chase each other.

**Artifacts** under `<log_dir>/power_control/<pool>/`: `node_agent_<node>.out`, `applier_<node>.out`, `applier_<node>.csv` (the enforced-limit trace), `epd/actuation/power_decision.json`, `epd/run_trace.csv` (requested / setpoint / enforced per cycle, FoM column non-zero once engaged), `epd/run_summary.json`, and `power_reset_<node>.out`. Runs compare only at equal total watts: report tokens/s/GPU and tokens/s/user at the same concurrency against a uniform static cap and a full-power reference, with the enforced-limit trace.

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
