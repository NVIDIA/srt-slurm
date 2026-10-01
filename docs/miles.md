# Miles RL post-training

srt-slurm launches a [Miles](https://github.com/radixark/miles) training run as a services-only job: a `ray` service brings up the cluster, and a `custom` benchmark step runs the bundled launcher `benchmarks/rl/miles/launch.sh`, which hands the cluster to Miles's own launch script. Miles pairs SGLang rollout engines with a Megatron-LM (or FSDP) trainer under Ray and owns every GPU. srt-slurm owns everything around it: the SLURM allocation, the container and mounts, Ray bring-up with readiness, the launcher step, per-job logs, tachometer telemetry, status reporting, and teardown.

Nothing in srt-slurm's core knows about Miles. The launcher is a script in the repo-root `benchmarks/rl/` folder, written against the environment every custom benchmark receives.

## Table of Contents

- [What runs where](#what-runs-where)
- [Recipe](#recipe)
- [Environment the launcher reads](#environment-the-launcher-reads)
- [Prerequisites](#prerequisites)
- [Agentic environments and sandboxes](#agentic-environments-and-sandboxes)
- [Gotchas](#gotchas)

## What runs where

The benchmark step is only the launcher and the log tail of the Ray job. Everything RL happens inside that job, on the raylets the `ray` service started.

```
sbatch: N nodes, one task per node
  1. services[train] (type: ray, nodes: N, placement: every worker node)
       node 0      ray start --head --node-ip-address <fabric ip> --port 6379 --dashboard-host 0.0.0.0 --num-gpus G --block
       node 1..N-1 ray start --address <node0 ip>:6379 --num-gpus G --block        (after the head answers /api/version)
       gate        /nodes?view=summary lists N raylets ALIVE
  2. exporters, tachometer (when enabled)
  3. benchmark (type: custom) on node 0, same image and mounts, no GPUs of its own:
       srtctl injects  SRT_SERVICE_TRAIN_IPS=<node0 ip>,...  SRT_SERVICE_TRAIN_NODE_COUNT=N  SRT_GPUS_PER_NODE=G
       launch.sh sets  MILES_SCRIPT_EXTERNAL_RAY=1  MASTER_ADDR=<node0 ip>  RAY_ADDRESS=http://<node0 ip>:8265
                       MILES_SCRIPT_NUM_NODES=N  MILES_SCRIPT_NUM_GPUS_PER_NODE=G
       and execs       cd /root/miles && python3 scripts/run_qwen3_dense.py      (every option comes from MILES_SCRIPT_* env)
       Miles runs `ray job submit -- python3 train.py ...` and streams the job's stdout into benchmark.out
  4. inside the Ray job (in the ray service's containers):
       MegatronTrainRayActor x (nodes x GPUs)   the trainer ranks
       CommandActor per engine                  each execs sglang.launch_server: the rollout engines
       router                                   spreads /generate over the engines
       RolloutExecutor                          samples prompts, generates, scores rewards: the rollout
       loop: rollout -> reward -> GRPO step -> update_weights into the engines -> repeat
  5. the launcher exits with the Ray job's status -> postprocess -> teardown: launcher, then the raylets (SIGTERM stops Ray cleanly)
```

Two consequences shape the recipe. The `ray` service carries the job's container, mounts and environment because that is where the trainer and engines actually run; the benchmark step is only a client of the dashboard port. And the Miles launcher runs `pkill -9 sglang` before it starts, which is harmless here because every SGLang engine is Miles's own and starts afterwards.

## Recipe

```yaml
schema: 2
name: "qwen3-4b-grpo"
model: { path: "qwen3-4b", container: "miles", precision: "bf16" }
resources: { gpu_type: "b200", gpus_per_node: 8 }
frontend: { type: none }
services:
  - name: train                              # the Ray cluster; owns the job's two nodes
    type: ray
    nodes: 2
    preamble: "cd /root/miles"
    metrics:                                 # both on the Ray head; see Engine metrics
      - { name: miles, port: 9090, nodes: first }
      - { name: engines, port: 31000, path: /engine_metrics, nodes: first }
benchmark:
  type: custom
  command: /benchmarks/rl/miles/launch.sh
  env:
    MILES_RECIPE: scripts/run_qwen3_dense.py
    MILES_SCRIPT_MODEL_NAME: Qwen3-4B
    MILES_SCRIPT_MODEL_DIR: /data/models
    MILES_SCRIPT_DATA_DIR: /data/datasets
    MILES_SCRIPT_OUTPUT_DIR: /data/miles-runs/qwen3-4b-grpo
    MILES_SCRIPT_ENABLE_EVAL: "true"
    MILES_SCRIPT_EXTRA_ARGS: >-
      --num-rollout 20 --colocate --actor-num-nodes 2 --actor-num-gpus-per-node 8 --rollout-num-gpus-per-engine 2
      --sglang-router-port 31000 --use-prometheus --prometheus-port 9090 --prometheus-run-name qwen3-4b-grpo
extra_mount:
  - "/data/models:/data/models"
  - "/data/datasets:/data/datasets"
  - "/data/miles-runs:/data/miles-runs"
environment: { NCCL_SOCKET_IFNAME: bond0 }
observability: { tachometer: { enabled: true } }
```

`frontend.type: none` says there is no router and no worker-count health gate; the services' own probes are the only gate. `services[].nodes` on the ray service is the node count of the job: with no engine roles, the service that owns the nodes sets the allocation size. The directories Miles reads and writes are host paths, so they are mounted at the same path into every step with `extra_mount`; the raylets need them because the trainer and engines run there. The `metrics` list on the ray service is what puts Miles's own rewards and timers and the engines' KV usage into tachometer's parquet next to the GPU telemetry; see [Engine metrics](#engine-metrics).

`examples/miles/qwen3-4b-grpo.yaml` is a runnable version.

## Environment the launcher reads

Injected by srtctl for every custom benchmark command (see [config reference, custom](config-reference.md#custom)):

| Variable | Meaning |
| --- | --- |
| `SRT_SERVICE_TRAIN_IPS`, `SRT_SERVICE_TRAIN_NODE_COUNT` | The ray service's nodes; the first IP is the head. `TRAIN` is the service name upper-cased |
| `SRT_GPUS_PER_NODE` | `resources.gpus_per_node` |

Set in `benchmark.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MILES_RECIPE` | required | Launch script, relative to `MILES_ROOT` or absolute in the container |
| `MILES_RAY_SERVICE` | `train` | Name of the `services[]` entry that is the Ray cluster |
| `MILES_ROOT` | `/root/miles` | The Miles checkout in the container |
| `MILES_SUBCOMMAND` | none | Positional for the multi-step launchers (`train`, `prepare`, `full-train`); the single-command recipes such as `run_qwen3_dense.py` take none |
| `MILES_RAY_PORT`, `MILES_RAY_DASHBOARD_PORT` | 6379, 8265 | Must match the ray service's `options` |
| `MILES_LAUNCH_DRY_RUN` | 0 | `1` prints the resolved environment and command and exits |
| `MILES_SCRIPT_*` | | Miles's own recipe options, passed through untouched. Every `ScriptArgs` field of a launch script is settable as `MILES_SCRIPT_<FIELD>`: `MODEL_NAME`, `MODEL_DIR`, `DATA_DIR`, `OUTPUT_DIR`, `ENABLE_EVAL`, `EXTRA_ARGS` (appended to `train.py`), `EXTRA_ENV_VARS` (JSON, reaches Ray's runtime env), `MEGATRON_PATH`, ... |

The launcher exports, unless already set, `MILES_SCRIPT_EXTERNAL_RAY=1`, `MASTER_ADDR`, `RAY_ADDRESS`, `MILES_SCRIPT_NUM_NODES` and `MILES_SCRIPT_NUM_GPUS_PER_NODE`, all derived from the ray service. It exits with the Ray job's status.

Give `MILES_SCRIPT_OUTPUT_DIR` a stable shared path: Miles resumes from the last checkpoint there when the same recipe is resubmitted. Global `environment:` reaches every step, including the raylets, so raylet-spawned processes inherit it; put NCCL and GLOO interface names there.

## Prerequisites

The image is `radixark/miles` (about 43 GB as a squashfs). It pins patched SGLang, Megatron-LM and sgl-router branches; use it for the whole job. Import once and alias it in `srtslurm.yaml`:

```bash
ENROOT_CACHE_PATH=/data/$USER/.cache/enroot enroot import -o /data/$USER/squash/miles-latest.sqsh docker://radixark/miles:latest
```

The recipe expects, on the shared filesystem, the HF weights, the converted Megatron checkpoint next to them, and the datasets:

```bash
hf download Qwen/Qwen3-4B --local-dir <model_dir>/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir <data_dir>/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir <data_dir>/aime-2024

# once, on a GPU node, inside the image:
cd /root/miles
read -ra MODEL_ARGS <<< "$(python3 miles/utils/external_utils/model_args_utils.py qwen3-4B)"
PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint <model_dir>/Qwen3-4B --save <model_dir>/Qwen3-4B_torch_dist
```

## Agentic environments and sandboxes

The run above scores answers with a verifier inside the rollout executor. Miles's agentic recipes instead drive an environment: an OpenEnv, NeMo Gym or Verifiers server the agent loop talks to over HTTP, or a per-task sandbox on E2B, Modal, Daytona or a Kubernetes-hosted AgentENV, orchestrated through Harbor or HUD. srt-slurm needs no code for any of these. They are Miles connectors configured through environment, and three things in the recipe carry that environment.

```yaml
services:
  - name: train
    type: ray
    nodes: 2
    # Credentials reach every process Miles spawns, because they inherit the raylet's
    # environment. Keep them in a mode-600 file on the shared filesystem, not in the recipe.
    preamble: "set -a; . /data/secrets/sandbox.env; set +a; cd /root/miles"
benchmark:
  type: custom
  command: /benchmarks/rl/miles/launch.sh
  env:
    MILES_RECIPE: examples/experimental/openenv/glm52_tbench2/run_glm5_2_744b_a40b_daytona.py
    MILES_SUBCOMMAND: train                 # the agentic launchers are multi-step; single-command recipes take none
    MILES_SCRIPT_MODEL_NAME: ...
    MILES_SCRIPT_EXTRA_ENV_VARS: '{"OPENENV_RUN_ID": "..."}'   # anything a recipe wants in Ray's runtime env
```

- **Credentials** (`E2B_API_KEY`, `MODAL_TOKEN_ID`, `DAYTONA_API_KEY`, an AgentENV token) go in the ray service `preamble` as a sourced file. Miles's own SLURM launcher does the same with a `DAYTONA_ENV_FILE`. Global `environment:` also works but puts the value in the recipe and in every log of the sbatch script.
- **An environment server you run yourself** (OpenEnv, NeMo Gym, Verifiers) is a `services[]` entry with its own `container:`; its address reaches the launcher as `SRT_SERVICE_<NAME>_IPS`, and a few lines in a copy of `launch.sh` turn that into the URL the recipe expects.
- **Outbound network** from the compute nodes to the provider is required; SaaS providers need internet, a Kubernetes AgentENV needs a route to its service.

Whether a provider works from a SLURM cluster depends on which direction the traffic flows:

| Mode | Traffic | Works from a private compute fabric |
| --- | --- | --- |
| In-process trials: the agent loop runs inside Miles's rollout executor and the sandbox only executes commands | compute node to provider, outbound only | Yes, with outbound network and a key |
| Sandbox-hosted agent: a coding agent runs inside the sandbox and calls the policy through Miles's session server or router | sandbox to a compute node's fabric IP, inbound | Only if that endpoint is reachable from the sandbox: a route back from a Kubernetes AgentENV, or an ingress or tunnel for SaaS sandboxes. Miles's `--pin-rollout-manager-to-head` exists for the Kubernetes version of this |

Before calling a recipe ready, check that its connector's Python dependencies are in the image. Miles's Harbor connector imports `harbor`, which its own README installs separately, so it may not be in `radixark/miles:latest`; steps run read-only containers, so the fix is a derived squashfs with the connector installed, or a venv on the shared filesystem added to `PYTHONPATH` in the ray service `preamble`. Per-task sandboxes hosted on the SLURM cluster itself are a different matter: pyxis containers have no Docker daemon, so that needs a daemonless provider such as an enroot-based sandbox service or rootless podman, which nothing here provides yet.

## Engine metrics

Miles's telemetry surfaces are all fixed-port endpoints on the Ray head once the recipe pins them, so a `metrics` list on the `ray` service is the whole integration; tachometer scrapes them like any other service (see [services.md](services.md#metrics)).

- **The engines, through the router.** Miles picks its SGLang engines' ports at runtime, but its default router, the sgl-model-gateway, serves `GET /engine_metrics`: the metrics of every registered engine, each sample labeled `worker_addr`. Pin the router with `--sglang-router-port` and scrape that. Names arrive normalized by the router (`sglang_num_running_reqs` rather than `sglang:num_running_reqs`).
- **Miles's training collector.** With `--use-prometheus` Miles runs a collector actor pinned to the driver node, the Ray head, that serves `miles_metric_*` gauges (rollout rewards, response lengths, actor and rollout timers, MFU) on `--prometheus-port`, 9090 by default, labeled `run_name`.

```yaml
services:
  - name: train
    type: ray
    nodes: 2
    metrics:
      - { name: miles, port: 9090, nodes: first }                          # miles_metric_*
      - { name: engines, port: 31000, path: /engine_metrics, nodes: first }  # sglang_* per engine, worker_addr label
benchmark:
  env:
    MILES_SCRIPT_EXTRA_ARGS: "... --sglang-router-port 31000 --use-prometheus --prometheus-port 9090 --prometheus-run-name qwen3-4b-grpo"
```

`nodes: first` scrapes only the service's first node, the head; the rows land as endpoints `miles_<head>` and `engines_<head>` with `service=train`. Both endpoints appear a few minutes into the job, once Miles has started the router and the collector, so the scraper logs failures until then. On sa-b200 (job 15881, Qwen3-4B GSM8K smoke, four TP2 engines) the parquet held 83 `miles_metric_*` families and 71 `sglang_*` families per engine at one-second resolution; KV usage peaked at about a quarter of the pool per engine and dropped to zero during training steps.


## Gotchas

- **Layout is Miles's to validate.** `--actor-num-nodes`, `--rollout-num-gpus` and friends live in `MILES_SCRIPT_EXTRA_ARGS`; srt-slurm does not check them against the allocation, Miles does when the job starts. Colocated: actor nodes equal the ray service's `nodes`. Most shipped recipes hardcode `--colocate`, and a store-true flag cannot be negated from `extra_args`.
- **Health timeouts.** The ray service allows ten minutes per node for the first `ray start`, which imports torch. Megatron checkpoint load happens inside the Miles job and is not gated by srt-slurm.
- **Fault tolerance is Miles's.** The raylets are `critical`, so a raylet dying fails the run. Engine restarts inside Miles (`--use-fault-tolerance`) do not involve srt-slurm.
- **Benign NCCL warnings.** `transport/p2p.cc NCCL WARN Cuda failure 1 'invalid argument'` during CUDA graph capture appeared in every successful run.
- **Live logs.** `tail -F outputs/<job>/logs/{sweep_<job>.log,benchmark.out}`; the driver log carries the Ray job's stdout, engines and actors included. `perf N` lines are the rollout side of iteration N, `step N` the trainer's.
