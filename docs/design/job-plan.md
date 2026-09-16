# Job plan: pods, kinds, and a graph under the recipe

Status: design sketch, 2026-09-15. Follows `sflow-engine.md` (DAG engine POC on `idhanani/sflow-engine-poc`) and `schema-v2-minimal.md`. Motivating case: launching Miles (radixark/miles, RL post-training on SGLang plus Megatron under Ray) from srt-slurm, and every other job that is not "engines behind one frontend with one client".

## Problem

The recipe path on `main` is a fixed stage sequence over an inference topology, and each coupling below blocks a job that has no engines, no frontend, or more than one of either:

- `SweepOrchestrator.run()` (`src/srtctl/cli/do_sweep.py:601`) always runs `start_frontend` (`:665`), and the benchmark stage returns 1 unless a frontend URL reports the expected worker counts (`src/srtctl/cli/mixins/benchmark_stage.py:238,302-308`). There is no `frontend.type: none` (`src/srtctl/frontends/base.py:23,110-129`).
- Topology is prefill, decode and agg only: `allocate_endpoints` (`src/srtctl/core/topology.py:279,449-467`), `ROLE_NAMES` (`src/srtctl/core/roles.py:66`), and `Nodes` reserving only the infra, frontend and bench singletons (`src/srtctl/core/runtime.py:143-152`). Every role must use the same engine (`roles.py:110`); there is no per-role model. The v1 role-count fields on `ResourceConfig` are read from 21 modules at 84 sites, so the assumption fans out everywhere.
- Services (`src/srtctl/services/config.py:178-249`) are already typed, placeable, probed and critical, but order by stage phase (`SERVICE_STARTS`, `config.py:34`) rather than by dependency, cannot reserve nodes, and have no GPU carve-out.
- No Ray, torchrun or generic multi-node step. The closest primitive is the TRT-LLM launch: one srun across N nodes with `ntasks` (`src/srtctl/cli/mixins/worker_stage.py:303,436-438`).

The DAG engine POC showed the runtime hosts a task graph well (2.9k lines, 108 tests, two live runs) and also showed that a graph written by hand loses what makes recipes safe unattended: worker-count health, port and rank math, validated engine flags, telemetry and postprocess. The conclusion there was to keep the graph additive and grow it toward recipes. This document takes the next step: the graph becomes an internal representation that compilers emit, and the semantics stay in the compilers.

## Concept

```
  recipe (schema 2)          workload kinds             workflow fragments (optional, sflow shape)
  roles / frontend / workload   miles, aiperf, ...        tasks + probes + depends_on
        │                          │                             │
        └──────── compilers: kinds render pod-shaped steps into one plan ────────┘
                                   │
                          plan.json  (pools, steps, probes, edges, terminal set)
                                   │
        engine: resolve node-derived refs, gate on edges and probes, launch, watch, tier teardown, event log
                                   │
             executors: srun inside an sbatch (today) | docker on one host | kubernetes
```

Three layers, each with one job:

- **Surface.** Opinionated and short. Roles are controllers with a pod template that a kind fills from a few typed fields. Workloads are terminal tasks. Frontends and services are optional kinds. Ordering is `after` edges with kind defaults.
- **Plan.** Data. Pools, pod-shaped steps, typed probes, edges, a terminal set. Serialized at submit time with node-derived values left as references. Printed by dry-run, diffed by golden tests, consumed by every executor.
- **Engine and executors.** One lifecycle for every step. Executors map a step onto srun, docker or a Kubernetes object.

## Surface

A role is a controller plus a pod template. This is the LeaderWorkerSet and RayCluster shape rather than a bare PodSpec because engines and Ray span nodes and pods do not.

```yaml
roles:
  <name>:
    kind: sglang | vllm | trtllm | mocker | ray | none   # who renders the template; default: top-level engine
    mode: prefill | decode | agg                        # engine kinds only; defaulted by the conventional names
    replicas: 4                                         # engine instances, or ray worker nodes
    size: 1                                             # nodes per replica; 2 = one TP16 engine over two nodes
    gpus: 2                                             # per node of the replica; sugar for resources.gpu
    nodes: 2                                            # sugar: replicas * size on whole-node roles
    args: { ... }                                       # kind-typed flags, rendered into the main container
    env: { ... }
    model: { path: ..., container: ... }                # optional per-role override of the top-level model
    placement: { colocate_with: prefill } | { dedicated: true } | { pool: <name> }
    critical: true
    after: [ <role or service names> ]                  # kind defaults apply when omitted
    pod:                                                # escape hatch, merged over what the kind rendered
      image: ...
      resources: { cpu: 32, memory: 256Gi, shm: 32Gi }
      volumes: [ { host: /data, mount: /data } ]
      initContainers: [ { name: convert, image: miles, command: [ ... ] } ]
      sidecars:       [ { name: kv-store, image: mooncake, command: [ ... ], ports: [ { name: store, port: auto } ] } ]
      containers:     [ { name: engine, env: { SGLANG_X: "1" }, readinessProbe: { httpGet: { path: /health, port: http } } } ]
      restartPolicy: Never
      slurm: { cpu_bind: none, mpi: pmix }              # executor annotations, ignored by other executors
```

Rules:

- `kind` selects a role kind. Engine names, `ray`, or `none` for a bare pool. Engine roles take `mode`. Known names `prefill`, `decode`, `agg` default their mode and stay as short as today.
- `frontend:` is optional. It is absent when no engine roles exist. `frontend.attach` defaults to every engine role.
- `workload:` is the terminal task. Types: `aiperf`, `sa-bench`, `lm-eval`, `custom`, `manual`, `miles`, and so on. `workload.on` names the pool the client runs on. `workload.attach` names the pool whose published endpoints it consumes; the default is the frontend when one exists.
- `services[]` keep today's shape and gain `after: [names]`. The three phases `infra`, `before_workers`, `after_frontend` map to canonical anchors so existing recipes are unchanged.
- `container`, `model`, `env` exist at the top level and per role, service and workload.
- Named ports (`ports[].name`) are the endpoint references other kinds consume. With host networking on SLURM the allocator assigns the numbers.
- Sweeps and overrides address typed fields. Paths into `pod.containers[...]` are legal and never emitted by the migrator.

Three recipes in this shape:

```yaml
# 1. Inference benchmark, same length as today
schema: 2
name: qwen3-235b-dynamo-disagg
engine: sglang
model: { path: /data/models/Qwen3-235B, container: sglang }
roles:
  prefill: { nodes: 2, workers: 2, gpus: 8 }
  decode:  { nodes: 4, workers: 4, gpus: 8 }
frontend: { type: dynamo, args: { router-mode: kv } }
workload: { type: aiperf, isl: 8192, osl: 1024, concurrencies: "64x128" }
observability: { tachometer: true }

# 2. Miles owns the GPUs (colocate or Miles-placed disaggregation)
schema: 2
name: qwen3-4b-grpo
container: miles
roles:
  train:
    kind: ray
    replicas: 2
    pod: { resources: { gpu: 8, shm: 64Gi }, volumes: [ { host: /data, mount: /data } ], env: { NCCL_SOCKET_IFNAME: enP6p9s0 } }
workload:
  type: miles
  on: train
  recipe: scripts/run_qwen3_dense.py
  model_name: Qwen3-4B
  dirs: { models: /data/models, datasets: /data/datasets, output: /data/miles-runs/qwen3-4b-grpo }
  layout: { colocate: true, rollout_gpus_per_engine: 2 }
  args: { enable-eval: false }
  extra_args: "--num-rollout 200"
  env: { WANDB_API_KEY: "..." }
observability: { tachometer: true }

# 3. Miles attached to an srt-slurm rollout fleet (needs Miles #2513 and #2514)
schema: 2
name: qwen3-4b-grpo-ext
engine: sglang
container: miles
model: { path: /data/models/Qwen3-4B }
roles:
  rollout: { mode: agg, nodes: 1, workers: 4, gpus: 2, args: { mem-fraction-static: 0.7 } }
  train:   { kind: ray, replicas: 1 }
workload:
  type: miles
  on: train
  attach: rollout
  recipe: scripts/run_qwen3_4b_fully_async.py
  layout: { colocate: false, actor_nodes: 1, actor_gpus_per_node: 8 }
services:
  - { type: mooncake-master }      # only for update-weight-transfer-mode p2p
```

## Plan

```python
@dataclass(frozen=True)
class Pool:
    name: str
    nodes: int | Literal["rest"]
    whole_node: bool                                   # False: the GPU ledger slices CUDA_VISIBLE_DEVICES

@dataclass(frozen=True)
class Placement:
    pool: str
    fanout: Literal["per_node", "span"]                # one step per node, or one step across the pool
    ntasks_per_node: int = 1
    gpus: int | None = None                            # per instance, from the ledger
    node_index: int | None = None                      # pin to pool[node_index]

@dataclass(frozen=True)
class Container:
    name: str; image: str; command: list[str]; args: list[str]
    env: dict[str, str]; mounts: dict[Path, Path]
    ports: dict[str, PortRequest]                      # name -> fixed or "auto"; filled per (node, step)
    resources: Resources                               # cpu, memory, shm, gpu count
    readiness: Probe | None; liveness: Probe | None

@dataclass(frozen=True)
class Step:                                            # pod-shaped
    name: str
    kind: Literal["service", "task"]
    placement: Placement
    init: tuple[Container, ...]                        # run to completion, in order, before main
    main: Container
    sidecars: tuple[Container, ...]
    after_ready: tuple[str, ...]; after_done: tuple[str, ...]
    critical: bool; shutdown_tier: int; stop_signal: str; terminate_timeout_s: int
    timeout_s: int | None
    annotations: dict[str, dict[str, str]]             # executor-specific: {"slurm": {"mpi": "pmix"}}

@dataclass(frozen=True)
class Probe:
    type: Literal["tcp", "http", "log", "exec", "dynamo_health", "sglang_router_health", "ray_fleet", "server_info"]
    params: dict[str, Any]; interval_s: int; timeout_s: int; failure_threshold: int

@dataclass(frozen=True)
class Plan:
    pools: tuple[Pool, ...]; steps: tuple[Step, ...]
    terminal: tuple[str, ...]                          # run ends when all complete; exit = worst exit code
    windows: tuple[str, ...]                           # measurement windows tasks may stamp
    provenance: dict[str, str]                         # recipe hash, srtctl version, cluster, image digests
```

Two-phase resolution as in the POC (`dag/engine.py:183` static, `:302` node-derived). Values that depend on the allocation stay as references in `plan.json`: `${pool.train[0].ip}`, `${endpoints(rollout, http)}`, `${step.ray-head.port.gcs}`. The engine resolves them inside the allocation once the nodelist is known.

A pod on SLURM is approximate and the subset is documented: `init`, `main` and `sidecars` are `srun --overlap` steps on the same node, they share the host network and bind mounts, they do not share a PID namespace. `emptyDir` is a per-job node-local directory. `shm` is the host `/dev/shm`. Nothing in the current sidecars needs more.

## Kinds

```python
class RoleKind(Protocol):            # sglang, vllm, trtllm, mocker, ray, none
    def steps(self, role: RoleSpec, pool: Pool, ctx: CompileContext) -> list[Step]: ...
    def endpoints(self, steps: list[Step]) -> dict[str, list[EndpointRef]]: ...   # "http", "bootstrap", "ray"

class FrontendKind(Protocol):        # dynamo, sglang-router, vllm-router, trtllm_serve
    def steps(self, fe: FrontendSpec, attached: list[RoleSpec], ctx: CompileContext) -> list[Step]: ...   # includes its semantic probe

class ServiceKind(Protocol):         # exists today: services/registry.py:65
    def steps(self, svc: ServiceSpec, ctx: CompileContext) -> list[Step]: ...

class WorkloadKind(Protocol):        # aiperf, sa-bench, lm-eval, custom, manual, miles
    def validate(self, wl: WorkloadSpec, draft: PlanDraft) -> list[str]: ...
    def steps(self, wl: WorkloadSpec, ctx: CompileContext) -> list[Step]: ...
    def progress(self, line: str) -> Event | None: ...   # optional log parser feeding status

class CollectorKind(Protocol):       # tachometer, dcgm-exporter, node-exporter, cpu power
    def steps(self, obs: ObservabilitySpec, pools: list[Pool], ctx: CompileContext) -> list[Step]: ...

class Probe(Protocol):
    def check(self, step: ResolvedStep) -> ProbeResult: ...
```

`CompileContext` owns the GPU ledger, the port allocator, the container alias and mount resolvers, cluster defaults from `srtslurm.yaml` such as `network_interface`, and deferred references. Kinds never touch srun, nodes or ports directly. Existing code becomes kind bodies: `build_worker_command` on the engine backends, the frontend classes, the service kinds, the benchmark runners, and the telemetry launchers.

Compile order: build pools from roles and placements; call role kinds; call the frontend kind with the attached roles; call service kinds; call collector kinds; call the workload kind with the draft plan so it can validate against pools and consume endpoints; add default edges (services before roles they serve, frontend after roles, workload after everything it attaches to and after collectors); set the terminal set; run validation; serialize.

## Engine guarantees

- **Lifecycle per step.** INITIATED, RUNNING, READY, COMPLETED, FAILED, CANCELLED. `after_ready` and `after_done` edges. Init containers gate the main container.
- **Async probes.** A thread pool, not the scheduler thread. The POC's synchronous `http_post` stalled every other task for up to `each_check_timeout`.
- **One event log.** `events.jsonl` in the run directory: every transition, probe result, window stamp and progress event. Status API, MCP job tools, ruter and the dashboard read it. Only the engine writes state, which removes the monitor-versus-engine attribution race from the POC.
- **Failure and teardown.** `critical` fails the run. Tiers, step names and stop signals reuse the graceful shutdown from #407 (`src/srtctl/core/processes.py:89,326`). Default `restartPolicy: Never`; `OnFailure` with a bounded count only when a job asks.
- **Resources.** GPU ledger per pool, choosing indices for NVLink and NUMA locality. Port allocator per node and step. Kubernetes gets counts, SLURM gets indices; the compiled step carries both.
- **Windows.** A task stamps `window <name> start|end` into the event log through a small `srtctl-mark` command. Postprocess and power reports read windows from the log rather than from benchmark-specific stamps.
- **Terminal semantics.** The run ends when every terminal step completes; exit code is the worst terminal exit; services are torn down by tier.
- **Dry-run is the plan.** `srtctl dry-run` prints the serialized plan with placeholders. `srtctl plan show <job>` prints what ran. Mock swaps the executor for `FakePopen`.

## Executors

| Role or step | Kubernetes object | SLURM realization |
|---|---|---|
| engine, `size: 1` | Deployment, or a DynamoGraphDeployment service when the frontend is Dynamo | one srun step per replica, GPU slice from the ledger |
| engine, `size: n` | LeaderWorkerSet | one spanning step with `ntasks` for MPI kinds, per-node steps with leader env otherwise |
| `ray` | RayCluster, head group plus worker group | head step plus a spanning worker step, `ray_fleet` probe |
| frontend | Deployment plus Service | one step, published `http` port |
| service | Deployment or StatefulSet | one step per placement node |
| collector | DaemonSet | per-node steps |
| workload | Job, or RayJob for Miles | terminal task step |
| `init` containers | init containers | ordered pre-steps on the same nodes |
| `sidecars` | extra containers | extra `srun --overlap` steps on the same node |

The Dynamo row is the strategic one: a Dynamo-fronted recipe compiles to a DynamoGraphDeployment on Kubernetes and to srun steps on SLURM from the same YAML, because DGD's `services.<name>.{replicas, resources, extraPodSpec}` is already this shape. The docker executor is the direct-host path with the same step model.

## Miles as the first non-inference workload

Facts that shape the kinds (all from radixark/miles `main` on 2026-09-15):

- A Miles job is a Ray cluster plus one driver. The launcher starts Ray and submits `train.py` (`miles/utils/external_utils/command_utils.py:163-257`). One placement group, PACK, sorted by node IP then GPU id; the actor takes the first bundles (`miles/ray/placement_group.py:92-105`). Engines are `python -m sglang.launch_server` subprocesses (`miles/backends/sglang_utils/sglang_engine.py:68`). Miles always launches its own router (`miles/ray/rollout/rollout_server.py:27`).
- Reference SLURM launcher: `examples/experimental/openenv/glm52_tbench2/launch_16node_slurm.sh`. Per-node `srun --overlap`, head `ray start --head --node-ip-address <fabric ip>`, workers `ray start --address <head>:6379 --block` with retry, then `MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head_ip> python3 <recipe> train --num-nodes N`. `MILES_SCRIPT_*` env vars and `--extra-args` are the documented tooling seam.
- Container `radixark/miles:latest` (47.6 GB, cu13) pins branch `sgl-project/sglang:sglang-miles` (58 commits, 139 files ahead of v0.5.19), `radixark/Megatron-LM:miles-main`, and a router fork. Rollout engines must come from this image unless plain broadcast weight sync without routing replay is acceptable, which is unverified.
- `--rollout-external` raises `NotImplementedError` on main (`miles/ray/rollout/server_cell.py:151-155`). Open PRs #2513 (`--rollout-external-engine-addrs host:port ...`, topology from `/server_info`, `--rollout-external-router-pd`) and #2514 (static engine provider, e2e test with plain `sglang.launch_server`) restore it. PR #3236 adds an opaque single external endpoint with disk-delta weight publication.
- `execute_train` runs `pkill -9 sglang; pkill -9 miles; pkill -9 redis` on the driver node before anything starts (`command_utils.py:186-199`).

Kinds:

- **`ray` role kind.** Renders `ray-head` (service, pool node 0, `ray start --head --node-ip-address ${pool[0].ip} --port 6379 --num-gpus G --block`, readiness `tcp 6379` then `ray_fleet(nodes=N, gpus=N*G)` against the dashboard, critical, tier 3) and `ray-workers` (service, span over pool nodes 1..N-1, `ntasks_per_node 1`, `ray start --address ${pool[0].ip}:6379 --num-gpus G --block` in a retry loop, `after_ready: ray-head`, critical, tier 2). Both export `CUDA_VISIBLE_DEVICES` for the full node, carry the runtime mounts plus the workload's directories, and carry the workload env so raylet-spawned processes inherit it. Publishes endpoints `gcs` and `dashboard`.
- **`miles` workload kind.** Renders `miles-driver` (task, pool node 0 of `workload.on`, same image, `after_ready` on the ray steps, the attached engine steps and collectors, tier 0). Env: `MILES_SCRIPT_EXTERNAL_RAY=1`, `MASTER_ADDR=${pool.train[0].ip}`, `MILES_SCRIPT_MODEL_DIR`, `MILES_SCRIPT_DATA_DIR`, `MILES_SCRIPT_OUTPUT_DIR`. Command: `python3 <miles_root>/<recipe> <subcommand> --num-nodes N --num-gpus-per-node G --model-name X <args> --extra-args "<extra_args> <layout flags> [--rollout-external-engine-addrs ${endpoints(attach, http)} --rollout-num-gpus R --rollout-num-gpus-per-engine T]" --extra-env-vars '<env as json>'`. Terminal. Progress parser: `rollout N: {...}` lines to status events.
- **Validation in the workload kind.** Colocate requires `actor_nodes == pool.nodes`. Disaggregated requires actor GPUs plus rollout GPUs plus eval GPUs to fit the pool and rollout GPUs divisible by GPUs per engine. Store-true flags cannot be negated, so `colocate: false` is refused when the recipe hardcodes `--colocate`. `attach` requires an engine role and no frontend; a frontend is allowed with a warning since Miles ignores it. `workload.on` must not intersect `attach`, which keeps Miles's `pkill -9 sglang` away from the engines.

Compiled plan for recipe 2 and the delta for recipe 3:

```
pools   train    nodes 2, whole                                      (recipe 3: train nodes 1, rollout nodes 1 sliced 4 x 2)
steps   ray-head       service  train[0]     tcp 6379, ray_fleet(2, 16)                      critical  tier 3
        ray-workers    service  train[1:]    span, ntasks_per_node 1, after_ready ray-head   critical  tier 2
        dcgm-exporter  service  train per_node                                                          tier 1
        tachometer     service  train[0]                                                                tier 1
        miles-driver   task     train[0]     after_ready ray-head, ray-workers, tachometer              tier 0
        rollout-0..3   service  rollout[0]   gpus 2, http /server_info, publishes http       (recipe 3)  tier 2
terminal  miles-driver
```

Gotchas for the first live run: GPU visibility inside `--overlap` steps on GRES partitions, `/dev/shm` size under enroot for the Ray object store, SIGTERM to the named ray steps producing `ray stop` rather than a SIGKILL of the raylet, health timeouts long enough for Megatron checkpoint load, and `dirs.output` on a stable shared path so a resubmitted recipe resumes.

## Future jobs

| Job | Pools | Steps | Probes and windows |
|---|---|---|---|
| Dynamo disagg benchmark | infra, workers, client | etcd, one step per worker process, frontend, exporters, bench task | `dynamo_health` with expected counts, bench stamps warmup and measure windows |
| Miles, Miles-placed GPUs | train | ray head, ray workers, driver | `ray_fleet` |
| Miles attached to a fleet | train, rollout | engines, ray, driver | `server_info` per engine |
| Policy plus reward plus judge with sandboxes | one pool per model, sandbox | three frontends, sandbox service, harness task | per-frontend health |
| Router A/B on one fleet | workers, client | two routers, bench tasks chained with `after_done` | health per router |
| Build then serve | any | build task, services `after_done` the build | log probe |
| Chaos | workers | a kill task at time T, `critical: false` on the victim | liveness probe on the fleet |
| Mocker or AISim | cpu | mocker services, bench task | http |
| Power study | workers | collectors and windows | windows |
| Same recipe on Kubernetes or one host | unchanged | unchanged | executor differs |

## Deliberately out

Replicas of tasks, retries beyond a bounded `OnFailure`, artifact storage and upload, a Jinja expression language for recipes, GPU sharing between steps on the same device, shared PID namespaces between co-located steps. Recipes keep `--set` and the fixed placeholder set. Only the optional workflow compiler speaks `${{ }}`, and only for fragments that attach to a recipe by step name.

## Migration on `main` after #407

1. **Extract the plan and engine** from the POC into `src/srtctl/plan/`: `DagEngine`, `TaskState`, the probes and the ledger (`dag/engine.py:446-777`, `dag/plan.py`, `dag/probes.py`). Leave the sflow schema, loader and expressions as the optional workflow compiler.
2. **Services first.** `ServiceConfig` is already a step: placement, readiness, `critical`, `start` becomes edges. This runs the engine against production traffic with no behavior change.
3. **Golden plans.** Serialize the plan for the examples and the InferenceMAX recipes before step 4 and diff after. Plan equality is behavior equality, and it is the guard the 2.0 refactor lacked.
4. **Roles, frontend, workload.** Stage mixins become kind bodies. The health gate becomes the frontend kind's probe. The benchmark becomes the terminal task. `run()` collapses to compile then execute. Open the role map, add `kind`, `mode`, `size`, `pod`, per-role `model` and `engine`, and `frontend.type: none`. The `ResourceConfig` role fields become a shim populated only by v1 recipes and the migrator.
5. **Collectors and windows.** Tachometer and exporters as collector kinds, windows as events, postprocess reading the event log. Power telemetry unfreezes here.
6. **Workload kinds.** `ray` role kind and `miles` workload kind. Validate on sa-b200: the Miles quick start on one node in colocate, two nodes disaggregated with a fully-async script, then a two-value sweep over `extra_args`.
7. **Executors.** Put `start_srun_process` (`src/srtctl/core/slurm.py:184`) behind an executor protocol; docker and Kubernetes follow, Kubernetes emitting DGD for Dynamo-fronted recipes.

| Piece | Size |
|---|---|
| Plan IR, engine extraction, event log, async probes | medium |
| Services as steps, edges replacing phases | small |
| Golden plan harness | small |
| Open roles, pod template, kind protocols, orchestrator collapse | large |
| Collectors and windows | medium |
| `ray` and `miles` kinds, example, docs, tests | small |
| Executor protocol, docker, Kubernetes | medium, then large |

## Open decisions and defaults

- Pools are named by recipes but populated only by kinds. Default: yes.
- A multi-node engine is one spanning step for MPI-shaped kinds and per-node steps with a leader otherwise; the kind decides. Default: kind decides.
- Workload env is delivered twice on purpose for Ray: on the ray steps so spawned processes inherit it, and through `--extra-env-vars` so Ray's runtime env agrees. Default: both.
- The driver runs as its own task step, never inside the head's shell. Default: own step.
- `nodes:` stays as sugar so SLURM users never have to think in replicas for whole-node roles, and dry-run always prints the node map. Default: keep.
- Vocabulary cost: replicas, sidecars and probes on a SLURM tool. Accepted, because the same users deploy Dynamo on Kubernetes and the roadmap already wants that executor.
