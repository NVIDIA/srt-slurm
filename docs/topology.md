# Topology and Placement

How a recipe asks for nodes and GPUs: `roles`, `placement`, `resources`, `slurm`, and the raw `sbatch_directives` / `srun_options` escape hatches.

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

Role names are `prefill`, `decode`, and `agg`. A recipe is disaggregated (prefill and decode) or aggregated (agg only), never both. Every role a recipe launches is declared here. Use either a shared top-level [engine](engines.md#engine) or `roles.<role>.engine` on every role, never both. Mixing the two forms fails preflight even when the engine types match. `roles.<role>.container` overrides `model.container`. Arguments and environment belong to the selected role engine. Different engines still need a compatible router and KV-transfer protocol.

For mixed-engine recipes, declare both engines explicitly and omit the top-level `engine`. Missing role engines fail preflight; options on one role never become another role's defaults. `model.container` remains the shared image for non-worker tasks; each worker can select its own image.

Per-role engines do not support Dynamo, sidecars, Slurm heterogeneous jobs,
profiling, failover, implicit Mooncake stores, vLLM discovery connectors, or SGLang
gRPC workers. Shared-engine SGLang gRPC recipes are unchanged.
Multi-node workers must occupy whole nodes; multi-node TRT-LLM is unsupported.
TileRT's point-to-point Mooncake transfer is supported; see [TileRT](tilert.md).

These restrictions are checked when `SrtConfig` loads, before Slurm submission,
including dry-run and preflight. `_validate_role_backends()` rejects per-role
engines with Dynamo or sidecars. `_validate_frontend()` checks every active role
against the frontend's `required_backend`, then calls its `validate(config)` hook.
For example, vLLM prefill plus SGLang decode is rejected with Dynamo, SGLang Router,
or vLLM Router. A frontend that permits mixed engines must define its own pairing
rules; passing configuration validation does not prove KV-transfer compatibility
between the installed engine versions.

Fields: [RoleConfig](schema-reference.md#roleconfig). `args` takes any flag the engine accepts, kebab-case or snake_case (the `trtllm` engine YAML, the mocker overrides); srtctl adds the topology flags itself (`disaggregation-mode`, ports, hosts, rank arguments), and [frontend](frontends.md#frontend) lists the keys each frontend owns. `extra_args` is TRT-LLM only: `trtllm-serve` flags for the OpenAI server layer that have no engine YAML key, such as `--tool_parser`. `kv_events`, `critical`, and `restart` are covered below; see [Native Sidecars](sidecars.md) for `sidecar`.

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

### restart

`restart` turns the process monitor into a supervisor for the role: when one of its workers exits, srtctl relaunches that worker in place (same nodes, GPUs and ports) after a backoff, the way a Kubernetes `restartPolicy` brings a container back. It is off by default.

```yaml
roles:
  decode:
    workers: 4
    restart: on-failure          # relaunch a decode worker that exits non-zero
  prefill:
    workers: 2
    restart:
      policy: always             # relaunch after any exit, clean or not
      max_restarts: 5            # per worker, over the whole job (default 3)
      backoff_seconds: 10        # first delay; doubles on each relaunch (default 10)
      max_backoff_seconds: 120   # cap on the doubled delay (default 300)
```

| Key | Default | Description |
| --- | --- | --- |
| `policy` | `never` | `never` leaves a worker exit to [critical](#critical). `on-failure` relaunches after a non-zero exit. `always` relaunches after any exit. |
| `max_restarts` | `3` | Relaunches allowed per worker over the life of the job. |
| `backoff_seconds` | `10` | Delay before the first relaunch. Doubles on each further relaunch of the same worker. |
| `max_backoff_seconds` | `300` | Cap on the doubled delay. |

How a relaunch behaves:

- The unit of restart is the logical worker. When one rank of a multi-node worker exits, the surviving ranks are stopped with SIGTERM and the whole worker comes back together. TRT-LLM endpoints already die as one step.
- The relaunched step is named `<mode>_<index>_<node>_r<n>` and writes the canonical worker log; the previous life's log is kept beside it as `<node>_<mode>_w<index>.out.<n>`.
- Under the Dynamo frontend, srtctl then polls the worker's `DYN_SYSTEM_PORT` `/health` (the engine's HTTP port under the other frontends) for as long as the initial health gate allows and records when the worker serves again. A worker that never answers is left running.
- Once `max_restarts` is spent, or when `on-failure` sees a clean exit, the worker goes back to the ordinary monitor and the role's `critical` flag decides whether the run fails.
- Every relaunch is recorded in `logs/worker_restarts.json` and copied into `recipe.lock.yaml` as `lock.worker_restarts`, so a result produced through restarts says so.

A relaunched worker registers with the Dynamo frontend as a new instance once the old lease expires; a static router sees the same URL come back. Requests in flight on the dead worker fail and the benchmark client records them; that is the point of a fault-tolerance probe, and the lockfile entry marks the run. The supervisor does not move a worker to another node: the allocation has no spare, and Slurm ends the job when a node fails.

`examples/features/worker-restart.yaml` is a runnable version: two workers serving with no benchmark, so you can SIGKILL a worker step with `scancel --signal=KILL <job>.<step>` and watch the relaunch in the sweep log. Its header walks through the timeline of a real run.

The v1 spelling of this section (`resources.prefill_nodes`, `resources.prefill_workers`, `resources.gpus_per_prefill`, `resources.prefill_critical`, `resources.decode_nodes: 0`, `backend.prefill_environment`, `backend.sglang_config.prefill`, `backend.prefill_extra_args`, `backend.kv_events_config`, and the `decode` and `aggregated` counterparts) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

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

The discovery plane (etcd, and NATS when a plane uses it) is placed through its services: an `etcd` or `nats` entry under [`services`](services.md) with `placement.node: dedicated`. See [Implicit Services](services.md#implicit-services).

The v1 spelling of this (`frontend.orchestrator_placement`, `frontend.dedicated_node`, `benchmark.client_placement`, `benchmark.client_dedicated_node`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

## resources

Cluster facts about the GPUs the job runs on. The worker topology (nodes, workers, GPUs per worker) lives under [roles](#roles).

```yaml
resources:
  gpu_type: "gb200"
  gpus_per_node: 4          # GPUs per node (default: from srtslurm.yaml)
  spread_workers: false     # one partial-node worker per node instead of packing
  het_jobs: null            # SLURM heterogeneous job for prefill and decode; null: cluster default
```

Fields: [ResourceConfig](schema-reference.md#resourceconfig). `het_jobs: null` defers to the cluster's `use_het_jobs`; see [slurm-faq.md](slurm-faq.md).

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

## slurm

SLURM job settings.

```yaml
slurm:
  time_limit: "04:00:00"    # Job time limit
  account: "my-account"     # SLURM account (overrides srtslurm.yaml)
  partition: "batch"        # SLURM partition (overrides srtslurm.yaml)
```

Fields: [SlurmConfig](schema-reference.md#slurmconfig). Unset values come from `default_account`, `default_partition`, and `default_time_limit` in `srtslurm.yaml`.

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

## srun_options

Additional srun options for job steps.

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

`srun_options` applies to every srun step the job launches (workers, frontends, benchmark, telemetry). Set `roles.<role>.srun_options` to override individual keys for that role's inference worker steps. Other keys are inherited from the recipe's `srun_options`; non-worker steps keep the recipe options.

```yaml
roles:
  prefill:
    # ... node, GPU, and engine settings ...
    srun_options:
      mem: "64G"                     # Host memory per node in each prefill step
  decode:
    # ... node, GPU, and engine settings ...
    srun_options:
      mem: "32G"                     # Host memory per node in each decode step
```
