# Pools: services that own nodes

A pool is a set of whole nodes that a service owns. Engine roles (`roles.prefill`, `roles.decode`, `roles.agg`) own their nodes through `nodes:`; a service owns nodes the same way, through `services[].nodes`. The allocation is the sum: the engine roles' nodes, then every pool in declaration order, plus the dedicated nodes the recipe reserves. A recipe may have any number of pools, next to engine roles or without them.

Pools are for work that is not an inference engine behind the job's frontend but still needs its own machines: a Ray cluster driving an RL trainer, a fleet of sandboxes, a torchrun job, a load generator that must not share a node with the engines it measures, or a placeholder that simply holds nodes.

## The recipe

```yaml
roles:
  agg:
    nodes: 1
    workers: 1
    gpus: 8

services:
  - name: train                  # owns two nodes: pool "train"
    type: generic
    command: ["sleep", "infinity"]
    nodes: 2
    container: "ubuntu:24.04"
  - name: napper                 # owns one node: pool "napper"
    type: generic
    command: ["sleep", "infinity"]
    nodes: 1
  - name: watcher                # rides on pool "train": one instance per node of it
    type: generic
    command: ["sleep", "infinity"]
    placement:
      pool: train

benchmark:
  type: custom
  command: "echo train pool: $SRT_SERVICE_TRAIN_NODES"
```

`srtctl dry-run` prints the node map before anything is submitted:

```
Nodes:
  engine roles: 1
  pool train (generic): 2
  pool napper (generic): 1
  total: 4
```

The full recipe is [examples/features/pools.yaml](../examples/features/pools.yaml).

## Owners and riders

A service that declares `nodes` is an **owner**. Its pool is named after it, it runs one instance per node of that pool, and its placement is `workers`, which for an owner means its own pool. Writing any other `placement.node` on an owner is refused: the nodes are the placement.

A service with `placement.pool: <owner>` is a **rider**. It runs one instance per node of the owner's pool, next to the owner's instances, in the same order. Riders are how a sidecar joins a fleet: an exporter, a log shipper, a sandbox agent, a helper that waits for the owner's port. A rider cannot itself declare `nodes`, and `placement.pool` names an owner in the same recipe or the recipe is refused.

Both share everything else a service has: `container`, `env`, `readiness`, `critical`, `start`, `source` and `build_command`, `preamble`. Readiness is checked per instance, so a pool of four is ready when all four instances passed their probe.

## How nodes are carved

`Nodes.from_slurm` splits the allocation in a fixed order:

1. Dedicated nodes the recipe reserves: `frontend.dedicated_node`, `benchmark.client_dedicated_node`, `placement.node: dedicated` on the discovery services.
2. The engine roles' nodes, as many as the roles add up to. These are the worker nodes (`Nodes.worker`).
3. Each pool, in `services:` order, taking the next `nodes` hostnames (`Nodes.pools`).

`Nodes.compute` is the worker nodes followed by every pool: every node that runs work. A recipe without pools carves exactly as before, and its `compute` set equals its worker set.

A job with no engine roles is a services-only job. It sets `frontend.type: none`, its pools are the whole allocation, and the SLURM head node (where the orchestrator runs) is the first node of the first pool. The benchmark step still runs; readiness is the services' own probes, since there is no frontend to report worker counts. `frontend.type: none` with engine roles is refused, because those roles would have no health gate.

## Placement vocabulary

| `placement.node` | Selects |
| --- | --- |
| `head`, `infra`, `dedicated` | One node. `dedicated` reserves the infra node exclusively; infra-class kinds only. |
| `prefill`, `decode`, `agg` | The distinct physical nodes that role's workers use. |
| `workers` | Every engine worker node. On an owner: its own pool. |
| `compute` | Every engine worker node plus every pool. Where the implied dcgm and node exporters run. |
| `all` | `compute` plus the head, infra and client nodes. |
| `placement.pool: <name>` | Every node of that pool. Replaces `node`. |

## What a pool instance knows

Every instance of a service gets the placeholders listed in [services.md](services.md#configuration-reference) rendered into `command`, `args`, `env` values and `preamble`. The pool placeholders describe the service's own node set:

| Placeholder | Value |
| --- | --- |
| `{index}` | This instance's position in the pool, `0..n-1`. |
| `{node}`, `{node_ip}` | This instance's node and IP. |
| `{pool_node}`, `{pool_ip}` | The first node of the pool and its IP. |
| `{pool_nodes}`, `{pool_ips}` | Every node of the pool, comma-separated, in order. |
| `{pool_node_count}` | The pool size. |
| `{gpus_per_node}` | `resources.gpus_per_node`. |

`{head_ip}` is the job head, the node the orchestrator runs on. When a pool sits next to engine roles that is an engine node, so a cluster that rendezvouses on it would point at the wrong machine. Use `{pool_ip}`.

A service's environment is its kind's defaults, the discovery variables, and its own `env:`. The recipe's top-level `environment:` goes to the engine workers and the benchmark step, not to services, so fabric settings such as `NCCL_SOCKET_IFNAME` belong in the service's `env:`.

## Forming a cluster on a pool

Instance 0 of a pool is the natural rendezvous. A generic owner can form a torchrun cluster with nothing but placeholders:

```yaml
services:
  - name: sft
    type: generic
    nodes: 4
    terminal: true              # the job ends when the run does, with its exit code
    container: "nvcr.io/nvidia/pytorch:25.06-py3"
    command:
      - torchrun
      - --nnodes={pool_node_count}
      - --nproc-per-node={gpus_per_node}
      - --node-rank={index}
      - --master-addr={pool_ip}
      - --master-port=29500
      - /workspace/train.py
    env:
      NCCL_SOCKET_IFNAME: bond0
      GLOO_SOCKET_IFNAME: bond0
```

Instances launch in pool order, and when a service has a `readiness` probe each instance must pass it before the next one starts. A torchrun rendezvous with `--master-addr` is static: rank 0 waits for every node to join, so a probe on rank 0 that fires only once the rendezvous completes deadlocks against instances that have not been launched yet. Leave `readiness` off for a self-forming cluster, or probe for something rank 0 prints before the others join. Log probes are case-sensitive regular expressions; check the exact line the program prints. A typed kind can do the cluster shape for you, rendering the head and member commands and gating on the fleet as a whole after every instance is up; the kinds table in [services.md](services.md#service-types) lists what exists.

## Ending the job with a pool

By default a service is a long-running process from the job's point of view: nothing waits for it to finish, and the job ends when the benchmark step ends, or in manual mode (no `benchmark:` block) when the job is stopped or hits its time limit. A service that finishes on its own is not a problem in itself: a clean exit is never a failure, and a non-zero exit fails the job only when the service is `critical`.

A pool that *is* the run, a training job or a test that runs to completion, sets `terminal: true`. The job then ends when every instance of every terminal service has exited, and the worst instance exit code becomes the job's exit code. Teardown follows as usual, so a sandbox fleet or an exporter riding next to the run is stopped when the run is done. Two terminal services compose: the job waits for both.

A terminal recipe has no benchmark step: `benchmark.type` stays at its default `manual`, and a recipe that combines `terminal` with a benchmark type is refused, since the job would have two ends. Use the benchmark step instead when something has to *drive* the pool from outside, a launcher that submits work to a cluster the pool runs, which is the next section.

## Driving a pool from the benchmark step

A custom benchmark command receives, for every launched service, `SRT_SERVICE_<NAME>_NODES`, `SRT_SERVICE_<NAME>_IPS` and `SRT_SERVICE_<NAME>_NODE_COUNT`, with `<NAME>` the service name upper-cased and non-alphanumerics replaced by `_`, plus `SRT_GPUS_PER_NODE` and `SRT_WORKER_NODES` (engine worker nodes, empty in a services-only job). The first IP is instance 0. This is how a launcher script submits work to a cluster the job brought up:

```bash
head_ip="${SRT_SERVICE_TRAIN_IPS%%,*}"
nodes="$SRT_SERVICE_TRAIN_NODE_COUNT"
```

See the `custom` section of [config-reference.md](config-reference.md#custom) for the full variable table.

## Telemetry and teardown

The implied dcgm and node exporters run on `compute`, so tachometer scrapes pool nodes like worker nodes and the parquet output covers the whole job. Pool services are `ManagedProcess`es with a step name like every other service: SIGTERM reaches the process inside the step on teardown, `critical: true` fails the job when an instance exits (the generic default is `false`, so a pool that may finish early needs nothing), and `shutdown_tier` orders them with the rest.

## Limits

- Pools are whole nodes. A service cannot take part of a node's GPUs next to an engine; place it on its own pool or on a role's nodes with `placement.node: <role>` and manage GPUs in its command.
- Pool sizes are fixed in the recipe. There is no `rest` or `auto`.
- `resources.het_jobs: true` (heterogeneous SLURM jobs) refuses pools; the two components already partition the allocation.
- An owner cannot be `dedicated`, `head` or `infra`, and cannot ride another pool.
- The allocation must be at least the engine nodes plus every pool. A short allocation fails at startup with `allocation has N non-reserved node(s) but the recipe needs M`.

## Checklist for a new pool recipe

1. Decide who owns nodes. Every service that needs its own machines gets `nodes:`; helpers that ride along get `placement.pool`.
2. Run `srtctl dry-run -f <recipe>` and read the `Nodes:` map. The total is what `sbatch --nodes` requests.
3. Give the owner a `readiness` probe that instance 0 passes on its own.
4. Decide how the job ends: `terminal: true` on the pool that is the run (no benchmark block), or a benchmark step that drives the pool and reads `SRT_SERVICE_<NAME>_IPS`; never hardcode a hostname.
5. Watch `outputs/<job>/logs/services/logs/service_<name>_<node>.out` for each instance (`service_<name>.out` for a one-node pool) and `sweep_<job>.log`, which logs each pool's nodes at start.
