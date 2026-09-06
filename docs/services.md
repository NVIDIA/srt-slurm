# Services

The top-level `services:` block declares long-running processes that srtctl launches and tracks next
to the inference workers, the frontend, and the benchmark client. One list, one shape, for anything
that is not a built-in component: an experimental router built from an unmerged PR, a standalone
Mooncake Store per worker node, a debugging HTTP server. Adding one is a recipe change, not a code
change.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Placement](#placement)
- [Start Order and Readiness](#start-order-and-readiness)
- [Environment](#environment)
- [Building From Source](#building-from-source)
- [Service Types](#service-types)
- [Example: a router from a PR](#example-a-router-from-a-pr)
- [Example: standalone Mooncake stores](#example-standalone-mooncake-stores)
- [Validation](#validation)
- [Limitations](#limitations)

## Quick Start

```yaml
services:
  - name: my-sidecar
    command:
      - python3
      - -m
      - my_package.my_sidecar
      - --port
      - "9000"
    readiness:
      port: 9000
```

`name` and `command` are the only required fields for the default `generic` type. The service runs in
the job container on the head node, starts once workers and the frontend are healthy, and the job
waits until port 9000 answers before moving on. Its log is `service_my-sidecar.out` in the job's log
directory. `examples/features/services.yaml` is a runnable version of this.

## Configuration Reference

```yaml
services:
  - name: my-sidecar             # required, unique across the list
    type: generic                # generic (default) | mooncake-store
    command:                     # argv, not shell-interpreted; required for generic
      - python3
      - -m
      - pkg
    args:                        # appended to command
      - --flag
    container: my-image          # image or srtslurm.yaml alias; default: job container
    env:                         # environment for the service process
      MY_FLAG: "1"
    placement:
      node: head                 # head | infra | prefill | decode | agg | workers
    start: after_frontend        # after_frontend | before_workers
    readiness:                   # optional TCP gate, checked on every service node
      port: 9000
      timeout_seconds: 120
    inherit_discovery_env: true  # inject ETCD_ENDPOINTS / NATS_SERVER
    critical: false              # a crash fails the run when true
    preamble: |                  # shell run before command, inside the container
      ulimit -n 1048576
    cpus_per_task: 8             # srun --cpus-per-task
    cpu_bind: none               # srun --cpu-bind
    srun_options:                # extra srun options for this service only
      exclusive: ""
    source:                      # clone before build/launch; single-node placements only
      git: https://github.com/org/repo
      rev: <commit-sha, tag, or refs/pull/N/head>
      path: subdir
    build_command:               # run once from the clone, inside the container
      - bash
      - -lc
      - pip install -e .
```

| Field | Default | Notes |
| --- | --- | --- |
| `name` | required | Unique. Names `service_<name>.out` and the tracked process. |
| `type` | `generic` | Selects a [service type](#service-types) that supplies defaults and environment. |
| `command` | type default | Argv passed directly to the process. `generic` has no default, so it is required there. |
| `args` | `[]` | Appended to `command`. Handy with typed services that supply the command. |
| `container` | type fallback, then job container | Aliases resolve through `srtslurm.yaml` like every other container key. |
| `env` | `{}` | Merged over the type's defaults; see [Environment](#environment). |
| `placement.node` | `head` | See [Placement](#placement). |
| `start` | type default | `generic`: `after_frontend`. `mooncake-store`: `before_workers`. |
| `readiness` | none | TCP port gate per node. Timing out terminates what this stage started and fails the job. |
| `inherit_discovery_env` | `true` | Inject the same `ETCD_ENDPOINTS` / `NATS_SERVER` the Dynamo frontend gets. |
| `critical` | type default | `generic`: `false`. `mooncake-store`: `true`. |
| `preamble` | none | Shell run after the environment is exported and before `command`. |
| `cpus_per_task`, `cpu_bind`, `srun_options` | none | Pass-through srun knobs for this service's launches. |
| `source`, `build_command` | none | See [Building From Source](#building-from-source). |

`command`, `args`, `env` values, and `preamble` may use these placeholders: `{node}`, `{node_ip}`,
`{node_id}` (position in the worker list), `{index}` (instance index within the service), `{role}`
(the `placement.node` value), `{head_node}`, `{head_ip}`, `{infra_node}`, `{infra_ip}`,
`{master_port}`, `{metadata_port}`. Only those names are substituted; other braces (JSON in an env
value) are left alone.

## Placement

`placement.node` picks the physical nodes. `head` and `infra` launch one instance. `prefill`,
`decode`, and `agg` launch one instance per distinct node that role's workers use, so two TP1 decode
workers on one node share one service. `workers` launches one instance per worker node.

When a service launches on more than one node its processes and logs get a node suffix:
`service_<name>_<node>`. Two services that declare the same `readiness.port` and land on the same
node are rejected before anything launches; give them disjoint placements or ports.

## Start Order and Readiness

Services launch in declaration order within a start phase:

- `before_workers`: after etcd/NATS and the Mooncake master, before any worker. For things workers
  connect to at startup.
- `after_frontend`: once workers and the frontend are healthy, before telemetry. For sidecars that
  register into a running job.

Within a phase, a service with `readiness` blocks until its port answers on each of its nodes; a
service without one is considered started when its `srun` is launched. Ongoing health is the shared
`ProcessRegistry` monitor, the same as every other process in the job. It tears the run down only for
`critical: true` services. Anything other components register under or route through should be
critical: a router that dies mid-run otherwise leaves the frontend silently talking to the raw
backend and the benchmark measuring something other than what it claims.

## Environment

The service process environment is built in layers, later ones winning:

1. Discovery env, when `inherit_discovery_env` is true: `ETCD_ENDPOINTS=http://<infra>:2379`,
   `NATS_SERVER=nats://<infra>:4222`.
2. The type's defaults (`mooncake-store` sets `MOONCAKE_LOCAL_HOSTNAME` to the node's IP).
3. The recipe's `env`, with placeholders substituted.
4. Values srtctl owns for the type (`mooncake-store`: `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`).
   A recipe value for these is ignored.

## Building From Source

`source` plus `build_command` clone and build once before launch. The clone runs on the bare host of
the service node (git and network access are host concerns, and the job container may lack git) into
`<log_dir>/services/<name>/src`, which every container sees at `/logs/services/<name>/src`.
`build_command` and `command` then run inside the service container from that directory (or
`source.path` under it).

`rev` must be immutable: a commit SHA, a tag, or `refs/pull/<n>/head` while iterating on an open
PR. `main`, `master`, and `HEAD` are rejected at load time. Because the build installs into one
container instance, `source` is only allowed with single-node placements (`head`, `infra`).

## Service Types

`type` selects a registered `ServiceKind` (`src/srtctl/services/`). A kind supplies defaults and the
environment its process needs; the launch path is shared by every kind. Register a new one with
`@register_service("<name>")`, the same pattern as `@register_benchmark`.

| Type | Default command | Start | Critical | Notes |
| --- | --- | --- | --- | --- |
| `generic` | none (required) | `after_frontend` | `false` | Launches exactly what you wrote. |
| `mooncake-store` | `python -m mooncake.mooncake_store_service` | `before_workers` | `true` | Requires `backend.mooncake_kv_store`. Container falls back to `mooncake_kv_store.container`. Injects the managed master's address. |

## Example: a router from a PR

```yaml
frontend:
  type: dynamo

services:
  - name: thunderagent-router
    source:
      git: https://github.com/ai-dynamo/dynamo
      rev: refs/pull/14000/head      # switch to a commit SHA once merged
    build_command:
      - bash
      - -lc
      - "cd lib/bindings/python && maturin develop --uv && cd ../../.. && pip install -e ."
    command:
      - python3
      - -m
      - dynamo.thunderagent_router
      - --endpoint
      - dyn://namespace.component.endpoint
      - --model-name
      - my-model
    # The router is the thing under test: other components register under its
    # endpoint, so running without it must fail the run, not degrade it.
    critical: true
```

`inherit_discovery_env` defaults to true, so the router sees the same etcd/NATS as the Dynamo
frontend and workers with no extra configuration.

## Example: standalone Mooncake stores

Inference workers run embedded Mooncake clients with `MOONCAKE_GLOBAL_SEGMENT_SIZE=0` while dedicated
per-node stores own the DRAM segments. Decode nodes contribute host memory without an in-process
HiCache pool. One entry per role gives each role its own segment size:

```yaml
backend:
  type: sglang
  prefill_environment:
    MOONCAKE_PROTOCOL: rdma
    MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
    MOONCAKE_GLOBAL_SEGMENT_SIZE: "0"
  decode_environment:
    MOONCAKE_PROTOCOL: rdma
    MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
    MOONCAKE_GLOBAL_SEGMENT_SIZE: "0"
  mooncake_kv_store:
    container: mooncake        # the master; also the stores' default container
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
    decode:
      disaggregation-transfer-backend: mooncake

services:
  - name: store-prefill
    type: mooncake-store
    placement:
      node: prefill
    args:
      - --port
      - "8800"
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
      MOONCAKE_GLOBAL_SEGMENT_SIZE: 100gb
    preamble: |
      ulimit -n 1048576
      ulimit -l unlimited
    cpus_per_task: 8
    readiness:
      port: 8800
  - name: store-decode
    type: mooncake-store
    placement:
      node: decode
    args:
      - --port
      - "8800"
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
      MOONCAKE_GLOBAL_SEGMENT_SIZE: 400gb
    preamble: |
      ulimit -n 1048576
      ulimit -l unlimited
    cpus_per_task: 8
    readiness:
      port: 8800
```

Stores start after the master is healthy and before workers. Each store gets `MOONCAKE_MASTER`,
`MOONCAKE_TE_META_DATA_SERVER`, and `MOONCAKE_LOCAL_HOSTNAME` from the runtime. If prefill and decode
share a node, the two entries above collide on port 8800 and the job fails before launching; use one
entry with `placement.node: workers` and a single segment size instead. See
[Mooncake KV Store](mooncake-kv-store.md) for the worker side.

## Validation

Rejected at load time, so `srtctl dry-run` catches them:

- Empty or duplicate `name`; unknown `type`.
- `generic` without `command`; a `command`/`args` entry that is blank; `build_command: []`.
- `placement.node` or `start` outside their vocabularies.
- `source` with a moving `rev`, or with a multi-node placement.
- `type: mooncake-store` without `backend.mooncake_kv_store`.

Rejected at launch, before any service starts: two services listening on the same port on one node.

`srtctl dry-run` prints every service's type, placement, start phase, criticality, command,
container, source, readiness, and env.

## Limitations

- Declared order is launch order, and `readiness` is the only wait. A service that needs another
  service to be ready polls for it in its own `command`.
- `source` builds are single-node only.
- Services run on the sbatch/SLURM path only; there is no local dev mode in 2.0.
