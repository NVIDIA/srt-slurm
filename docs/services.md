# Services

The top-level `services:` block declares long-running processes that srtctl launches and tracks next
to the inference workers, the frontend, and the benchmark client. One list, one shape, for everything
that is not a worker or the frontend: the discovery plane (etcd, NATS), the Mooncake master, the
metrics exporters tachometer scrapes, an experimental router built from an unmerged PR, a standalone
Mooncake Store per worker node, a debugging HTTP server. The built-in ones are implied by the rest of
the recipe and need no entry; declaring one by name takes it over. Adding a new one is a recipe
change, not a code change.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Implicit Services](#implicit-services)
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
    type: generic                # generic (default) | etcd | nats | mooncake-master | dcgm-exporter | node-exporter | mooncake-store
    enabled: true                # false drops the service (the way to switch an implied one off)
    external: null               # typed kinds only: use an instance that already runs at this address
    command:                     # argv, not shell-interpreted; required for generic
      - python3
      - -m
      - pkg
    args:                        # appended to command
      - --flag
    container: my-image          # image or srtslurm.yaml alias; default: job container
    env:                         # environment for the service process
      MY_FLAG: "1"
    options:                     # kind-specific knobs (nats: max_payload_mb; exporters: port, collect_interval_ms)
      max_payload_mb: 24
    placement:
      node: head                 # head | infra | dedicated | prefill | decode | agg | workers
    start: after_frontend        # infra | before_workers | after_frontend
    readiness:                   # optional probe, checked on every service node; typed kinds have default ports
      port: 9000                 # or tcp: {port} / http: {port, path, status} / log: {pattern}
      timeout_seconds: 120
      interval_seconds: 2
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
| `type` | `generic` | Selects a [service type](#service-types) that supplies defaults, and for the typed kinds the command. |
| `enabled` | `true` | `false` drops the service. Declaring an implied name with `enabled: false` switches it off. |
| `external` | none | `etcd`, `nats`, `mooncake-master` only: an address of an already-running instance. Nothing launches; the address is injected where the job's own would have been. |
| `options` | `{}` | Kind-specific knobs; unknown keys are rejected. `nats`: `max_payload_mb`. `dcgm-exporter`, `node-exporter`: `port`, `collect_interval_ms`. `mooncake-master`: `store_config` (vLLM). |
| `command` | type default | Argv passed directly to the process. `generic` has no default, so it is required there. |
| `args` | `[]` | Appended to `command`. Handy with typed services that supply the command. |
| `container` | type fallback, then job container | Aliases resolve through `srtslurm.yaml` like every other container key. |
| `env` | `{}` | Merged over the type's defaults; see [Environment](#environment). |
| `placement.node` | type default | `generic`: `head`. See [Placement](#placement). |
| `start` | type default | `etcd`, `nats`: `infra`. `mooncake-master`, `mooncake-store`: `before_workers`. `generic` and the exporters: `after_frontend`. |
| `readiness` | type default | One probe per node: `port` / `tcp`, `http`, or `log`, plus `timeout_seconds` and `interval_seconds`. The typed kinds gate on their well-known ports when no probe is written. See [Start Order and Readiness](#start-order-and-readiness). Timing out terminates what this stage started and fails the job. |
| `inherit_discovery_env` | `true` | Inject the same `ETCD_ENDPOINTS` / `NATS_SERVER` the Dynamo frontend gets. |
| `critical` | type default | `generic`: `false`. `mooncake-store`: `true`. |
| `preamble` | none | Shell run after the environment is exported and before `command`. |
| `cpus_per_task`, `cpu_bind`, `srun_options` | none | Pass-through srun knobs for this service's launches. |
| `source`, `build_command` | none | See [Building From Source](#building-from-source). |
| `build_timeout_seconds` | `1800` | `build_command` is killed when this runs out, so a hung build cannot hold the allocation. |

`command`, `args`, `env` values, and `preamble` may use these placeholders: `{node}`, `{node_ip}`,
`{node_id}` (position in the worker list), `{index}` (instance index within the service), `{role}`
(the `placement.node` value), `{head_node}`, `{head_ip}`, `{infra_node}`, `{infra_ip}`,
`{master_port}`, `{metadata_port}`. Only those names are substituted; other braces (JSON in an env
value) are left alone.

## Implicit Services

Three things the recipe asks for elsewhere are services the job runs without an entry:

| Implied by | Services | Where |
| --- | --- | --- |
| `frontend.type: dynamo` | `etcd`, `nats` | the infra node, phase `infra` |
| `backend.mooncake_kv_store` (v1) or a declared `mooncake-master` | `mooncake-master` | the infra node, phase `before_workers` |
| tachometer on (the default; `observability.tachometer.enabled`) | `dcgm-exporter`, `node-exporter` | every worker node, phase `after_frontend` |

`srtctl dry-run` lists them next to the declared ones, marked `implied by:`. A declared entry with
the same `name` replaces the implied one, so the recipe only says what differs:

```yaml
services:
  - name: etcd
    type: etcd
    placement:
      node: dedicated             # reserve a node for the discovery plane
  - name: nats
    type: nats
    placement:
      node: dedicated             # etcd and nats share the infra node: both or neither
    options:
      max_payload_mb: 24
  - name: dcgm-exporter
    type: dcgm-exporter
    container: mirror/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04   # air-gapped cluster
  - name: node-exporter
    type: node-exporter
    enabled: false                # drop it
```

`external` uses an instance that already runs: nothing launches, and the workers and frontend get its
address in `ETCD_ENDPOINTS` / `NATS_SERVER` (or `MOONCAKE_MASTER`) instead of the infra node's.

```yaml
services:
  - name: etcd
    type: etcd
    external: http://etcd.shared.example:2379
```

The power-telemetry path (`telemetry.enabled`) launches and owns its own DCGM exporter; the implied
`dcgm-exporter` steps aside when it is on.

The v1 spellings still load and mean the same thing: `infra.etcd_nats_dedicated_node` and
`infra.nats_max_payload_mb` are the etcd/nats entries above, `backend.mooncake_kv_store` is a
`mooncake-master` entry plus Mooncake env on the roles. `srtctl migrate` rewrites them.

## Placement

`placement.node` picks the physical nodes. `head`, `infra`, and `dedicated` launch one instance;
`dedicated` reserves a node for the infra services (the job asks Slurm for one more node) and is
accepted by `etcd`, `nats`, and `mooncake-master`. `prefill`, `decode`, and `agg` launch one instance
per distinct node that role's workers use, so two TP1 decode workers on one node share one service.
`workers` launches one instance per worker node.

When a service launches on more than one node its processes and logs get a node suffix:
`service_<name>_<node>`. Two services that declare the same `readiness.port` and land on the same
node are rejected before anything launches; give them disjoint placements or ports.

## Start Order and Readiness

Services launch in three phases; within a phase, implied services first, then declared ones in
declaration order:

- `infra`: the discovery plane (etcd, NATS). Nothing else should need this phase.
- `before_workers`: after the discovery plane, before any worker. The Mooncake master, standalone
  stores, anything workers connect to at startup.
- `after_frontend`: once workers and the frontend are healthy, before the scraper. The exporters,
  sidecars that register into a running job.

Within a phase, a service with `readiness` blocks until its probe passes on each of its nodes. The
typed kinds gate on their well-known ports by default (etcd 2379, NATS 4222, the Mooncake master
8700, 8701, and 8702, the exporters none); a `generic` service without a probe is considered started
when its `srun` is launched. Three probes are available, and a `readiness` block names exactly one:

```yaml
readiness:
  port: 9000                 # shorthand for tcp
  timeout_seconds: 120       # per node; the job fails when it runs out
  interval_seconds: 2        # between attempts
```

```yaml
readiness:
  tcp:
    port: 9000               # a TCP connection is accepted
```

```yaml
readiness:
  http:
    port: 8000
    path: /ready             # GET http://<node>:8000/ready
    status: 200              # returns this status
```

```yaml
readiness:
  log:
    pattern: 'Uvicorn running on .*:\d+'   # a regex matched against service_<name>.out
```

The probe is re-run every `interval_seconds` until it passes. Between attempts the stage checks that
the process is still alive, so a service that crashes fails the job at once with its exit code rather
than after the full timeout. Ongoing health is the shared
`ProcessRegistry` monitor, the same as every other process in the job. It tears the run down only for
`critical: true` services. Anything other components register under or route through should be
critical: a router that dies mid-run otherwise leaves the frontend silently talking to the raw
backend and the benchmark measuring something other than what it claims.

## Environment

The service process environment is built in layers, later ones winning:

1. Discovery env, when `inherit_discovery_env` is true: `ETCD_ENDPOINTS=http://<infra>:2379`,
   `NATS_SERVER=nats://<infra>:4222` (or the `external` addresses). etcd and NATS themselves never get
   it.
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

`source` is the same shape `dynamo.source` uses. At submit time `srtctl apply` resolves a non-commit
`rev` with `git ls-remote` and records the commit as `source.sha` in the submitted `config.yaml`
(the recipe on disk is untouched), so the job checks out exactly the commit the lockfile names even
if the PR is pushed to again while the job waits in the queue. `--json` output lists what was pinned
under `pinned_sources`. `srtctl dry-run` never touches the network.

## Service Types

`type` selects a registered `ServiceKind` (`src/srtctl/services/`). A kind supplies defaults and the
environment its process needs; the launch path is shared by every kind. Register a new one with
`@register_service("<name>")`, the same pattern as `@register_benchmark`.

| Type | Default command | Start | Critical | Notes |
| --- | --- | --- | --- | --- |
| `generic` | none (required) | `after_frontend` | `false` | Launches exactly what you wrote. |
| `etcd` | `/configs/etcd` from the job container, advertising the node's IP | `infra` | `true` | Implied by the Dynamo frontend. Placement `head`, `infra`, or `dedicated`; supports `external`. Fresh data dir on node-local `/tmp` each job. |
| `nats` | `/configs/nats-server -js` from the job container | `infra` | `true` | Implied by the Dynamo frontend. `options.max_payload_mb` writes a server config. Same placements as etcd; supports `external`. |
| `mooncake-master` | `mooncake_master` with the RPC, HTTP metadata, and metrics ports srtctl owns | `before_workers` | `true` | Implied by `backend.mooncake_kv_store`; declaring it is the 2.0 spelling. `args` are appended. Container falls back to `mooncake_kv_store.container`. Supports `dedicated` and `external`. |
| `dcgm-exporter` | `dcgm-exporter --collect-interval=<ms> --address :9401` in `nvcr.io/nvidia/k8s/dcgm-exporter` | `after_frontend` | `false` | Implied on worker nodes while tachometer runs. Shell-less (distroless image). `options`: `port`, `collect_interval_ms`. |
| `node-exporter` | `/bin/node_exporter` with the cpu, infiniband, and meminfo collectors on 9101 in `quay.io/prometheus/node-exporter` | `after_frontend` | `false` | Implied on worker nodes while tachometer runs. Shell-less. `options`: `port`. |
| `mooncake-store` | `python -m mooncake.mooncake_store_service` | `before_workers` | `true` | Requires a Mooncake master (either spelling). Container falls back to the master's. Injects the master's address. |

The bespoke launch paths these replace (`start_head_infrastructure` with its own readiness loop, a
Mooncake-master stage, exporter launches inside the tachometer stage) are gone; every one of these is
a `ManagedProcess` from the same stage, with the same registry, cleanup, and dry-run output.

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
- `placement.node` or `start` outside their vocabularies; `dedicated` or `external` on a kind that does
  not support them; an `options` key the kind does not know.
- `source` with a moving `rev`, or with a multi-node placement.
- `type: mooncake-store` without a Mooncake master; two `mooncake-master` entries; a `mooncake-master`
  entry next to `backend.mooncake_kv_store`.
- `etcd` and `nats` disagreeing on `dedicated` (they share the infra node).

Rejected at launch, before any service starts: two services listening on the same port on one node.

`srtctl dry-run` prints every service's type, placement, start phase, criticality, command,
container, source, readiness, and env.

## Cleanup

Nothing a service launches outlives the job:

- Every long-running service is a named Slurm step (`service_<name>`), so cleanup delivers SIGTERM
  through `scancel --signal=TERM` and the process gets 30 seconds to flush (etcd its WAL, a scraper
  its parquet) before the step is killed. Signalling the `srun` client directly would have killed
  the task outright.
- Every `srun` the stage starts, including the one-shot clone and build steps, is registered with the
  job's `ProcessRegistry` the moment it exists, not when the stage returns. The registry's cleanup
  runs on normal completion, on any failed stage, from the SIGTERM handler (`scancel`), and from the
  crash monitor when a critical process dies, and it terminates then kills each tracked `srun`. Slurm
  cancels the step, which kills the whole step cgroup inside the container, so forked or daemonized
  children of the service go with it.
- A readiness gate that fails, or a signal that arrives during one, terminates everything the stage
  already launched before the error propagates.
- A service whose process exits before its readiness port answers fails immediately with its exit
  code instead of waiting out the readiness timeout.
- The clone step is bounded by the `timeout` on each git command (600s each) and the build step by
  `build_timeout_seconds`; a step that overruns is killed and the job fails with a pointer to its log.
- When the batch script exits, Slurm releases the allocation and reaps any remaining step, so even a
  cleanup path srtctl never reaches cannot leave a service running on a compute node.

## Limitations

- Declared order is launch order, and `readiness` is the only wait. A service that needs another
  service to be ready polls for it in its own `command`.
- `source` builds are single-node only.
- Services run on the sbatch/SLURM path only; there is no local dev mode in 2.0.
