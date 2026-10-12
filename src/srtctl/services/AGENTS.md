# Services

Rules for `src/srtctl/services/` and `cli/mixins/service_stage.py`. `docs/services.md` and `docs/pools.md` are the user-facing reference. Reuse services and `placement` before adding a bespoke launcher (Design Rules in the root `CLAUDE.md`).

## Services (etcd, NATS, Mooncake master, exporters)

Everything that is not a worker or the frontend is a service (`src/srtctl/services/`, launched by `ServiceStageMixin`). The Dynamo frontend implies `etcd` and `nats`, `backend.mooncake_kv_store` implies `mooncake-master`, tachometer implies `dcgm-exporter` and `node-exporter` on every worker node (`services/implicit.py`). A recipe declares one by name only to change it:

```yaml
services:
  - name: etcd
    type: etcd
    placement:
      node: dedicated    # reserve a node for the discovery plane
  - name: nats
    type: nats
    placement:
      node: dedicated
    options:
      max_payload_mb: 24
```

`services/normalize.py` maps declared etcd/nats/mooncake-master entries back onto `infra` and `backend.mooncake_kv_store` before schema load, so the runtime reads one set of fields. Adding a kind: subclass `ServiceKind` in `services/`, `@register_service("<type>")`, import it from `services/__init__.py`; if the rest of the recipe should imply it, add it to `implied_services`. See `docs/services.md`.

## Service kinds and metrics

The top-level `services:` list declares long-running processes launched next to the job (see `docs/services.md`). Each entry has a `type` that selects a `ServiceKind` registered in `src/srtctl/services/` with `@register_service("<name>")`; the kind supplies defaults (command, start phase, criticality) and the env it injects, and `ServiceStageMixin` (`src/srtctl/cli/mixins/service_stage.py`) launches every kind the same way: resolve `placement.node` to physical nodes, optional clone/build of `source`, one `srun` per node, optional TCP `readiness` gate, `ManagedProcess` into the shared registry. `start_services("before_workers")` runs after the Mooncake master; `start_services("after_frontend")` runs after the frontend is healthy.

```yaml
services:
  - name: store
    type: mooncake-store       # generic (default) | mooncake-store
    placement:
      node: workers            # head | infra | prefill | decode | agg | workers | compute | all
                               # or pool: <owner> to ride on the nodes another service owns
    env:
      MOONCAKE_GLOBAL_SEGMENT_SIZE: 100gb
    readiness:
      port: 8800
```

Adding a kind: subclass `ServiceKind`, set `default_command` / `default_start` / `default_critical`, override `validate`, `container_fallback`, `default_environment`, `forced_environment` as needed, decorate, and import it from `src/srtctl/services/__init__.py`. `srtctl dry-run` prints every service; add a `tests/test_dry_run.py` case when a kind adds visible fields.

**Metrics.** A service that serves Prometheus metrics declares `metrics: {port, path, nodes, name}` or a list of them (the scrape annotation; `nodes: first` for a cluster head, `name` required with several endpoints); `TelemetryStageMixin._service_metrics_targets` turns every endpoint into one tachometer target per node it runs on (`ServiceMetricsTarget`, endpoint `<name>_<node>`, name defaulting to the service). Kinds that always publish return their default from `ServiceKind.metrics()` and set `metrics_filter` / `metrics_endpoint_prefix` / `metrics_gpu_metadata`; the dcgm, node and process exporters are scraped this way (`core/telemetry.py` has no exporter special case left; only workers and the frontend keep their own target logic).

## Pools and services-only jobs

A service that declares `nodes: N` owns a **pool** of N whole nodes, added to the allocation after the engine roles' nodes, in `services:` order (see `docs/pools.md`). Any number of services may own nodes, next to engine roles or without them. An owner runs one instance per node of its pool (`placement.node: workers` means its own pool); `placement.pool: <owner>` makes a **rider** that runs one instance per node of that pool. A job with only pools sets `frontend.type: none`: no frontend, no worker-count health gate, the services' readiness probes are the gate before the benchmark step.

```yaml
services:
  - name: train
    type: generic
    command: ["torchrun", "--nnodes={pool_node_count}", "--nproc-per-node={gpus_per_node}",
              "--node-rank={index}", "--master-addr={pool_ip}", "train.py"]
    nodes: 2                   # pool "train": 2 nodes; instance 0 is the rendezvous; no readiness probe
                               # (instances launch one after another, gated on readiness, and a static
                               # torchrun rendezvous only completes once every node has joined)
    terminal: true             # the job ends when every instance has exited, worst exit code; no benchmark block
  - name: watcher
    type: generic
    command: ["sleep", "infinity"]
    placement:
      pool: train              # rides on the train pool
```

Where things live:

- Carving: `Nodes.from_slurm(engine_nodes=..., pools=[(name, count), ...])` in `src/srtctl/core/runtime.py`; `Nodes.worker` is the engine nodes, `Nodes.pools` the carve, `Nodes.compute` both. Recipes without pools carve exactly as before.
- Counts: `SrtConfig.engine_node_count`, `services_node_count`, `total_nodes`, `pool_services` in `src/srtctl/core/schema.py`; rules in `_validate_services_only` and `ServiceConfig.__post_init__` (`src/srtctl/services/config.py`).
- Placement: `ServiceStageMixin.service_nodes` resolves `effective_pool`, then `placement.node` (`compute` = engine worker nodes plus every pool; `all` adds head, infra, client). The implied dcgm/node exporters run on `compute`.
- Placeholders per instance (`ServiceLaunchContext.template_vars`): `{index}`, `{node_ip}`, `{pool_node}`, `{pool_ip}`, `{pool_nodes}`, `{pool_ips}`, `{pool_node_count}`, `{gpus_per_node}`. `{head_ip}` is the job head, an engine node when a pool sits next to roles, so a cluster rendezvous uses `{pool_ip}`. Services do not inherit the recipe's top-level `environment:` (workers and the benchmark do); fabric env such as `NCCL_SOCKET_IFNAME` goes in the service's `env:`.
- Custom benchmark env (`BenchmarkStageMixin._get_service_env`): `SRT_SERVICE_<NAME>_NODES` / `_IPS` / `_NODE_COUNT` per launched service, `SRT_GPUS_PER_NODE`, `SRT_WORKER_NODES`. This is how a launcher script drives a pool.
- Terminal services (`services[].terminal: true`): the job's run. `ServiceStageMixin.terminal_processes` collects their `ManagedProcess`es; the manual loop in `BenchmarkStageMixin.run_benchmark` returns once every one has exited, with the worst exit code, instead of holding until Ctrl+C. Refused together with a non-`manual` benchmark type or on an `external` service. Without a terminal service, manual mode holds the allocation as before.
- Readers of "every node doing work" use `runtime.nodes.compute`, not `.worker`: host setup, resource snapshot, download node, exporters. New code that means all work nodes should do the same.
- Limits: whole nodes only, fixed sizes, refused with `resources.het_jobs`. `srtctl dry-run` prints the `Nodes:` map; `tests/test_pools.py` is the acceptance suite (toy recipe through the mock orchestrator on four nodes).
