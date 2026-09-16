# CLAUDE.md

Development guide for working on this codebase.

## Quick Reference

```bash
# Run lint + tests (recommended)
make check

# Just lint
make lint

# Just tests
make test

# Run single test file
uv run pytest tests/test_e2e.py -v

# Run single test
uv run pytest tests/test_e2e.py::TestH100Cluster::test_endpoint_allocation -v

# Auto-fix lint issues
uv run ruff check --fix src/srtctl/
uv run ruff format src/srtctl/
```

## Code Style

- **Python 3.10+** - use modern syntax (`|` unions, `match` statements)
- **Ruff** for linting and formatting (config in `pyproject.toml`)
- **Type hints** everywhere - use `ty` for type checking
- **Frozen dataclasses** for configs (`@dataclass(frozen=True)`)
- **Line length**: 120 characters

## Python Patterns

Follow these patterns when extending the codebase:

- **Frozen dataclasses for config** - Use `@dataclass(frozen=True)` for all configuration objects. Immutability prevents accidental mutation and makes code easier to reason about.
- **Protocol over ABC** - Prefer `typing.Protocol` for interface definitions (see `BackendProtocol`). Enables duck typing without inheritance coupling.
- **marshmallow_dataclass for validation** - Combine dataclasses with marshmallow schemas for type-safe config loading with validation. Custom fields (e.g., `BackendConfigField`) handle polymorphic deserialization.
- **Factory classmethods** - Use `@classmethod` named `from_*` for construction (e.g., `RuntimeContext.from_config()`, `RunMetadata.from_json()`). Keep `__init__` simple.
- **TYPE_CHECKING guard** - Import type-only dependencies under `if TYPE_CHECKING:` to avoid circular imports. Use string annotations for forward refs.
- **Computed properties** - Use `@property` for derived values instead of storing computed state. See `ResourceConfig.gpus_per_prefill`, `RunMetadata.topology_label`.
- **Registry pattern** - Use decorators for extensible registration (`@register_benchmark("sa-bench")`). New implementations just decorate and import.
- **TypedDict for external data** - Use `TypedDict` for typing dicts from JSON/external sources where you can't control the structure.
- **Single source of truth** - Create context objects (like `RuntimeContext`) that compute all derived paths/values once at startup rather than recomputing.
- **testing** - when we make a new significant feature change, we should always add a new test

## Key Concepts

### RuntimeContext

Single source of truth for computed paths. Created once at job start:

```python
runtime = RuntimeContext.from_config(config, job_id)
runtime.log_dir          # /path/to/logs/12345_1P_4D_...
runtime.head_node_ip     # 10.0.0.1
runtime.container_mounts # List of mount strings
```

### Endpoint Allocation

Maps logical workers to physical nodes/GPUs:

```python
endpoints = allocate_endpoints(
    num_prefill=2, num_decode=4, num_agg=0,
    gpus_per_prefill=8, gpus_per_decode=4, gpus_per_agg=0,
    gpus_per_node=8,
    available_nodes=("node0", "node1", "node2"),
)
# Returns List[Endpoint] with node assignments and GPU indices
```

### Health Checks

Two patterns for checking worker readiness:

```python
# Dynamo backend
check_dynamo_health(response_json, expected_prefill=2, expected_decode=4)

# SGLang router
check_sglang_router_health(response_json, expected_prefill=2, expected_decode=4)
```

For aggregated mode, pass `expected_prefill=0, expected_decode=num_agg`.

### Status Reporting

Optional fire-and-forget HTTP status reporting to one or more collectors. Configure in `srtslurm.yaml`:

```yaml
# Cluster-level config (srtslurm.yaml)
cluster: "bruh"  # Cluster name for dashboard display
reporting:
  status:
    endpoint: "http://login-node:8080"
```

**srtctl status-server** is the in-repo collector for that endpoint (`src/srtctl/status_server/`: `store.py` is the SQLite side, `server.py` the stdlib HTTP side validating with `srtctl.contract`, `ui/index.html` the dependency-free single-page UI served at `/`, which fetches the same `/api` routes with the read token from `localStorage`). It appends an event whenever `(status, stage, message)` changes, creates a placeholder row for a PUT whose POST never arrived, and serves cursor-based feeds at `/api/events` and `/api/jobs/{id}/events`. `make_server(store, port=0)` gives tests a real server on an ephemeral port (`tests/test_status_server.py` drives it with the real `StatusReporter`). The wire contract is `docs/status-api-spec.md`; a payload field changes in `srtctl.contract`, the server, and the spec together.

**StatusReporter** - Used in `do_sweep.py` to report job lifecycle:

```python
from srtctl.core.status import StatusReporter, JobStatus, JobStage

reporter = StatusReporter.from_config(config.reporting, job_id)
reporter.report_started(config, runtime)  # Job started, with model/resources/head_node/log_dir metadata
reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
reporter.report_completed(exit_code, logs_url=s3_url)  # Final status
```

**Status lifecycle** (status is the stage being entered, not readiness):
```
submitted → starting → workers → frontend → benchmark → completed | failed
stages: starting, head_infrastructure, preflight, workers, frontend, benchmark, cleanup
```

**create_job_record()** - Standalone function for job submission:

```python
from srtctl.core.status import create_job_record

# Called in submit.py after sbatch succeeds
create_job_record(
    reporting=config.reporting,
    job_id=job_id,
    job_name=config.name,
    cluster=get_srtslurm_setting("cluster"),
    recipe=str(config_path),
    metadata=metadata,  # Tags go in metadata["tags"]
)
```

**Key behaviors:**
- All HTTP requests have 5-second timeout
- Failures are logged at DEBUG and silently ignored
- Job execution is never blocked by status reporting
- Tags are passed via `metadata["tags"]` (not a separate field)
- `metadata["log_dir"]` (from `report_started`) is the run's log directory on the cluster filesystem; `logs_url` is only set when `reporting.s3` uploads it
- Auth is a bearer token read from `$SRTCTL_STATUS_TOKEN` on both sides (`reporting.status.token_env` renames the variable). Never add a literal token field: `SrtConfig.Schema().dump` lands in the lockfile and resolved configs are copied into `logs/` and synced to S3. The reporter never follows redirects and warns on 3xx/401/403; the server refuses to listen beyond loopback without a token unless `--allow-unauthenticated`

### Services (etcd, NATS, Mooncake master, exporters)

Everything that is not a worker or the frontend is a service (`src/srtctl/services/`, launched by `ServiceStageMixin`). The Dynamo frontend implies `etcd` and `nats`, `backend.mooncake_kv_store` implies `mooncake-master`, tachometer implies `dcgm-exporter` and `node-exporter` on every worker node (`services/implicit.py`). A recipe declares one by name only to change it:

```yaml
services:
  - name: etcd
    type: etcd
    placement:
      node: dedicated    # reserve a node for the discovery plane (v1: infra.etcd_nats_dedicated_node)
  - name: nats
    type: nats
    placement:
      node: dedicated
    options:
      max_payload_mb: 24 # v1: infra.nats_max_payload_mb
```

`services/normalize.py` maps declared etcd/nats/mooncake-master entries back onto `infra` and `backend.mooncake_kv_store` before schema load, so the runtime reads one set of fields. Adding a kind: subclass `ServiceKind` in `services/`, `@register_service("<type>")`, import it from `services/__init__.py`; if the rest of the recipe should imply it, add it to `implied_services`. See `docs/services.md`.

### Mooncake KV Store

When `mooncake_kv_store` is set under an SGLang or vLLM backend, srtslurm:
1. Launches `mooncake_master` on the infra node (same node as etcd/nats)
2. Injects `MOONCAKE_MASTER=<infra_ip>:8700` on all workers automatically
3. Passes through any env vars in `mooncake_kv_store.env` to all workers
4. For vLLM, also renders `mooncake_kv_store.store_config` into the JSON file
   pointed to by `MOONCAKE_CONFIG_PATH` (vLLM's `MooncakeStoreConnector` reads
   its config from JSON, not env vars). See `docs/mooncake-kv-store.md`.

```yaml
backend:
  type: sglang
  mooncake_kv_store:
    container: nvcr.io/nvidia/mooncake:latest  # optional, defaults to job container
    env:                                        # direct MOONCAKE_* / SGLANG_* env vars
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: mlx5_0
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake  # user still sets this
      disaggregation-ib-device: "mlx5_0,mlx5_1"
    decode:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
```

`MOONCAKE_MASTER` is always computed from the runtime infra node IP — do not set it manually in `env`.

`MOONCAKE_LOCAL_HOSTNAME` is auto-resolved per-worker to that worker's own IP (using `runtime.network_interface`), so multi-node peer transfers don't fall back to `localhost`. If you need a specific NIC IP, set `MOONCAKE_LOCAL_HOSTNAME` in `env` to override the default.

**Validation:** In disaggregated mode, srtslurm rejects configs that set `mooncake_kv_store` without `disaggregation-transfer-backend: mooncake` on `sglang_config.prefill` or `sglang_config.decode`. This catches the common misconfiguration where the master process gets launched but workers fall back to default transport.

### Services

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

### Pools and services-only jobs

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

### Process cleanup and graceful shutdown

Every long-running `srun` (workers, frontends, nginx, services, tachometer) is launched with a `step_name` and tracked as a `ManagedProcess` carrying the same name. `ProcessRegistry.cleanup()` stops processes by `shutdown_tier`: tier 0 (workers, frontends, sidecars) is SIGTERMed all at once through `scancel --signal=TERM --full <job>.<step>`, waited for up to each process's `terminate_timeout`, and killed if still up; then tier 1 (Mooncake master, stores), then tier 2 (etcd, NATS). SIGTERM aimed at the `srun` client itself aborts the step and SIGKILLs the task, which is why the step name matters: it is the only way the engine, router, or scraper sees the signal and gets to deregister, drain, or flush. New launch sites must pass `step_name` to both `start_srun_process` and `ManagedProcess`.

### Host Setup

`host_setup` runs commands on each node's **bare host, outside the container**, before any
worker starts — the counterpart to `setup_script`, which runs *inside* the container. Use it
for node state a container cannot reach (GPU clocks, kernel modules).

```yaml
host_setup:
  commands: ["sudo -n nvidia-smi -lmc <min>,<max>"]
  teardown: ["sudo -n nvidia-smi -rmc"]   # runs on the cleanup path, success or failure
  nodes: all                              # all | workers
```

Implemented in `SweepOrchestrator._run_host_setup()` / `._run_host_teardown()` as one
container-less `start_srun_process(container_image=None, ...)` per node. Cluster-wide default
lives in `srtslurm.yaml` as `default_host_setup` (whole-block replace, like
`default_health_check`). Commands run as the submitting user, so privileged ones need
passwordless sudo. Always pair a `commands` entry that sets persistent state with a
`teardown` — otherwise it leaks to the next job on that node.

### ResourceConfig

Supports explicit GPUs per worker (overrides computed values):

```python
resources:
  gpu_type: "gb200"
  prefill_nodes: 2
  prefill_workers: 4
  decode_nodes: 4
  decode_workers: 8
  gpus_per_prefill: 4  # Optional: explicit override
  gpus_per_decode: 2   # Optional: explicit override
```

## Testing

Tests are located in `tests/`. Run `make check` to run lint + all tests.

### Mocking SLURM

```python
class H100Rack:
    NUM_NODES = 13
    GPUS_PER_NODE = 8

    @classmethod
    def slurm_env(cls):
        return {
            "SLURM_JOB_ID": "12345",
            "SLURM_NODELIST": "h100-[01-13]",
            ...
        }

with patch.dict(os.environ, H100Rack.slurm_env()):
    with patch("subprocess.run", H100Rack.mock_scontrol()):
        # Test code here
```

## Common Tasks

### Adding a New Backend

1. Create `backends/mybackend.py` with a dataclass implementing `BackendProtocol`
2. Implement required methods:
   - `get_srun_config()` - MPI settings and launch strategy
   - `get_config_for_mode(mode)` - Mode-specific configuration
   - `get_environment_for_mode(mode)` - Environment variables
   - `allocate_endpoints()` - Logical worker allocation
   - `endpoints_to_processes()` - Physical process mapping
   - `build_worker_command(process, runtime)` - Command construction
3. Export from `backends/__init__.py`
4. Add polymorphic deserialization in `BackendConfigField` in `schema.py`

**Current backends:**
- **SGLang**: Per-process srun launching, supports prefill/decode/aggregated modes
- **TRTLLM**: MPI-style launching (one srun per endpoint with all nodes), prefill/decode only

### Adding a New Benchmark

1. Create `benchmarks/mybench.py` inheriting from `BenchmarkRunner`
2. Implement `run(config, log_dir)` method
3. Add bash script to `benchmarks/scripts/mybench/bench.sh`
4. Register in benchmark type mapping

### Adding or Changing Any Config Field

`docs/schema-reference.md` is generated from the dataclasses in `core/schema.py` and `backends/`. After adding, renaming, or re-typing a field, run `uv run srtctl schema-docs` and commit the result; CI and `tests/test_schema_docs.py` fail when the file is stale. Put the field's description in the class docstring `Attributes:` block or in a `#` comment directly above the field so it lands in the generated table.

### Adding Config That Affects srun (Mounts, Env Vars, Options)

When adding new config fields that affect what gets passed to srun (environment variables, container mounts, srun options), you must also update:

1. `show_config_details()` in `src/srtctl/cli/submit.py` -- this renders all mounts/env/options in `srtctl dry-run` output so users can verify config before submitting
2. `tests/test_dry_run.py` -- add test cases verifying the new config appears in dry-run output

Config sources that feed into dry-run display:
- **Mounts**: `config.extra_mount`, `config.container_mounts`, `default_mounts` from srtslurm.yaml
- **Env vars**: `config.environment` (global), `backend.prefill_environment`, `backend.decode_environment`, `backend.aggregated_environment`
- **srun options**: `config.srun_options`
- **Host setup**: `config.host_setup`, `default_host_setup` from srtslurm.yaml

## Debugging

### Check Generated Commands

`srtctl dry-run` shows the sbatch script, all container mounts (with source labels), environment variables (global and per-mode), and srun options:

```bash
srtctl dry-run -f config.yaml
```

### Find Full srun Commands at Runtime

The full srun command (with all mounts, env vars, and flags) is logged at INFO level in the sweep log:

```bash
tail -f outputs/<job_id>/logs/sweep_<job_id>.log | grep "srun command"
```

Per-worker env vars and commands are also logged individually (search for `Env:` and `Command:` lines).

