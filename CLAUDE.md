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

Optional fire-and-forget HTTP status reporting to external APIs. Configure in `srtslurm.yaml`:

```yaml
# Cluster-level config (srtslurm.yaml)
cluster: "bruh"  # Cluster name for dashboard display
reporting:
  status:
    endpoint: "test-endpoint.com"
```

**StatusReporter** - Used in `do_sweep.py` to report job lifecycle:

```python
from srtctl.core.status import StatusReporter, JobStatus, JobStage

reporter = StatusReporter.from_config(config.reporting, job_id)
reporter.report_started(runtime)  # Job started with metadata
reporter.report(JobStatus.WORKERS_READY, JobStage.WORKERS, "All workers healthy")
reporter.report_completed(exit_code)  # Final status
```

**Status lifecycle:**
```
submitted → starting → head_ready → workers_starting → workers_ready
         → frontend_starting → frontend_ready → benchmark → completed | failed
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

### InfraConfig

Controls infrastructure placement (etcd/nats):

```python
infra:
  etcd_nats_dedicated_node: true  # Reserve first node for infra services
```

### Mooncake KV Store (SGLang only)

When `mooncake_kv_store` is set under an SGLang backend, srtslurm:
1. Launches `mooncake_master` on the infra node (same node as etcd/nats)
2. Injects `MOONCAKE_MASTER=<infra_ip>:50051` on all workers automatically
3. Passes through any env vars in `mooncake_kv_store.env` to all workers

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

### Adding Config That Affects srun (Mounts, Env Vars, Options)

When adding new config fields that affect what gets passed to srun (environment variables, container mounts, srun options), you must also update:

1. `show_config_details()` in `src/srtctl/cli/submit.py` -- this renders all mounts/env/options in `srtctl dry-run` output so users can verify config before submitting
2. `tests/test_dry_run.py` -- add test cases verifying the new config appears in dry-run output

Config sources that feed into dry-run display:
- **Mounts**: `config.extra_mount`, `config.container_mounts`, `default_mounts` from srtslurm.yaml
- **Env vars**: `config.environment` (global), `backend.prefill_environment`, `backend.decode_environment`, `backend.aggregated_environment`
- **srun options**: `config.srun_options`

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

