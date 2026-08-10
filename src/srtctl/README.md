# srtctl - Python-first SLURM Orchestration

This package provides Python-first orchestration for LLM inference benchmarks
on SLURM clusters, replacing the previous Jinja/bash-heavy approach.

## Architecture

```
srtctl/
├── __init__.py              # Package exports
├── cli/
│   ├── submit.py            # srtctl apply - job submission
│   ├── do_sweep.py          # srtctl-sweep - main orchestrator
│   └── setup_head.py        # Head node infrastructure (NATS/etcd)
├── core/
│   ├── config.py            # Config loading and srtslurm.yaml resolution
│   ├── runtime.py           # RuntimeContext - single source of truth
│   ├── topology.py          # Endpoint/Process allocation for workers
│   ├── processes.py         # ProcessRegistry - lifecycle management
│   ├── slurm.py             # SLURM srun launching and node resolution
│   ├── health.py            # Health checks (HTTP polling, worker readiness)
│   ├── schema.py            # Frozen dataclass schemas
│   ├── sweep.py             # Sweep parameter handling
│   └── ip_utils/            # Bash-based IP resolution utilities
│       ├── __init__.py      # Python wrappers for bash functions
│       └── get_node_ip.sh   # IP detection bash functions
├── backends/
│   ├── base.py              # BackendProtocol interface
│   └── sglang.py            # SGLang implementation
├── benchmarks/
│   ├── base.py              # BenchmarkRunner ABC
│   ├── sa_bench.py          # Serving benchmark
│   ├── router.py            # Router benchmark
│   └── ...                  # Other benchmark types
└── templates/               # Jinja2 templates for sbatch scripts
```

## Usage

```bash
srtctl apply -f config.yaml
```

## Key Concepts

### RuntimeContext

Single source of truth for all computed paths and values. Replaces bash
variables scattered throughout Jinja templates.

```python
runtime = RuntimeContext.from_config(config, job_id)
print(runtime.log_dir)  # Computed once
print(runtime.model_path)  # Resolved from config
print(runtime.head_node_ip)  # From SLURM
```

### Endpoints and Processes

Typed Python replaces bash array math:

```python
# Old (Jinja/bash):
# for i in $(seq 0 $((PREFILL_WORKERS - 1))); do
#     leader_idx=$((WORKER_NODE_OFFSET + i * PREFILL_NODES_PER_WORKER))
# done

# New (Python):
endpoints = allocate_endpoints(
    num_prefill=2,
    num_decode=4,
    num_agg=0,
    gpus_per_prefill=8,
    gpus_per_decode=4,
    gpus_per_agg=8,
    gpus_per_node=8,
    available_nodes=nodes,
)
for endpoint in endpoints:
    print(f"{endpoint.mode} worker {endpoint.index} on {endpoint.nodes}")
```

### ProcessRegistry

Manages process lifecycle with health monitoring:

```python
registry = ProcessRegistry(job_id)
registry.add_process(worker_proc)

# Background thread monitors for failures
if registry.check_failures():
    registry.cleanup()  # Graceful shutdown
```

### Health Checks

HTTP-based health checking for different frontends:

```python
from srtctl.core.health import wait_for_model

# Wait for all workers to register
wait_for_model(
    host=head_ip,
    port=8000,
    n_prefill=2,
    n_decode=4,
    frontend_type="sglang",  # or "dynamo"
    timeout=300,
)
```

For aggregated mode, pass `n_prefill=0, n_decode=num_agg`.

### BackendProtocol

Interface for different serving frameworks:

```python
class BackendProtocol(Protocol):
    @property
    def type(self) -> BackendType: ...
    def build_worker_command(self, process, runtime) -> list[str]: ...
```

### Multiple Workers Per Node

The allocator automatically handles placing multiple workers on a single node:

```yaml
resources:
  gpus_per_node: 8
  decode_workers: 2
  gpus_per_decode: 4 # 2 workers × 4 GPUs = 8 GPUs = 1 node
```

`CUDA_VISIBLE_DEVICES` is automatically set per worker (e.g., `0,1,2,3` and `4,5,6,7`).

## Files Overview

| File                 | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `core/config.py`     | YAML loading, srtslurm.yaml resolution   |
| `core/runtime.py`    | Computed paths/values (RuntimeContext)   |
| `core/topology.py`   | Worker topology and GPU allocation       |
| `core/processes.py`  | Process lifecycle management             |
| `core/slurm.py`      | SLURM srun launching, node IP resolution |
| `core/health.py`     | Health checks, worker readiness polling  |
| `core/ip_utils/`     | Bash-based IP detection utilities        |
| `cli/do_sweep.py`    | Main orchestrator (runs on head node)    |
| `backends/sglang.py` | SGLang backend implementation            |
| `benchmarks/base.py` | BenchmarkRunner ABC                      |
