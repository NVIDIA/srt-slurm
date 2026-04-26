# srtctl Architecture Documentation

**Version**: 1.1
**Last Updated**: 2026-01-27

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Design Philosophy](#design-philosophy)
3. [System Components](#system-components)
4. [Architecture Layers](#architecture-layers)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Process Architecture on SLURM Cluster](#process-architecture-on-slurm-cluster)
7. [Key Abstractions](#key-abstractions)
8. [Extension Points](#extension-points)
9. [Module Dependencies](#module-dependencies)
10. [Directory Structure](#directory-structure)

---

## High-Level Overview

### What is srtctl?

srtctl (SLURM Runtime Control) is a Python-first orchestration framework for LLM inference benchmarks on SLURM clusters. It provides:

- **Configuration-driven deployment**: YAML configs define model, resources, backends, and benchmarks
- **Multi-backend support**: SGLang and TRTLLM with prefill/decode disaggregation
- **Automated orchestration**: Handles infrastructure setup, worker spawning, health checks, and benchmarking
- **Container-based execution**: Workers run inside containers with proper mounts and environment

### Problem Statement

Running distributed LLM inference workloads on SLURM clusters involves significant complexity:

1. **Resource Allocation**: Mapping GPU workers to nodes with proper tensor parallelism
2. **Process Coordination**: Starting services in the correct order with health checks
3. **Configuration Management**: Handling model paths, container images, and environment variables
4. **Monitoring & Cleanup**: Tracking process health and graceful shutdown

srtctl abstracts this complexity into a simple YAML interface while providing extensibility for different backends, frontends, and benchmarks.

### Architecture Overview

```
+------------------------------------------------------------------+
|                         USER INTERFACE                            |
|  srtctl apply -f config.yaml    |    srtctl dry-run -f config.yaml|
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                        CLI LAYER                                  |
|   submit.py (job submission)  |  interactive.py (TUI)            |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    CONFIGURATION LAYER                            |
|   schema.py (frozen dataclasses)  |  config.py (YAML loading)    |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                   ORCHESTRATION LAYER                             |
|         SweepOrchestrator + Stage Mixins (Worker/Frontend/Bench)  |
+------------------------------------------------------------------+
                |               |                |
                v               v                v
+---------------+  +---------------+  +------------------+
|  BACKEND      |  |   FRONTEND    |  |   BENCHMARK      |
|  BackendProto |  |   FrontendProto|  |   BenchmarkRunner|
|  (SGLang)     |  |   (Dynamo/SGL) |  |   (SA-Bench/MMLU)|
+---------------+  +---------------+  +------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    INFRASTRUCTURE LAYER                           |
|   SLURM (srun)  |  Containers (Enroot)  |  NATS/etcd             |
+------------------------------------------------------------------+
```

---

## Design Philosophy

### 1. Single Source of Truth

The `RuntimeContext` computes all paths and values **once** at startup. This eliminates:

- Scattered bash variable expansion
- Inconsistent path computation
- Configuration drift during execution

```python
# All runtime values computed in one place
runtime = RuntimeContext.from_config(config, job_id)
runtime.log_dir          # /path/to/logs/12345/logs
runtime.head_node_ip     # 10.0.0.1
runtime.container_mounts # Dict[Path, Path]
```

### 2. Frozen Dataclasses

All configuration objects are **immutable** after creation using `@dataclass(frozen=True)`:

```python
@dataclass(frozen=True)
class SrtConfig:
    name: str
    model: ModelConfig
    resources: ResourceConfig
    # ... all fields immutable
```

Benefits:

- Prevents accidental mutation
- Easier to reason about state
- Safe to pass around without defensive copying
- Thread-safe by default

### 3. Protocol Pattern (Duck Typing)

Using `typing.Protocol` instead of ABC for interfaces enables duck typing without inheritance:

```python
class BackendProtocol(Protocol):
    def build_worker_command(...) -> list[str]: ...

# SGLangProtocol is a frozen dataclass that implements this
# No inheritance required - just implement the methods
@dataclass(frozen=True)
class SGLangProtocol:
    def build_worker_command(...) -> list[str]:
        # implementation
```

### 4. Registry Pattern

Extensible component registration via decorators:

```python
@register_benchmark("sa-bench")
class SABenchRunner(BenchmarkRunner):
    ...

# Later: get_runner("sa-bench") returns instance
runner = get_runner("sa-bench")
```

### 5. Factory Classmethods

Use `@classmethod` named `from_*` for construction:

```python
RuntimeContext.from_config(config, job_id)
Nodes.from_slurm(benchmark_on_separate_node)
SrtConfig.from_yaml(yaml_path)
```

---

## System Components

### CLI Layer

```
src/srtctl/cli/
|-- __init__.py
|-- submit.py        # Main entry point: srtctl apply/dry-run
|-- do_sweep.py      # SweepOrchestrator - runs inside SLURM job
|-- setup_head.py    # Head node infrastructure (NATS, etcd)
|-- interactive.py   # TUI for job management
|-- mixins/
    |-- worker_stage.py    # Backend worker startup
    |-- frontend_stage.py  # Frontend/nginx startup
    |-- benchmark_stage.py # Benchmark execution
```

#### submit.py - Job Submission

Entry point for `srtctl apply|dry-run -f config.yaml`:

```
User runs: srtctl apply -f config.yaml
                |
                v
    +---------------------------+
    | 1. Parse CLI arguments    |
    | 2. load_config(path)      |
    | 3. Generate sbatch script |
    | 4. Submit via sbatch      |
    +---------------------------+
                |
                v
    +---------------------------+
    | SLURM allocates nodes     |
    | Runs sbatch script        |
    +---------------------------+
```

#### do_sweep.py - SweepOrchestrator

The main orchestration class that runs inside the SLURM job:

```python
@dataclass
class SweepOrchestrator(WorkerStageMixin, FrontendStageMixin, BenchmarkStageMixin):
    config: SrtConfig
    runtime: RuntimeContext

    def run(self) -> int:
        """Run the complete benchmark sweep."""
        # Stage 1: Start head infrastructure (NATS, etcd)
        # Stage 2: Start backend workers
        # Stage 3: Start frontends
        # Stage 4: Run benchmark
        # Cleanup
```

### Configuration Layer

```
src/srtctl/core/
|-- schema.py       # Frozen dataclass definitions
|-- config.py       # YAML loading with cluster defaults
|-- formatting.py   # FormattablePath/String wrappers
|-- runtime.py      # RuntimeContext - single source of truth
```

#### schema.py - Configuration Dataclasses

All configs are **frozen dataclasses** with marshmallow validation:

| Class             | Purpose             | Key Fields                                            |
| ----------------- | ------------------- | ----------------------------------------------------- |
| `SrtConfig`       | Main job config     | name, model, resources, backend, frontend, benchmark  |
| `ModelConfig`     | Model settings      | path, container, precision                            |
| `ResourceConfig`  | GPU/node allocation | gpu_type, gpus_per_node, prefill/decode nodes/workers |
| `BackendConfig`   | Polymorphic backend | type, sglang_config, environment per mode             |
| `FrontendConfig`  | Router settings     | type, enable_multiple_frontends, args, env            |
| `BenchmarkConfig` | Benchmark params    | type, isl, osl, concurrencies, sweep                  |
| `ProfilingConfig` | Profiling settings  | type (nsys/torch), phase configs                      |

#### runtime.py - RuntimeContext

The **single source of truth** for all runtime values:

```python
@dataclass(frozen=True)
class RuntimeContext:
    job_id: str
    run_name: str
    nodes: Nodes
    head_node_ip: str
    log_dir: Path
    model_path: Path
    container_image: Path
    gpus_per_node: int
    network_interface: str | None
    container_mounts: dict[Path, Path]
    srun_options: dict[str, str]
    environment: dict[str, str]
    frontend_port: int = 8000

    @classmethod
    def from_config(cls, config: SrtConfig, job_id: str) -> RuntimeContext:
        """All path computation happens here, once at startup."""
```

### Backend Layer

```
src/srtctl/backends/
|-- __init__.py     # Exports BackendConfig, protocols
|-- base.py         # BackendProtocol definition
|-- sglang.py       # SGLangProtocol implementation
|-- trtllm.py       # TRTLLMProtocol implementation
```

#### BackendProtocol

```python
class BackendProtocol(Protocol):
    @property
    def type(self) -> str: ...

    def get_config_for_mode(self, mode: str) -> dict[str, Any]: ...
    def get_environment_for_mode(self, mode: str) -> dict[str, str]: ...

    def allocate_endpoints(...) -> list[Endpoint]: ...
    def endpoints_to_processes(...) -> list[Process]: ...

    def build_worker_command(
        self, process, endpoint_processes, runtime,
        frontend_type, nsys_prefix, dump_config_path
    ) -> list[str]: ...
```

#### SGLangProtocol

Implements BackendProtocol for SGLang with P/D disaggregation:

```python
@dataclass(frozen=True)
class SGLangProtocol:
    type: Literal["sglang"] = "sglang"

    # Per-mode environment
    prefill_environment: dict[str, str]
    decode_environment: dict[str, str]
    aggregated_environment: dict[str, str]

    # SGLang CLI config per mode
    sglang_config: SGLangServerConfig | None = None

    # KV events config
    kv_events_config: bool | dict[str, Any] | None = None
```

**Launch strategy**: Per-process srun launching (one srun per worker process).

#### TRTLLMProtocol

Implements BackendProtocol for TRTLLM with MPI-style launching:

```python
@dataclass(frozen=True)
class TRTLLMProtocol:
    type: Literal["trtllm"] = "trtllm"

    # Per-mode environment
    prefill_environment: dict[str, str]
    decode_environment: dict[str, str]

    # TRTLLM CLI config per mode
    trtllm_config: TRTLLMServerConfig | None = None
```

**Launch strategy**: MPI-style launching (one srun per endpoint with all nodes together). Uses `trtllm-llmapi-launch` for distributed launching.

**Key differences from SGLang**:
- No aggregated mode support
- Uses UUID-based EPLB shared memory naming (`TRTLLM_EPLB_SHM_NAME`)
- MPI launch with `--mpi=pmix` and `--cpu-bind=verbose,none`
- Configuration written to YAML file and passed via `--extra-engine-args`

### Frontend Layer

```
src/srtctl/frontends/
|-- __init__.py     # Exports FrontendProtocol
|-- base.py         # FrontendProtocol definition
|-- dynamo.py       # DynamoFrontend (NATS/etcd)
|-- sglang.py       # SGLangFrontend (direct router)
```

#### FrontendProtocol

```python
class FrontendProtocol(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def health_endpoint(self) -> str: ...

    def parse_health(response_json, expected_prefill, expected_decode) -> WorkerHealthResult: ...

    def start_frontends(
        topology, runtime, config, backend, backend_processes
    ) -> list[ManagedProcess]: ...
```

### Infrastructure Layer

```
src/srtctl/core/
|-- slurm.py        # SLURM utilities (srun, nodelist parsing)
|-- processes.py    # Process lifecycle management
|-- health.py       # HTTP health checks
|-- topology.py     # Endpoint/Process allocation
```

---

## Architecture Layers

### Layer Diagram

```
+------------------------------------------------------------------+
|                          CLI LAYER                                |
+------------------------------------------------------------------+
| submit.py       | do_sweep.py    | interactive.py | setup_head.py|
|                 |                |                |               |
| - Parse args    | - Orchestrate  | - TUI mode     | - Start NATS |
| - Load config   | - Stage mixins | - Job browser  | - Start etcd |
| - Submit sbatch | - Run stages   |                |               |
+------------------------------------------------------------------+
                                |
                                | SrtConfig, RuntimeContext
                                v
+------------------------------------------------------------------+
|                      CONFIGURATION LAYER                          |
+------------------------------------------------------------------+
| schema.py                    | config.py        | runtime.py     |
|                              |                  |                 |
| - SrtConfig (frozen)         | - load_config()  | - RuntimeContext|
| - ModelConfig (frozen)       | - YAML parsing   | - from_config() |
| - ResourceConfig (frozen)    | - Cluster defaults| - Path compute |
| - BackendConfig (polymorphic)| - Validation     |                 |
+------------------------------------------------------------------+
                                |
                                | Endpoints, Processes
                                v
+------------------------------------------------------------------+
|                     ORCHESTRATION LAYER                           |
+------------------------------------------------------------------+
| SweepOrchestrator                                                 |
|   +-- WorkerStageMixin   (start_worker, start_all_workers)       |
|   +-- FrontendStageMixin (start_nginx, start_frontend)           |
|   +-- BenchmarkStageMixin (run_benchmark)                         |
|                                                                   |
| ProcessRegistry        | ManagedProcess     | Signal Handlers    |
| - add_process()        | - name, popen      | - SIGTERM/SIGINT   |
| - check_failures()     | - log_file, node   | - Graceful cleanup |
| - cleanup()            | - terminate()      |                    |
+------------------------------------------------------------------+
                                |
                                | Commands, Health Checks
                                v
+------------------------------------------------------------------+
|                        BACKEND LAYER                              |
+------------------------------------------------------------------+
| BackendProtocol                    | Implementations:             |
| - get_srun_config()                | - SGLangProtocol             |
| - allocate_endpoints()             |   (per-process srun)         |
| - endpoints_to_processes()         | - TRTLLMProtocol             |
| - build_worker_command()           |   (MPI-style srun)           |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                       FRONTEND LAYER                              |
+------------------------------------------------------------------+
| FrontendProtocol                   | DynamoFrontend | SGLangFrontend|
| - start_frontends()                | - /health      | - /workers    |
| - parse_health()                   | - NATS/etcd    | - Direct conn |
| - health_endpoint                  |                |               |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                     INFRASTRUCTURE LAYER                          |
+------------------------------------------------------------------+
| slurm.py                 | processes.py       | health.py        |
| - start_srun_process()   | - ManagedProcess   | - wait_for_port()|
| - get_slurm_nodelist()   | - ProcessRegistry  | - wait_for_model()|
| - get_hostname_ip()      | - Signal handlers  | - Health parsers |
+------------------------------------------------------------------+
```

---

## Data Flow Diagrams

### Config Loading Flow

```
+------------+     +-------------+     +--------------+
| YAML Config| --> | load_config | --> | SrtConfig    |
+------------+     +-------------+     | (frozen DC)  |
                                       +--------------+
                                              |
                                              v
+------------+     +------------------+     +----------------+
| SLURM Env  | --> | RuntimeContext   | <-- | SrtConfig      |
| (job_id,   |     | .from_config()   |     |                |
| nodelist)  |     +------------------+     +----------------+
+------------+              |
                            v
               +-----------------------+
               | RuntimeContext        |
               | - job_id, run_name    |
               | - nodes (head, worker)|
               | - log_dir, model_path |
               | - container_mounts    |
               +-----------------------+
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
   +-------------+   +-------------+   +-------------+
   | allocate_   |   | Backend.    |   | Frontend.   |
   | endpoints() |   | build_cmd() |   | start()     |
   +-------------+   +-------------+   +-------------+
          |                 |                 |
          v                 v                 v
   +-------------+   +-------------+   +-------------+
   | Endpoints   |   | Worker      |   | Router      |
   | + Processes |   | Processes   |   | Processes   |
   +-------------+   +-------------+   +-------------+
```

### Job Submission Flow

```
User runs: srtctl apply -f config.yaml
                |
                v
+-----------------------------+
| cli/submit.py::main()       |
| 1. Parse CLI args           |
| 2. load_config(path)        |
| 3. submit_with_orchestrator |
+-----------------------------+
                |
                v
+-----------------------------+
| submit_with_orchestrator()  |
| 1. generate_sbatch_script() |
| 2. Write to temp file       |
| 3. sbatch script_path       |
| 4. Copy config to outputs/  |
+-----------------------------+
                |
                v
+-----------------------------+
| SLURM allocates nodes       |
| Runs sbatch script          |
+-----------------------------+
                |
                v
+-----------------------------+
| job_script_minimal.j2       |
| 1. mkdir output dirs        |
| 2. pip install srtctl       |
| 3. python -m srtctl.cli.do_sweep |
+-----------------------------+
                |
                v
+-----------------------------+
| cli/do_sweep.py::main()     |
| 1. load_config()            |
| 2. get_slurm_job_id()       |
| 3. RuntimeContext.from_config() |
| 4. SweepOrchestrator(config, runtime) |
| 5. orchestrator.run()       |
+-----------------------------+
```

### Worker Startup Flow

```
SweepOrchestrator.start_all_workers()
                |
                v
+------------------------------------+
| For each Process in backend_processes: |
|   1. Get endpoint_processes        |
|   2. Build bash preamble           |
|      - Custom setup script         |
|      - Dynamo installation         |
|   3. Build worker command          |
|      - backend.build_worker_command()|
|   4. Set environment variables     |
|      - HEAD_NODE_IP                |
|      - ETCD_ENDPOINTS              |
|      - NATS_SERVER                 |
|      - DYN_SYSTEM_PORT             |
|      - CUDA_VISIBLE_DEVICES        |
|   5. start_srun_process()          |
|   6. Create ManagedProcess         |
+------------------------------------+
                |
                v
+------------------------------------+
| start_srun_process()               |
|   1. Build srun command            |
|      --overlap                     |
|      --nodes, --ntasks             |
|      --nodelist                    |
|      --output                      |
|      --container-image             |
|      --container-mounts            |
|   2. Wrap in bash -c               |
|      - Export env vars             |
|      - Run preamble                |
|      - Execute main command        |
|   3. subprocess.Popen()            |
+------------------------------------+
```

### Health Check Flow

```
SweepOrchestrator.run_benchmark()
         |
         v
wait_for_model(host, port, n_prefill, n_decode, frontend_type)
         |
         +-----> GET http://host:port/{health_endpoint}
         |              |
         |       +------+------+
         |       |             |
         |       v             v
         |   /health       /workers
         |   (dynamo)      (sglang)
         |       |             |
         |       v             v
         |  check_dynamo   check_sglang
         |  _health()      _router_health()
         |       |             |
         |       v             v
         |   WorkerHealthResult
         |   - ready: bool
         |   - prefill_ready vs expected
         |   - decode_ready vs expected
         |
         +<--- Loop until ready or timeout
```

---

## Process Architecture on SLURM Cluster

### Physical Layout

```
+------------------------------------------------------------------+
|                        SLURM JOB ALLOCATION                       |
+------------------------------------------------------------------+
|                                                                    |
|  HEAD NODE (node0)                                                 |
|  +------------------------------------------------------------+   |
|  | sbatch script (HOST)                                        |   |
|  |   -> python -m srtctl.cli.do_sweep (orchestrator)          |   |
|  +------------------------------------------------------------+   |
|  | srun container: setup_head.py                               |   |
|  |   -> NATS server (:4222)                                    |   |
|  |   -> etcd server (:2379)                                    |   |
|  +------------------------------------------------------------+   |
|  | srun container: nginx (if multiple frontends)               |   |
|  |   -> Load balancer (:8000)                                  |   |
|  +------------------------------------------------------------+   |
|  | srun container: frontend_0                                  |   |
|  |   -> dynamo.frontend or sglang_router (:8080)              |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  WORKER NODE (node1) - Prefill                                    |
|  +------------------------------------------------------------+   |
|  | srun container: prefill_0                                   |   |
|  |   -> dynamo.sglang or sglang.launch_server                 |   |
|  |   -> GPUs 0-7 (TP=8)                                       |   |
|  |   -> HTTP port 30000                                        |   |
|  |   -> Bootstrap port 31000                                   |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  WORKER NODE (node2) - Decode                                     |
|  +------------------------------------------------------------+   |
|  | srun container: decode_0 (GPUs 0-3)                         |   |
|  |   -> HTTP port 30000                                        |   |
|  +------------------------------------------------------------+   |
|  | srun container: decode_1 (GPUs 4-7)                         |   |
|  |   -> HTTP port 30001                                        |   |
|  +------------------------------------------------------------+   |
|                                                                    |
+------------------------------------------------------------------+
```

### Port Allocation Strategy

```
+------------------+------------+----------------------------------+
| Port Type        | Range      | Description                      |
+------------------+------------+----------------------------------+
| HTTP ports       | 30000+     | Per-node, incremental            |
| Bootstrap ports  | 31000+     | Per-node, prefill only           |
| KV events ports  | 5550+      | Global, incremental              |
| System ports     | 8081+      | Per-process, incremental         |
| Frontend public  | 8000       | Public-facing (nginx or direct)  |
| Frontend internal| 8080       | Behind nginx                     |
| NATS             | 4222       | Message broker                   |
| etcd             | 2379       | Key-value store                  |
+------------------+------------+----------------------------------+
```

### Process Relationships

```
                    +------------------+
                    |  ORCHESTRATOR    |
                    | (do_sweep.py)    |
                    | runs on HEAD     |
                    +--------+---------+
                             |
           +-----------------+------------------+
           |                 |                  |
           v                 v                  v
+----------+----+   +--------+-------+   +------+--------+
| HEAD INFRA    |   | WORKERS        |   | FRONTENDS     |
| - NATS        |   | - prefill_0..N |   | - router_0..N |
| - etcd        |   | - decode_0..N  |   | - nginx (opt) |
+---------------+   | - agg_0..N     |   +---------------+
                    +----------------+
                             |
                             | NATS pub/sub
                             | etcd registration
                             v
                    +----------------+
                    | FRONTENDS      |
                    | discover workers|
                    | via NATS/etcd  |
                    +----------------+
```

---

## Key Abstractions

### RuntimeContext

The **single source of truth** for all runtime values. Created once at job start:

```python
@dataclass(frozen=True)
class RuntimeContext:
    # Runtime identifiers
    job_id: str
    run_name: str

    # Node topology
    nodes: Nodes          # head, bench, worker tuple
    head_node_ip: str

    # Computed paths (all absolute)
    log_dir: Path
    model_path: Path
    container_image: Path

    # Resource configuration
    gpus_per_node: int
    network_interface: str | None

    # Container mounts: host_path -> container_path
    container_mounts: dict[Path, Path]

    @classmethod
    def from_config(cls, config: SrtConfig, job_id: str) -> RuntimeContext:
        """All path computation happens here, once at startup."""
```

### Endpoint vs Process

```
+---------------+       +---------------+
|   ENDPOINT    |       |    PROCESS    |
+---------------+       +---------------+
| Logical unit  |       | Physical unit |
| May span nodes|  -->  | Runs on 1 node|
| Has mode/index|       | Has ports     |
+---------------+       +---------------+

Example: TP=16 endpoint on 8-GPU nodes
+-----------------------------------+
| Endpoint (prefill, index=0)       |
| - nodes: (node1, node2)           |
| - gpu_indices: {0..7}             |
+-----------------------------------+
        |
        +---> Process (node1, rank=0, leader)
        |        - http_port: 30000
        |        - bootstrap_port: 31000
        |
        +---> Process (node2, rank=1, follower)
                 - http_port: 0 (not exposed)
```

### NodePortAllocator

Manages per-node port assignments to avoid conflicts:

```python
@dataclass
class NodePortAllocator:
    base_http_port: int = 30000
    base_bootstrap_port: int = 31000
    base_kv_events_port: int = 5550

    def next_http_port(self, node: str) -> int:
        """Get next available HTTP port for a node."""

    def next_bootstrap_port(self, node: str) -> int:
        """Get next available bootstrap port for a node."""

    def next_kv_events_port(self) -> int:
        """Get next available kv-events port (globally unique)."""
```

### ProcessRegistry

Lifecycle management for all spawned processes:

```python
class ProcessRegistry:
    def add_process(self, process: ManagedProcess) -> None:
        """Add a process to the registry."""

    def check_failures(self) -> bool:
        """Check if any critical process has failed."""

    def cleanup(self) -> None:
        """Terminate all registered processes."""

    def print_failure_details(self, tail_lines: int = 50) -> None:
        """Print detailed failure info with log tails."""
```

### ManagedProcess

```python
@dataclass
class ManagedProcess:
    name: str                    # e.g., "prefill_0", "decode_1"
    popen: subprocess.Popen
    log_file: Path | None
    node: str | None
    critical: bool = True        # Failure triggers cleanup

    @property
    def is_running(self) -> bool: ...

    def terminate(self, timeout: float = 10.0) -> None:
        """Terminate gracefully, then kill if needed."""
```

---

## Extension Points

### How to Add a New Backend

1. **Create backend module** at `backends/mybackend.py`:

```python
from dataclasses import dataclass
from marshmallow_dataclass import dataclass as marshmallow_dataclass

@marshmallow_dataclass(frozen=True)
class MyBackendProtocol:
    type: Literal["mybackend"] = "mybackend"

    # Configuration fields
    my_option: str | None = None

    def get_config_for_mode(self, mode: str) -> dict[str, Any]:
        """Return config dict for worker mode."""
        ...

    def get_environment_for_mode(self, mode: str) -> dict[str, str]:
        """Return env vars for worker mode."""
        ...

    def allocate_endpoints(self, ...) -> list[Endpoint]:
        """Allocate logical endpoints to nodes."""
        from srtctl.core.topology import allocate_endpoints
        return allocate_endpoints(...)

    def endpoints_to_processes(self, endpoints, base_sys_port=8081) -> list[Process]:
        """Convert endpoints to physical processes."""
        from srtctl.core.topology import endpoints_to_processes
        return endpoints_to_processes(endpoints, base_sys_port)

    def build_worker_command(self, process, endpoint_processes, runtime, ...) -> list[str]:
        """Build command to start worker process."""
        cmd = ["python3", "-m", "mybackend.server", ...]
        return cmd
```

2. **Register in `backends/__init__.py`**:

```python
from .mybackend import MyBackendProtocol

BackendConfig = SGLangProtocol | MyBackendProtocol
```

3. **Update BackendConfigField in schema.py** to handle polymorphic deserialization.

### How to Add a New Frontend

1. **Create frontend module** at `frontends/myfrontend.py`:

```python
class MyFrontend:
    @property
    def type(self) -> str:
        return "myfrontend"

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def parse_health(self, response_json, expected_prefill, expected_decode) -> WorkerHealthResult:
        """Parse health check response."""
        ...

    def start_frontends(self, topology, runtime, config, backend, backend_processes) -> list[ManagedProcess]:
        """Start frontend processes."""
        ...

    def get_frontend_args_list(self, args: dict | None) -> list[str]:
        """Convert args dict to CLI arguments."""
        ...
```

2. **Register in `frontends/base.py`**:

```python
def get_frontend(frontend_type: str) -> FrontendProtocol:
    if frontend_type == "myfrontend":
        from srtctl.frontends.myfrontend import MyFrontend
        return MyFrontend()
    # ... existing frontends
```

### How to Add a New Benchmark

1. **Create benchmark module** at `benchmarks/mybench.py`:

```python
from srtctl.benchmarks.base import BenchmarkRunner, register_benchmark

@register_benchmark("mybench")
class MyBenchRunner(BenchmarkRunner):
    @property
    def name(self) -> str:
        return "My Benchmark"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/mybench/run.sh"

    def validate_config(self, config: SrtConfig) -> list[str]:
        """Return list of validation errors (empty if valid)."""
        errors = []
        if config.benchmark.my_required_field is None:
            errors.append("benchmark.my_required_field is required")
        return errors

    def build_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        """Build benchmark command."""
        return [
            "python3", self.script_path,
            "--host", runtime.nodes.head,
            "--port", str(runtime.frontend_port),
            # ... other args
        ]
```

2. **Add benchmark script** at `benchmarks/scripts/mybench/run.sh`

3. **Import in `benchmarks/__init__.py`** to trigger registration:

```python
from . import mybench  # noqa: F401
```

---

## Module Dependencies

### Import Hierarchy

```
                    +-------------+
                    |  __init__   |
                    +------+------+
                           |
         +-----------------+------------------+
         |                 |                  |
    +----v----+      +-----v-----+      +-----v-----+
    |   cli   |      |   core    |      | backends  |
    +---------+      +-----------+      +-----------+
         |                 |                  |
         |           +-----+-----+            |
         |           |           |            |
    +----v----+ +----v----+ +----v----+  +----v----+
    | do_sweep| | schema  | | runtime |  | sglang  |
    +---------+ +---------+ +---------+  +---------+
         |           |           |
    +----v----+ +----v----+ +----v----+
    | mixins  | | config  | | topology|
    +---------+ +---------+ +---------+
         |           |           |
    +----v----+ +----v----+ +----v----+
    |frontends| | health  | |processes|
    +---------+ +---------+ +---------+
         |           |           |
    +----v----+ +----v----+ +----v----+
    |benchmrks| | slurm   | |formatting|
    +---------+ +---------+ +---------+
```

### Circular Import Prevention

1. **TYPE_CHECKING guard** - Import type-only dependencies:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
```

2. **Lazy imports** - Import at function call time:

```python
def get_frontend(frontend_type: str) -> FrontendProtocol:
    # Import here to avoid circular imports
    from srtctl.frontends.dynamo import DynamoFrontend
    from srtctl.frontends.sglang import SGLangFrontend
    ...
```

3. **Forward references** - Use string annotations:

```python
def from_config(cls, config: "SrtConfig", job_id: str) -> "RuntimeContext":
    ...
```

---

## Directory Structure

```
src/srtctl/
|-- __init__.py              # Package exports and version
|-- logging_utils.py         # Logging configuration
|
|-- core/                    # Core infrastructure
|   |-- __init__.py          # Core exports
|   |-- config.py            # Config loading with cluster defaults
|   |-- schema.py            # Frozen dataclass definitions
|   |-- runtime.py           # RuntimeContext - single source of truth
|   |-- topology.py          # Endpoint/Process allocation
|   |-- processes.py         # Process lifecycle management
|   |-- slurm.py             # SLURM utilities (srun, nodelist)
|   |-- health.py            # HTTP health checks
|   |-- formatting.py        # FormattablePath/String wrappers
|   |-- sweep.py             # Parameter sweep generation
|   |-- ip_utils/            # IP resolution utilities
|       |-- __init__.py
|       |-- get_node_ip.sh
|
|-- backends/                # Backend implementations
|   |-- __init__.py          # Exports BackendConfig, protocols
|   |-- base.py              # BackendProtocol definition
|   |-- sglang.py            # SGLangProtocol implementation
|   |-- trtllm.py            # TRTLLMProtocol implementation
|
|-- frontends/               # Frontend implementations
|   |-- __init__.py          # Exports FrontendProtocol
|   |-- base.py              # FrontendProtocol definition
|   |-- dynamo.py            # DynamoFrontend (NATS/etcd)
|   |-- sglang.py            # SGLangFrontend (direct router)
|
|-- cli/                     # CLI entry points
|   |-- __init__.py
|   |-- submit.py            # srtctl apply/dry-run commands
|   |-- do_sweep.py          # SweepOrchestrator
|   |-- setup_head.py        # Head node infrastructure
|   |-- interactive.py       # Interactive mode
|   |-- mixins/              # Orchestrator stage mixins
|       |-- __init__.py
|       |-- worker_stage.py      # Backend worker startup
|       |-- frontend_stage.py    # Frontend/nginx startup
|       |-- benchmark_stage.py   # Benchmark execution
|
|-- benchmarks/              # Benchmark runners
|   |-- __init__.py          # Registry and exports
|   |-- base.py              # BenchmarkRunner ABC, register_benchmark
|   |-- sa_bench.py          # SA-Bench throughput benchmark
|   |-- aime.py              # AIME math accuracy benchmark
|   |-- mmlu.py              # MMLU accuracy benchmark
|   |-- gpqa.py              # GPQA benchmark
|   |-- longbenchv2.py       # LongBench v2 benchmark
|   |-- router.py            # Router benchmark
|   |-- mooncake_router.py   # Mooncake router benchmark
|   |-- profiling.py         # Profiling benchmark
|   |-- scripts/             # Benchmark shell scripts
|       |-- sa-bench/
|           |-- bench.sh
|           |-- benchmark_serving.py
|           |-- ...
|
|-- templates/               # Jinja2 templates
    |-- job_script_minimal.j2    # sbatch script template
    |-- nginx.conf.j2            # nginx load balancer config
```

---

## Summary

srtctl is a well-architected orchestration framework with:

- **Clean separation of concerns**: Config, runtime, backend, frontend, benchmark layers
- **Strong typing**: Frozen dataclasses with marshmallow validation
- **Extensibility**: Protocol-based backends/frontends, decorator-based benchmark registration
- **Robust process management**: Registry, monitoring, graceful cleanup
- **SLURM integration**: Proper container mounts, srun launching, nodelist parsing
- **Modern Python**: 3.10+ syntax, comprehensive type hints, clear module structure

The codebase follows Python best practices and provides a solid foundation for orchestrating complex LLM inference workloads on SLURM clusters.
