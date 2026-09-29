# Backends

Rules for `src/srtctl/backends/`. Every consumer asks a backend through `BackendProtocol` (`backends/base.py`); see the Design Rules in the root `CLAUDE.md`.

## Adding a New Backend

1. Create `backends/<name>.py` with a dataclass implementing `BackendProtocol`
2. Implement every member of `BackendProtocol` (`backends/base.py`), including the ones your engine answers with a neutral default:
   - `get_srun_config()` - MPI settings and launch strategy (`launch_per_endpoint`, `sequential_node_start`)
   - `get_config_for_mode(mode)` - Mode-specific configuration
   - `get_environment_for_mode(mode)` - Environment variables
   - `allocate_endpoints()` - Logical worker allocation
   - `endpoints_to_processes()` - Physical process mapping; every port through `NodePortAllocator`
   - `build_worker_command(process, runtime)` - Command construction
   - `get_process_environment(process)` - Per-process env derived from `Process` ports (side channels, scan bases)
   - `mooncake_kv_store` / `get_mooncake_worker_env(...)` - the Mooncake block and its worker env; `None` / `{}` without one
   - `failover` / `get_failover_environment(...)` - shadow engine recovery; `None` / `{}` without it
   - `should_set_visible_devices()` - `True` unless the engine takes its devices on the command line; the variable is the cluster's `visible_devices_env`
   - `get_served_model_name(default)`
3. Export from `backends/__init__.py`
4. Add polymorphic deserialization in `BackendConfigField` in `schema.py`

**Current backends:**
- **ATOM**: Native ROCm servers behind AToMesh, with one Slurm node per logical worker and allocator-owned Mooncake handshake ports
- **SGLang**: Per-process srun launching, supports prefill/decode/aggregated modes
- **TRTLLM**: MPI-style launching (one srun per endpoint with all nodes), prefill/decode only
- **vLLM**: Per-process srun launching, prefill/decode/aggregated, `per_node` DP; `frontend_type` selects Dynamo registration or a direct `vllm serve` server, and `_CONNECTOR_MAP` owns the KV connector table

## Mooncake KV Store

`docs/mooncake-kv-store.md` is the reference, including the schema 2 recipe shape. Rules for code:

- The master is the `mooncake-master` service (`services/`); a v1 `backend.mooncake_kv_store` block is normalized onto it by `services/normalize.py` before schema load.
- srtslurm stamps `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, and `MOONCAKE_LOCAL_HOSTNAME` on every worker; `MOONCAKE_LOCAL_HOSTNAME` is the worker's own IP on `runtime.network_interface`. A value in a role's `env` pins the NIC; `MOONCAKE_MASTER` is never set by hand.
- vLLM reads its store config from JSON: `store_config` is rendered into the file `MOONCAKE_CONFIG_PATH` names.
- SGLang disaggregated recipes must set `disaggregation-transfer-backend: mooncake` in the prefill and decode `args`; the validator rejects a master without it, because workers would silently fall back to the default transport.
- Consumers read `backend.mooncake_kv_store` / `backend.get_mooncake_worker_env(...)`; a backend without Mooncake returns `None` / `{}`.

## Shadow engine recovery (vLLM, `engine.failover`)

`VLLMProtocol.failover` (recipe key `engine.failover`, dataclass `VLLMFailoverConfig` in `backends/vllm.py`) runs Dynamo's shadow engine recovery on SLURM without DRA. It implies the `gms` service (`services/gms.py`, `placement.per: worker`): one GPU Memory Service instance per vLLM worker, in the `before_workers` phase, running one `python3 -m gpu_memory_service --device k` per GPU of the worker and gated on its `GMS ready:` log line. The worker stage then launches `1 + shadow_engines` engine steps per worker and node (`<role>_<index>_<node>` and `..._e<k>`) with `--load-format gms --gms-shadow-mode`. Each engine is its own `Process` (`Process.engine_id`, emitted by `endpoints_to_processes(engines_per_process=...)`) so ports come from the usual allocators; the gms instance and the engines of a worker get the same pinned `CUDA_VISIBLE_DEVICES` (no `--device-ids`) so "device k" is the same GPU for the servers and the engines, and the sockets and `failover.lock` live under `<shared_dir>/srtctl-<job_id>/<role>_<index>` (`/dev/shm`: enroot bind-mounts the host's; `/tmp` is per container). Relaunching a dead engine is `roles.<role>.restart`'s job (the supervisor's unit is one engine, not the worker). Validation (`_validate_vllm_failover`): Dynamo frontend, no sidecar mode, no DP, `load-format` gms or unset. See `docs/shadow-engine-recovery.md`; `tests/test_failover.py` is the acceptance suite.

`placement.per: worker` is the general mechanism behind the gms kind: `ServiceStageMixin.service_instances` attaches one instance to each engine-0 `Process` on the placed nodes, `ServiceLaunchContext.process` / `.config` carry the worker and the recipe to the kind, the stage pins `CUDA_VISIBLE_DEVICES`, and the step is `service_<name>_<role>_<index>_<node>`. `tests/test_service_per_worker.py` covers it for a generic service.
