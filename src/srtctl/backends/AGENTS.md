# Backends

Rules for `src/srtctl/backends/`. Every consumer asks a backend through `Backend` (`backends/base.py`); see the Design Rules in the root `CLAUDE.md`.

## Adding a New Backend

1. Create `backends/<name>.py` with a frozen dataclass inheriting `Backend`. Keep schema fields, including the bound `roles`, on the concrete dataclass.
2. Supply `type` and implement the required abstract launch hooks:
   - `allocate_endpoints()` - Logical worker allocation
   - `endpoints_to_processes()` - Physical process mapping; every port through `NodePortAllocator`
   - `build_worker_command(process, runtime)` - Command construction
3. Inherit optional defaults and override only where the engine needs different behavior:
   - `get_srun_config()` - Defaults to per-process launching without MPI; override MPI settings and launch strategy as needed
   - `get_config_for_mode(mode)` / `get_environment_for_mode(mode)` - Read the bound role's arguments and environment
   - `get_process_environment(process)` - Defaults to `{}`; override for per-process env derived from `Process` ports
   - `mooncake_kv_store` / `get_mooncake_worker_env(...)` - Default to `None` / `{}`
   - `failover` / `get_failover_environment(...)` - Default to `None` / `{}`
   - `should_set_visible_devices()` - Defaults to `True`; override if the engine takes its devices on the command line
   - `get_served_model_name(default)` - Defaults to the supplied model name
   - `fatal_log_patterns(mode)` - Defaults to `()`; override for launchers that survive engine failure
4. Export from `backends/__init__.py`
5. Add polymorphic deserialization in `BackendConfigField` in `schema.py`

**Current backends:**
- **ATOM**: Native ROCm servers behind AToMesh, with one Slurm node per logical worker and allocator-owned Mooncake handshake ports
- **SGLang**: Per-process srun launching, supports prefill/decode/aggregated modes
- **TileRT**: Decode workers behind `tilert-router`, paired with an explicit prefill engine and image.
- **TRTLLM**: MPI-style launching (one srun per endpoint with all nodes), prefill/decode only
- **vLLM**: Per-process srun launching, prefill/decode/aggregated, `per_node` DP; `frontend_type` selects Dynamo registration or a direct `vllm serve` server, and `_CONNECTOR_MAP` owns the KV connector table

## Mooncake KV Store

`docs/mooncake-kv-store.md` is the reference, including the schema 2 recipe shape. Rules for code:

- The master is the `mooncake-master` service (`services/`); `services/normalize.py` maps a declared entry onto the internal `backend.mooncake_kv_store` field before schema load, and `engine.mooncake_kv_store` sets that field directly.
- srtslurm stamps `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, and `MOONCAKE_LOCAL_HOSTNAME` on every worker; `MOONCAKE_LOCAL_HOSTNAME` is the worker's own IP on `runtime.network_interface`. A value in a role's `env` pins the NIC; `MOONCAKE_MASTER` is never set by hand.
- vLLM reads its store config from JSON: `store_config` is rendered into the file `MOONCAKE_CONFIG_PATH` names.
- SGLang disaggregated recipes must set `disaggregation-transfer-backend: mooncake` in the prefill and decode `args`; the validator rejects a master without it, because workers would silently fall back to the default transport.
- Consumers read `backend.mooncake_kv_store` / `backend.get_mooncake_worker_env(...)`; a backend without Mooncake returns `None` / `{}`.

## Shadow engine recovery (vLLM, `engine.failover`)

`VLLMBackend.failover` (recipe key `engine.failover`, dataclass `VLLMFailoverConfig` in `backends/vllm.py`) runs Dynamo's shadow engine recovery on SLURM without DRA. It implies the `gms` service (`services/gms.py`, `placement.per: worker`): one GPU Memory Service instance per vLLM worker, in the `before_workers` phase, running one `python3 -m gpu_memory_service --device k` per GPU of the worker and gated on its `GMS ready:` log line. The worker stage then launches `1 + shadow_engines` engine steps per worker and node (`<role>_<index>_<node>` and `..._e<k>`) with `--load-format gms --gms-shadow-mode`. Each engine is its own `Process` (`Process.engine_id`, emitted by `endpoints_to_processes(engines_per_process=...)`) so ports come from the usual allocators; the gms instance and the engines of a worker get the same pinned `CUDA_VISIBLE_DEVICES` (no `--device-ids`) so "device k" is the same GPU for the servers and the engines, and the sockets and `failover.lock` live under `<shared_dir>/srtctl-<job_id>/<role>_<index>` (`/dev/shm`: enroot bind-mounts the host's; `/tmp` is per container). Relaunching a dead engine is `roles.<role>.restart`'s job (the supervisor's unit is one engine, not the worker). Validation (`_validate_vllm_failover`): Dynamo frontend, no sidecar mode, no DP, `load-format` gms or unset. See `docs/shadow-engine-recovery.md`; `tests/test_failover.py` is the acceptance suite.

`placement.per: worker` is the general mechanism behind the gms kind: `ServiceStageMixin.service_instances` attaches one instance to each engine-0 `Process` on the placed nodes, `ServiceLaunchContext.process` / `.config` carry the worker and the recipe to the kind, the stage pins `CUDA_VISIBLE_DEVICES`, and the step is `service_<name>_<role>_<index>_<node>`. `tests/test_service_per_worker.py` covers it for a generic service.
