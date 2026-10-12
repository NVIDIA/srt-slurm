# Legacy (v1) recipe layout

Recipes without `schema: 2` (or with `schema: 1`) use the pre-2.0 layout: worker topology under `resources`, the engine and its per-mode settings under `backend`, the discovery plane under `infra`, and placement as per-block booleans. **This layout no longer loads.** `srtctl apply`, `srtctl dry-run`, and every other reader reject a recipe that has no `schema: 2` key, and reject any of the keys below even under `schema: 2`, with an error that names the keys and points here. `srtctl migrate -f <recipe> --in-place` rewrites a v1 recipe into the 2.0 layout, preserving comments and key order (a directory is walked recursively); it is the only part of srtctl that still reads this layout. The 2.0 layout is documented in [schema-reference.md](schema-reference.md) and [config-reference.md](config-reference.md).

Internally, srtctl still stores the resolved configuration in fields with these names: the 2.0 vocabularies (`engine:`, `roles:`, `placement:`, `services:`, `dynamo.source`) expand into them at load. That is an implementation detail; a recipe cannot reach those fields by their old spelling.

## v1 keys and what replaced them

| v1 key | 2.0 spelling |
|---|---|
| `backend` (top level) | `engine:` (the type, plus engine-wide knobs) and `roles.<role>.args` / `.env` |
| `infra` (top level) | `services:` entries of type `etcd` and `nats` (`placement.node: dedicated`, `options.max_payload_mb`) |
| `resources.prefill_nodes` | `roles.prefill.nodes` |
| `resources.prefill_workers` | `roles.prefill.workers` |
| `resources.gpus_per_prefill` | `roles.prefill.gpus` |
| `resources.prefill_critical` | `roles.prefill.critical` |
| `resources.decode_nodes` | `roles.decode.nodes` (`colocate` replaces the `0` sentinel) |
| `resources.decode_workers` | `roles.decode.workers` |
| `resources.gpus_per_decode` | `roles.decode.gpus` |
| `resources.decode_critical` | `roles.decode.critical` |
| `resources.agg_nodes` | `roles.agg.nodes` |
| `resources.agg_workers` | `roles.agg.workers` |
| `resources.gpus_per_agg` | `roles.agg.gpus` |
| `resources.agg_critical` | `roles.agg.critical` |
| `frontend.orchestrator_placement` | `frontend.placement.node: <location>` |
| `frontend.dedicated_node` | `frontend.placement.node: dedicated` |
| `benchmark.client_placement` | `benchmark.placement.node: <location>` |
| `benchmark.client_dedicated_node` | `benchmark.placement.node: dedicated` |
| `dynamo.version` | `dynamo.source.pypi` |
| `dynamo.wheel` | `dynamo.source.wheel` |
| `dynamo.hash` | `dynamo.source.git` + `dynamo.source.rev` (a commit) |
| `dynamo.cargo_patches` | `dynamo.source.patches` |
| `backend.prefill_environment` | `roles.prefill.env` |
| `backend.decode_environment` | `roles.decode.env` |
| `backend.aggregated_environment` | `roles.agg.env` |
| `backend.sglang_config` | `roles.<role>.args` (one mapping per role; the `prefill` / `decode` / `aggregated` keys) |
| `backend.kv_events_config` | `roles.<role>.kv_events` |
| `backend.mooncake_kv_store` | a `services:` entry of type `mooncake-master` plus the worker env on `roles.<role>.env` (see [mooncake-kv-store.md](mooncake-kv-store.md)) |
| `backend.prefill_extra_args` | `roles.prefill.extra_args` |
| `backend.decode_extra_args` | `roles.decode.extra_args` |
| `backend.aggregated_extra_args` | `roles.agg.extra_args` |
| `backend.trtllm_config` | `roles.<role>.args` (one mapping per role; the `prefill` / `decode` / `aggregated` keys) |
| `backend.vllm_config` | `roles.<role>.args` (one mapping per role; the `prefill` / `decode` / `aggregated` keys) |
| `backend.mocker_config` | `roles.<role>.args` (one mapping per role; the `prefill` / `decode` / `aggregated` keys) |

`dynamo.top_of_tree` is not in this table: it is still a 2.0 key. It has no immutable equivalent under `dynamo.source`, so the migrator leaves it in place and prints a note; pin a commit in `dynamo.source.rev` when you can.

## v1 values that changed meaning

These keys exist in both layouts, but the value means something else in 2.0. `srtctl migrate` writes the 2.0 spelling for a schema 1 document; a `schema: 2` document keeps the 2.0 meaning.

| Key | schema 1 value | 2.0 spelling | 2.0 meaning of the old value |
|---|---|---|---|
| `frontend.type` | `sglang` (the SGLang Model Gateway router) | `sglang-router` | `sglang` is the router-free single `sglang.launch_server` worker |

## What the migrator does not carry over

- Benchmark fields the recipe's `benchmark.type` never reads (`isl` on `gsm8k`, `num_shots` on `sa-bench`) are removed. They were silent no-ops in v1; schema 2 rejects them.
- `infra` under a frontend that runs no etcd or NATS (`sglang-router`, `vllm-router`, the direct frontends) is not turned into services: `nats_max_payload_mb` is dropped (it had no effect) and `etcd_nats_dedicated_node: true` is left as is with a note, because it still reserves a node. Drop it by hand to give the node back to the workers.
- A v1 recipe that never named a Dynamo to install pip-installed PyPI 0.8.0 implicitly. That default still applies when `dynamo.source` is absent; set `dynamo.source` or `dynamo.install: false` to make the choice explicit.

## Legacy fields in retained sections

Types and defaults of the v1 keys, as the last loader that accepted them saw them, for translating by hand when a recipe cannot go through `srtctl migrate`.

### resources

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_nodes` | int \| None | `None` | Disaggregated mode |
| `decode_nodes` | int \| None | `None` | `0` shared the prefill nodes (2.0: `roles.decode.nodes: colocate`) |
| `prefill_workers` | int \| None | `None` |  |
| `decode_workers` | int \| None | `None` |  |
| `agg_nodes` | int \| None | `None` | Aggregated mode |
| `agg_workers` | int \| None | `None` |  |
| `gpus_per_prefill` | int \| None | `None` | Explicit GPUs per worker (override computed values) |
| `gpus_per_decode` | int \| None | `None` |  |
| `gpus_per_agg` | int \| None | `None` |  |
| `prefill_critical` | bool | `True` | A worker of this role exiting fails the run |
| `decode_critical` | bool | `True` |  |
| `agg_critical` | bool | `True` |  |

### frontend

| Key | Type | Default | Description |
|---|---|---|---|
| `orchestrator_placement` | str | `'head'` | trtllm_serve: which node runs the disaggregated orchestrator. "head" (default) -> nodes.head (first prefill/CTX node) "first_decode" -> first decode/GEN worker-leader node |
| `dedicated_node` | bool | `False` | If True, reserve a node exclusively for the frontend/orchestrator instead of running it on a worker node. Requires at least 2 nodes. Not supported together with resources.het_jobs: true. Default: False. |

### benchmark

| Key | Type | Default | Description |
|---|---|---|---|
| `client_placement` | str | `'head'` | Which node runs the benchmark client: "head" (default) -> nodes.head (co-located with orchestrator by default) "last_decode" -> last decode/GEN worker-leader node (isolate the client off the CTX/orchestrator node). When the client lands on a different node than the orchestrator, use the injected $SRT_FRONTEND_HOST env in the benchmark command's URL. |
| `client_dedicated_node` | bool | `False` | If True, reserve a node exclusively for the benchmark client instead of running it on a worker node. Requires at least 2 nodes. Not supported together with resources.het_jobs: true. Default: False. |

### dynamo

| Key | Type | Default | Description |
|---|---|---|---|
| `version` | str \| None | `'0.8.0'` | PyPI release to pip-install |
| `hash` | str \| None | `None` | Commit to clone and build from |
| `wheel` | str \| None | `None` | ai-dynamo package version to install via staged wheels |
| `cargo_patches` | list[str] \| None | `None` | Optional dependency-declaration overrides applied to the dynamo Cargo.toml tree before a source build (requires `hash`). Each entry is a full `<crate> = <spec>` TOML line, e.g. 'dynamo-tokenizers = { git = "https://github.com/ai-dynamo/frontend-crates", branch = "..." }' The crate's existing declaration is replaced tree-wide, letting a source build pull a crate from an unmerged branch without waiting for a crates.io release. |

## backend

`backend.type` selected the engine; the 2.0 layout writes `engine:` instead and moves the per-mode keys below onto `roles.<role>`. The engine-wide knobs (everything not listed here) are unchanged and documented under Engine types in [schema-reference.md](schema-reference.md#engine-types).

### `backend.type: sglang`

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `sglang_config` | mapping with `prefill` / `decode` / `aggregated` keys | `None` | SGLang server CLI config per mode |
| `kv_events_config` | bool \| dict[str, Any] \| None | `None` | KV events config - enables --kv-events-config with auto-allocated ports Per-mode: {"prefill": true, "decode": {"publisher": "zmq", "topic": "custom"}} Or global: true (enables for prefill+decode with defaults) |
| `mooncake_kv_store` | mapping (`container`, `env`, `master_extra_args`) | `None` | Mooncake KV store - launches mooncake_master on infra node and injects MOONCAKE_MASTER env var on all workers automatically |

### `backend.type: trtllm`

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_environment` | dict[str, str] | `{}` |  |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `prefill_extra_args` | list[str] | `[]` | Extra `trtllm-serve` CLI flags per mode, appended verbatim to the worker command (frontend.type: trtllm_serve only). Used for options that configure the OpenAI server layer rather than the engine, such as `--tool_parser`. |
| `decode_extra_args` | list[str] | `[]` |  |
| `aggregated_extra_args` | list[str] | `[]` |  |
| `trtllm_config` | mapping with `prefill` / `decode` / `aggregated` keys | `None` |  |

### `backend.type: vllm`

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `vllm_config` | mapping with `prefill` / `decode` / `aggregated` keys | `None` | vLLM server CLI config per mode |
| `mooncake_kv_store` | mapping (`container`, `env`, `master_extra_args`, `store_config`) | `None` | Mooncake KV store; `store_config` values are JSON-serialized into MOONCAKE_CONFIG_PATH for vLLM's `MooncakeStoreConfig`. |
| `kv_events_config` | bool \| dict[str, Any] \| None | `None` | KV events config - enables --kv-events-config with auto-allocated ports. Required for Dynamo's event-driven KV-aware routing. Global true enables defaults for prefill and decode workers. Per-mode: {"prefill": true, "decode": {"topic": "custom"}} |

### `backend.type: mocker`

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `mocker_config` | mapping with `prefill` / `decode` / `aggregated` keys | `None` | Per-mode CLI overrides |

## infra

| Key | Type | Default | Description |
|---|---|---|---|
| `etcd_nats_dedicated_node` | bool | `False` | If True, run etcd and nats on a dedicated node instead of the head node. This reserves the first node exclusively for infrastructure services. Default: False. |
| `nats_max_payload_mb` | int \| None | `None` | Maximum NATS message payload in MB. Default: None (uses NATS default of 1MB). Set to 24+ for disaggregated serving with long ISL (e.g. 65K+ tokens where prompt data exceeds 1MB in NATS messages). |
