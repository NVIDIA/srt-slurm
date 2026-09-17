# Legacy (v1) recipe layout

<!-- GENERATED FILE. Do not edit by hand. Regenerate with `srtctl schema-docs`; CI fails when this file is stale. -->

Recipes without `schema: 2` (or with `schema: 1`) use the pre-2.0 layout: worker topology under `resources`, the engine and its per-mode settings under `backend`, the discovery plane under `infra`, and placement as per-block booleans. Every field below still loads, because the 2.0 vocabularies (`engine:`, `roles:`, `placement:`, `services:`, `dynamo.source`) are normalized into exactly these fields before validation. Nothing here should appear in a new recipe: `srtctl migrate -f <recipe> --in-place` rewrites it, preserving comments and key order, and `srtctl migrate --verify -f <path>` proves the v1 and migrated documents resolve to the same config. The 2.0 layout is documented in [schema-reference.md](schema-reference.md) and [config-reference.md](config-reference.md).

## v1 keys and what replaced them

| v1 key | 2.0 spelling |
|---|---|
| `backend` (top level) | `engine:` (the type, plus engine-wide knobs) and `roles.<role>.args` / `.env` |
| `infra` (top level) | `services:` entries of type `etcd` and `nats` (`placement.node: dedicated`, `options.max_payload_mb`) |
| `resources.prefill_nodes` | `roles.prefill.nodes` |
| `resources.prefill_workers` | `roles.prefill.workers` |
| `resources.gpus_per_prefill` | `roles.prefill.gpus` |
| `resources.decode_nodes` | `roles.decode.nodes` (`colocate` replaces the `0` sentinel) |
| `resources.decode_workers` | `roles.decode.workers` |
| `resources.gpus_per_decode` | `roles.decode.gpus` |
| `resources.agg_nodes` | `roles.agg.nodes` |
| `resources.agg_workers` | `roles.agg.workers` |
| `resources.gpus_per_agg` | `roles.agg.gpus` |
| `frontend.orchestrator_placement` | `frontend.placement.node: <location>` |
| `frontend.dedicated_node` | `frontend.placement.node: dedicated` |
| `benchmark.client_placement` | `benchmark.placement.node: <location>` |
| `benchmark.client_dedicated_node` | `benchmark.placement.node: dedicated` |
| `dynamo.version` | `dynamo.source.pypi` |
| `dynamo.wheel` | `dynamo.source.wheel` |
| `dynamo.hash` | `dynamo.source.git` + `dynamo.source.rev` (a commit) |
| `dynamo.top_of_tree` | `dynamo.source.git` + `dynamo.source.top_of_tree: true` |
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

## v1 values that changed meaning

These keys exist in both layouts, but the value means something else in 2.0. A schema 1 recipe keeps its historical meaning at load; `srtctl migrate` writes the 2.0 spelling.

| Key | schema 1 value | 2.0 spelling | 2.0 meaning of the old value |
|---|---|---|---|
| `frontend.type` | `sglang` (the SGLang Model Gateway router) | `sglang-router` | `sglang` is the router-free single `sglang.launch_server` worker |

## Legacy fields in retained sections

### resources (ResourceConfig)

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_nodes` | int \| None | `None` | Disaggregated mode |
| `decode_nodes` | int \| None | `None` |  |
| `prefill_workers` | int \| None | `None` |  |
| `decode_workers` | int \| None | `None` |  |
| `agg_nodes` | int \| None | `None` | Aggregated mode |
| `agg_workers` | int \| None | `None` |  |
| `gpus_per_prefill` | int \| None | `None` | Explicit GPUs per worker (override computed values) Use data_key to map from YAML field names to internal attribute names |
| `gpus_per_decode` | int \| None | `None` |  |
| `gpus_per_agg` | int \| None | `None` |  |

### frontend (FrontendConfig)

| Key | Type | Default | Description |
|---|---|---|---|
| `orchestrator_placement` | str | `'head'` | trtllm_serve: which node runs the disaggregated orchestrator. "head" (default) -> nodes.head (first prefill/CTX node) "first_decode" -> first decode/GEN worker-leader node |
| `dedicated_node` | bool | `False` | If True, reserve a node exclusively for the frontend/orchestrator instead of running it on a worker node. Requires at least 2 nodes. Not supported together with resources.het_jobs: true. Default: False. |

### benchmark (BenchmarkConfig)

| Key | Type | Default | Description |
|---|---|---|---|
| `client_placement` | str | `'head'` | Which node runs the benchmark client: "head" (default) -> nodes.head (co-located with orchestrator by default) "last_decode" -> last decode/GEN worker-leader node (isolate the client off the CTX/orchestrator node). When the client lands on a different node than the orchestrator, use the injected $SRT_FRONTEND_HOST env in the benchmark command's URL. |
| `client_dedicated_node` | bool | `False` | If True, reserve a node exclusively for the benchmark client instead of running it on a worker node. Requires at least 2 nodes. Not supported together with resources.het_jobs: true. Default: False. |

### dynamo (DynamoConfig)

| Key | Type | Default | Description |
|---|---|---|---|
| `version` | str \| None | `'0.8.0'` |  |
| `hash` | str \| None | `None` |  |
| `top_of_tree` | bool | `False` |  |
| `wheel` | str \| None | `None` |  |
| `cargo_patches` | list[str] \| None | `None` | Optional dependency-declaration overrides applied to the dynamo Cargo.toml tree before a source build (requires `hash`). Each entry is a full `<crate> = <spec>` TOML line, e.g. 'dynamo-tokenizers = { git = "https://github.com/ai-dynamo/frontend-crates", branch = "..." }' The crate's existing declaration is replaced tree-wide, letting a source build pull a crate from an unmerged branch without waiting for a crates.io release. |

## backend

`backend.type` selected the engine; the 2.0 layout writes `engine:` instead and moves the per-mode keys below onto `roles.<role>`. The engine-wide knobs (everything not listed here) are unchanged and documented under Engine types in [schema-reference.md](schema-reference.md#engine-types).

### SGLangProtocol

`backend.type: sglang`

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `sglang_config` | [SGLangServerConfig](#sglangserverconfig) \| None | `None` | SGLang server CLI config per mode |
| `kv_events_config` | bool \| dict[str, Any] \| None | `None` | KV events config - enables --kv-events-config with auto-allocated ports Per-mode: {"prefill": true, "decode": {"publisher": "zmq", "topic": "custom"}} Or global: true (enables for prefill+decode with defaults) |
| `mooncake_kv_store` | [MooncakeKVStoreConfig](#mooncakekvstoreconfig) \| None | `None` | Mooncake KV store - launches mooncake_master on infra node and injects MOONCAKE_MASTER env var on all workers automatically |

### TRTLLMProtocol

`backend.type: trtllm`

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_environment` | dict[str, str] | `{}` |  |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `prefill_extra_args` | list[str] | `[]` | Extra `trtllm-serve` CLI flags per mode, appended verbatim to the worker command (frontend.type: trtllm_serve only -- dynamo.trtllm takes a different CLI). `trtllm_config` already covers everything that belongs in the engine YAML, which is nearly everything: trtllm-serve merges that file into LlmArgs. But a few of its options configure the OpenAI SERVER layer rather than the engine and have no LlmArgs field, so no YAML key can reach them. The one that matters in practice is `--tool_parser` (a click.Choice consumed directly by the server constructor); note that its sibling `--reasoning_parser` IS forwarded into get_llm_args() and so remains settable from `trtllm_config`. backend: type: trtllm prefill_extra_args: ["--tool_parser", "glm47"] decode_extra_args: ["--tool_parser", "glm47"] |
| `decode_extra_args` | list[str] | `[]` |  |
| `aggregated_extra_args` | list[str] | `[]` |  |
| `trtllm_config` | [TRTLLMServerConfig](#trtllmserverconfig) \| None | `None` |  |

### VLLMProtocol

`backend.type: vllm`

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `vllm_config` | [VLLMServerConfig](#vllmserverconfig) \| None | `None` | vLLM server CLI config per mode |
| `mooncake_kv_store` | [VLLMMooncakeKVStoreConfig](#vllmmooncakekvstoreconfig) \| None | `None` | Mooncake KV store — when set, srtslurm launches mooncake_master on the infra node and auto-injects MOONCAKE_MASTER / MOONCAKE_TE_META_DATA_SERVER / MOONCAKE_LOCAL_HOSTNAME on every vLLM worker. |
| `kv_events_config` | bool \| dict[str, Any] \| None | `None` | KV events config - enables --kv-events-config with auto-allocated ports. Required for Dynamo's event-driven KV-aware routing. Global true enables defaults for prefill and decode workers. Per-mode: {"prefill": true, "decode": {"topic": "custom"}} |

### MockerProtocol

`backend.type: mocker`

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill_environment` | dict[str, str] | `{}` | Environment variables per mode |
| `decode_environment` | dict[str, str] | `{}` |  |
| `aggregated_environment` | dict[str, str] | `{}` |  |
| `mocker_config` | [MockerServerConfig](#mockerserverconfig) \| None | `None` | Per-mode CLI overrides |

### SGLangServerConfig

SGLang server CLI configuration per mode (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill` | dict[str, Any] \| None | `None` |  |
| `decode` | dict[str, Any] \| None | `None` |  |
| `aggregated` | dict[str, Any] \| None | `None` |  |

### MooncakeKVStoreConfig

Mooncake KV store configuration.

| Key | Type | Default | Description |
|---|---|---|---|
| `container` | str \| None | `None` |  |
| `env` | dict[str, str] | `{}` |  |
| `master_extra_args` | list[str] | `[]` |  |

### TRTLLMServerConfig

SGLang server CLI configuration per mode (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill` | dict[str, Any] \| None | `None` |  |
| `decode` | dict[str, Any] \| None | `None` |  |
| `aggregated` | dict[str, Any] \| None | `None` |  |

### VLLMServerConfig

vLLM server CLI configuration per mode (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill` | dict[str, Any] \| None | `None` |  |
| `decode` | dict[str, Any] \| None | `None` |  |
| `aggregated` | dict[str, Any] \| None | `None` |  |

### VLLMMooncakeKVStoreConfig

Mooncake KV store config for the vLLM backend.

| Key | Type | Default | Description |
|---|---|---|---|
| `container` | str \| None | `None` |  |
| `env` | dict[str, str] | `{}` |  |
| `master_extra_args` | list[str] | `[]` |  |
| `store_config` | dict[str, Any] \| None | `None` | ``store_config`` values are JSON-serialized into MOONCAKE_CONFIG_PATH and parsed by vLLM's ``MooncakeStoreConfig`` dataclass — fields are a mix of str (e.g. ``protocol``), int (e.g. ``port``), and human-readable sizes (e.g. ``"4GB"``). Type as ``dict[str, Any]`` to avoid forcing users to quote numeric values. |

### MockerServerConfig

Mocker CLI configuration per mode (prefill/decode/aggregated).

| Key | Type | Default | Description |
|---|---|---|---|
| `prefill` | dict[str, Any] \| None | `None` |  |
| `decode` | dict[str, Any] \| None | `None` |  |
| `aggregated` | dict[str, Any] \| None | `None` |  |

## infra

### InfraConfig

Infrastructure configuration for etcd/nats placement.

| Key | Type | Default | Description |
|---|---|---|---|
| `etcd_nats_dedicated_node` | bool | `False` | If True, run etcd and nats on a dedicated node instead of the head node. This reserves the first node exclusively for infrastructure services. Default: False. |
| `nats_max_payload_mb` | int \| None | `None` | Maximum NATS message payload in MB. Default: None (uses NATS default of 1MB). Set to 24+ for disaggregated serving with long ISL (e.g. 65K+ tokens where prompt data exceeds 1MB in NATS messages). |
