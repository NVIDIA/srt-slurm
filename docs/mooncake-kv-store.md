# Mooncake KV Store

## Per-process vLLM device selection

For nodes with a known physical-GPU-to-HCA mapping, opt in with the `mooncake-master` service's `device_names_by_gpu` option. The map describes the node's hardware, so it is one per job; each Mooncake role keeps its own client config:

```yaml
roles:
  prefill:
    args:
      kv-transfer-config: '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
    mooncake_store_config: &mooncake
      metadata_server: P2PHANDSHAKE
      global_segment_size: 150GB
      local_buffer_size: 4GB
      protocol: rdma
  decode:
    args:
      kv-transfer-config: '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
    mooncake_store_config: *mooncake
services:
  - name: mooncake-master
    type: mooncake-master
    options:
      device_names_by_gpu: [mlx5_0, mlx5_1, mlx5_2, mlx5_3]
```

`device_names_by_gpu` is also accepted under `engine.mooncake_kv_store` (or legacy `backend.mooncake_kv_store`), but do not combine that form with a declared `mooncake-master` service. `srtctl migrate` moves the legacy mapping into the service's `options.device_names_by_gpu` without changing its meaning.

This mapping is indexed by **physical GPU ID on each node**, not by the CUDA ordinal inside a container. A process on GPU 2 receives `device_name: mlx5_2`; two TP2 processes on GPUs `[0,1]` and `[2,3]` receive disjoint device subsets. Only the configs needed by the allocated processes of roles that use Mooncake are generated in `/logs`, and each such worker receives its own `MOONCAKE_CONFIG_PATH`. Other store settings, including segment capacity and the managed master address, are preserved. When the map is nonempty, it overrides the role's `device_name` in each process-local JSON (`/logs/mooncake_store_config_<role>_gpu<ids>.json`); the role's own configuration is not mutated. The map must cover every physical GPU on the node; repeated HCA names are allowed for machines where multiple GPUs share an HCA. Without the option, or with an empty list, each role's JSON is used unchanged. Config loading (including dry-run) rejects wrong-length maps and entries that are empty, contain whitespace, or contain a comma, before allocating nodes.

This is **process-local**, not nested-rank-local: all TP ranks spawned inside one process still inherit its subset. A one-HCA-per-nested-rank policy requires separate connector support. This option does not patch vLLM, set private vLLM environment variables, or change store capacity. Use homogeneous mappings across nodes, and verify device locality on the target cluster. Restricting a process to its assigned HCAs can avoid unnecessary RDMA registration fanout; performance equivalence to other HCA policies must be measured, not assumed.

First-class support for [Mooncake](https://github.com/kvcache-ai/Mooncake) as the KV transfer backend for prefill-decode disaggregation. A `mooncake-master` entry under `services:` in an SGLang, vLLM, or TensorRT-LLM recipe makes srtslurm launch and configure the mooncake master automatically and wire up worker env vars so peer-to-peer transfers work across multiple nodes.

## Table of Contents

- [Overview](#overview)
- [Quick Start (SGLang)](#quick-start-sglang)
- [Quick Start (vLLM)](#quick-start-vllm)
- [Quick Start (TensorRT-LLM)](#quick-start-tensorrt-llm)
- [What srtslurm Owns vs What You Set](#what-srtslurm-owns-vs-what-you-set)
- [Configuration Reference](#configuration-reference)
- [Standalone Store Services](#standalone-store-services)
- [Master Metrics Endpoint](#master-metrics-endpoint)
- [Validation](#validation)
- [Common Configurations](#common-configurations)
  - [RDMA / InfiniBand](#rdma-infiniband)
  - [TCP](#tcp)
  - [Custom Master Container](#custom-master-container)
- [Troubleshooting](#troubleshooting)

---

## Overview

SGLang supports several KV transfer backends for prefill-decode disaggregation: `mooncake`, `nixl`, `ascend`, `mori`, and `fake`. Mooncake is the default and uses RDMA/TCP for high-throughput transfers backed by a central master process.

Without first-class support, running mooncake with srtslurm meant:

1. Launching `mooncake_master` somewhere yourself (no integration with the SLURM job)
2. Setting `MOONCAKE_MASTER`, `MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE`, etc. as env vars on every prefill and decode worker manually
3. Resolving each worker's own IP for `MOONCAKE_LOCAL_HOSTNAME` so multi-node transfers don't fall back to `localhost`
4. Adding `disaggregation-transfer-backend: mooncake` to the prefill and decode `args`

The `mooncake-master` service automates 1 and the srtslurm-owned parts of 2 and 3. The Mooncake tuning env (`MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE`, ...) goes in each role's `env`. You still set the SGLang flags in step 4 because they're SGLang's CLI surface, not srtslurm's, but srtslurm validates that you did.

The master is a [service](services.md#implicit-services) and the worker-side env lives on the roles. The v1 layout (`backend.mooncake_kv_store`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

```yaml
engine: sglang
roles:
  prefill:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: mlx5_0
    args:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
  decode:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: mlx5_0
    args:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
services:
  - name: mooncake-master
    type: mooncake-master
    container: nvcr.io/nvidia/mooncake:latest   # optional; default: job container
    args: []                                     # appended to mooncake_master
    # options.store_config: ...                  # deprecated: use roles.<role>.mooncake_store_config
    # placement.node: dedicated                  # share the reserved infra node with etcd/nats
    # external: 10.0.0.5:8700                    # a master that already runs; nothing launches
```

`MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, and `MOONCAKE_LOCAL_HOSTNAME` are stamped on every worker. A `MOONCAKE_LOCAL_HOSTNAME` in a role's `env` pins the NIC.

## Quick Start (SGLang)

Minimum config to run mooncake:

```yaml
engine: sglang
roles:
  prefill:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
    args:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
  decode:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
    args:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
services:
  - name: mooncake-master
    type: mooncake-master
```

Even more minimal, just declare the master and let everything else default:

```yaml
engine: sglang
roles:
  prefill:
    args:
      disaggregation-transfer-backend: mooncake
  decode:
    args:
      disaggregation-transfer-backend: mooncake
services:
  - name: mooncake-master
    type: mooncake-master
```

## Quick Start (vLLM)

vLLM's `MooncakeStoreConnector` reads its configuration from a JSON file pointed to by `MOONCAKE_CONFIG_PATH` rather than directly from env vars, so each role that uses it takes a `mooncake_store_config:` section that srtslurm renders into that role's JSON at job start:

```yaml
engine: vllm
roles:
  prefill:
    env:                                  # in-process Mooncake C++ knobs, per role
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
      MC_STORE_CLIENT_METRIC: "1"
    args:
      kv-transfer-config: '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
    mooncake_store_config:                # rendered into this role's MOONCAKE_CONFIG_PATH JSON
      metadata_server: "P2PHANDSHAKE"
      global_segment_size: "100GB"
      local_buffer_size: "4GB"
      protocol: "rdma"
      device_name: "mlx5_0,mlx5_1"
  decode:
    env:
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
      MC_STORE_CLIENT_METRIC: "1"
    args:
      kv-transfer-config: '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
    mooncake_store_config:
      metadata_server: "P2PHANDSHAKE"
      global_segment_size: "100GB"
      local_buffer_size: "4GB"
      protocol: "rdma"
      device_name: "mlx5_0,mlx5_1"
services:
  - name: mooncake-master
    type: mooncake-master
```

Real-world production form: `MultiConnector` wraps `NixlConnector` (P2P transfer between prefill and decode) **and** `MooncakeStoreConnector` (shared store for cross-instance reuse), both with `kv_role: "kv_both"` so prefill and decode workers run identical connector stacks. srtslurm's validator accepts `MooncakeStoreConnector` standalone or wrapped in `MultiConnector`.

srtslurm stamps `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, and `MOONCAKE_LOCAL_HOSTNAME` on every worker, and `MOONCAKE_CONFIG_PATH=/logs/mooncake_store_config_<role>.json` on the workers of each role whose KV connector uses Mooncake; you supply the rest. The connector is resolved as the worker command resolves it: the role's `kv-transfer-config`, else `roles.<role>.args.connector`, else `engine.connector`, including connectors a `MultiConnector` wraps. An aggregated worker behind a frontend other than Dynamo runs only the connector its role names, so give such a role its Mooncake connector in `roles.agg.args`. `master_server_address` is also auto-filled from the infra node IP and ignored if set by hand. `metadata_server` and `protocol` must have the same value in every role.

To write the keys the roles share once, put a YAML anchor on the first role's mapping and merge it into the others; a role's own keys after the merge win. [`examples/vllm/dynamo-disagg-mooncake.yaml`](https://github.com/NVIDIA/srt-slurm/blob/main/examples/vllm/dynamo-disagg-mooncake.yaml) does this:

```yaml
roles:
  prefill:
    mooncake_store_config: &mooncake
      protocol: rdma
      metadata_server: P2PHANDSHAKE
      global_segment_size: 8GB
  decode:
    mooncake_store_config:
      <<: *mooncake
      global_segment_size: 16GB          # decode ranks lend more
```

Older recipes set these keys once, in the `mooncake-master` service's `options.store_config`. That still works: the keys apply under every role's own `mooncake_store_config`, and srtslurm logs a deprecation warning. `srtctl migrate -f recipe.yaml` moves them into each role whose connector uses Mooncake, keeping the role's own keys. It leaves an override file (`base` plus `override_*` variants) and a sweep file as they are, because a variant or sweep point can change which roles use Mooncake, and it leaves `engine.mooncake_kv_store.store_config` for you to move by hand.

The role `env:` maps are injected on the vLLM workers (not on the standalone `mooncake_master` daemon; the master gets the service entry's own `env`). Use them for in-process Mooncake C++ knobs like `MC_ENABLE_DEST_DEVICE_AFFINITY`, `MC_STORE_CLIENT_METRIC`, `MC_TE_METRIC`.

## Quick Start (TensorRT-LLM)

TensorRT-LLM's `mooncake-store` KV connector shares one Mooncake Store pool between context and generation servers. Use a TensorRT-LLM build that supports loading the connector's client JSON through `MOONCAKE_CONFIG_PATH` and the `capacity` role. Every rank reads that file before it looks at `kv_connector_config.mooncake_store`. srtslurm runs the same `mooncake_master` it runs for SGLang and vLLM, and gives each role its own client config:

```yaml
engine: trtllm
roles:
  prefill:
    args:
      kv_cache_config:
        use_kv_cache_manager_v2: true    # the connector needs KV cache manager v2; "auto" may not pick it
      cache_transceiver_config:          # v2 in disaggregated serving needs the Python NIXL transceiver
        backend: NIXL
        transceiver_runtime: PYTHON
      kv_connector_config:
        connector: mooncake-store        # no mooncake_store block
    mooncake_store_config:               # this role's Mooncake client config
      protocol: rdma
      device_name: "mlx5_0,mlx5_1"       # optional; empty lets Mooncake discover RDMA devices
      global_segment_size: 16GiB         # host memory each rank lends; byte counts or binary units
      stage_through_host: true           # copy through host memory; no GPUDirect RDMA needed
  decode:
    args:
      kv_cache_config:
        use_kv_cache_manager_v2: true
      cache_transceiver_config:
        backend: NIXL
        transceiver_runtime: PYTHON
      kv_connector_config:
        connector: mooncake-store
    mooncake_store_config:
      protocol: rdma                     # keys the whole pool shares match across roles
      device_name: "mlx5_0,mlx5_1"
      global_segment_size: 32GiB         # decode ranks lend more
      stage_through_host: true
services:
  - name: mooncake-master
    type: mooncake-master
```

Each role's `global_segment_size` is the host memory each of that role's ranks lends to the pool: in this example, each prefill rank lends **16 GiB** and each decode rank lends **32 GiB**. A role does not inherit another role's keys; a key a role leaves out takes TensorRT-LLM's default. Use `global_segment_size` in these mappings: `segment_size` belongs to TensorRT-LLM's `kv_connector_config.mooncake_store` block and is rejected here because the client JSON would ignore it.

For every role whose `args` select the connector, srtslurm writes `/logs/mooncake_store_config_<role>.json` before workers start and sets `MOONCAKE_CONFIG_PATH` to it on that role's srun task, so every rank `trtllm-llmapi-launch` starts inherits it. The file layers, later wins: a per-role `role` default and `model_key` (the served model name), then `roles.<role>.mooncake_store_config` (over the deprecated service `options.store_config`, if any); `master_server_address` is always `<infra_node_ip>:8700`:

| Role | Default `role` | What the server does with the pool |
| --- | --- | --- |
| `prefill`, `agg` | `both` | Looks up prefixes before computing them and stores what it computed. |
| `decode` | `capacity` | Lends its ranks' segments and never transfers: prompt KV reaches it over the cache transceiver. |

`model_key` names the checkpoint in the pool's keys. TensorRT-LLM gives it no default, so that two checkpoints sharing a pool cannot read each other's KV. An srtslurm pool belongs to one job, so recipes can omit it. All roles receive the same key, chosen in this order:

1. `model_key` in every connector role's `mooncake_store_config`, with the same value in each.
2. `engine.served_model_name`, when set.
3. The basename of the resolved model path: `/models/checkpoint-revision-42` becomes `checkpoint-revision-42`, and `hf:Qwen/Qwen3-0.6B` becomes `Qwen3-0.6B`. Local paths use the checkpoint directory name before it is mounted at `/model` in workers.

Keys every server of the pool must agree on (`model_key`, `namespace`, `metadata_server`, `protocol`) must have the same value in every role that sets them, and a recipe that sets one in some roles but not others is refused. The rest of the keys pass through to TensorRT-LLM's reader, which supplies its own defaults (`metadata_server: P2PHANDSHAKE`, `protocol: rdma`, `namespace: trtllm`, ...). Notes:

- The same recipe works with `frontend.type: trtllm_serve` and `frontend.type: dynamo`. trtllm-serve skips its own pool provisioning when `MOONCAKE_CONFIG_PATH` is already set, and `dynamo.trtllm` provisions nothing, so no wrapper is needed around either worker.
- srtslurm launches the native `mooncake_master` binary directly. A `capacity` role's ranks lend the memory, so separate donor processes are unnecessary.
- `global_segment_size` is per rank: a node needs its ranks times the segment size of host memory. TensorRT-LLM refuses `GB`/`MB`, because vLLM's reader of the same file scales them by 1024 and TensorRT-LLM's by 1000; write `GiB` or a byte count.
- Leave `local_hostname` unset so each rank uses its own address; a value in `mooncake_store_config` would apply to every rank of the role. TensorRT-LLM does not read `MOONCAKE_LOCAL_HOSTNAME`, `MOONCAKE_MASTER`, or `MOONCAKE_TE_META_DATA_SERVER`; srtslurm still stamps them, and they are inert for these workers.
- Engine-side limits depend on the TensorRT-LLM build. These examples select KV cache manager v2 and the Python NIXL transceiver with `cache_transceiver_config: {backend: NIXL, transceiver_runtime: PYTHON}`. Check the installed engine's support before enabling attention DP or speculative decoding. On a cluster without GPUDirect RDMA, set `stage_through_host: true`.

## What srtslurm Owns vs What You Set

| Concern                                         | Owner     | Notes                                                                                                |
| ----------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| Launching `mooncake_master`                     | srtslurm  | The `mooncake-master` service: runs on the infra node (same node as etcd/nats; `placement.node: dedicated` shares their reserved node) before workers, gated on RPC `8700`, HTTP metadata `8701`, and admin HTTP `8702`. Its log is `service_mooncake-master.out`. |
| `MOONCAKE_MASTER` env var on workers            | srtslurm  | Always computed as `<infra_node_ip>:8700`. User values in role `env` are overridden.                  |
| `MOONCAKE_TE_META_DATA_SERVER` env var          | srtslurm  | Always computed as `http://<infra_node_ip>:8701/metadata`.                                            |
| `MOONCAKE_LOCAL_HOSTNAME` env var               | srtslurm  | Auto-resolved per-worker via `runtime.network_interface`. User can override in role `env` for custom NICs. |
| `MOONCAKE_CONFIG_PATH` (vLLM, TensorRT-LLM)      | srtslurm  | Points to the JSON file srtslurm renders from the role's `mooncake_store_config`, one per role: `/logs/mooncake_store_config_<role>.json`. Mounted under `/logs` in every worker. |
| `master_server_address` in the client config    | srtslurm  | (vLLM, TensorRT-LLM) Always overridden with `<infra_node_ip>:8700`. User values are ignored.          |
| `role` in the TensorRT-LLM client config        | srtslurm  | Defaults to `both` for prefill and agg and `capacity` for decode; set it in `roles.<role>.mooncake_store_config` to override. |
| `model_key` in the TensorRT-LLM client config   | srtslurm  | Defaults to the served model name; set it in every role's `mooncake_store_config` to override. |
| `MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE`, etc.    | User      | Set in `roles.prefill.env` and `roles.decode.env`.                                                    |
| `disaggregation-transfer-backend: mooncake`     | User      | (SGLang only) Set in `roles.prefill.args` and `roles.decode.args`. srtslurm validates this is present. |
| `disaggregation-ib-device`                      | User      | (SGLang only) Set in `roles.prefill.args` and `roles.decode.args`. Format: `"mlx5_0,mlx5_1"` or JSON map. |
| `kv-transfer-config`                            | User      | (vLLM only) Set in `roles.prefill.args` and `roles.decode.args` to wire vLLM's `MooncakeStoreConnector`. |
| `kv_connector_config: {connector: mooncake-store}` | User   | (TensorRT-LLM only) Set in the `args` of each role that joins the pool, without a `mooncake_store` block. |

## Configuration Reference

### SGLang

```yaml
engine: sglang
roles:
  prefill:
    env:                                        # optional, per role
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: mlx5_0
      MOONCAKE_TE_META_DATA_SERVER: P2PHANDSHAKE
      # SGLang-specific staging buffer knobs:
      SGLANG_DISAGG_STAGING_BUFFER: "true"
      SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB: "64"
      SGLANG_DISAGG_STAGING_POOL_SIZE_MB: "4096"
    args:
      disaggregation-transfer-backend: mooncake
  decode:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: mlx5_0
    args:
      disaggregation-transfer-backend: mooncake
services:
  - name: mooncake-master
    type: mooncake-master
    container: nvcr.io/nvidia/mooncake:latest  # optional, default: job container
    args: []                                    # optional, appended to mooncake_master
```

### vLLM

```yaml
engine: vllm
roles:
  prefill:
    env:                                 # optional, in-process Mooncake C++ knobs on the vLLM workers
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
      MC_STORE_CLIENT_METRIC: "1"        # default 1 (enabled)
      MC_TE_METRIC: "0"                  # default 0 (disabled)
    args:
      kv-transfer-config: '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
    mooncake_store_config:               # rendered into this role's MOONCAKE_CONFIG_PATH JSON
      metadata_server: "P2PHANDSHAKE"
      global_segment_size: "100GB"
      local_buffer_size: "4GB"
      protocol: "rdma"
      device_name: "mlx5_0,mlx5_1"
  decode:
    env:
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
    args:
      kv-transfer-config: '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
    mooncake_store_config:
      metadata_server: "P2PHANDSHAKE"
      global_segment_size: "200GB"       # decode ranks lend more
      local_buffer_size: "4GB"
      protocol: "rdma"
      device_name: "mlx5_0,mlx5_1"
services:
  - name: mooncake-master
    type: mooncake-master
    container: ...                       # optional, default: job container
    args: []                             # optional, appended to mooncake_master
```

### Fields

On the `mooncake-master` service entry (the generic service fields such as `placement`, `external`, and `readiness` are in [Services](services.md#configuration-reference)):

- **`container`** (`str`, optional): Container image used for the `mooncake_master` srun. Defaults to the job container if unset. Useful when mooncake needs a different runtime than your worker container.
- **`args`** (`list[str]`, optional): Extra arguments appended to the standalone `mooncake_master` command. Use this for flags supported only by the Mooncake version in the selected container. Do not use it to override the RPC, HTTP metadata, or metrics ports because srtslurm configures worker endpoints and readiness checks from its own port values.
- **`options.device_names_by_gpu`** (vLLM, `list[str]`, optional): The physical-GPU-to-HCA map; see [Per-process vLLM device selection](#per-process-vllm-device-selection).
- **`options.store_config`** (vLLM and TensorRT-LLM, `dict[str, Any]`, deprecated): The client config keys every role used to share. They still apply under each role's `roles.<role>.mooncake_store_config`, and srtslurm logs a deprecation warning; the field will be removed in a future release. `srtctl migrate` moves them to the roles.

On the roles:

- **`roles.<role>.mooncake_store_config`** (vLLM and TensorRT-LLM, `dict[str, Any]`, optional): The Mooncake client config for that role's workers, rendered as JSON into the file `MOONCAKE_CONFIG_PATH` names. The role must use Mooncake (vLLM: a Mooncake connector in its `kv-transfer-config`, `connector`, or `engine.connector`; TensorRT-LLM: `kv_connector_config.connector: mooncake-store`), and a `mooncake-master` service must exist. Keys every client of the pool must agree on have to match across roles (vLLM: `metadata_server`, `protocol`; TensorRT-LLM: also `model_key` and `namespace`). SGLang takes per-role Mooncake settings in `roles.<role>.env` instead.
  - For **vLLM**, keys map 1:1 to vLLM's `MooncakeStoreConfig` dataclass, a mix of `str` (e.g. `protocol`), `int` (e.g. `port`), and human-readable size strings (e.g. `"4GB"`). srtslurm does not default these fields; values like `global_segment_size`, `protocol`, and `device_name` are hardware-specific and silently using a srtslurm-picked default is worse than failing loudly, so set them explicitly.
  - For **TensorRT-LLM**, the keys are those of its Mooncake client config (`protocol`, `device_name`, `global_segment_size`, `local_buffer_size`, `namespace`, `metadata_server`, `transfer_batch_size`, `stage_through_host`, and `model_key`, which defaults to the served model name); sizes are byte counts or binary units, `role` defaults per role, and `mooncake_store`-block names (`segment_size`, `pool`, `run_dir`, `master_timeout`) are refused because the client config would ignore them.
  - `master_server_address` is auto-filled and any user value is ignored.
- **`roles.<role>.env`** (`dict[str, str]`, optional): Env vars injected on that role's workers.
  - For **SGLang**, keys map directly to mooncake's environment variable names. See the [SGLang server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/environ.py) and [mooncake_store.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py) for the full list.
  - For **vLLM**, this is for in-process Mooncake C++ knobs (`MC_*`) read by the transfer engine / store client. vLLM's connector itself reads configuration from `MOONCAKE_CONFIG_PATH` (the JSON rendered from the role's `mooncake_store_config`), not from these env vars.
  - Setting `MOONCAKE_MASTER` or `MOONCAKE_TE_META_DATA_SERVER` here is a no-op (srtslurm always wins). The same holds for `MOONCAKE_CONFIG_PATH` on a role that uses a `mooncake-master` service's pool; a role without one keeps its own value.

Mooncake `v0.3.11+` adds NoF SSD-tier eviction flags. Enable them only in recipes using a compatible Mooncake image:

```yaml
services:
  - name: mooncake-master
    type: mooncake-master
    args:
      - --nof_eviction_high_watermark_ratio=0.9
```

Older Mooncake versions do not recognize this option, so leave it out of those recipes. The existing `--eviction_high_watermark_ratio` controls memory eviction; the `--nof_...` option independently controls the NVMe-over-Fabrics SSD tier.

## Standalone Store Services

Mooncake can run the Store as a standalone process per node, so workers use embedded clients with `MOONCAKE_GLOBAL_SEGMENT_SIZE=0` while dedicated stores own the DRAM segments. In srtslurm that is a `services:` entry with `type: mooncake-store`: it starts after the master is healthy and before workers, gets `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, and `MOONCAKE_LOCAL_HOSTNAME` from the runtime, and defaults its container to the `mooncake-master` entry's `container`.

```yaml
services:
  - name: mooncake-master
    type: mooncake-master
  - name: store
    type: mooncake-store
    placement:
      node: workers          # or prefill / decode for per-role segment sizes
    args:
      - --port
      - "8800"
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
      MOONCAKE_GLOBAL_SEGMENT_SIZE: 100gb
    readiness:
      port: 8800
```

The worker side stays in the roles' `env` (`roles.prefill.env` / `roles.decode.env`). See [Services](services.md#example-standalone-mooncake-stores) for the full shape, per-role entries, and the co-location rules.

## Master Metrics Endpoint

The `mooncake_master` admin HTTP server is always exposed on port `8702` on the infra node and starts before workers do (srtslurm waits for it). It serves:

- `GET /metrics`: Prometheus text format (master + transfer-engine counters)
- `GET /metrics/summary`: human-readable summary
- `GET /health`, `/role`, `/ha_status`, `/leader`
- `GET /query_key`: used by Dynamo's KV router shared-cache path

To scrape from outside the cluster, point your collector at `http://<infra_node_ip>:8702/metrics`. The infra node IP is logged at job start.

## Validation

srtslurm rejects disaggregated recipes that declare a `mooncake-master` service without a matching `disaggregation-transfer-backend: mooncake` in `roles.prefill.args` or `roles.decode.args`. This catches the common mistake where the master process launches but workers fall back to default transport:

```text
ValidationError: a mooncake-master service is configured but neither roles.prefill.args
nor roles.decode.args has 'disaggregation-transfer-backend: mooncake'.
Add it to both roles (and 'disaggregation-ib-device') so workers actually
use the mooncake master srtslurm launches for you.
```

Both dash and underscore forms (`disaggregation-transfer-backend`, `disaggregation_transfer_backend`) are accepted. The vLLM check looks for a Mooncake connector in each role's resolved KV connector: its `kv-transfer-config`, else `connector`, else `engine.connector`, including connectors a `MultiConnector` wraps.

For every engine, `roles.<role>.mooncake_store_config` needs a role whose workers read a Mooncake client config file (refused on SGLang roles and on vLLM or TensorRT-LLM roles without a Mooncake connector) and a `mooncake-master` service, and the keys the pool shares must match across roles.

For TensorRT-LLM, srtslurm checks at load what the workers would otherwise reject in every rank after the model has loaded:

- A `mooncake-master` service needs at least one role with `kv_connector_config.connector: mooncake-store`, and those roles must not also set a `mooncake_store` block: TensorRT-LLM would read the srtslurm config instead of it.
- Each rendered config needs a non-empty `model_key` and a known `role`; sizes must be byte counts or binary units (`local_buffer_size` at least 1, `global_segment_size` at least 0), `transfer_batch_size` must be positive, and `stage_through_host` must not be a string (TensorRT-LLM reads `"false"` as true).
- `roles.<role>.mooncake_store_config` and the deprecated `options.store_config` must not carry `mooncake_store`-block keys (`segment_size`, `pool`, `run_dir`, `master_timeout`), which the client config would ignore.
- A role that selects the connector without a `mooncake-master` service must name its pool itself: `MOONCAKE_CONFIG_PATH` in its `env`, or, when trtllm-serve runs the engine (`frontend.type: trtllm_serve`, or a Dynamo sidecar), a `mooncake_store` block with its own `run_dir` and one worker. Every TensorRT-LLM rank runs under `trtllm-llmapi-launch`, so it reads a config trtllm-serve rendered only back from `run_dir`, which one server owns; `dynamo.trtllm` renders none.
- `external` on the `mooncake-master` service is refused for TensorRT-LLM: the rendered `master_server_address` always names the infra node.

## Common Configurations

### RDMA / InfiniBand

The most common production setup:

```yaml
engine: sglang
roles:
  prefill:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
    args:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
  decode:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
    args:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
services:
  - name: mooncake-master
    type: mooncake-master
```

For a per-GPU IB device map, pass JSON to `disaggregation-ib-device`:

```yaml
roles:
  prefill:
    args:
      disaggregation-ib-device: '{"0": "mlx5_0", "1": "mlx5_1", "2": "mlx5_2", "3": "mlx5_3"}'
```

### TCP

For development / clusters without RDMA:

```yaml
engine: sglang
roles:
  prefill:
    env:
      MOONCAKE_PROTOCOL: tcp
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
    args:
      disaggregation-transfer-backend: mooncake
  decode:
    env:
      MOONCAKE_PROTOCOL: tcp
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
    args:
      disaggregation-transfer-backend: mooncake
services:
  - name: mooncake-master
    type: mooncake-master
```

### Custom Master Container

Pin a specific mooncake build for the master process:

```yaml
services:
  - name: mooncake-master
    type: mooncake-master
    container: nvcr.io/nvidia/mooncake:24.10
```

The workers continue to use the job's main container; only the master process uses the override.

## Troubleshooting

### Master fails to start within 120s

srtslurm waits up to 120 seconds for `mooncake_master` to bind on port 8700. If it times out, check:

- `service_mooncake-master.out` in the run's log directory, which usually shows a binary-not-found or RDMA setup error
- Whether `mooncake_master` is on `$PATH` inside the master container. If you're using a custom container, verify it has the mooncake binaries installed.
- Whether port 8700 is already in use on the infra node from a previous failed run (rare, but can happen if cleanup was interrupted)

### Workers connect but transfers stall

Almost always a `MOONCAKE_LOCAL_HOSTNAME` resolution issue. srtslurm auto-resolves it via `runtime.network_interface`. Verify in the worker log's `Env:` line that each worker has its own node's IP, not `localhost` or another worker's IP.

If your cluster uses a separate RDMA NIC from the primary interface, override it with the right IP. A `MOONCAKE_LOCAL_HOSTNAME` in a role's `env` applies to every worker of that role, so for true per-worker overrides you'd need to set `runtime.network_interface` cluster-wide via `srtslurm.yaml`.

### "Either MOONCAKE_MASTER or MOONCAKE_CLIENT is not set"

This error from SGLang means the worker started before `MOONCAKE_MASTER` was injected. Check that a `mooncake-master` service is declared in the recipe; the env var is only auto-set when that entry exists. Run `srtctl dry-run -f recipe.yaml` and look for `mooncake` in the env table.

### "ValidationError: a mooncake-master service is configured but neither..."

You declared a `mooncake-master` service but forgot `disaggregation-transfer-backend: mooncake` in `roles.prefill.args` and `roles.decode.args`. Add it to both roles; see [Validation](#validation) above.
