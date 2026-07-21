# Mooncake KV Store

First-class support for [Mooncake](https://github.com/kvcache-ai/Mooncake) as the KV transfer backend for prefill-decode disaggregation. When `mooncake_kv_store` is set under an SGLang or vLLM backend, srtslurm launches and configures the mooncake master automatically and wires up worker env vars so peer-to-peer transfers work across multiple nodes.

## Table of Contents

- [Overview](#overview)
- [Quick Start (SGLang)](#quick-start-sglang)
- [Quick Start (vLLM)](#quick-start-vllm)
- [What srtslurm Owns vs What You Set](#what-srtslurm-owns-vs-what-you-set)
- [Configuration Reference](#configuration-reference)
- [Standalone Store Services](#standalone-store-services)
- [Master Metrics Endpoint](#master-metrics-endpoint)
- [Validation](#validation)
- [Common Configurations](#common-configurations)
  - [RDMA / InfiniBand](#rdma--infiniband)
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
4. Adding `disaggregation-transfer-backend: mooncake` to `sglang_config`

The `mooncake_kv_store` block automates 1–3. You still set the SGLang flags in step 4 because they're SGLang's CLI surface, not srtslurm's — but srtslurm validates that you did.

## Quick Start (SGLang)

Minimum config to run mooncake:

```yaml
backend:
  type: sglang
  mooncake_kv_store:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
    decode:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
```

Even more minimal — just enable mooncake and let everything else default:

```yaml
backend:
  type: sglang
  mooncake_kv_store: {}
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
    decode:
      disaggregation-transfer-backend: mooncake
```

## Quick Start (vLLM)

vLLM's `MooncakeStoreConnector` reads its configuration from a JSON file pointed to by `MOONCAKE_CONFIG_PATH` rather than directly from env vars, so the vLLM block takes an extra `store_config:` section that srtslurm renders into that JSON at job start:

```yaml
backend:
  type: vllm
  mooncake_kv_store:
    env:                                  # injected on every vLLM worker
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
      MC_STORE_CLIENT_METRIC: "1"
    store_config:                         # → MOONCAKE_CONFIG_PATH JSON
      metadata_server: "P2PHANDSHAKE"
      global_segment_size: "100GB"
      local_buffer_size: "4GB"
      protocol: "rdma"
      device_name: "mlx5_0,mlx5_1"
  vllm_config:
    prefill:
      kv-transfer-config: '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
    decode:
      kv-transfer-config: '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
```

Real-world production form: `MultiConnector` wraps `NixlConnector` (P2P transfer between prefill→decode) **and** `MooncakeStoreConnector` (shared store for cross-instance reuse), both with `kv_role: "kv_both"` so prefill and decode workers run identical connector stacks. srtslurm's validator accepts `MooncakeStoreConnector` standalone or wrapped in `MultiConnector`.

srtslurm stamps `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, `MOONCAKE_LOCAL_HOSTNAME`, and `MOONCAKE_CONFIG_PATH` on every worker; you supply the rest. `master_server_address` in `store_config` is also auto-filled from the infra node IP and ignored if set by hand.

The `env:` map is injected on every vLLM worker (not on the standalone `mooncake_master` daemon — the master srun passes no env). Use it for in-process Mooncake C++ knobs like `MC_ENABLE_DEST_DEVICE_AFFINITY`, `MC_STORE_CLIENT_METRIC`, `MC_TE_METRIC`.

## What srtslurm Owns vs What You Set

| Concern                                         | Owner     | Notes                                                                                                |
| ----------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| Launching `mooncake_master`                     | srtslurm  | Runs on the infra node (same node as etcd/nats; respects `infra.etcd_nats_dedicated_node`). RPC `8700`, HTTP metadata `8701`, admin HTTP `8702`. |
| `MOONCAKE_MASTER` env var on workers            | srtslurm  | Always computed as `<infra_node_ip>:8700`. User values in `env` are overridden.                       |
| `MOONCAKE_TE_META_DATA_SERVER` env var          | srtslurm  | Always computed as `http://<infra_node_ip>:8701/metadata`.                                            |
| `MOONCAKE_LOCAL_HOSTNAME` env var               | srtslurm  | Auto-resolved per-worker via `runtime.network_interface`. User can override in `env` for custom NICs. |
| `MOONCAKE_CONFIG_PATH` (vLLM only)               | srtslurm  | Always points to the JSON file srtslurm renders from `store_config:`. Mounted under `/logs` in every worker. |
| `master_server_address` in `store_config` (vLLM)| srtslurm  | Always overridden with `<infra_node_ip>:8700`. User values are ignored.                               |
| `MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE`, etc.    | User      | Passed through `mooncake_kv_store.env` to all workers.                                               |
| `disaggregation-transfer-backend: mooncake`     | User      | (SGLang only) Set on `sglang_config.prefill` and `sglang_config.decode`. srtslurm validates this is present. |
| `disaggregation-ib-device`                      | User      | (SGLang only) Set on `sglang_config.prefill` and `sglang_config.decode`. Format: `"mlx5_0,mlx5_1"` or JSON map. |
| `kv-transfer-config`                            | User      | (vLLM only) Set on `vllm_config.prefill` and `vllm_config.decode` to wire vLLM's `MooncakeStoreConnector`. |
| Standalone Store service command/env            | User      | Optional managed service, launched once per selected physical P/D/aggregated node before workers.       |
| Master/metadata/local-hostname env on standalone services | srtslurm | Computed per service node; user values for master and metadata endpoints are overridden.              |

## Configuration Reference

### SGLang

```yaml
backend:
  type: sglang
  mooncake_kv_store:
    container: nvcr.io/nvidia/mooncake:latest  # optional, default: job container
    env:                                        # optional, default: {}
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: mlx5_0
      MOONCAKE_TE_META_DATA_SERVER: P2PHANDSHAKE
      # SGLang-specific staging buffer knobs:
      SGLANG_DISAGG_STAGING_BUFFER: "true"
      SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB: "64"
      SGLANG_DISAGG_STAGING_POOL_SIZE_MB: "4096"
```

### vLLM

```yaml
backend:
  type: vllm
  mooncake_kv_store:
    container: ...                       # optional, default: job container
    env:                                 # optional, injected on every vLLM worker (in-process Mooncake C++ knobs)
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
      MC_STORE_CLIENT_METRIC: "1"        # default 1 (enabled)
      MC_TE_METRIC: "0"                  # default 0 (disabled)
    store_config:                        # required for vLLM; rendered into MOONCAKE_CONFIG_PATH JSON
      metadata_server: "P2PHANDSHAKE"
      global_segment_size: "100GB"
      local_buffer_size: "4GB"
      protocol: "rdma"
      device_name: "mlx5_0,mlx5_1"
```

### Fields

- **`container`** (`str`, optional): Container image used for the `mooncake_master` srun and as the fallback for standalone Store services. Defaults to the job container if unset. Useful when Mooncake needs a different runtime than the inference workers.
- **`env`** (`dict[str, str]`, optional): Pass-through env vars injected on every prefill and decode worker.
  - For **SGLang**, keys map directly to mooncake's environment variable names — see the [SGLang server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/environ.py) and [mooncake_store.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py) for the full list.
  - For **vLLM**, this is for in-process Mooncake C++ knobs (`MC_*`) read by the transfer engine / store client. vLLM's connector itself reads configuration from `MOONCAKE_CONFIG_PATH` (the JSON rendered from `store_config:`), not from these env vars.
  - Setting `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, or `MOONCAKE_CONFIG_PATH` here is a no-op (srtslurm always wins).
- **`store_config`** (vLLM only, `dict[str, Any]`): Pass-through dict rendered as JSON into the file pointed to by `MOONCAKE_CONFIG_PATH`. Keys map 1:1 to vLLM's `MooncakeStoreConfig` dataclass — a mix of `str` (e.g. `protocol`), `int` (e.g. `port`), and human-readable size strings (e.g. `"4GB"`). srtslurm does not default these fields — values like `global_segment_size`, `protocol`, and `device_name` are hardware-specific and silently using a srtslurm-picked default is worse than failing loudly, so set them explicitly. `master_server_address` is auto-filled and any user value is ignored.
- **`standalone`** (SGLang or vLLM, optional): Launch managed standalone Mooncake Store services on selected worker nodes. See [Standalone Store Services](#standalone-store-services).

## Standalone Store Services

Mooncake can run a real Store client as a standalone process. This lets inference workers use embedded clients with `MOONCAKE_GLOBAL_SEGMENT_SIZE=0`, while dedicated per-node services own the DRAM segments. Decode nodes can therefore contribute host memory without enabling an in-process SGLang HiCache pool.

The worker environment remains in the backend's existing per-mode sections. The standalone process has its own command, arguments, and environment:

```yaml
backend:
  type: sglang

  # Embedded clients connect to the shared Store but do not contribute a
  # second memory segment.
  prefill_environment:
    MOONCAKE_PROTOCOL: rdma
    MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
    MC_GID_INDEX: "3"
    MOONCAKE_GLOBAL_SEGMENT_SIZE: "0"
  decode_environment:
    MOONCAKE_PROTOCOL: rdma
    MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
    MC_GID_INDEX: "3"
    MOONCAKE_GLOBAL_SEGMENT_SIZE: "0"

  mooncake_kv_store:
    container: mooncake.sqsh
    standalone:
      # Passed through without interpreting Mooncake-specific flags.
      command: [python, -m, mooncake.mooncake_store_service]
      args: [--port, "8800", --max-wait-time, "120"]
      env:
        MOONCAKE_PROTOCOL: rdma
        MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
        MC_GID_INDEX: "3"
        MOONCAKE_LOCAL_BUFFER_SIZE: "0"
        MC_TE_METRIC: "true"
      placements:
        prefill:
          env:
            MOONCAKE_GLOBAL_SEGMENT_SIZE: "100gb"
        decode:
          env:
            MOONCAKE_GLOBAL_SEGMENT_SIZE: "400gb"
      preamble: |
        ulimit -n 1048576
        ulimit -l unlimited
      cpus_per_task: 8
      health_check:
        port: 8800
        timeout_seconds: 120
```

srtslurm launches the standalone services after `mooncake_master` is healthy and before inference workers start. It starts at most one service per selected physical node, even when multiple endpoints of the same role share that node. Prefill/decode co-location is supported only when their placement `env` and `args` are identical; conflicting definitions fail before any service is launched.

The following environment variables are added automatically to every standalone service:

- `MOONCAKE_MASTER=<infra_ip>:8700`
- `MOONCAKE_TE_META_DATA_SERVER=http://<infra_ip>:8701/metadata`
- `MOONCAKE_LOCAL_HOSTNAME=<service_node_ip>`

`MOONCAKE_MASTER` and `MOONCAKE_TE_META_DATA_SERVER` always use the managed master. A user-supplied `MOONCAKE_LOCAL_HOSTNAME` may override the resolved node IP when a specific network identity is required.

### Standalone fields

| Field | Default | Description |
| ----- | ------- | ----------- |
| `enabled` | `true` | Disable the service block without deleting it. |
| `container` | master container, then job container | Optional service image override. |
| `command` | `[python, -m, mooncake.mooncake_store_service]` | Executable and fixed command prefix. |
| `args` | `[]` | Common arguments appended to `command`. |
| `env` | `{}` | Common service environment. |
| `placements` | required | Role map: `prefill`, `decode`, and/or `aggregated`; each accepts `enabled`, `env`, and appended `args`. |
| `preamble` | null | Shell commands run after environment export and before the service command. |
| `cpus_per_task` | null | Optional `srun --cpus-per-task`. |
| `cpu_bind` | null | Optional `srun --cpu-bind`. |
| `srun_options` | `{}` | Additional options for service sruns only. |
| `health_check` | TCP port `8080`, 120s | Set to null to disable readiness checking. |
| `critical` | `true` | Treat a non-zero service exit as a job failure. |

Values in `command`, `args`, `env`, and `preamble` support `{node}`, `{node_ip}`, `{node_id}`, `{role}`, `{index}`, `{infra_node}`, `{infra_ip}`, `{master_port}`, and `{metadata_port}` templates. Keep numeric Mooncake Store settings in environment variables or a generated JSON configuration when the installed `mooncake_store_service` CLI does not parse typed `-D` overrides.

## Master Metrics Endpoint

The `mooncake_master` admin HTTP server is always exposed on port `8702` on the infra node and starts before workers do (srtslurm waits for it). It serves:

- `GET /metrics` — Prometheus text format (master + transfer-engine counters)
- `GET /metrics/summary` — human-readable summary
- `GET /health`, `/role`, `/ha_status`, `/leader`
- `GET /query_key` — used by Dynamo's KV router shared-cache path

To scrape from outside the cluster, point your collector at `http://<infra_node_ip>:8702/metrics`. The infra node IP is logged at job start.

## Validation

srtslurm rejects configs that set `mooncake_kv_store` in disaggregated mode without a matching `disaggregation-transfer-backend: mooncake` on `sglang_config.prefill` or `sglang_config.decode`. This catches the common mistake where the master process launches but workers fall back to default transport.

```text
ValidationError: mooncake_kv_store is set but neither sglang_config.prefill
nor sglang_config.decode has 'disaggregation-transfer-backend: mooncake'.
Add it to both modes (and 'disaggregation-ib-device') so workers actually
use the mooncake master srtslurm launches for you.
```

Both dash and underscore forms (`disaggregation-transfer-backend`, `disaggregation_transfer_backend`) are accepted.

## Common Configurations

### RDMA / InfiniBand

The most common production setup:

```yaml
backend:
  type: sglang
  mooncake_kv_store:
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
      MOONCAKE_DEVICE: "mlx5_0,mlx5_1"
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
    decode:
      disaggregation-transfer-backend: mooncake
      disaggregation-ib-device: "mlx5_0,mlx5_1"
```

For a per-GPU IB device map, pass JSON to `disaggregation-ib-device`:

```yaml
sglang_config:
  prefill:
    disaggregation-ib-device: '{"0": "mlx5_0", "1": "mlx5_1", "2": "mlx5_2", "3": "mlx5_3"}'
```

### TCP

For development / clusters without RDMA:

```yaml
backend:
  type: sglang
  mooncake_kv_store:
    env:
      MOONCAKE_PROTOCOL: tcp
      MOONCAKE_GLOBAL_SEGMENT_SIZE: "4gb"
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
    decode:
      disaggregation-transfer-backend: mooncake
```

### Custom Master Container

Pin a specific mooncake build for the master process:

```yaml
backend:
  type: sglang
  mooncake_kv_store:
    container: nvcr.io/nvidia/mooncake:24.10
    env:
      MOONCAKE_PROTOCOL: rdma
```

Inference workers continue to use the job's main container. The override applies to the master and to standalone Store services that do not set their own `standalone.container`.

## Troubleshooting

### Master fails to start within 120s

srtslurm waits up to 120 seconds for `mooncake_master` to bind on port 8700. If it times out, check:

- `mooncake_master.out` in the run's log directory — usually shows a binary-not-found or RDMA setup error
- Whether `mooncake_master` is on `$PATH` inside the master container. If you're using a custom container, verify it has the mooncake binaries installed.
- Whether port 8700 is already in use on the infra node from a previous failed run (rare, but can happen if cleanup was interrupted)

### Workers connect but transfers stall

Almost always a `MOONCAKE_LOCAL_HOSTNAME` resolution issue. srtslurm auto-resolves it via `runtime.network_interface`. Verify in the worker log's `Env:` line that each worker has its own node's IP, not `localhost` or another worker's IP.

If your cluster uses a separate RDMA NIC from the primary interface, override per-worker with the right IP — but note that `mooncake_kv_store.env` applies the same value everywhere, so for true per-worker overrides you'd need to set `runtime.network_interface` cluster-wide via `srtslurm.yaml`.

### "Either MOONCAKE_MASTER or MOONCAKE_CLIENT is not set"

This error from SGLang means the worker started before `MOONCAKE_MASTER` was injected. Check that `mooncake_kv_store` is present in the recipe — the env var is only auto-set when this block exists. Run `srtctl dry-run -f recipe.yaml` and look for `mooncake` in the env table.

### "ValidationError: mooncake_kv_store is set but neither..."

You added `mooncake_kv_store` but forgot `disaggregation-transfer-backend: mooncake` in `sglang_config.prefill` and `sglang_config.decode`. Add it to both modes — see [Validation](#validation) above.
