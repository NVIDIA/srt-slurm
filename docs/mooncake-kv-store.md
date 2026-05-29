# Mooncake KV Store

First-class support for [Mooncake](https://github.com/kvcache-ai/Mooncake) as the KV transfer backend for SGLang prefill-decode disaggregation. When `mooncake_kv_store` is set under an SGLang backend, srtslurm launches and configures the mooncake master automatically and wires up worker env vars so peer-to-peer transfers work across multiple nodes.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [What srtslurm Owns vs What You Set](#what-srtslurm-owns-vs-what-you-set)
- [Configuration Reference](#configuration-reference)
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

## Quick Start

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

## What srtslurm Owns vs What You Set

| Concern                                         | Owner     | Notes                                                                                                |
| ----------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| Launching `mooncake_master`                     | srtslurm  | Runs on the infra node (same node as etcd/nats; respects `infra.etcd_nats_dedicated_node`). Port 50051. |
| `MOONCAKE_MASTER` env var on workers            | srtslurm  | Always computed as `<infra_node_ip>:50051`. User values in `env` are overridden.                      |
| `MOONCAKE_LOCAL_HOSTNAME` env var               | srtslurm  | Auto-resolved per-worker via `runtime.network_interface`. User can override in `env` for custom NICs. |
| `MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE`, etc.    | User      | Passed through `mooncake_kv_store.env` to all workers.                                               |
| `disaggregation-transfer-backend: mooncake`     | User      | Set on `sglang_config.prefill` and `sglang_config.decode`. srtslurm validates this is present.       |
| `disaggregation-ib-device`                      | User      | Set on `sglang_config.prefill` and `sglang_config.decode`. Format: `"mlx5_0,mlx5_1"` or JSON map.    |

## Configuration Reference

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
      MOONCAKE_MASTER_METRICS_PORT: "9003"
      # SGLang-specific staging buffer knobs:
      SGLANG_DISAGG_STAGING_BUFFER: "true"
      SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB: "64"
      SGLANG_DISAGG_STAGING_POOL_SIZE_MB: "4096"
```

### Fields

- **`container`** (`str`, optional): Container image used for the `mooncake_master` srun. Defaults to the job container if unset. Useful when mooncake needs a different runtime than your SGLang container.
- **`env`** (`dict[str, str]`, optional): Pass-through env vars injected on every prefill and decode worker. Keys map directly to mooncake's environment variable names — see the [SGLang server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/environ.py) and [mooncake_store.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py) for the full list. Setting `MOONCAKE_MASTER` here is a no-op (srtslurm always wins).

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

The workers continue to use the job's main container — only the master process uses the override.

## Troubleshooting

### Master fails to start within 120s

srtslurm waits up to 120 seconds for `mooncake_master` to bind on port 50051. If it times out, check:

- `mooncake_master.out` in the run's log directory — usually shows a binary-not-found or RDMA setup error
- Whether `mooncake_master` is on `$PATH` inside the master container. If you're using a custom container, verify it has the mooncake binaries installed.
- Whether port 50051 is already in use on the infra node from a previous failed run (rare, but can happen if cleanup was interrupted)

### Workers connect but transfers stall

Almost always a `MOONCAKE_LOCAL_HOSTNAME` resolution issue. srtslurm auto-resolves it via `runtime.network_interface`. Verify in the worker log's `Env:` line that each worker has its own node's IP, not `localhost` or another worker's IP.

If your cluster uses a separate RDMA NIC from the primary interface, override per-worker with the right IP — but note that `mooncake_kv_store.env` applies the same value everywhere, so for true per-worker overrides you'd need to set `runtime.network_interface` cluster-wide via `srtslurm.yaml`.

### "Either MOONCAKE_MASTER or MOONCAKE_CLIENT is not set"

This error from SGLang means the worker started before `MOONCAKE_MASTER` was injected. Check that `mooncake_kv_store` is present in the recipe — the env var is only auto-set when this block exists. Run `srtctl dry-run -f recipe.yaml` and look for `mooncake` in the env table.

### "ValidationError: mooncake_kv_store is set but neither..."

You added `mooncake_kv_store` but forgot `disaggregation-transfer-backend: mooncake` in `sglang_config.prefill` and `sglang_config.decode`. Add it to both modes — see [Validation](#validation) above.
