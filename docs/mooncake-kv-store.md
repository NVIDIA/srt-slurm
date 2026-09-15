# Mooncake KV Store

First-class support for [Mooncake](https://github.com/kvcache-ai/Mooncake) as the KV transfer backend for prefill-decode disaggregation. A `mooncake-master` entry under `services:` in an SGLang or vLLM recipe makes srtslurm launch and configure the mooncake master automatically and wire up worker env vars so peer-to-peer transfers work across multiple nodes.

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
    # options.store_config: ...                  # vLLM only; rendered into MOONCAKE_CONFIG_PATH
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

vLLM's `MooncakeStoreConnector` reads its configuration from a JSON file pointed to by `MOONCAKE_CONFIG_PATH` rather than directly from env vars, so the vLLM master entry takes an extra `options.store_config:` section that srtslurm renders into that JSON at job start:

```yaml
engine: vllm
roles:
  prefill:
    env:                                  # in-process Mooncake C++ knobs, per role
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
      MC_STORE_CLIENT_METRIC: "1"
    args:
      kv-transfer-config: '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
  decode:
    env:
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
      MC_STORE_CLIENT_METRIC: "1"
    args:
      kv-transfer-config: '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
services:
  - name: mooncake-master
    type: mooncake-master
    options:
      store_config:                       # rendered into the MOONCAKE_CONFIG_PATH JSON
        metadata_server: "P2PHANDSHAKE"
        global_segment_size: "100GB"
        local_buffer_size: "4GB"
        protocol: "rdma"
        device_name: "mlx5_0,mlx5_1"
```

Real-world production form: `MultiConnector` wraps `NixlConnector` (P2P transfer between prefill and decode) **and** `MooncakeStoreConnector` (shared store for cross-instance reuse), both with `kv_role: "kv_both"` so prefill and decode workers run identical connector stacks. srtslurm's validator accepts `MooncakeStoreConnector` standalone or wrapped in `MultiConnector`.

srtslurm stamps `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, `MOONCAKE_LOCAL_HOSTNAME`, and `MOONCAKE_CONFIG_PATH` on every worker; you supply the rest. `master_server_address` in `store_config` is also auto-filled from the infra node IP and ignored if set by hand.

The role `env:` maps are injected on the vLLM workers (not on the standalone `mooncake_master` daemon; the master gets the service entry's own `env`). Use them for in-process Mooncake C++ knobs like `MC_ENABLE_DEST_DEVICE_AFFINITY`, `MC_STORE_CLIENT_METRIC`, `MC_TE_METRIC`.

## What srtslurm Owns vs What You Set

| Concern                                         | Owner     | Notes                                                                                                |
| ----------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| Launching `mooncake_master`                     | srtslurm  | The `mooncake-master` service: runs on the infra node (same node as etcd/nats; `placement.node: dedicated` shares their reserved node) before workers, gated on RPC `8700`, HTTP metadata `8701`, and admin HTTP `8702`. Its log is `service_mooncake-master.out`. |
| `MOONCAKE_MASTER` env var on workers            | srtslurm  | Always computed as `<infra_node_ip>:8700`. User values in role `env` are overridden.                  |
| `MOONCAKE_TE_META_DATA_SERVER` env var          | srtslurm  | Always computed as `http://<infra_node_ip>:8701/metadata`.                                            |
| `MOONCAKE_LOCAL_HOSTNAME` env var               | srtslurm  | Auto-resolved per-worker via `runtime.network_interface`. User can override in role `env` for custom NICs. |
| `MOONCAKE_CONFIG_PATH` (vLLM only)               | srtslurm  | Always points to the JSON file srtslurm renders from `options.store_config:`. Mounted under `/logs` in every worker. |
| `master_server_address` in `store_config` (vLLM)| srtslurm  | Always overridden with `<infra_node_ip>:8700`. User values are ignored.                               |
| `MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE`, etc.    | User      | Set in `roles.prefill.env` and `roles.decode.env`.                                                    |
| `disaggregation-transfer-backend: mooncake`     | User      | (SGLang only) Set in `roles.prefill.args` and `roles.decode.args`. srtslurm validates this is present. |
| `disaggregation-ib-device`                      | User      | (SGLang only) Set in `roles.prefill.args` and `roles.decode.args`. Format: `"mlx5_0,mlx5_1"` or JSON map. |
| `kv-transfer-config`                            | User      | (vLLM only) Set in `roles.prefill.args` and `roles.decode.args` to wire vLLM's `MooncakeStoreConnector`. |

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
  decode:
    env:
      MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
    args:
      kv-transfer-config: '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
services:
  - name: mooncake-master
    type: mooncake-master
    container: ...                       # optional, default: job container
    args: []                             # optional, appended to mooncake_master
    options:
      store_config:                      # required for vLLM; rendered into MOONCAKE_CONFIG_PATH JSON
        metadata_server: "P2PHANDSHAKE"
        global_segment_size: "100GB"
        local_buffer_size: "4GB"
        protocol: "rdma"
        device_name: "mlx5_0,mlx5_1"
```

### Fields

On the `mooncake-master` service entry (the generic service fields such as `placement`, `external`, and `readiness` are in [Services](services.md#configuration-reference)):

- **`container`** (`str`, optional): Container image used for the `mooncake_master` srun. Defaults to the job container if unset. Useful when mooncake needs a different runtime than your worker container.
- **`args`** (`list[str]`, optional): Extra arguments appended to the standalone `mooncake_master` command. Use this for flags supported only by the Mooncake version in the selected container. Do not use it to override the RPC, HTTP metadata, or metrics ports because srtslurm configures worker endpoints and readiness checks from its own port values.
- **`options.store_config`** (vLLM only, `dict[str, Any]`): Pass-through dict rendered as JSON into the file pointed to by `MOONCAKE_CONFIG_PATH`. Keys map 1:1 to vLLM's `MooncakeStoreConfig` dataclass, a mix of `str` (e.g. `protocol`), `int` (e.g. `port`), and human-readable size strings (e.g. `"4GB"`). srtslurm does not default these fields; values like `global_segment_size`, `protocol`, and `device_name` are hardware-specific and silently using a srtslurm-picked default is worse than failing loudly, so set them explicitly. `master_server_address` is auto-filled and any user value is ignored.

On the roles:

- **`roles.<role>.env`** (`dict[str, str]`, optional): Env vars injected on that role's workers.
  - For **SGLang**, keys map directly to mooncake's environment variable names. See the [SGLang server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/environ.py) and [mooncake_store.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py) for the full list.
  - For **vLLM**, this is for in-process Mooncake C++ knobs (`MC_*`) read by the transfer engine / store client. vLLM's connector itself reads configuration from `MOONCAKE_CONFIG_PATH` (the JSON rendered from `options.store_config:`), not from these env vars.
  - Setting `MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER`, or `MOONCAKE_CONFIG_PATH` here is a no-op (srtslurm always wins).

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

Both dash and underscore forms (`disaggregation-transfer-backend`, `disaggregation_transfer_backend`) are accepted. The vLLM check looks for a Mooncake connector in each role's `kv-transfer-config`.

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
