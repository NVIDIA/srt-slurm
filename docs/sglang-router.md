# SGLang Router Mode

This page explains the sglang router mode for prefill-decode (PD) disaggregation, an alternative to the default Dynamo frontend architecture.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
  - [Router Arguments](#router-arguments)
  - [Frontend Environment Variables](#frontend-environment-variables)
- [Architecture Modes](#architecture-modes)
  - [Single Router](#single-router-enable_multiple_frontends-false)
  - [Multiple Routers](#multiple-routers-enable_multiple_frontends-true-default)
- [How Router Distribution Works](#how-router-distribution-works)
- [Port Configuration](#port-configuration)
  - [Bootstrap Port](#bootstrap-port)
  - [Server Port](#server-port)
- [Complete Example](#complete-example)
- [Troubleshooting](#troubleshooting)
- [Comparison with Dynamo](#comparison-with-dynamo)

---

## Overview

By default, srtctl uses **Dynamo frontends** to coordinate between prefill and decode workers. This requires NATS/ETCD infrastructure and the `dynamo` package.

**SGLang Router** is an alternative that uses sglang's native `sglang_router` for PD disaggregation.

| Feature        | Dynamo Frontends           | SGLang Router              |
| -------------- | -------------------------- | -------------------------- |
| Infrastructure | NATS + ETCD + dynamo       | sglang_router only         |
| Routing        | Dynamo's coordination      | sglang's native PD routing |
| Scaling        | nginx + multiple frontends | nginx + multiple routers   |

## Configuration

Enable sglang router in your recipe's `frontend` section:

```yaml
frontend:
  type: sglang
```

That's it. The workers will launch with `sglang.launch_server` instead of `dynamo.sglang`, and the router will handle request distribution.

### Router Arguments

Pass extra CLI args to the router:

```yaml
frontend:
  type: sglang
  args:
    kv-overlap-score-weight: 1
    router-temperature: 0
    no-kv-events: true # boolean flags (no value)
    router-ttl: 120.0
```

For dynamo frontend, use the same `args` field:

```yaml
frontend:
  type: dynamo
  args:
    router-mode: "kv"
    router-reset-states: true
```

### Frontend Environment Variables

Pass environment variables to frontend processes:

```yaml
frontend:
  type: sglang
  env:
    MY_CUSTOM_VAR: "value"
```

## Architecture Modes

### Single Router (`enable_multiple_frontends: false`)

The simplest mode - one router on node 0, no nginx:

```yaml
frontend:
  type: sglang
  enable_multiple_frontends: false
```

```
┌─────────────────────────────────────────────────────────┐
│  Node 0                                                 │
│  ┌──────────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  sglang-router   │  │   Prefill   │  │   Decode   │ │
│  │    :8000         │──│   Worker    │──│   Worker   │ │
│  └──────────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

- Router directly on port 8000
- Good for testing or small deployments
- No load balancing overhead

### Multiple Routers (`enable_multiple_frontends: true`, default)

Nginx load balances across multiple router instances:

```yaml
frontend:
  type: sglang
  enable_multiple_frontends: true # default
  num_additional_frontends: 9 # default, total = 1 + 9 = 10 routers
```

```
┌──────────────────────────────────────────────────────────────────────┐
│  Node 0                               Node 1          Node 2         │
│  ┌─────────┐  ┌────────────────┐     ┌──────────┐    ┌──────────┐   │
│  │  nginx  │  │ sglang-router  │     │ sglang-  │    │ sglang-  │   │
│  │  :8000  │──│    :30080      │     │ router   │    │ router   │   │
│  └────┬────┘  └────────────────┘     │ :30080   │    │ :30080   │   │
│       │                               └──────────┘    └──────────┘   │
│       └──────────────────────────────────┴───────────────┴───────────┘
│                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Prefill   │  │   Prefill   │  │   Decode    │  │   Decode    │  │
│  │   Worker 0  │  │   Worker 1  │  │   Worker 0  │  │   Worker 1  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

- nginx on node 0 listens on port 8000 (public)
- Routers listen on port 30080 (internal)
- nginx round-robins requests to routers
- Routers distributed across nodes using same logic as Dynamo frontends

If the cluster rejects raising open-file limits inside the nginx container, keep the default (`frontend.nginx_raise_ulimit: false` or unset). If you need the previous high-nofile behavior, set `frontend.nginx_raise_ulimit: true` or a cluster default in `srtslurm.yaml` — see [Configuration Reference](config-reference.md#frontend).

## How Router Distribution Works

The `num_additional_frontends` setting controls how many additional routers spawn beyond the first:

| Setting                       | Total Routers | Distribution                     |
| ----------------------------- | ------------- | -------------------------------- |
| `num_additional_frontends: 0` | 1             | Node 0 only                      |
| `num_additional_frontends: 4` | 5             | Node 0 + 4 distributed           |
| `num_additional_frontends: 9` | 10            | Node 0 + 9 distributed (default) |

Routers are distributed across available nodes using ceiling division:

```
nodes_per_router = ceil((total_nodes - 1) / num_additional_frontends)
```

## Port Configuration

### Bootstrap Port

The sglang router needs the **disaggregation bootstrap port** to connect to prefill workers. This must match the `disaggregation-bootstrap-port` in your sglang config:

```yaml
backend:
  sglang_config:
    prefill:
      disaggregation-bootstrap-port: 30001 # Must match
      # ... other config
    decode:
      disaggregation-bootstrap-port: 30001 # Must match
      # ... other config
```

The default bootstrap port is `30001` (matching most recipes). If you use a different port, ensure it's consistent across prefill and decode configs.

### Server Port

Workers listen on port `30000` by default. This is standard sglang behavior and doesn't need configuration.

## Complete Example

Here's a full recipe using sglang router:

```yaml
name: "deepseek-r1-sglang-router"

model:
  path: "deepseek-r1-fp4"
  container: "sglang-latest"
  precision: "fp4"

resources:
  gpu_type: "gb300"
  gpus_per_node: 4
  prefill_nodes: 2
  prefill_workers: 2
  decode_nodes: 2
  decode_workers: 2

frontend:
  type: sglang
  enable_multiple_frontends: true
  num_additional_frontends: 3 # 4 total routers

backend:
  sglang_config:
    prefill:
      model-path: /model/
      tensor-parallel-size: 4
      disaggregation-mode: prefill
      disaggregation-bootstrap-port: 30001
      disaggregation-transfer-backend: nixl
      # ... other prefill settings

    decode:
      model-path: /model/
      tensor-parallel-size: 4
      disaggregation-mode: decode
      disaggregation-bootstrap-port: 30001
      disaggregation-transfer-backend: nixl
      # ... other decode settings

benchmark:
  type: "sa-bench"
  isl: 128000
  osl: 8000
  concurrencies: "16x32"
```

## Troubleshooting

### Port Conflicts

If you see `bind() to 0.0.0.0:8000 failed (Address already in use)`:

- This means nginx and a router are both trying to use port 8000
- Ensure you're using the latest template (routers use port 30080 internally)

### Router Not Connecting to Workers

Check that:

1. `disaggregation-bootstrap-port` matches in prefill/decode configs
2. Workers are fully started before router tries to connect
3. Network connectivity between router and worker nodes

### Benchmark Can't Reach Endpoint

The benchmark connects to `http://<node0>:8000`. Ensure:

- nginx is running (if `enable_multiple_frontends: true`)
- Router is running (if `enable_multiple_frontends: false`)
- Port 8000 is accessible

## Comparison with Dynamo

| Aspect         | Dynamo Frontends                    | SGLang Router            |
| -------------- | ----------------------------------- | ------------------------ |
| **Startup**    | Slower (NATS/ETCD + dynamo install) | Faster (just sglang)     |
| **Complexity** | More moving parts                   | Simpler                  |
| **Maturity**   | Production-tested                   | Newer                    |
| **Config**     | Via dynamo.sglang                   | Via sglang.launch_server |
| **Scaling**    | Same nginx approach                 | Same nginx approach      |

Both modes support the same `enable_multiple_frontends` and `num_additional_frontends` settings for horizontal scaling.
