# Router comparison: three arms, one topology

Three recipes in [`recipes/qwen3-32b/router-comparison/`](../recipes/qwen3-32b/router-comparison/)
serve one model over the same two aggregate TP4 SGLang workers on one 8×H100
host, under the same AgentX AIPerf workload. Everything except routing is
identical, and every difference is declared in YAML.

## What is shared

| Setting | Value | Where |
| --- | --- | --- |
| Model | `Qwen/Qwen3-32B-FP8` | `model.path`, `backend.sglang_config.aggregated.served-model-name` |
| Topology | 1 node, 2 aggregate workers, 4 GPUs each | `resources.agg_nodes`, `resources.agg_workers`, `resources.gpus_per_agg` |
| Engine flags | page size 64, TP4, metrics on, 40960 context | `backend.sglang_config.aggregated` |
| KV events | SGLang ZMQ publisher on every worker | `backend.kv_events_config.aggregated: true` |
| Telemetry | Tachometer scrape of router and workers | `observability.tachometer.enabled` |
| Workload | AgentX trace replayed by AIPerf on a fixed schedule | `benchmark.type: custom` + `benchmark.command` |

## What differs

Exactly these fields, and nothing else:

| Field | Arm 1 — raw experimental router | Arm 2 — stock Dynamo | Arm 3 — Dynamo + SGL policy |
| --- | --- | --- | --- |
| `frontend.type` | `sgl-router` | `dynamo` | `dynamo` |
| `frontend.source` | `$ROUTER_ARM_SGLANG_SOURCE/experimental/sgl-router` | — | — |
| `frontend.args` | `policy: cache_aware_zmq` | `router-mode: kv` | `router-mode: kv` |
| `dynamo.install` | `false` | (default `true`) | (default `true`) |
| `dynamo.hash` | — | `4dca1626…` | `4dca1626…` |
| `dynamo.event_plane` | — | `zmq` | `zmq` |
| `dynamo.sidecar` | — | `true` | `true` |
| `dynamo.policy_catalog` | — | — | `sgl-router-cache-aware-dynamo-policy` @ `411b4648…` |

Everything else follows from those fields:

- `frontend.type: sgl-router` switches the workers from `dynamo.sglang` to
  `sglang.launch_server`, drops etcd/NATS and the Dynamo install entirely, and
  moves Tachometer's worker scrape from the Dynamo system port to the serving
  port. Readiness gates on each worker's `/health` and then the router's
  `/readyz`.
- `dynamo.sidecar: true` replaces the Python Dynamo worker with a native
  `sglang.launch_server` engine plus `dynamo.sglang.sidecar`, coupled so the
  sidecar starts only once the engine's gRPC port binds and either process
  exiting stops the other. Co-located workers get deterministic gRPC port
  offsets from `dynamo.sidecar_port`.
- `dynamo.policy_catalog` builds the external catalog crate against the same
  `dynamo.hash`, links it through Dynamo's `custom-policy` cargo feature, and
  passes the crate's own YAML to `--router-policy-config`. See
  [Configuration Reference](config-reference.md#external-worker-selection-policy-catalog).

Arms 2 and 3 render byte-identical worker commands; only the frontend's
`--router-policy-config` and the wheel cache key differ.

## Run

```bash
export ROUTER_ARM_MODEL_PATH=/absolute/path/to/Qwen3-32B-FP8
export ROUTER_ARM_SGLANG_SOURCE=/absolute/path/to/sglang
export ROUTER_ARM_TRACE=/absolute/path/to/agentx_trace.jsonl

make setup ARCH="$(uname -m)"
uv sync --extra ruter

ARM=recipes/qwen3-32b/router-comparison/arm1-sgl-router.yaml

# Inspect
uv run srtctl dry-run -f "$ARM"

# SLURM
uv run srtctl apply -f "$ARM"

# One GPU host
uv run srtctl apply -f "$ARM" --bash > /tmp/arm1.sh
bash -n /tmp/arm1.sh
bash /tmp/arm1.sh
```

Arm 1 requires `experimental/sgl-router` inside `ROUTER_ARM_SGLANG_SOURCE`; both
lifecycles build it with cargo rather than consuming a prebuilt binary.
