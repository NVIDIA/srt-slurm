# Router comparison: three equivalent aggregate SGLang deployments

Three recipes serve the same model, on the same node, over the same two
aggregate TP4 SGLang workers, under the same AgentX AIPerf workload. The only
declared differences are how requests are routed.

| Arm | Recipe | Router |
| --- | --- | --- |
| 1 | [`arm1-sgl-router.yaml`](arm1-sgl-router.yaml) | Experimental Rust `sgl-router`, `cache_aware_zmq` |
| 2 | [`arm2-dynamo-sidecar.yaml`](arm2-dynamo-sidecar.yaml) | Dynamo frontend, stock KV router, native SGLang sidecars |
| 3 | [`arm3-dynamo-sgl-policy.yaml`](arm3-dynamo-sgl-policy.yaml) | Arm 2 plus the external `sgl-router-cache-aware` selection policy |

## Required environment

```bash
export ROUTER_ARM_MODEL_PATH=/absolute/path/to/Qwen3-32B-FP8
export ROUTER_ARM_SGLANG_SOURCE=/absolute/path/to/sglang
export ROUTER_ARM_TRACE=/absolute/path/to/agentx_trace.jsonl
```

`ROUTER_ARM_SGLANG_SOURCE` must contain `experimental/sgl-router`; arm 1 builds
that crate from source rather than consuming a prebuilt binary.

## Run

```bash
make setup ARCH="$(uname -m)"
uv sync --extra ruter

# SLURM
uv run srtctl apply -f recipes/qwen3-32b/router-comparison/arm1-sgl-router.yaml

# One GPU host
uv run srtctl apply -f recipes/qwen3-32b/router-comparison/arm1-sgl-router.yaml --bash > arm1.sh
bash arm1.sh
```

See [../../../docs/router-comparison.md](../../../docs/router-comparison.md) for
the field-by-field difference between the arms.
