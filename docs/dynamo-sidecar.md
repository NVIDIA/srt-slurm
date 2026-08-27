# Dynamo sidecar engines

`dynamo.engine_mode: sidecar` runs SGLang as its normal Python engine process and starts Dynamo's standalone Rust connector beside it in the same Slurm task. The connector talks to SGLang's native gRPC service over loopback; it does not import `dynamo.sglang` or require Dynamo Python bindings in the engine process.

```yaml
frontend:
  type: dynamo

backend:
  type: sglang

dynamo:
  engine_mode: sidecar
  version: "1.3.0"
```

The first worker that needs a revision builds `dynamo-sglang-sidecar` with Cargo and stores it under `/configs/dynamo-wheels/sidecars/`. Later workers reuse the executable for that Dynamo revision and CPU architecture.

## Dynamo source selection

Sidecar mode resolves one immutable source commit during job setup and supplies it to every frontend and worker launch.

| YAML selection | Frontend install | Sidecar source |
| --- | --- | --- |
| `dynamo.hash` | source-built runtime wheel | the supplied commit |
| `dynamo.version` | `ai-dynamo` and `ai-dynamo-runtime` from PyPI | the verified `v<version>` Dynamo tag |
| `dynamo.top_of_tree: true` | source-built runtime wheel | the SHA resolved from `main` once at job setup |

`dynamo.wheel` is intentionally rejected in sidecar mode because a staged wheel has no trustworthy source-revision mapping. Use `version`, `hash`, or `top_of_tree` instead.

## Scope

This initial path supports SGLang on Slurm, including aggregate and prefill/decode layouts. It needs stock SGLang v0.5.16 or later with the native `--grpc-port` server support. vLLM and TensorRT-LLM have different native engine launch contracts (`vllm-rs` and `tensorrt_llm.commands.serve`, respectively), so srtctl rejects those combinations rather than silently launching their old in-process Dynamo modules. Kubernetes sidecar delivery remains a separate image/artifact-distribution problem.
