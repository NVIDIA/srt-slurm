# Dynamo sidecar engines

`dynamo.engine_mode: sidecar` runs SGLang as its normal Python engine process and starts Dynamo's standalone Rust connector beside it in the same Slurm task or direct-host worker. The connector talks to SGLang's native gRPC service over loopback; it does not import `dynamo.sglang` or require Dynamo Python bindings in the engine process.

```yaml
frontend:
  type: dynamo

backend:
  type: sglang

dynamo:
  engine_mode: sidecar
  version: "1.3.0"
```

The first worker that needs a revision builds `dynamo-sglang-sidecar` with Cargo. Slurm stores it under `/configs/dynamo-wheels/sidecars/`; `srtctl apply --bash` stores it under `<output-base>/.srtctl-cache/dynamo-wheels/sidecars/`. Later workers reuse the executable for that Dynamo revision and CPU architecture.

## Dynamo source selection

The Dynamo frontend keeps its existing installation behavior. Worker-side builds use the configured source directly; mutable refs are resolved inside the sidecar build and cached by the resulting commit.

| YAML selection | Frontend install | Sidecar source |
| --- | --- | --- |
| `dynamo.hash` | source-built runtime wheel | the supplied commit |
| `dynamo.version` | `ai-dynamo` and `ai-dynamo-runtime` from PyPI | the matching `v<version>` Dynamo tag |
| `dynamo.top_of_tree: true` | source-built runtime wheel | `main`, resolved in the worker build |

`dynamo.wheel` is intentionally rejected in sidecar mode because a staged wheel has no trustworthy source-revision mapping. Use `version`, `hash`, or `top_of_tree` instead.

## Scope

This initial path supports SGLang on Slurm, including aggregate and prefill/decode layouts, plus the single-node `srtctl apply --bash` lifecycle. Direct Bash retains its existing `dynamo.hash` or `dynamo.top_of_tree: true` source requirement. It needs stock SGLang v0.5.16 or later with the native `--grpc-port` server support. vLLM and TensorRT-LLM have different native engine launch contracts (`vllm-rs` and `tensorrt_llm.commands.serve`, respectively), so srtctl rejects those combinations rather than silently launching their old in-process Dynamo modules. Kubernetes sidecar delivery remains a separate image/artifact-distribution problem.
