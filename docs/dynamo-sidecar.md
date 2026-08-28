# Dynamo sidecar engines

`dynamo.engine_mode: sidecar` runs the selected backend through its native engine launcher and starts the matching Dynamo connector beside it in the same Slurm task. The connector talks to the engine's native gRPC service over loopback instead of using the legacy `dynamo.sglang`, `dynamo.vllm`, or `dynamo.trtllm` Python worker.

```yaml
frontend:
  type: dynamo

backend:
  type: vllm

dynamo:
  engine_mode: sidecar
  wheel: "1.5.0.dev20260828"
```

Sidecars are included in `ai-dynamo` and launched as `python3 -m dynamo.<backend>.sidecar`. srtctl installs Dynamo through the normal `version`, `wheel`, `hash`, or `top_of_tree` path; there is no separate Cargo build or sidecar cache. Set `dynamo.install: false` when the container already includes an `ai-dynamo` build with the required launcher.

## Dynamo source selection

The Dynamo frontend and sidecars use the same selected package or source checkout.

| YAML selection | Frontend and sidecar install |
| --- | --- |
| `dynamo.wheel` | staged `ai-dynamo` and `ai-dynamo-runtime` wheels |
| `dynamo.version` | matching `ai-dynamo` and `ai-dynamo-runtime` packages from PyPI |
| `dynamo.hash` | source-built runtime wheel plus the supplied Python checkout |
| `dynamo.top_of_tree: true` | source-built runtime wheel plus the current `main` checkout |

The selected Dynamo build must contain the sidecar launcher for the configured backend. The first nightly expected to contain all three launchers is `1.5.0.dev20260828`.

## Scope

Slurm supports SGLang and vLLM in aggregate and prefill/decode layouts. TensorRT-LLM currently supports aggregate sidecar workers. The native engine contracts are `sglang.launch_server`, `vllm-rs serve`, and `tensorrt_llm.commands.serve`, respectively. SGLang needs native `--grpc-port` support, and the vLLM and TensorRT-LLM containers must include their native launchers.

The single-node `srtctl apply --bash` lifecycle remains SGLang-only and launches `dynamo.sglang.sidecar` from the installed source checkout. Kubernetes sidecar delivery remains a separate image/artifact-distribution problem.
