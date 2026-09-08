# Examples

Small, runnable starting points, one per frontend and topology. Every example serves the same model (Qwen3-0.6B) on one node so the files differ only in the frontend and the prefill/decode layout, and a full matrix run finishes in minutes. They are not performance claims. Copy one, change the model, GPU type, topology, and engine flags to match your target, then `srtctl dry-run -f <config>` before submitting.

## Matrix

| Backend | Dynamo frontend | Native router | Router-free direct |
| --- | --- | --- | --- |
| SGLang | `sglang/dynamo-agg.yaml`, `sglang/dynamo-disagg.yaml` | `sglang/sglang-router-agg.yaml`, `sglang/sglang-router-disagg.yaml` | |
| vLLM | `vllm/dynamo-agg.yaml`, `vllm/dynamo-disagg.yaml` | `vllm/vllm-router-agg.yaml`, `vllm/vllm-router-disagg.yaml` | `vllm/vllm-direct-agg.yaml` |
| TRT-LLM | `trtllm/dynamo-agg.yaml`, `trtllm/dynamo-disagg.yaml` | `trtllm/trtllm-serve-disagg.yaml` | `trtllm/trtllm-serve-agg.yaml` |
| Mocker | `mocker/dynamo-agg.yaml` | | |

- **Dynamo frontend**: workers register with etcd/NATS and the Dynamo frontend routes (KV-aware here). Dynamo is installed at job start via `dynamo.version` unless the container ships it (`dynamo.install: false`, as the TRT-LLM examples do).
- **Native router**: the engine's own router in front of plain engine workers. No Dynamo, NATS, or etcd. SGLang uses the Model Gateway (`sglang_router`), vLLM the official vLLM Router (`vllm-router`), TRT-LLM `trtllm-serve disaggregated` with a generated `ser.yaml`.
- **Router-free direct**: one worker owns the public port. `frontend.type: vllm` and `frontend.type: trtllm_serve` in aggregate mode launch no router process.
- **Mocker**: `dynamo.mocker` stands in for an engine, so the whole orchestration path runs without loading weights. The fastest way to validate a cluster config.

Aggregated examples run two TP1 workers; disaggregated examples run one TP1 prefill and one TP1 decode worker on the same node (`decode_nodes: 0` places decode on the prefill node's spare GPUs).

Every example is written in the 2.0 layout: `engine:` names the engine (a string, or a mapping with engine-wide knobs), `roles:` holds everything about each worker role, and there is no `backend:` block. `features/override.yaml` is the exception, kept in the v1 layout to show that it still loads.

## Features

| File | Shows |
| --- | --- |
| `features/sweep.yaml` | `sweep:` plus `{placeholder}` substitution; one job per combination |
| `features/override.yaml` | `base` plus `override_*` and `zip_override_*` variants in one file |
| `features/profiling.yaml` | `profiling:` torch capture on an aggregated worker |
| `features/services.yaml` | `services:` sidecar (an HTTP log browser on the head node) with a `readiness:` port gate |
| `features/infra-services.yaml` | etcd and NATS as declared services on a dedicated node with a NATS payload limit; the implied exporters overridden or switched off |
| `features/dynamo-source.yaml` | `dynamo.source:` building Dynamo from a git tag (or a PR head via `--set dynamo.source.rev=refs/pull/<n>/head`), pinned to a commit at submit |

## Cluster aliases

The examples reference two kinds of aliases that `srtslurm.yaml` resolves:

```yaml
model_paths:
  qwen3-0.6b: /path/to/Qwen3-0.6B          # or use path: "hf:Qwen/Qwen3-0.6B" in the recipe

containers:
  sglang: /path/to/sglang.sqsh              # SGLang image; Dynamo examples pip-install ai-dynamo into it
  vllm: /path/to/vllm.sqsh                  # vLLM image with the vllm-router executable
  trtllm: /path/to/tensorrtllm-runtime.sqsh # Dynamo TRT-LLM runtime image (ships ai-dynamo and trtllm-serve)
```

`resources.gpu_type` and `gpus_per_node` are set to `h100` and `8`; change them to match the partition you submit to.

## Validation

CI validates every file under `examples/`, expanding sweep and override files into their variants. Run the same check locally with:

```bash
uv run python -c "
from pathlib import Path
from srtctl.core.config import validate_config_file
for p in sorted(Path('examples').rglob('*.yaml')):
    print(validate_config_file(p) or f'ok {p}')
"
```
