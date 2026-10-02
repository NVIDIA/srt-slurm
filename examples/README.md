# Examples

Small, runnable starting points, one per frontend and topology. Every example serves the same model (Qwen3-0.6B) on one node so the files differ only in the frontend and the prefill/decode layout, and a full matrix run finishes in minutes. They are not performance claims. Copy one, change the model, GPU type, topology, and engine flags to match your target, then `srtctl dry-run -f <config>` before submitting.

## Matrix

| Backend | Dynamo frontend | Native router | Router-free direct |
| --- | --- | --- | --- |
| SGLang | `sglang/dynamo-agg.yaml`, `sglang/dynamo-disagg.yaml` | `sglang/sglang-router-agg.yaml`, `sglang/sglang-router-disagg.yaml` | `sglang/sglang-direct-agg.yaml` |
| vLLM | `vllm/dynamo-agg.yaml`, `vllm/dynamo-disagg.yaml` | `vllm/vllm-router-agg.yaml`, `vllm/vllm-router-disagg.yaml`, `vllm/vllm-router-moriio-disagg.yaml` (ROCm, MoRI-IO discovery) | `vllm/vllm-direct-agg.yaml` |
| TRT-LLM | `trtllm/dynamo-agg.yaml`, `trtllm/dynamo-disagg.yaml` | `trtllm/trtllm-serve-disagg.yaml` | `trtllm/trtllm-serve-agg.yaml` |
| Mocker | `mocker/dynamo-agg.yaml` | | |

- **Dynamo frontend**: workers register with etcd and the Dynamo frontend routes (KV-aware here); the request plane is tcp and NATS is not started unless a plane asks for it. Dynamo is installed at job start via `dynamo.source` (`pypi:` here) unless the container ships it (`dynamo.install: false`, as the TRT-LLM examples do).
- **Native router**: the engine's own router in front of plain engine workers. No Dynamo, NATS, or etcd. SGLang uses the Model Gateway (`frontend.type: sglang-router`), vLLM the official vLLM Router (`vllm-router`), TRT-LLM `trtllm-serve disaggregated` with a generated `ser.yaml`.
- **Router-free direct**: one worker owns the public port. `frontend.type: sglang`, `frontend.type: vllm`, and `frontend.type: trtllm_serve` in aggregate mode launch no router process.
- **Mocker**: `dynamo.mocker` stands in for an engine, so the whole orchestration path runs without loading weights. The fastest way to validate a cluster config.

Aggregated examples run two TP1 workers; disaggregated examples run one TP1 prefill and one TP1 decode worker on the same node (`decode.nodes: colocate` places decode on the prefill node's spare GPUs).

Every example is written in the 2.0 layout: `engine:` names the engine (a string, or a mapping with engine-wide knobs), `roles:` holds everything about each worker role, and there is no `backend:` block. The v1 layout is documented in [../docs/legacy-v1.md](../docs/legacy-v1.md); `srtctl migrate -f <recipe> --in-place` rewrites a v1 recipe.

## Features

| File | Shows |
| --- | --- |
| `features/sweep.yaml` | `sweep:` plus `{placeholder}` substitution; one job per combination |
| `features/override.yaml` | `base` plus `override_*` and `zip_override_*` variants in one file |
| `features/profiling.yaml` | `profiling:` torch capture on an aggregated worker |
| [`features/amd-power-telemetry.yaml`](features/amd-power-telemetry.yaml) | SGLang on MI300X with AMD device-metrics-exporter power collection. Set the `sglang-rocm` container alias and `visible_devices_env: ROCR_VISIBLE_DEVICES` in `srtslurm.yaml`. See [power telemetry](../docs/power-telemetry.md#amd-amd-device-metrics) |
| `features/services.yaml` | `services:` sidecar (an HTTP log browser on the head node) with a `readiness:` port gate |
| `features/mlperf-client.yaml` | `benchmark.type: custom` driving the MLPerf inference-endpoint client in its own image; placeholder paths, a reference rather than a runnable example |
| `features/infra-services.yaml` | etcd and NATS as declared services on a dedicated node with a NATS payload limit; the implied exporters overridden or switched off |
| `features/dynamo-source.yaml` | `dynamo.source:` building Dynamo from a git tag (or a PR head via `--set dynamo.source.rev=refs/pull/<n>/head`), pinned to a commit at submit |
| `features/node-hooks.yaml` | `host_setup:` pre-run and post-run commands on each node's bare host, driven by `configs/node-hooks.sh`: arbitrary shell commands, one per line, from `HOOK_PRE` / `HOOK_POST` block scalars in the recipe environment, with per-command logging and a state snapshot |
| `features/lmcache-server.yaml` | `services[].type: lmcache-server` plus `connector: lmcache-mp`: an LMCache DRAM tier, one server per worker node started before the workers and gated on its healthcheck; LMCache flags go under `args`. Needs a vLLM image that ships LMCache (the `vllm-lmcache` alias). See [../docs/services.md](../docs/services.md#service-types) |
| `features/lmcache-server-disagg.yaml` | The same tier on the prefill side only: the server placed on prefill nodes and a hand-written `MultiConnector` (NIXL plus LMCache MP) in the prefill args, which srtctl passes through untouched |
| `features/lmcache-server-sglang.yaml` | The same server next to aggregated SGLang workers: SGLang's `enable-lmcache` flag, with srtctl pointing each worker at the server on its node (`LMCACHE_MP_HOST`/`LMCACHE_MP_PORT`). Needs an SGLang image that ships LMCache (the `sglang-lmcache` alias) |
| `features/vllm-failover.yaml` | `engine.failover:` shadow engine recovery: a GPU Memory Service sidecar and a parked standby engine per vLLM worker, relaunched in place after a crash. Needs a container that ships `gpu_memory_service` (the `dynamo-vllm` alias, an `nvcr.io/nvidia/ai-dynamo/vllm-runtime` image). See [../docs/shadow-engine-recovery.md](../docs/shadow-engine-recovery.md) |
| `features/worker-restart.yaml` | `roles.<role>.restart:` relaunching a killed worker in place with backoff; serves with `benchmark.type: manual` so you can SIGKILL a worker step and watch the sweep log |
| `features/sglang-weight-cache.yaml` | SGLang fast engine recovery on the existing machinery: a generic per-worker service runs SGLang's weight cache daemon (tensors kept in HBM, handed out over CUDA IPC), the engines start with `--weight-cache-mode client` and skip the disk load, and `roles.<role>.restart` relaunches a killed engine in place. Needs SGLang v0.5.19+; see [docs/sglang-weight-cache.md](../docs/sglang-weight-cache.md) |

## Cluster aliases

The examples reference two kinds of aliases that `srtslurm.yaml` resolves:

```yaml
model_paths:
  qwen3-0.6b: /path/to/Qwen3-0.6B          # or use path: "hf:Qwen/Qwen3-0.6B" in the recipe

containers:
  sglang: /path/to/sglang.sqsh              # SGLang image; Dynamo examples pip-install ai-dynamo into it
  sglang-rocm: /path/to/sglang-rocm.sqsh    # SGLang ROCm image, for features/amd-power-telemetry.yaml
  vllm: /path/to/vllm.sqsh                  # vLLM image with the vllm-router executable
  trtllm: /path/to/tensorrtllm-runtime.sqsh # Dynamo TRT-LLM runtime image (ships ai-dynamo and trtllm-serve)
  dynamo-vllm: /path/to/vllm-runtime.sqsh   # Dynamo vLLM runtime image (ships ai-dynamo and gpu_memory_service), for features/vllm-failover.yaml
  vllm-lmcache: /path/to/vllm-lmcache.sqsh  # vLLM image with LMCache installed, for features/lmcache-server.yaml and -disagg.yaml
  sglang-lmcache: /path/to/sglang-lmcache.sqsh  # SGLang image with LMCache installed, for features/lmcache-server-sglang.yaml
```

`resources.gpu_type` and `gpus_per_node` are set to `h100` and `8`; change them to match the partition you submit to.

## Optional GPU temperature

To retain temperature with the existing power collector, add this telemetry block
to a NVIDIA example and resolve `dcgm-exporter` to the exporter image in your
cluster configuration:

```yaml
telemetry:
  enabled: true
  provider: dcgm-power
  required: false
  dcgm_exporter:
    container_image: dcgm-exporter
```

The same exporter response supplies `DCGM_FI_DEV_POWER_USAGE` and optional
`DCGM_FI_DEV_GPU_TEMP`. Samples v3 retain Celsius in `temperature_c`; an exporter
without that metric leaves the cell empty. No extra sampler is launched.
Temperature does not establish a valid energy measurement: that still requires
benchmark measurement windows and power validation. Deploy v3-compatible readers
before upgrading the collector. See [power telemetry](../docs/power-telemetry.md).

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

### TileRT

[`tilert/glm5-disagg.yaml`](tilert/glm5-disagg.yaml) (B200, NIXL) and
[`tilert/glm5-rocm-mooncake-disagg.yaml`](tilert/glm5-rocm-mooncake-disagg.yaml) (MI355X, Mooncake)
use vLLM prefill and TileRT decode on 8-GPU nodes. See [TileRT setup](../docs/tilert.md)
for image and weight requirements.

### ATOM and AToMesh

[`atom/atomesh-disagg.yaml`](atom/atomesh-disagg.yaml) launches native ATOM prefill
and decode workers with the AToMesh router. Each logical worker fits on one node;
the cluster configuration selects ROCm device visibility.
