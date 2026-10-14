# GLM-5.3 AgentX Pareto recipes with Dynamo on Vera Rubin

These eight recipes are Dynamo counterparts of
[`../trtllm-serve`](../trtllm-serve/README.md). They launch `dynamo.trtllm`
prefill/decode workers behind `dynamo.frontend`, with Dynamo pinned to
[`1.6.0.dev20261013`](https://pypi.org/project/ai-dynamo/1.6.0.dev20261013/).
Filenames and point names match the native recipes.

The conversion keeps the worker counts, GPU allocations, TRT-LLM engine tuning,
kernel environment, MTP acceptance controls, prefill Mooncake capacity
and inline AIPerf command. The native recipes remain in their original directory.

All recipes set `engine.numa_memory_bind: local` and `engine.numa_cpu_bind: true`
to bind worker memory and CPUs to the task GPU's NUMA node.

| Recipe | Prefill workers / GPUs each | Decode workers / GPUs each | Concurrency | Prefill affinity | Decode affinity |
| --- | --- | --- | --- | --- | --- |
| [1p1d / tp16](glm5.3-disagg-1p1d-tp4-tp16-c1.yaml) | 1 / 4 | 1 / 16 | 1 | session | none |
| [1p2d / tp4](glm5.3-disagg-1p2d-tp4-tp4-c36.yaml) | 1 / 4 | 2 / 4 | 36 | session | none |
| [1p3d / tp8](glm5.3-disagg-1p3d-tp4-tp8-c40.yaml) | 1 / 4 | 3 / 8 | 40 | session | none |
| [1p3d / tp4](glm5.3-disagg-1p3d-tp4-tp4-c64.yaml) | 1 / 4 | 3 / 4 | 64 | session | none |
| [3p1d / dep32](glm5.3-disagg-3p1d-tp4-dep32-c560.yaml) | 3 / 4 | 1 / 32 | 560 | none | none |
| [3p1d / dep16 / c640](glm5.3-disagg-3p1d-tp4-dep16-c640.yaml) | 3 / 4 | 1 / 16 | 640 | none | none |
| [3p1d / dep16 / c576](glm5.3-disagg-3p1d-tp4-dep16-c576.yaml) | 3 / 4 | 1 / 16 | 576 | none | none |
| [3p1d / dep8](glm5.3-disagg-3p1d-tp4-dep8-c560.yaml) | 3 / 4 | 1 / 8 | 560 | none | session |

## Dynamo installation and cluster settings

Every recipe stages and installs the matching `ai-dynamo` and `ai-dynamo-runtime`
nightly wheels through srt-slurm's existing installer:

```yaml
dynamo:
  install: true
  source:
    wheel: 1.6.0.dev20261013
  request_plane: tcp
  event_plane: zmq
```

Keep the custom TRT-LLM image used by the native recipes. Resolve these aliases
in `srtslurm.yaml`:

```yaml
model_paths:
  glm5.3: /models/GLM-5.3-NVFP4
containers:
  trtllm-glm5.3-rubin: /images/trtllm-7890925e.sqsh
gpus_per_node: 4
use_segment_sbatch_directive: true
```

The image must contain tekit revision
`7890925ea10e613cd787c82a406a8fdc088eabc8` and its dependencies for the Rubin
kernels, MTP prefill, Python NIXL transceiver and Mooncake donor. Installing the
base Dynamo packages leaves this custom TRT-LLM build in place; do not install
`ai-dynamo[trtllm]`, which pins a different TRT-LLM release.

srt-slurm supplies etcd discovery, TCP request transport and ZMQ events. These
recipes do not require a NATS service.
Keep the site's account, partition, network interface, cache mounts and other
cluster settings in `srtslurm.yaml`. As with the native recipes, the full
allocation must fit within one NVL72 domain. These RapLab recipes omit the
`sudo nvidia-smi` host clock setup and teardown.

Their top-level `environment` sets `PIP_CONFIG_FILE=/dev/null`,
`PIP_INDEX_URL=https://pypi.org/simple` and
`PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com`. This overrides the image's
`/etc/environment` pip index at `urm.nvidia.com`, which RapLab cannot resolve.

```bash
srtctl dry-run -f recipes/trtllm/vrnvl72/agentic/raplab-dynamo/glm5.3-disagg-3p1d-tp4-dep8-c560.yaml
```

## Routing and parsing

All eight recipes run a single frontend on a dedicated node,
with the benchmark client colocated on that same node. Both set
`placement.node: dedicated`; `benchmark.colocate_with_frontend` defaults to
`true`, so they share one extra node. In the four recipes without Mooncake,
the frontend uses `router-mode: kv` and a 14,400-second session-affinity TTL.
Prefill workers inherit that frontend policy. Decode workers advertise
`DYN_ROUTER_MODE=least-loaded`, without session affinity, overriding the frontend
policy for the decode tier. The existing AIPerf session-ID headers are retained.

These frontends use an etcd lease TTL of 120 seconds, the `fastokens` tokenizer,
an 8,000,000,000-byte tokenizer cache budget, and a 30-second TCP request timeout.
`DYN_ROUTER_USE_KV_EVENTS=0` is this Dynamo version's supported switch for routing
without worker KV events. `OTEL_EXPORT_ENABLED=false` overrides srt-slurm's
frontend tracing environment. Default temperature, cache enablement, JSONL and
unset admission thresholds are omitted, as is the TCP request-plane setting
that srt-slurm already injects.

The four Mooncake recipes keep their existing frontend and worker policies:
`router-mode: least-loaded`, with 3,600-second hard session affinity only on
roles corresponding to the native recipes' `conversation` policy.

Worker roles using session affinity set `DYN_ENGINE_CONV_AFFINITY=1`: prefill
in the four recipes without Mooncake, and decode in the DEP8 Mooncake recipe.
This enables the corresponding TRT-LLM engine conversation-affinity handling.

`DYN_TOOL_CALL_PARSER=glm47` on both worker roles advertises the parser in their
model cards for Dynamo's Rust chat processor. Native `--tool_parser` arguments
are removed. Dynamo reads the KV block size from the initialized engine;
`kv_cache_config.tokens_per_block` remains the source of truth. Both roles set
`enable_iter_perf_stats: true`, `iter_perf_stats_interval: 40` and
`kv_cache_config.iteration_stats_interval: 40`. Engine-level `return_perf_metrics`
is retained from the native recipes; the native frontend's `server_config_extra` settings are
replaced by Dynamo's frontend configuration. `num_postprocess_workers` is
removed because this Dynamo version strips that setting and owns postprocessing.

These options were checked against the published `ai-dynamo` wheel's
`dynamo/common/configuration/groups/router_args.py`, `runtime_args.py`, and
`dynamo/trtllm/workers/llm_worker.py` at the pinned version.

## Mooncake and workload

The four TP decode recipes retain private prefill host caches. The four DEP
recipes retain their Mooncake master and one `trtllm-serve mooncake_donor` per
decode node. All donors within a recipe use its largest original segment size;
the dedicated frontend/client node frees the memory previously reserved on the
first decode node:

| Decode topology | Donor segment size per decode node |
| --- | --- |
| DEP32 | 768GiB |
| DEP16 (c576 and c640) | 1050GiB |
| DEP8 | 1100GiB |

The image's Mooncake Python store wheel is pinned to 0.3.13. With RDMA, it
[discovers NIC NUMA nodes and splits the segment across them](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/mooncake-store/src/real_client.cpp#L886),
then [binds each memory region to its NUMA node](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/mooncake-store/src/utils.cpp#L378).
The donor keeps the complete NIC allowlist and relies on that allocation policy.
This does not impose per-NUMA process CPU binding.

Decode workers remain connector-free; prefill workers receive
`MOONCAKE_CONFIG_PATH` and use the
shared store. The donor remains a separate capacity process even though the
inference workers now use Dynamo. Prefill workers and donors both set
`MC_ENABLE_PARALLEL_REG_MR=0`, `MC_MS_AUTO_DISC=0`,
`MC_TE_FILTERS=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11` and
`MC_ENABLE_DEST_DEVICE_AFFINITY=1` in their process environments so Mooncake's
in-process transport receives them. The donor also sets `TLLM_LOG_LEVEL=INFO`
and `PYTHONUNBUFFERED=1` for logging; inference tuning stays in the worker
environments. Other store settings come from the config and CLI.

The master starts before workers. Donors use `start: after_frontend`: Dynamo's
worker and frontend launches return before model readiness, allowing donor
registration to overlap worker loading. Donors publish
`/logs/mooncake_donor_{index}.ready`; the `readiness.file` gate waits for them
before benchmarking. Donors now share the workers' shutdown tier, while the
master shuts down afterward. This requires srt-slurm's
[file-readiness support](https://github.com/NVIDIA/srt-slurm/pull/608).
See the [native Mooncake notes](../trtllm-serve/README.md#mooncake) for prefill
segment sizes and other inherited pool settings.

Each benchmark still installs AIPerf commit
`b7b16cf851885567988a643282266bce74e34437` in a temporary environment and invokes
`aiperf profile` directly against the discovered frontend. The AgentX scenario,
dataset, concurrency, session headers, warmup and 3,600-second measurement are
unchanged. Results remain under `/logs/agentic/aiperf_results`.

The benchmark client passes the runtime-provided `AIPERF_SERVER_METRICS_URLS`
list to `aiperf profile --server-metrics`, collecting each logical prefill and
decode worker's `/metrics` endpoint on its `DYN_SYSTEM_PORT`. Worker metrics
publication is enabled by default. The command requires that endpoint list
and exports JSON, CSV, Parquet and JSONL under the same artifact directory;
JSONL retains individual scrape snapshots, with `endpoint_url` identifying the
worker. See the [custom benchmark endpoint variables](../../../../../docs/benchmarks.md#custom).

For every actual cluster run, write `<run_dir>/logs/perf-analysis.md` following
[`docs/perf-analysis.md`](../../../../../docs/perf-analysis.md).
