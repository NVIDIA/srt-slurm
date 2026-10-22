# GLM-5.3 AgentX Pareto recipes with Dynamo on Vera Rubin

These eight recipes are Dynamo counterparts of
[`../trtllm-serve`](../trtllm-serve/README.md). They launch `dynamo.trtllm`
prefill/decode workers behind `dynamo.frontend`, using locally supplied Dynamo wheels.
Filenames and point names match the native recipes.

The conversion keeps the worker counts, GPU allocations, TRT-LLM engine tuning,
kernel environment, MTP acceptance controls, prefill Mooncake capacity
and inline AIPerf command. The native recipes remain in their original directory.

TRT-LLM engine settings, per-role arguments and TRT-LLM environment tuning match
the corresponding RapLab Dynamo recipes. Hecate relies on its default
`NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=0` and retains its Lustre paths,
cache and wheel locations, host setup and network configuration, including
`UCX_NET_DEVICES=all`. The Mooncake recipes omit destination-device affinity
overrides and additional Transfer Engine device filters. DEP8 also omits the
Store's `device_name` NIC list; the other Mooncake recipes retain that list.

Seven recipes set `engine.numa_memory_bind: local` and `engine.numa_cpu_bind: true`
to bind worker memory and CPUs to the task GPU's NUMA node. The DEP8 recipe omits
these overrides while investigating host OOMs; it uses the default memory binding
across NUMA nodes 0 and 1 without srt-slurm's GPU-local CPU-binding wrapper.

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

Every recipe mounts the local Dynamo wheel directory at `/dynamo_wheels` and
installs its wheels through `configs/install_dynamo_wheel.sh`. The built-in
Dynamo installer is disabled:

```yaml
dynamo:
  install: false
  request_plane: tcp
  event_plane: zmq
setup_script: install_dynamo_wheel.sh
reuse_container: true
extra_mount:
- /lustre:/lustre
- /lustre/fsw/coreai_comparch_inferencex/rihuo/dynamo_wheels:/dynamo_wheels
```

Provide compatible `ai-dynamo` and `ai-dynamo-runtime` wheels in that directory.

All eight recipes enable `reuse_container: true` to reduce duplicate image
extractions on each node. Workers and colocated services using the same image
and root-remapping mode share a job-scoped Pyxis root filesystem. This flag
defaults to `false` elsewhere. Filesystem changes, including wheel installations,
are shared, so setup scripts must be idempotent and serialize concurrent package
installs. Model, host KV cache and Mooncake pool memory remain separate costs.
See [container filesystem memory](../../../../../docs/slurm-faq.md#host-memory-used-by-container-filesystems).

Every recipe specifies the model and custom TRT-LLM image paths directly:

```yaml
model:
  path: /lustre/fsw/coreai_comparch_inferencex/rihuo/GLM-5.3-NVFP4
  container: /lustre/fsw/coreai_comparch_inferencex/rihuo/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh
```

Keep GPU and segment settings in `srtslurm.yaml`:

```yaml
gpus_per_node: 4
use_segment_sbatch_directive: true
```

Every recipe uses `host_setup` on worker nodes to lock GPU memory clocks to
4752 MHz with `sudo -n nvidia-smi -lmc 4752,4752` and reset them during teardown
with `sudo -n nvidia-smi -rmc`. Setup failures abort the run
(`ignore_failure: false`).

The image must support the Rubin kernels, MTP prefill, Python NIXL transceiver
and Mooncake donor, with their required dependencies. Installing the
base Dynamo packages leaves this custom TRT-LLM build in place; do not install
`ai-dynamo[trtllm]`, which pins a different TRT-LLM release.

srt-slurm supplies etcd discovery, TCP request transport and ZMQ events. These
recipes do not require a NATS service.
Keep the site's account, partition, network interface, cache mounts and other
cluster settings in `srtslurm.yaml`. As with the native recipes, the full
allocation must fit within one NVL72 domain.

```bash
srtctl dry-run -f recipes/trtllm/vrnvl72/agentic/dynamo/glm5.3-disagg-3p1d-tp4-dep8-c560.yaml
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
The local Dynamo wheels must support `DYN_ROUTER_USE_KV_EVENTS=0` for routing
without worker KV events. All eight recipes set `observability.enable_otel: false`
and disable the `otel-collector` service, suppressing the default lifecycle collector
and tracing environment. The `node-exporter`, `process-exporter` and `dcgm-exporter`
services are also disabled for now. Default temperature, cache enablement, JSONL and
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

These options were originally checked against the published `ai-dynamo`
`1.6.0.dev20261013` wheel's
`dynamo/common/configuration/groups/router_args.py`, `runtime_args.py`, and
`dynamo/trtllm/workers/llm_worker.py`. The locally supplied wheels must retain
support for these options.

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

Decode workers remain connector-free; prefill workers receive
`MOONCAKE_CONFIG_PATH` and use the
shared store. The donor remains a separate capacity process even though the
inference workers now use Dynamo. All recipes set `MC_ENABLE_PARALLEL_REG_MR=0`
in the shared worker environment and explicitly in each donor's environment,
so the setting reaches prefill, decode and donor processes. Donor logging uses
`TLLM_LOG_LEVEL=INFO` and `PYTHONUNBUFFERED=1`; inference tuning stays in the
worker environments. Store settings come from the config and CLI.

The master and donors start before workers. Donors use `start: before_workers`
and publish `/logs/mooncake_donor_{index}.ready`; the `readiness.file` gate waits
for these files before worker launch. Workers shut down before the donors and
master. This requires srt-slurm's
[file-readiness support](https://github.com/NVIDIA/srt-slurm/pull/608).
See the [native Mooncake notes](../trtllm-serve/README.md#mooncake) for prefill
segment sizes and other inherited pool settings.

Each recipe mounts `/lustre:/lustre`. The benchmark installs AIPerf
from `/lustre/fsw/coreai_comparch_inferencex/rihuo/agentx-harness` in a temporary environment and invokes
`aiperf profile` directly against the discovered frontend. The AgentX scenario,
dataset, concurrency, session headers, warmup and 3,600-second measurement are
unchanged. Results remain under `/logs/agentic/aiperf_results`.

Prefill and decode workers set both `FLASHINFER_CACHE_DIR` and
`FLASHINFER_WORKSPACE_BASE` to
`/lustre/fsw/coreai_comparch_inferencex/rihuo/flashinfer-cache`. The benchmark
sets `AIPERF_DATASET_MMAP_CACHE_DIR` to
`/lustre/fsw/coreai_comparch_inferencex/rihuo/aiperf_mmap`, keeping both caches
on the mounted Lustre filesystem.

The benchmark client passes the runtime-provided `AIPERF_SERVER_METRICS_URLS`
list to `aiperf profile --server-metrics`, collecting each logical prefill and
decode worker's `/metrics` endpoint on its `DYN_SYSTEM_PORT`. Worker metrics
publication is enabled by default. The command requires that endpoint list
and exports JSON, CSV, Parquet and JSONL under the same artifact directory;
JSONL retains individual scrape snapshots, with `endpoint_url` identifying the
worker. See the [custom benchmark endpoint variables](../../../../../docs/benchmarks.md#custom).

For every actual cluster run, write `<run_dir>/logs/perf-analysis.md` following
[`docs/perf-analysis.md`](../../../../../docs/perf-analysis.md).

## EGM over NVLink

[`glm5.3-disagg-3p1d-tp4-dep8-c560-egm.yaml`](glm5.3-disagg-3p1d-tp4-dep8-c560-egm.yaml)
is an EGM variant derived directly from this directory's DEP8 c560 recipe.
It preserves its Lustre paths, image, kernel and engine tuning, host clock setup,
container reuse, NIXL prefill/decode handoff, disabled exporters and full AgentX
workload. Nominal Mooncake capacity remains `5120GiB`: one `1040GiB` EGM provider
on each of three prefill nodes replaces four `260GiB` rank pools, and each of the
two decode nodes contributes `1000GiB`.
The EGM variant sets `engine.numa_cpu_bind: true` and
`engine.numa_memory_bind: local` to bind inference worker CPUs and memory to
their task GPU's NUMA node.

Prefill clients use `protocol: nvlink`, `stage_through_host: true`, a `2GiB`
host staging budget and zero ordinary Store capacity. TRT-LLM registers pinned
host slots with Mooncake; it copies active GPU KV into these slots before a
Store write and copies back after a read. The native providers allocate
Fabric-exportable host DRAM with `enable_egm_store_pool=true`,
`egm_numa_nodes=auto` and zero local buffer. Both ends of the Mooncake payload
transfer therefore use host memory. The master starts before providers, which
become ready before inference workers start.
For a two-node check before serving, use the
[EGM/NVLink smoke test](../../../../../tools/check_mooncake_egm.md), which
requires pinned host source/destination buffers, verifies their CUDA memory
type, and compares the complete returned payload on the CPU.

Supply exactly one ARM64 CUDA 13 Mooncake wheel built from
the EGM stack at `db195f7f7cad61a0a3c3141b8ecf6ca399a54fca` or its upstream
squash `9c0528d9`, with `WITH_STORE=ON`, `USE_CUDA=ON`, `USE_MNNVL=ON` and
`USE_VRAM_SEGMENT=OFF`, in
`/lustre/fsw/coreai_comparch_inferencex/rihuo/mooncake_egm_wheels/`.
The directory is covered by the existing Lustre mount. To change it, update
`SRT_MOONCAKE_EGM_WHEEL_DIR` in the global environment and all three Mooncake
service environments. The setup script installs this wheel alongside the
existing Dynamo wheels; the published Mooncake `0.3.13` wheel lacks EGM pooling.
`USE_VRAM_SEGMENT=OFF` keeps the connector's separate `64MiB` Store local buffer
in host memory; the upstream ARM64 CUDA 13 nightly profile uses this default.
Host staging avoids registering GPU KV pages with Mooncake. Active model/KV
memory and CUDA context/stream resources still use the GPUs.
See the [EGM integration and hardware requirements](../raplab-dynamo/README.md#egm-over-nvlink)
for the source trace, IMEX/Fabric prerequisites and setup checks.

The allocation requests six exclusive nodes in one segment and all allocatable
memory. Account and partition come from cluster configuration. Its explicit
two-hour walltime uses the job-sizing fallback for the retained 60-minute
measurement, warmup, startup and drain; no comparable EGM serving history is
available. Validate the launch with:

```bash
srtctl dry-run -f recipes/trtllm/vrnvl72/agentic/dynamo/glm5.3-disagg-3p1d-tp4-dep8-c560-egm.yaml
```
