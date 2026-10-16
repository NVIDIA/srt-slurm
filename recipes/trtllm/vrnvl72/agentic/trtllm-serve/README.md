# GLM-5.3 AgentX Pareto recipes on Vera Rubin

These eight schema-2 recipes run native `trtllm-serve` prefill/decode workers
behind `trtllm-serve disaggregated`. They preserve the BTK `e2e_pareto` engine
settings, per-role kernel switches, routing policies and AgentX workload from
the original GLM-5.2 recipes, with model identifiers changed to GLM-5.3.

Filenames use `glm5.3-disagg-<P>p<D>d-tp<PTP>-<decode>-c<C>.yaml`.
`P` and `D` are worker counts, `PTP` is the prefill `tensor_parallel_size`, and
`C` is benchmark concurrency. The decode token is `tp<N>` for tensor parallel
attention or `dep<N>` for attention data parallelism with expert parallel size
`N`. The links below retain the original Pareto point labels for source mapping.

The conversion uses benchmark-recipe revision
`b7810e6a799a04e60270443724c8d12daa35c6b7`, tekit revision
`7890925ea10e613cd787c82a406a8fdc088eabc8`, and the local AIPerf
checkout at `/home/rihuo/agentx-harness`. Use an image built from that tekit
revision: the recipes depend on its Rubin kernels, Python NIXL transceiver,
MTP prefill support and Mooncake connector.

| Source point / recipe | Prefill workers / GPUs each | Decode workers / GPUs each | Total GPUs | Concurrency | KV dtype |
| --- | --- | --- | --- | --- | --- |
| [pareto01](glm5.3-disagg-1p1d-tp4-tp16-c1.yaml) | 1 / 4 | 1 / 16 | 20 | 1 | fp8 |
| [pareto02](glm5.3-disagg-1p2d-tp4-tp4-c36.yaml) | 1 / 4 | 2 / 4 | 12 | 36 | fp8 |
| [pareto03](glm5.3-disagg-1p3d-tp4-tp8-c40.yaml) | 1 / 4 | 3 / 8 | 28 | 40 | fp8 |
| [pareto04](glm5.3-disagg-1p3d-tp4-tp4-c64.yaml) | 1 / 4 | 3 / 4 | 16 | 64 | fp8 |
| [pareto05](glm5.3-disagg-3p1d-tp4-dep32-c560.yaml) | 3 / 4 | 1 / 32 | 44 | 560 | nvfp4 |
| [pareto06](glm5.3-disagg-3p1d-tp4-dep16-c640.yaml) | 3 / 4 | 1 / 16 | 28 | 640 | nvfp4 |
| [pareto07](glm5.3-disagg-3p1d-tp4-dep16-c576.yaml) | 3 / 4 | 1 / 16 | 28 | 576 | nvfp4 |
| [pareto08](glm5.3-disagg-3p1d-tp4-dep8-c560.yaml) | 3 / 4 | 1 / 8 | 20 | 560 | nvfp4 |

## Upstream configuration comparison

Compared all eight points with `agentx/glm_5.2/verarubin/e2e_pareto` at
benchmark-recipe revision `578962ec04a2c370c55e8f22458831337b8b15f6`.
The upstream recipe directory is unchanged from the original conversion
revision above. After translating BTK sweep fields into TRT-LLM YAML, the
following match for every point:

- Worker counts, GPU allocation, TP/PP/EP sizes and concurrency.
- Engine arguments, including cache sizes and dtype, MTP, scheduler, graph,
  attention and transfer settings; the Mooncake client block is rendered
  separately through `MOONCAKE_CONFIG_PATH`.
- Original kernel environment values and forced acceptance, after resolving
  offline, FlashInfer, UCX device and acceptance placeholders.
- Native context/generation routers, tool parser, postprocessing workers and
  frontend server configuration.
- AgentX dataset, duration, seed, context limit, streaming, token counts,
  trajectory ratios, warmup, idle-gap cap and statistics cadence.
- Mooncake prefill segment sizes, transport/client settings, master lease,
  eviction ratio and allocation strategy.

Deployment differences are explicit: GLM-5.3 model/image paths and model name;
local AIPerf installation and mounted caches; RapLab NIC/MPI/NCCL and pip
settings; GPU-local NUMA CPU/memory binding; iteration statistics and worker
metrics export; a shared dedicated frontend/client node; donors starting
`after_frontend` with uniform maximum segment sizes; no HBM clock setup;
removal of `TRTLLM_SERVE_ENABLE_MSGSPEC`;
and removal of the Dynamo-specific benchmark session-ID variable. Existing
srt-slurm executor/unbuffered environment additions and service readiness,
ports and output paths remain part of the conversion. The recipes therefore
preserve upstream inference/workload settings with these deployment overrides,
rather than being literal copies of the BTK files.

## Cluster configuration

The recipes use model path `/home/rihuo/GLM-5.3-NVFP4` and image
`/home/_shared/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh`.
Configure cluster settings in `srtslurm.yaml`, for example:

```yaml
gpus_per_node: 4
use_segment_sbatch_directive: true
```

Set your account, partition, fabric interface, Hugging Face cache mounts and
other cluster paths there. The original cluster-path placeholders are replaced
by the explicit recipe paths and the mounted `/home/rihuo/aiperf_mmap` cache directory.
The tokenizer uses the mounted `/model`. Worker environments match the shared
RapLab settings: explicit UCX/MPI NIC lists, MPI transports, NCCL HCA mapping,
UCX diagnostics and offline worker settings. The top-level pip environment
uses PyPI and NVIDIA indexes with `PIP_CONFIG_FILE=/dev/null`. The whole allocation must fit within one NVL72 domain;
`resources.het_jobs: false` preserves the source's single-segment allocation.

```bash
srtctl dry-run -f recipes/trtllm/vrnvl72/agentic/trtllm-serve/glm5.3-disagg-3p1d-tp4-dep8-c560.yaml
```

All recipes use `engine.numa_cpu_bind: true` and `engine.numa_memory_bind: local`
to bind worker CPUs and memory to the GPU NUMA node through native `numactl`.
Both roles enable iteration statistics at 40-iteration intervals. Native
routing, `--tool_parser`, postprocessing workers and frontend performance
settings remain in place; Dynamo installation, transport and `DYN_*` variables
are omitted.

The frontend and benchmark client share one dedicated node in every recipe.
Both use `placement.node: dedicated`, with
`benchmark.colocate_with_frontend: true`. This adds one node beyond the worker
allocation and keeps client/frontend memory use off the decode nodes.

## Mooncake

Points 01–04 retain a 240 GiB private host cache per prefill rank and have no
Mooncake service. Points 05–08 retain a GPU-only prefill cache manager, a shared
Mooncake store, and connector-free decode workers. The master runs on the first
prefill node; one `trtllm-serve mooncake_donor` runs per decode node.

| Source point / recipe | Segment per prefill rank | Donor segment per decode node |
| --- | --- | --- |
| [pareto05](glm5.3-disagg-3p1d-tp4-dep32-c560.yaml) | 224 GiB | 768 GiB on each of eight nodes |
| [pareto06](glm5.3-disagg-3p1d-tp4-dep16-c640.yaml) | 256 GiB | 1050 GiB on each of four nodes |
| [pareto07](glm5.3-disagg-3p1d-tp4-dep16-c576.yaml) | 256 GiB | 1050 GiB on each of four nodes |
| [pareto08](glm5.3-disagg-3p1d-tp4-dep8-c560.yaml) | 260 GiB | 1100 GiB on each of two nodes |

srt-slurm renders the prefill client JSON and injects `MOONCAKE_CONFIG_PATH`.
Only prefill selects `kv_connector_config`; no worker uses `role: capacity`,
which the pinned tekit revision does not implement. The donors read the same
JSON for the master and transport, then apply one uniform segment size to every decode node.
Tekit writes `/logs/mooncake_donor_<index>.ready` after segment registration.
The `readiness.file` probes wait for these markers before benchmarking;
srt-slurm removes an old marker before launching each donor.

Prefill workers and donors use the same Mooncake transport environment as
the RapLab recipes: `MC_ENABLE_PARALLEL_REG_MR=0`, `MC_MS_AUTO_DISC=0`,
`MC_ENABLE_DEST_DEVICE_AFFINITY=1` and the eight-NIC `MC_TE_FILTERS` allowlist.
Donors use the recipe's largest original segment size on every decode node
and `start: after_frontend`; their
environment contains transport and logging settings.

The master command uses this Mooncake build's `--rpc_port` spelling, random
allocation, a 60-second lease TTL and a 0.02 eviction ratio. The RPC/metrics
ports become srt-slurm's 8700/8702. Internal host/disk caches are distinct from
the Mooncake pool and staging buffers; budget all of them together.

## AgentX workload

Each recipe mounts `/home/rihuo/:/home/rihuo/`. Its `benchmark.command` creates
a temporary virtual environment, installs AIPerf from
`/home/rihuo/agentx-harness`, and invokes
`aiperf profile` directly against the runtime-discovered frontend. Dependency installation
and public trace loading require network access from the benchmark container.
The temporary client environment is removed when the command exits.

The workload retains the `inferencex-agentx-mvp` scenario, WEKA 062126 dataset,
393 entries, seed 42, trajectory start ratios 0.25–0.75, 1M-token context cap,
streaming, server token counts, 10 warmup requests per lane, 1800-second warmup
grace, 3600-second measurement and 300-second idle-gap cap. Conversation-aware
Dynamo routing, unsafe override and GPU telemetry remain disabled. Session
headers and AIPerf's `--failed-request-threshold 0.1` setting are retained.

AIPerf receives every logical worker's `/prometheus/metrics` URL through
`AIPERF_SERVER_METRICS_URLS` and exports JSON, CSV, Parquet and JSONL.
`return_perf_metrics: true` remains enabled for the native workers.

Forced MTP acceptance is retained from the GLM-5.2 source `acceptance_length.json`:
depth 7 uses `3.91 - 1 = 2.91` accepted draft tokens and depth 5 uses
`3.61 - 1 = 2.61`. These are synthetic acceptance controls, not measured accuracy.

Each recipe retains the source's three-hour allocation budget, covering client
preparation, worker startup, warmup, the full measurement, drain and cleanup.
Results go under `/logs/agentic/aiperf_results`. The BTK-specific rollup upload
plugin is omitted. For every actual cluster run, write the required
`<run_dir>/logs/perf-analysis.md` according to
[`docs/perf-analysis.md`](../../../../../docs/perf-analysis.md).
