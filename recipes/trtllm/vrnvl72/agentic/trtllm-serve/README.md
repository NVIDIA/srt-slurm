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
`7890925ea10e613cd787c82a406a8fdc088eabc8`, and AIPerf revision
`b7b16cf851885567988a643282266bce74e34437`. Use an image built from that tekit
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

## Cluster configuration

Resolve the recipe aliases in your `srtslurm.yaml`, for example:

```yaml
model_paths:
  glm5.3: /models/GLM-5.3-NVFP4
containers:
  trtllm-glm5.3-rubin: /images/trtllm-7890925e.sqsh
gpus_per_node: 4
use_segment_sbatch_directive: true
```

Set your account, partition, fabric interface, Hugging Face cache mounts and
other cluster paths there. The original cluster-path placeholders are replaced
by these aliases and node-local `/tmp/flashinfer-cache` and
`/tmp/aiperf-mmap-cache` directories. The tokenizer uses the mounted `/model`.
The source's `UCX_TLS=all`, `UCX_NET_DEVICES=all` and offline worker settings
are retained. The whole allocation must fit within one NVL72 domain;
`resources.het_jobs: false` preserves the source's single-segment allocation.

The host setup locks HBM clocks to 4752 MHz and resets them on cleanup. This
requires the site's passwordless `sudo` permission for `nvidia-smi`. The worker
NUMA memory policy remains `numactl -m 0,1`.

```bash
srtctl dry-run -f recipes/trtllm/vrnvl72/agentic/trtllm-serve/glm5.3-disagg-3p1d-tp4-dep8-c560.yaml
```

## Mooncake

Points 01–04 retain a 240 GiB private host cache per prefill rank and have no
Mooncake service. Points 05–08 retain a GPU-only prefill cache manager, a shared
Mooncake store, and connector-free decode workers. The master runs on the first
prefill node; one `trtllm-serve mooncake_donor` runs per decode node.

| Source point / recipe | Segment per prefill rank | Donor segment per decode node, in placement order |
| --- | --- | --- |
| [pareto05](glm5.3-disagg-3p1d-tp4-dep32-c560.yaml) | 224 GiB | 768 GiB on each of eight nodes |
| [pareto06](glm5.3-disagg-3p1d-tp4-dep16-c640.yaml) | 256 GiB | 900, 1050, 1050, 1050 GiB |
| [pareto07](glm5.3-disagg-3p1d-tp4-dep16-c576.yaml) | 256 GiB | 900, 1050, 1050, 1050 GiB |
| [pareto08](glm5.3-disagg-3p1d-tp4-dep8-c560.yaml) | 260 GiB | 960, 1100 GiB |

srt-slurm renders the prefill client JSON and injects `MOONCAKE_CONFIG_PATH`.
Only prefill selects `kv_connector_config`; no worker uses `role: capacity`,
which the pinned tekit revision does not implement. The donors read the same
JSON for the master and transport, then apply their explicit per-node sizes.
Tekit writes `/logs/mooncake_donor_<index>.ready` after segment registration.
The `readiness.file` probes wait for these markers before workers start;
srt-slurm removes an old marker before launching each donor.

The master command uses this Mooncake build's `--rpc_port` spelling, random
allocation, a 60-second lease TTL and a 0.02 eviction ratio. The RPC/metrics
ports become srt-slurm's 8700/8702. Internal host/disk caches are distinct from
the Mooncake pool and staging buffers; budget all of them together.

## AgentX workload

Each recipe's `benchmark.command` creates a temporary virtual environment,
installs AIPerf at commit `b7b16cf851885567988a643282266bce74e34437`, and invokes
`aiperf profile` directly against the runtime-discovered frontend. Installation
and public trace loading require network access from the benchmark container.
The temporary client environment is removed when the command exits.

The workload retains the `inferencex-agentx-mvp` scenario, WEKA 062126 dataset,
393 entries, seed 42, trajectory start ratios 0.25–0.75, 1M-token context cap,
streaming, server token counts, 10 warmup requests per lane, 1800-second warmup
grace, 3600-second measurement and 300-second idle-gap cap. Conversation-aware
Dynamo routing, unsafe override and GPU telemetry remain disabled. Session
headers and AIPerf's `--failed-request-threshold 0.1` setting are retained.

Forced MTP acceptance is retained from the GLM-5.2 source `acceptance_length.json`:
depth 7 uses `3.91 - 1 = 2.91` accepted draft tokens and depth 5 uses
`3.61 - 1 = 2.61`. These are synthetic acceptance controls, not measured accuracy.

Each recipe retains the source's three-hour allocation budget, covering client
preparation, worker startup, warmup, the full measurement, drain and cleanup.
Results go under `/logs/agentic/aiperf_results`. The BTK-specific rollup upload
plugin is omitted. For every actual cluster run, write the required
`<run_dir>/logs/perf-analysis.md` according to
[`docs/perf-analysis.md`](../../../../../docs/perf-analysis.md).
