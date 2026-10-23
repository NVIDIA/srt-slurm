# GLM-5.3 with Dynamo on Vera Rubin

These 8 recipes are ported from
[`rihuo/glm5.3-mooncake-w2/recipes/trtllm/vrnvl72/agentic/raplab-dynamo`](https://github.com/NVIDIA/srt-slurm/tree/0e4b9b8a66e03494bf25cf5b669c48580da6bc24/recipes/trtllm/vrnvl72/agentic/raplab-dynamo).
They retain the source's model and container paths, worker tuning, NUMA binding,
routing, concurrency and complete AgentX workload. Cache capacities are listed
below. Update the
model, image, cache, Dynamo wheel and AgentX checkout paths for your cluster.

## Baselines

Each node has four GPUs. Node counts include the dedicated Dynamo frontend and
benchmark client, which share one node.

| Recipe | Prefill workers / GPUs each | Decode workers / GPUs each | Concurrency | Nodes |
| --- | --- | --- | --- | --- |
| [TP16 c1](glm5.3-disagg-1p1d-tp4-tp16-c1.yaml) | 1 / 4 | 1 / 16 | 1 | 6 |
| [TP4 c36](glm5.3-disagg-1p2d-tp4-tp4-c36.yaml) | 1 / 4 | 2 / 4 | 36 | 4 |
| [TP8 c40](glm5.3-disagg-1p3d-tp4-tp8-c40.yaml) | 1 / 4 | 3 / 8 | 40 | 8 |
| [TP4 c64](glm5.3-disagg-1p3d-tp4-tp4-c64.yaml) | 1 / 4 | 3 / 4 | 64 | 5 |

The TP baselines explicitly advertise `kv` routing and session affinity
on the prefill tier, with a 14,400-second idle TTL matching the frontend.
Session affinity uses Dynamo's default `hard` mode. KV events are disabled for
that tier and engine conversation affinity is enabled. Decode workers advertise
`least-loaded` routing.

The TP decode baselines use private prefill host caches.

## EGM over NVLink

These variants use three TP4 prefill workers and one DEP decode worker. Cache
capacity lives in separate native EGM providers; inference clients use pinned host staging
and zero ordinary Store capacity.

| Recipe | Prefill pools per node | Decode pool per node | Total capacity | Nodes |
| --- | --- | --- | --- | --- |
| [DEP8 c560 EGM 4x260](glm5.3-disagg-3p1d-tp4-dep8-c560-egm-4x260.yaml) | 4 × 260GiB | 1100GiB | 5320GiB | 6 |
| [DEP16 c576 EGM](glm5.3-disagg-3p1d-tp4-dep16-c576-egm-4x256.yaml) | 4 × 256GiB | 1050GiB | 7272GiB | 8 |
| [DEP16 c640 EGM](glm5.3-disagg-3p1d-tp4-dep16-c640-egm-4x256.yaml) | 4 × 256GiB | 1050GiB | 7272GiB | 8 |
| [DEP32 c560 EGM](glm5.3-disagg-3p1d-tp4-dep32-c560-egm-4x224.yaml) | 4 × 224GiB | 768GiB | 8832GiB | 12 |

The DEP8 `4x260` recipe uses **260GiB per prefill pool**, increased from the
source revision's 224GiB. Four-pool providers expose physical
GPUs `0`, `1`, `2`, `3` respectively and bind their complete host pools to NUMA
nodes `0`, `0`, `1`, `1`. Decode providers expose all four GPUs
and discover NUMA nodes automatically. Providers start in `before_workers` and
stop after inference clients; the master starts in `infra` and stops last.

The image must already contain the EGM-enabled ARM64 CUDA 13 Mooncake build,
including the matching `mooncake.cli` master. There are no Mooncake wheel mounts,
installers or reinstall steps in these recipes. See
[EGM prerequisites and transport](../../../../docs/mooncake-kv-store.md#egm-host-memory-over-nvlink-tensorrt-llm)
for build, HOST_NUMA, Fabric/IMEX and pinned-staging requirements. Leave
`MC_USE_NVLINK_IPC` unset for inference workers.

## Setup and workload

The DEP8 `4x260` recipe uses `least-loaded` routing at the frontend and both
worker tiers, with Dynamo session affinity disabled.

The recipes retain the local Dynamo wheel installation through
[`configs/install_dynamo_wheel.sh`](../../../../configs/install_dynamo_wheel.sh).
It serializes installs per Python prefix and caches completion by wheel content.
`dynamo.install: false` prevents the built-in installer from replacing that
build. The image must supply the custom TRT-LLM Rubin kernels, MTP and Python
NIXL transceiver.

The source's optional `reuse_container: true` in the DEP8 four-pool EGM recipe
is omitted because this branch uses standard per-step containers. Its topology
and provider allocation are unchanged.

The benchmark creates a temporary environment from the mounted AgentX checkout,
then runs `inferencex-agentx-mvp` for 3600 measured seconds with the source
dataset, concurrency and warmup settings. It collects each
logical worker's Prometheus endpoint through `AIPERF_SERVER_METRICS_URLS`,
exporting JSON, CSV, Parquet and JSONL to `/logs/agentic/aiperf_results`.

Set account, partition, network interface, allocation policy and cache mounts in
`srtslurm.yaml`. Keep participating nodes in one NVLink domain. Allocation
walltime is inherited from cluster configuration or supplied at submission,
as in the source recipes; include model startup, dataset preparation, warmup,
the full measurement, drain and artifact preservation. With no comparable
timing history, the [job-sizing fallback](../../../../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md)
is two hours for a 60-minute measurement. Dry-run before submitting:

```bash
srtctl dry-run -f recipes/trtllm/glm5.3/vrnvl72-lab/glm5.3-disagg-3p1d-tp4-dep16-c576-egm-4x256.yaml
```

For every actual cluster run, write `<run_dir>/logs/perf-analysis.md` using the
[per-run analysis requirements](../../../../docs/perf-analysis.md).
