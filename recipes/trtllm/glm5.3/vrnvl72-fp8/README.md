# GLM-5.3 with Dynamo on Vera Rubin (FP8 KV)

This directory holds the hecate form of the RapLab
[DEP16 c620 EGM FP8 KV recipe](../vrnvl72-lab-w2-fp8/glm5.3-disagg-5p1d-tp4-dep16-c620-egm-4x256.yaml):
GLM-5.3 BTK pareto05 with FP8 KV, five TP4 prefill workers, one DEP16 decode
worker and 620 users. Topology, engine arguments, worker tuning, routing,
concurrency and the AgentX workload are unchanged. Cluster settings follow the
[wave 2 recipes](../vrnvl72-w2/README.md) and the
[EGM recipes](../vrnvl72/README.md#egm-over-nvlink); see
[Hecate settings](#hecate-settings). Update model, image, cache, Dynamo wheel
and AgentX checkout paths for your cluster.

## Recipe

Each node has four GPUs. The node count includes the dedicated Dynamo frontend
and benchmark client, which share one node.

| Recipe | Source | Prefill workers / GPUs each | Decode workers / GPUs each | Concurrency | Nodes |
| --- | --- | --- | --- | --- | --- |
| [DEP16 c620 EGM FP8 KV](glm5.3-disagg-5p1d-tp4-dep16-c620-egm-4x256.yaml) | pareto05 (FP8 KV) | 5 / 4 | 1 / 16 | 620 | 10 |

## EGM over NVLink

The recipe keeps BTK's pool size but serves the pool from native EGM providers
over NVLink instead of RDMA. Each prefill rank's 256GiB BTK segment becomes one of
four 256GiB providers per prefill node; providers expose physical GPUs `0`, `1`,
`2`, `3` respectively and bind their host pools to NUMA nodes `0`, `0`, `1`, `1`.
Each decode node runs one 1050GiB provider that exposes all four GPUs and
discovers NUMA nodes automatically, for 9320GiB in total. Providers start in
`before_workers` and stop after inference clients; the master starts in `infra`
and stops last. Leave `MC_USE_NVLINK_IPC` unset for inference workers. See
[EGM prerequisites and transport](../../../../docs/mooncake-kv-store.md#egm-host-memory-over-nvlink-tensorrt-llm)
for build, HOST_NUMA and Fabric/IMEX requirements.

Three Mooncake settings differ from BTK:

- **Direct KV copies** (`stage_through_host: false`). Prefill workers register
  their GPU KV pools with Mooncake, which copies pages directly between the
  providers' host DRAM and the KV cache. BTK passes every page through pinned
  host slots, which its RDMA transport needs without GPUDirect. On RapLab, staged
  EGM loads peaked near 13GiB/s per prefill rank, and waiting on them was most of
  TTFT. This mode has not run over EGM before: if registration fails, prefill
  workers stop at startup with `MooncakeDistributedStore.register_buffer failed`.
  To run the staged path for comparison, set
  `TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST: '1'` in `roles.prefill.env`; the
  recipe's `staging_buffer_bytes: 4GiB` then gives each load worker RapLab's
  1GiB.
- **Four async-load workers** (BTK: two). FP8 KV loads (p50 4.7GiB, p90 14.7GiB)
  queued 1.5-2.6s behind two workers.
- **A 300-second object lease** (BTK: 60 seconds). Lookups renew the lease, and
  AgentX sessions sit idle for up to 300 seconds. With a 60-second lease,
  eviction removed idle sessions' prefixes, which they then recomputed.

## Hecate settings

- Model, image, flashinfer cache, Dynamo wheels, the AgentX checkout and the
  aiperf mmap cache live under `/lustre/fsw/coreai_comparch_inferencex/rihuo`;
  `/lustre` is mounted at the same path.
- The image is wave 2's
  `tensorrt-llm-arm64-rubin-glm5-4bad965-feat_rubin-glm5-mega-30454-mooncake-cuda13-0.3.14.dev20261021.sqsh`.
  RapLab uses its `-glm53mk` variant, which adds only the GLM-5 megakernel fix;
  this point does not enable the megakernel.
- Worker nodes lock GPU memory clocks to 4752 MHz with
  `sudo -n nvidia-smi -lmc 4752,4752` and reset them during teardown.
- UCX uses every NIC (`UCX_NET_DEVICES: all`) instead of RapLab's NIC, MPI and
  NCCL device lists, and `MC_ENABLE_PARALLEL_REG_MR` is `0` for every step.
- Exclusive nodes with all host memory (`exclusive`, `mem: 0`) hold the EGM
  pools, and `segment: '10'` keeps all ten nodes in one NVLink domain.
- The OpenTelemetry collector and the node, process and DCGM exporters are
  disabled.
- The time limit is 2.5 hours. RapLab runs of this point took 1h55m: about 30
  minutes of startup, 15 minutes of warmup, the 60-minute measurement and 9
  minutes of drain and artifact copy.

## Setup and workload

The recipe installs the local Dynamo wheel through
[`configs/install_dynamo_wheel.sh`](../../../../configs/install_dynamo_wheel.sh);
`dynamo.install: false` prevents the built-in installer from replacing that
build. `reuse_container: true` shares one Pyxis container per image on each node
instead of extracting the root filesystem for every step; see
[host memory used by container filesystems](../../../../docs/slurm-faq.md#host-memory-used-by-container-filesystems).
The benchmark creates a temporary environment from the mounted AgentX checkout,
then runs `inferencex-agentx-mvp` for 3600 measured seconds with BTK's dataset,
concurrency and warmup settings, collecting each logical worker's Prometheus
endpoint through `AIPERF_SERVER_METRICS_URLS`.

Set account, partition and network interface in `srtslurm.yaml`. Dry-run before
submitting:

```bash
srtctl dry-run -f recipes/trtllm/glm5.3/vrnvl72-fp8/glm5.3-disagg-5p1d-tp4-dep16-c620-egm-4x256.yaml
```

For every actual cluster run, write `<run_dir>/logs/perf-analysis.md` using the
[per-run analysis requirements](../../../../docs/perf-analysis.md).
