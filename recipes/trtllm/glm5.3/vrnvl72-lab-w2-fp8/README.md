# GLM-5.3 with Dynamo on Vera Rubin RapLab (wave 2)

These recipes are the RapLab form of the [wave 2 recipes](../vrnvl72-w2/README.md),
which port the second GLM-5.2 BTK `e2e_pareto` frontier (pareto01-07) to
srt-slurm. Topology, engine arguments, worker tuning, routing, concurrency and the
AgentX workload follow wave 2. The c620 FP8 recipe also enables the
[native EGM optimizations](#optimized-c620-fp8-egm-recipe) described below; the
other differences are the RapLab settings used by the
[lab recipes](../vrnvl72-lab/README.md), listed below. Update model,
image, cache, Dynamo wheel and AgentX checkout paths for your cluster.

## Recipes

Each node has four GPUs. Node counts include the dedicated Dynamo frontend and
benchmark client, which share one node.

| Recipe | Source | Prefill workers / GPUs each | Decode workers / GPUs each | Concurrency | Nodes |
| --- | --- | --- | --- | --- | --- |
| [TP16 c1](glm5.3-disagg-1p1d-tp4-tp16-c1.yaml) | pareto01 | 1 / 4 | 1 / 16 | 1 | 6 |
| [TP16 c1 trtllm-serve](glm5.3-disagg-1p1d-tp4-tp16-c1-trtllm-serve.yaml) | pareto01 | 1 / 4 | 1 / 16 | 1 | 6 |
| [TP4 c56](glm5.3-disagg-1p3d-tp4-tp4-c56.yaml) | pareto02 | 1 / 4 | 3 / 4 | 56 | 5 |
| [TP8 c50](glm5.3-disagg-1p3d-tp4-tp8-c50.yaml) | pareto03 | 1 / 4 | 3 / 8 | 50 | 8 |
| [TP8 c28](glm5.3-disagg-1p3d-tp4-tp8-c28.yaml) | pareto04 | 1 / 4 | 3 / 8 | 28 | 8 |
| [DEP16 c620 EGM FP8 KV](glm5.3-disagg-5p1d-tp4-dep16-c620-egm-4x256.yaml) | pareto05 (FP8 KV) | 5 / 4 | 1 / 16 | 620 | 10 |
| [DEP16 c720 EGM](glm5.3-disagg-4p1d-tp4-dep16-c720-egm-4x256.yaml) | pareto06 | 4 / 4 | 1 / 16 | 720 | 9 |
| [DEP16 c920 EGM](glm5.3-disagg-5p1d-tp4-dep16-c920-egm-4x256.yaml) | pareto07 | 5 / 4 | 1 / 16 | 920 | 10 |

See the [wave 2 README](../vrnvl72-w2/README.md) for routing and the image. The
lab recipes use the same image with the GLM-5 megakernel fixed for
GLM-5.3-NVFP4 (`-glm53mk.sqsh`), except for the optimized c620 FP8 image below;
see [RapLab differences](#raplab-differences).

## EGM over NVLink

The DEP16 recipes keep wave 2's topology and BTK pool sizes but serve the
Mooncake pool from native EGM providers over NVLink instead of RDMA, as the
[lab EGM recipes](../vrnvl72-lab/README.md#egm-over-nvlink) do. Inference
clients contribute no Store capacity. The optimized c620 FP8 recipe uses direct
KV copies; the other DEP16 recipes use pinned host staging.

| Recipe | Prefill pools per node | Decode pool per node | Total capacity | Nodes |
| --- | --- | --- | --- | --- |
| [DEP16 c620 EGM FP8 KV](glm5.3-disagg-5p1d-tp4-dep16-c620-egm-4x256.yaml) | 4 × 256GiB | 1050GiB | 9320GiB | 10 |
| [DEP16 c720 EGM](glm5.3-disagg-4p1d-tp4-dep16-c720-egm-4x256.yaml) | 4 × 256GiB | 1050GiB | 8296GiB | 9 |
| [DEP16 c920 EGM](glm5.3-disagg-5p1d-tp4-dep16-c920-egm-4x256.yaml) | 4 × 256GiB | 1050GiB | 9320GiB | 10 |

Each prefill rank's 256GiB BTK segment becomes one of four 256GiB providers per
prefill node; providers expose physical GPUs `0`, `1`, `2`, `3` respectively and
bind their host pools to NUMA nodes `0`, `0`, `1`, `1`. Each decode node runs one
1050GiB provider that exposes all four GPUs and discovers NUMA nodes
automatically. Providers start in `before_workers` and stop after inference
clients; the master starts in `infra` and stops last. Leave `MC_USE_NVLINK_IPC`
unset for inference workers. See
[EGM prerequisites and transport](../../../../docs/mooncake-kv-store.md#egm-host-memory-over-nvlink-tensorrt-llm)
for build, HOST_NUMA, Fabric/IMEX and staging requirements.

### Optimized c620 FP8 EGM recipe

The c620 FP8 recipe uses
`/home/rihuo/tensorrt-llm-arm64-feat_rubin-glm5-mega-30454-mooncake-egm-optimization.sqsh`.
It supplies the native EGM connector,
ARM64 CUDA 13 Mooncake `0.3.14rc3` client and Release sf4 master.
Pyxis reads this image from its host-side path.

The connector's `global_segment_size: 0` leaves all capacity in these external
providers. Keep that value and the existing provider services together. The
image's `store_host_donor` late-lending retry and accounting fixes apply to
worker-owned donations; this fixed-provider recipe does not use that mode.

The native connector implementation from tekit `bcc8ec9e65` is enabled explicitly
in `roles.prefill.env`:

| Setting | Value | Effect |
| --- | --- | --- |
| `TRTLLM_M31_STORE_EGM` | `1` | Select native NVLink EGM transport. |
| `TRTLLM_M31_DISABLE_EGM_DIRECT` | `0` | Copy EGM pages directly into KV device pointers without registering GPU pools or staging through host slots. |
| `TRTLLM_M31_EGM_DIRECT_SAVE` | `1` | Keep saves direct as well as loads. |
| `TRTLLM_M31_EGM_GROUPED_LOADS` | `1` | Group reads by source segment. |
| `TRTLLM_M31_EGM_GROUPED_SAVES` | `1` | Group writes by destination segment and prefer node-local allocation. |
| `TRTLLM_M31_EGM_LOAD_STREAMS` | `2` | Use up to two copy streams per asynchronous load; small loads stay on one stream. |
| `TRTLLM_M31_EGM_PRIVATE_LOOP_STREAM` | `1` | Keep executor waits off the legacy default stream so they do not hold up Mooncake's per-thread default streams. |

The `TRTLLM_M31_*` names are retained by tekit; these transport paths also support
GLM. The recipe keeps `transfer_batch_size: 512`, four asynchronous load workers,
and the 300-second object lease.

For a staged comparison, change `TRTLLM_M31_DISABLE_EGM_DIRECT` to `1` in
`roles.prefill.env`. To keep direct loads and stage saves alone, keep that switch
at `0` and set `TRTLLM_M31_EGM_DIRECT_SAVE` to `0`. Both use the configured
`staging_buffer_bytes: 4GiB` per active transfer direction; the native worker
sizes its pools for the selected directions.
`TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST` does not override the native EGM
transport selection.

The master service invokes `mooncake_master` from the image's PATH so it uses the
Release `0.3.14rc3+sf4` binary with four-bit allocator size classes. Its port,
eviction ratio, allocation strategy and 300-second lease remain configured by
the recipe.

## Native trtllm-serve variant

[TP16 c1 trtllm-serve](glm5.3-disagg-1p1d-tp4-tp16-c1-trtllm-serve.yaml) runs
the TP16 c1 workers as native `trtllm-serve` servers behind
`trtllm-serve disaggregated` (`frontend.type: trtllm_serve`), as BTK pareto01
does, instead of Dynamo. Topology, worker env (minus `DYN_*` and
`TRTLLM_SERVE_ENABLE_MSGSPEC`), engine arguments
and the AgentX workload match the Dynamo recipe. The orchestrator follows BTK's
`server_config`: `conversation` routers on both tiers, `gen_strip_message_history`,
`gen_tokids_ctxbytes`, a 3600-second keep-alive, a 3600-second request timeout
and a 45-minute start timeout. Workers pass `--tool_parser glm47` and set BTK's
`num_postprocess_workers` (8 prefill, 4 decode), which native `trtllm-serve`
honors. No Dynamo wheel is installed.

## RapLab differences

- Model, image, flashinfer cache, Dynamo wheels and the AgentX checkout live
  under `/home/rihuo`, which is mounted at the same path; the aiperf mmap cache
  is `/tmp/aiperf-mmap-cache`.
- Except for the optimized c620 FP8 recipe above, the image is wave 2's image
  with `glm52_plain_layer.py` and
  `glm52_plain_layer.cu` replaced by a TensorRT-LLM patch that applies the MoE
  activation global scales in the megakernel. The stock megakernel asserts
  `fc31_input_scale == 1`, which holds for GLM-5.2-NVFP4 but not for
  GLM-5.3-NVFP4, so the TP16 c1 decode worker (`TRTLLM_GLM5_MEGAKERNEL=1`) fails
  at startup on the stock image.
- pip is pointed at `pypi.org` and `pypi.nvidia.com`, because the image's
  `/etc/environment` names a package index RapLab cannot resolve.
- Workers pin UCX, MPI and NCCL to the RapLab NICs (`mlx5_0`-`mlx5_2`,
  `mlx5_5`-`mlx5_8`, `mlx5_11`).
- The DEP16 recipes request exclusive nodes with all host memory
  (`exclusive`, `mem: 0`) for the EGM pools and cap enroot at 8 processors.
- There is no GPU memory-clock host setup, and only the disabled OpenTelemetry
  collector is declared.

## Setup and workload

The recipes install the local Dynamo wheel through
[`configs/install_dynamo_wheel.sh`](../../../../configs/install_dynamo_wheel.sh);
`dynamo.install: false` prevents the built-in installer from replacing that
build. `reuse_container: true` shares one Pyxis container per image on each node
instead of extracting the root filesystem for every step; see
[host memory used by container filesystems](../../../../docs/slurm-faq.md#host-memory-used-by-container-filesystems).
The benchmark runs `inferencex-agentx-mvp` for 3600 measured seconds, as
in wave 2.

Set account, partition, network interface and allocation policy in
`srtslurm.yaml`, and keep participating nodes in one NVLink domain. Walltime is
inherited from cluster configuration or submission; with no comparable timing
history, the [job-sizing fallback](../../../../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md)
is two hours for a 60-minute measurement. Dry-run before submitting:

```bash
srtctl dry-run -f recipes/trtllm/glm5.3/vrnvl72-lab-w2-fp8/glm5.3-disagg-5p1d-tp4-dep16-c620-egm-4x256.yaml
```

For every actual cluster run, write `<run_dir>/logs/perf-analysis.md` using the
[per-run analysis requirements](../../../../docs/perf-analysis.md).
