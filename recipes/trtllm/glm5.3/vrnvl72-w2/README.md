# GLM-5.3 with Dynamo on Vera Rubin (wave 2)

These 7 recipes port the second GLM-5.2 BTK `e2e_pareto` frontier (pareto01-07,
TensorRT-LLM `feat/rubin-glm5-mega`) to srt-slurm. Worker env, engine arguments,
MTP depth, routing, concurrency and the AgentX workload follow the BTK points;
everything else (paths, frontend, host setup, benchmark client) follows the
[vrnvl72 recipes](../vrnvl72/README.md). Update model, image, Dynamo wheel and
AgentX checkout paths for your cluster.

## Recipes

Each node has four GPUs. Node counts include the dedicated Dynamo frontend and
benchmark client, which share one node.

| Recipe | Source | Prefill workers / GPUs each | Decode workers / GPUs each | Concurrency | Nodes |
| --- | --- | --- | --- | --- | --- |
| [TP16 c1](glm5.3-disagg-1p1d-tp4-tp16-c1.yaml) | pareto01 | 1 / 4 | 1 / 16 | 1 | 6 |
| [TP4 c56](glm5.3-disagg-1p3d-tp4-tp4-c56.yaml) | pareto02 | 1 / 4 | 3 / 4 | 56 | 5 |
| [TP8 c50](glm5.3-disagg-1p3d-tp4-tp8-c50.yaml) | pareto03 | 1 / 4 | 3 / 8 | 50 | 8 |
| [TP8 c28](glm5.3-disagg-1p3d-tp4-tp8-c28.yaml) | pareto04 | 1 / 4 | 3 / 8 | 28 | 8 |
| [DEP16 c504](glm5.3-disagg-3p1d-tp4-dep16-c504.yaml) | pareto05 | 3 / 4 | 1 / 16 | 504 | 8 |
| [DEP16 c720](glm5.3-disagg-4p1d-tp4-dep16-c720.yaml) | pareto06 | 4 / 4 | 1 / 16 | 720 | 9 |
| [DEP16 c920](glm5.3-disagg-5p1d-tp4-dep16-c920.yaml) | pareto07 | 5 / 4 | 1 / 16 | 920 | 10 |

The TP recipes match BTK's conversation context router: the frontend uses `kv`
routing with session affinity (14,400-second idle TTL) and KV events disabled,
prefill workers advertise the same and set `DYN_ENGINE_CONV_AFFINITY: '1'`, and
decode workers advertise `least-loaded` (BTK's `load_balancing` generation
router). The TP16 c1 recipe differs: BTK pareto01 routes generation by
conversation too, so neither tier sets router env and both inherit the
frontend's routing; both tiers set `DYN_ENGINE_CONV_AFFINITY: '1'`. The TP
recipes use private prefill host caches.

The DEP16 recipes use `least-loaded` routing on the frontend and both tiers and
share prefill KV through Mooncake over RDMA: each prefill rank contributes a
256GiB segment, and one `trtllm-serve mooncake_donor` per decode node adds
1050GiB. The BTK pool gives its first donor only 900GiB because that node also
hosts the frontend, client and Mooncake master; here those share the dedicated
frontend node, so every donor uses the largest BTK size.
Donors start in `before_workers` and publish a readiness file before inference
workers launch. Prefill workers enable Mooncake early admission.

## Image

The recipes run on
`tensorrt-llm-arm64-rubin-glm5-4bad965-feat_rubin-glm5-mega-30454-mooncake-cuda13-0.3.14.dev20261021.sqsh`,
built from the TensorRT-LLM commit the BTK points pin (`feat/rubin-glm5-mega` @
`4bad9655`) with the CUDA 13 Mooncake wheel. Engine arguments and worker env are
passed through unchanged, including pareto01's `TRTLLM_GLM5_MEGAKERNEL=1` and the
DEP16 points' breakable prefill CUDA graphs (`prefill_cuda_graph_backend`,
`prefill_capture_num_tokens`). `num_postprocess_workers` is not set, because
Dynamo's TensorRT-LLM worker ignores it.

## Setup and workload

The recipes install the local Dynamo wheel through
[`configs/install_dynamo_wheel.sh`](../../../../configs/install_dynamo_wheel.sh);
`dynamo.install: false` prevents the built-in installer from replacing that
build. `reuse_container: true` shares one Pyxis container per image on each node
instead of extracting the root filesystem for every step; see
[host memory used by container filesystems](../../../../docs/slurm-faq.md#host-memory-used-by-container-filesystems).
Worker nodes lock GPU memory clocks to 4752 MHz with
`sudo -n nvidia-smi -lmc 4752,4752` and reset them during teardown. Workers use
GPU-local CPU and memory binding.

The benchmark creates a temporary environment from the mounted AgentX checkout,
then runs `inferencex-agentx-mvp` for 3600 measured seconds with the BTK
dataset, concurrency and warmup settings, collecting each logical worker's
Prometheus endpoint through `AIPERF_SERVER_METRICS_URLS`.

Set account, partition, network interface, allocation policy and cache mounts in
`srtslurm.yaml`, and keep participating nodes in one NVLink domain. Walltime is
inherited from cluster configuration or submission; with no comparable timing
history, the [job-sizing fallback](../../../../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md)
is two hours for a 60-minute measurement. Dry-run before submitting:

```bash
srtctl dry-run -f recipes/trtllm/glm5.3/vrnvl72-w2/glm5.3-disagg-3p1d-tp4-dep16-c504.yaml
```

For every actual cluster run, write `<run_dir>/logs/perf-analysis.md` using the
[per-run analysis requirements](../../../../docs/perf-analysis.md).
