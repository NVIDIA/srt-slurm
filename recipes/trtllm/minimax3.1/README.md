# MiniMax-M3.1 Flash on VR200 R37 (vLLM + Dynamo 1.5.0)

Standalone srt-slurm replays of the first seven-point sweep of InferenceMAX PR 1859: CI run
[42170068985](https://github.com/NVIDIA/InferenceMAX/actions/runs/42170068985)
(commit `4455c46e`, srt-slurm `1ca857b2`). The engine is vLLM even though the folder sits under
`trtllm/`.

| Recipe | Topology | Nodes | CI result (TPS/GPU, ITL p50 / p90, TTFT p90) |
| --- | --- | ---: | --- |
| `4pdep2-1ddep8-c640.yaml` | 4 x DEP2 prefill + DEP8 decode, C640 (GLM-5.3 client, see below) | 5 | none: aborted in warmup (see below) |
| `1pdep2-1ddep8-c240.yaml` | DEP2 prefill + DEP8 decode, C240 | 3 | 174,106; 2.29 / 3.03 ms; 2.24 s |
| `1pdep2-1ddep8-c126.yaml` | DEP2 prefill + DEP8 decode, C126 (959020) | 3 | 84,262; 1.91 / 2.10 ms; 4.14 s |
| `tp2-c24.yaml` | TP2 aggregate, DRAM KV offload, C24 | 1 | 88,385; 1.95 / 2.17 ms; 9.31 s |
| `tp2-c14.yaml` | TP2 aggregate, C14 | 1 | 40,382; 1.06 / 1.14 ms; 7.16 s |
| `tp4-c14.yaml` | TP4 aggregate, C14 | 1 | 22,989; 0.84 / 0.91 ms; 3.79 s |
| `tp8-c14.yaml` | TP8 aggregate, C14 | 2 | 11,725; 0.73 / 0.79 ms; 5.49 s |

**C640 in this sweep:** the run aborted during warmup after the frontend logged Dynamo
`TCP request timeout` errors to the decode worker. This recipe keeps that configuration, without
`DYN_TCP_REQUEST_TIMEOUT`. The next sweep (run 42183317699, commit `12c6b381`) changed only this:
it adds frontend `DYN_TCP_REQUEST_TIMEOUT: '120'`, and C640 then completed. The other six configs
are identical in both sweeps.

## What is identical to CI

Every serving setting is the `config.yaml` srtctl resolved in that run: role layout, GPU
placement, vLLM args (including CI's dspark `synthetic_acceptance_length: 4.04` override),
worker and frontend env, Mooncake services and segment sizes, the `host_setup` memory-clock check
(4752 MHz), `identity` (Dynamo 1.5.0) and `srun_options.cpu-bind: none`. The Slurm request matches
the CI `#SBATCH` lines: account `nv-shared-1`, partition `batch-r37`, `--exclusive`,
`--cpus-per-task=352`, `--gpus-per-node=4`, 2 h. The container mounts match too, except for the
CI-only InferenceMAX workspace.

- Image: `/home/rihuo/m31f-vr200-final-n1-397e38ee-xdr.sqsh` (Dynamo 1.5.0 baked in, `dynamo.install: false`)
- Model: `/raid/common-models/minimax-m3.1-flash` (node-local; mounted at `/model`, draft at `/model/dspark`)

## What differs from CI

- **Client:** CI ran InferenceMAX's `benchmarks/srt_agentic.sh`. These recipes install the same
  `agentx-harness==1.0.13` and `datasets==5.1.0` into a uv Python 3.11 venv and run
  `agentx profile` with CI's exact flags and `AIPERF_*` environment. The CPU pinning
  (`taskset`) is unchanged. InferenceMAX's post-processing (result aggregation, power audit,
  validation) is not run; raw artifacts land in `/logs/agentic/conc_<C>/aiperf_artifacts`.
- **C640 client:** `4pdep2-1ddep8-c640.yaml` instead uses the client setup of
  `recipes/trtllm/glm5.3/vrnvl72-lab-w2-fp8/glm5.3-disagg-5p1d-tp4-dep16-c620.yaml`: the local
  AgentX checkout `/home/rihuo/agentx-harness` runs
  `aiperf profile --scenario inferencex-agentx-mvp` with that recipe's flags and `AIPERF_*`
  environment. As in that recipe, the Dynamo frontend and the client share one dedicated node;
  the frontend drops CI's `INFMAX_API_CPUS` pinning. It sends
  `X-Dynamo-Session-ID`, which CI did not, so the KV router's session affinity applies. Artifacts
  land in `/logs/agentic/aiperf_results`.
- **Dropped:** CI-only `post_eval` and `srun_options.container-workdir: /infmax-workspace`.
- **Names:** `name: m31f-r37-<recipe>` instead of the CI runner name.

## Requirements

1. srtctl at `1ca857b2` or later. Older checkouts reject the disaggregated recipes: per-role
   `mooncake_store_config` for vLLM arrived in that commit.
2. R37 `srtslurm.yaml` with the cluster settings CI used for that run:

   ```yaml
   network_interface: eth0
   use_segment_sbatch_directive: false   # CI's sbatch had no --segment
   default_bash_preamble: export PATH=/configs/m31-launch-bin:$PATH
   ```

   The preamble makes `python3` resolve to `configs/m31-launch-bin/python3`, the launch adapter
   that applies `INFMAX_API_CPUS` (pins the frontend and API processes) and
   `INFMAX_NUMA_PHYSICAL` (C640 prefill: maps `numa-bind-*` lists to the worker's physical GPUs).
   Without it those variables are ignored, and C640 prefill workers on GPUs 2,3 bind to NUMA
   node 0. The adapter passes any other command through unchanged.
3. Read access to `/home/nv-shared-1/{ai-perf-cache,hf-hub-cache}` (the aiperf mmap cache and HF
   dataset cache) and internet access from compute nodes for the client venv (as in CI).
4. For the C640 recipe, an AgentX checkout at `/home/rihuo/agentx-harness` on R37.

## Run

```bash
srtctl apply -f recipes/trtllm/minimax3.1/4pdep2-1ddep8-c640.yaml --no-preflight
```

`--no-preflight` is only needed when `/raid/common-models` is not visible on the login node.
