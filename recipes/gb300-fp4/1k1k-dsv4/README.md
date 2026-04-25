# DeepSeek-V4-Pro (1.6T MoE, MXFP4) — 1k/1k on GB300

This directory contains NVIDIA-verified SGLang recipes for **DeepSeek-V4-Pro**
(1.6T-parameter MoE with MXFP4 MoE weights + FP8 KV, UE8M0 scales) on **GB300**
(ARM64 Grace + Blackwell, 4 GPU per node), 1024 input / 1024 output workload.
Both **aggregated** (single-node SGLang) and **disaggregated** (1P+1D dynamo +
NIXL) serving modes are covered.

## Container

All recipes reference the `dsv4-grace-blackwell` alias defined in
`srtslurm.yaml.example`. Pull + convert:

```bash
enroot import --output sglang-deepseek-v4-grace-blackwell.sqsh \
  docker://lmsysorg/sglang:deepseek-v4-grace-blackwell
```

(Use the `deepseek-v4-blackwell` image for B200 x86_64, or `deepseek-v4-hopper` for H200.)

## Model checkpoint

```bash
hf download deepseek-ai/DeepSeek-V4-Pro --local-dir /shared/models/deepseek/DeepSeek-V4-Pro
```

## Recipes

### Aggregated (single SGLang server)

| file | parallelism | MTP | target | notes |
|---|---|---|---|---|
| `agg-low-latency.yaml`  | TP=4                        | EAGLE 3/4 | minimum TPOT / best per-user latency | GB300 1 node |
| `agg-nomtp.yaml`        | TP=4                        | —         | baseline throughput, no spec decoding | GB300 1 node |
| `agg-balanced-tep.yaml` | TP=4 + DP=4 + DP-attn + DeepEP | EAGLE 1/2 | Pareto mid-curve                     | GB300 1 node |
| `agg-max-tpt-tep.yaml`  | TP=4 + DP=4 + DP-attn + DeepEP | —         | maximum TPS/GPU                      | GB300 1 node |
| `agg-2n-low-latency.yaml` | TP=8                      | EAGLE 3/4 | low-latency, 2× memory headroom     | GB300 2 nodes |
| `agg-2n-nomtp.yaml`     | TP=8                        | —         | throughput, 2× memory headroom       | GB300 2 nodes |

### Disaggregated (dynamo frontend, NIXL KV transfer)

| file | parallelism | MTP | target | notes |
|---|---|---|---|---|
| `disagg-1p1d-tp4-mxfp4.yaml` | 1P + 1D, both TP=4 | — | steady decode TPOT at high concurrency | GB300 2 nodes (1P + 1D) |

## Key flags (derived from the SGLang DSv4 cookbook)

- `moe-runner-backend: flashinfer_mxfp4` — MXFP4 MoE kernels (Blackwell only).
- `chunked-prefill-size: 4096` + `disable-flashinfer-autotune: true` — cookbook recipe.
- `disable-radix-cache: true` — synthetic benchmark best practice; also
  reduces contiguous-allocator fragmentation at weight-reorder time.
- `mem-fraction-static: 0.78` — leaves headroom for the MXFP4
  `reorder_w1w3_to_w3w1` path (0.82 intermittently OOMs on GB300).
- TEP recipes: `enable-dp-attention + moe-a2a-backend: deepep` plus
  `deepep-config num_sms=96` (DeepEP `DEEPEP_LARGE_SMS_FLAG` for single-node
  Blackwell per cookbook).

## References

- [SGLang cookbook: `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)
- [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- Upstream SGLang PR: sgl-project/sglang#23600
