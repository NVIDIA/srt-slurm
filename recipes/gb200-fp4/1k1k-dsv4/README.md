# DeepSeek-V4-Pro (1.6T MoE, MXFP4) — 1k/1k aggregated on GB200

NVIDIA-verified SGLang recipes for **DeepSeek-V4-Pro** (MXFP4) on **GB200**
(ARM64 Grace + Blackwell, 4 GPU per node), aggregated mode, 1k / 1k workload.
GB200 HBM per GPU is smaller than GB300, so the 1.6T MXFP4 checkpoint only fits
across **2 nodes (8 GPUs) at TP=8**.

## Container

Same Grace+Blackwell aarch64 image as GB300 (shared enroot sqsh alias
`dsv4-grace-blackwell` in `srtslurm.yaml.example`).

## Recipes

| file | parallelism | MTP | notes |
|---|---|---|---|
| `agg-2n-low-latency.yaml` | TP=8 | EAGLE 3/4 | low-latency, 2-node |
| `agg-2n-nomtp.yaml`       | TP=8 | —         | throughput, 2-node  |

See `recipes/gb300-fp4/1k1k-dsv4/README.md` for the full flag rationale —
flags are identical to the GB300 2-node recipes apart from the partition.
