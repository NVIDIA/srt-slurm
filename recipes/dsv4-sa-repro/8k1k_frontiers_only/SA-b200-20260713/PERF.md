# Reference perf — DeepSeek-V4-Pro SA disaggregated, B200 (bench e2e redo2/redo4 baseline)

Native **trtllm-serve** via srt-slurm (srtctl), STOCK `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc15.post1`
(= what the bench flow actually executes — its runtime wheel install silently no-ops), sa-bench client,
corrected `random-v2` dataset (`dataset_name: custom`), steady main run = 10×conc @ max-concurrency, ignore_eos.

Definitions: **tps/user** = `1000/mean_TPOT` · **tps/gpu** = `output_throughput/(ctx·8+gen·8)` ·
TTFT/TPOT are means (median/p99 in `tables/`). **Δ%** = (srt−bench)/bench on tps/gpu.

Full data (all measured points, b200_tables schema + bench comparison columns):
`tables/SRT-E2E-8k_scatter.csv`; per-variant frontiers in `tables/SRT-E2E-8k_frontier_{mtp0,mtp}.csv`.


## 8k1k (ISL/OSL 8192/1024) — 67 measured points


### Pareto frontier — mtp0 (10 points)

| recipe | GPUs | tps/user srt | tps/user bench | tps/gpu srt | tps/gpu bench | Δ% | TPOT srt | TPOT bench | TTFT srt | TTFT bench |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `disagg-b200-8k1k-1p5d-dep8-tep8-b32-mtp0 @c5` | 48 | 95.3 | — | 9 | — | — | 10.5 | — | 809 | — |
| `disagg-b200-8k1k-1p4d-dep8-tep8-b4-mtp0 @c16` | 40 | 85.1 | 85.0 | 31 | 30 | +0.1 | 11.7 | 11.7 | 968 | 1,023 |
| `disagg-b200-8k1k-1p5d-dep8-tep8-b32-mtp0 @c40` | 48 | 77.7 | 77.6 | 57 | 57 | +0.1 | 12.9 | 12.9 | 1,177 | 1,211 |
| `disagg-b200-8k1k-1p5d-dep8-tep8-b32-mtp0 @c175` | 48 | 59.6 | 59.3 | 178 | 178 | -0.0 | 16.8 | 16.8 | 2,687 | 2,708 |
| `disagg-b200-8k1k-1p3d-dep8-dep8-b8-eplb384-mtp0 @c210` | 32 | 50.3 | 50.1 | 259 | 257 | +0.6 | 19.9 | 19.9 | 3,996 | 4,088 |
| `disagg-b200-8k1k-2p3d-dep8-dep8-b16-mtp0 @c423` | 40 | 44.1 | 44.2 | 388 | 388 | -0.1 | 22.7 | 22.6 | 3,133 | 3,202 |
| `disagg-b200-8k1k-3p2d-dep8-dep8-b32-mtp0 @c564` | 40 | 39.6 | 39.5 | 466 | 468 | -0.4 | 25.3 | 25.3 | 3,279 | 3,316 |
| `disagg-b200-8k1k-2p1d-dep8-dep8-b64-eplb384-mtp0 @c563` | 24 | 34.8 | 34.2 | 679 | 669 | +1.5 | 28.8 | 29.1 | 4,011 | 4,155 |
| `disagg-b200-8k1k-3p1d-dep8-dep8-b128-eplb384-mtp0 @c1126` | 32 | 27.6 | 27.1 | 789 | 787 | +0.3 | 36.2 | 36.4 | 5,984 | 6,069 |
| `disagg-b200-8k1k-5p1d-dep8-dep8-b256-eplb384-mtp0 @c2150` | 48 | 20.4 | 20.1 | 799 | 797 | +0.3 | 48.9 | 48.9 | 4,677 | 4,829 |

### Pareto frontier — mtp (mtp1/mtp3) (10 points)

| recipe | GPUs | tps/user srt | tps/user bench | tps/gpu srt | tps/gpu bench | Δ% | TPOT srt | TPOT bench | TTFT srt | TTFT bench |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `disagg-b200-8k1k-1p5d-dep8-tep8-b8-mtp3 @c5` | 48 | 220.7 | — | 19 | — | — | 4.5 | — | 818 | — |
| `disagg-b200-8k1k-1p5d-dep8-tep8-b8-mtp3 @c10` | 48 | 202.7 | — | 34 | — | — | 4.9 | — | 891 | — |
| `disagg-b200-8k1k-1p4d-dep8-tep8-b4-mtp3 @c16` | 40 | 178.9 | 179.0 | 58 | 58 | +0.7 | 5.6 | 5.6 | 999 | 1,073 |
| `disagg-b200-8k1k-1p5d-dep8-tep8-b8-mtp3 @c45` | 48 | 154.3 | 153.9 | 112 | 112 | +0.4 | 6.5 | 6.5 | 1,414 | 1,452 |
| `disagg-b200-8k1k-1p4d-dep8-dep8-b2-eplb384-mtp3 @c72` | 40 | 130.1 | 130.3 | 180 | 180 | +0.2 | 7.7 | 7.7 | 1,708 | 1,727 |
| `disagg-b200-8k1k-1p2d-dep8-dep8-b4-eplb384-mtp3 @c70` | 24 | 115.6 | 115.3 | 269 | 267 | +0.7 | 8.7 | 8.7 | 1,590 | 1,649 |
| `disagg-b200-8k1k-3p2d-dep8-dep8-b16-mtp3 @c282` | 40 | 91.0 | 91.2 | 525 | 524 | +0.2 | 11.0 | 10.9 | 1,717 | 1,774 |
| `disagg-b200-8k1k-2p1d-dep8-dep8-b32-mtp3 @c282` | 24 | 72.8 | 73.1 | 698 | 698 | +0.1 | 13.7 | 13.6 | 2,259 | 2,377 |
| `disagg-b200-8k1k-3p1d-dep8-dep8-b64-eplb384-mtp3 @c563` | 32 | 55.9 | 55.1 | 765 | 756 | +1.3 | 17.9 | 17.9 | 3,878 | 4,093 |
| `disagg-b200-8k1k-5p1d-dep8-dep8-b128-eplb384-mtp3 @c1126` | 48 | 39.7 | 38.9 | 775 | 773 | +0.3 | 25.2 | 25.2 | 3,743 | 3,777 |

All 60 bench-matched 8k1k points — srt÷bench ratios:
- tps/gpu: mean 1.003 · median 1.002 · min 0.993 · max 1.036
- tps/user: mean 1.004 · median 1.002 · min 0.994 · max 1.020
- TPOT (lower=faster): mean 1.000 · median 1.000 · min 0.984 · max 1.011
- TTFT: mean 0.978 · median 0.977 · min 0.836 · max 1.133 (load-dependent; see per-point table)


## Notes & known issues

- **Pruned (2026-07-14):** selected frontier points were dropped and all non-frontier
  scatter recipes removed from `recipes/`. The `(N points)` headers reflect the pruning,
  but the "## measured points" totals and the aggregate srt÷bench ratio summaries above
  predate it and are no longer exact.

- **Run with srt-slurm `main` + the trtllm-serve frontend.** srtctl-side fixes used for these
  runs: pre-create `outputs/<job>/logs/`; no CLI `--tp/--ep/--pp` for trtllm_serve; sa-bench
  profiling-lib dispatch by `PROFILING_BACKEND`.
- **`frontend.args {server_start_timeout: 7200, request_timeout: 7200}` is load-bearing** — the
  180s default triggers an orchestrator retry storm at high concurrency (every request
  re-issued, 2× ctx load, locked slow equilibrium).
- **ctx `free_gpu_memory_fraction` exported as 0.5** (bench configs say 0.7): the stock
  `trtllm-serve disaggregated` orchestrator needs a GPU (~5.3GB) on the prefill head node and
  srtctl starts it after the workers fill memory; bench only survives 0.7 because its
  orchestrator starts during model load. Verified ratio 1.00 at 0.5.
- **Provenance correction:** bench's own disagg SOL/E2E runs execute the STOCK NGC image — the
  runtime `pip install <wheel>[devel]` silently no-ops ("already satisfied"). The custom
  feat/deepseek_v4 wheel (d6d1b48) has a ~3.2× disagg-ctx host-side regression and is NOT what
  either side's numbers measure.
- **Known intermittent failure (stock engine bug):** MTP3 + CUDA graph (tep8, gen frac 0.9) can
  hit a probabilistic `CUDA illegal memory access` in decode sampling
  (`currentStreamCaptureStatusMayInitCtx`; surfaces as Xid 31). Matches bench bringup §6.4/§6.9.
  Retry the point; if it recurs, drop that config's gen frac to 0.8.
- 8k1k `ctx5_gen1_dep8_batch128_eplb0_mtp3` has NO bench baseline (bench's own job died);
  srt-slurm numbers are absolute-only.

Generated by `srt-slurm/tools/gen_dsv4_recipes.py` (per-config recipes, engine configs verbatim
from the bench run dirs) → `tools/make_srt_tables.py` (b200_tables-style CSVs) →
`tools/export_sa_b200.py` (this folder).
