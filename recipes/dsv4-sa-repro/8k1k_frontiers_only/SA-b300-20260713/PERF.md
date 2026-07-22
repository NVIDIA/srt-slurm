# Reference perf — DeepSeek-V4-Pro SA-bench disaggregated, bia B300 (SA-b300-20260713)

Native trtllm-serve (srt-slurm) E2E, migrated from bench-trtllm-disagg. **8k frontier is
best-of trtllm + megamoe** (8 megamoe points, at the high-tps/gpu end where MEGAMOE_DEEPGEMM
beats TRTLLM). backend column marks each point.
tps/user=1000/median_TPOT, tps/gpu=total_tps/(ctx_num·ctx_rank+gen_num·gen_rank). Frontier =
per-config saturation-collapse then Pareto (per mtp variant).

## 8k/1k ISL

### 8k · mtp0 — best-of frontier (10 pts)

| config | backend | conc | GPUs | tps/user | tps/gpu | total_tps | median TPOT | median TTFT |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 10p1d-dep4-dep8-b512 | **mm** | 4301 | 48 | 15.6 | 10,977 | 526,874 | 64.06 | 3,803 |
| 7p1d-dep4-dep8-b256 | **mm** | 2253 | 36 | 21.7 | 10,152 | 365,476 | 46.19 | 4,349 |
| 5p1d-dep4-dep8-b128 | **trtllm** | 1229 | 28 | 28.1 | 8,581 | 240,267 | 35.59 | 6,648 |
| 3p1d-dep4-dep8-b64 | **trtllm** | 666 | 20 | 35.0 | 7,486 | 149,727 | 28.57 | 7,973 |
| 2p1d-dep4-dep8-b32 | **trtllm** | 282 | 16 | 39.9 | 5,320 | 85,118 | 25.04 | 2,325 |
| 1p1d-dep4-tep8-b128 | **trtllm** | 180 | 12 | 45.5 | 4,110 | 49,319 | 21.98 | 8,313 |
| 2p3d-dep4-tep8-b64 | **trtllm** | 270 | 32 | 54.0 | 2,737 | 87,578 | 18.53 | 7,056 |
| 1p3d-dep4-tep8-b32 | **trtllm** | 144 | 28 | 63.1 | 1,824 | 51,062 | 15.84 | 7,493 |
| 1p4d-dep4-tep8-b8 | **trtllm** | 44 | 36 | 82.0 | 620 | 22,321 | 12.19 | 3,442 |
| 1p4d-dep4-tep8-b2 | **trtllm** | 12 | 36 | 96.3 | 186 | 6,681 | 10.39 | 4,616 |

*vs bench (same backend) Δ%:*

| config | backend | conc | Δ tps/gpu | Δ tps/user | Δ TPOT |
|---|---|--:|--:|--:|--:|
| 10p1d-dep4-dep8-b512 | mm | 4301 | +0.7% | +0.2% | -0.2% |
| 7p1d-dep4-dep8-b256 | mm | 2253 | -0.1% | +0.1% | -0.1% |
| 5p1d-dep4-dep8-b128 | trtllm | 1229 | -0.1% | -0.0% | +0.0% |
| 3p1d-dep4-dep8-b64 | trtllm | 666 | -0.2% | -0.2% | +0.2% |
| 2p1d-dep4-dep8-b32 | trtllm | 282 | +0.7% | +0.7% | -0.7% |
| 1p1d-dep4-tep8-b128 | trtllm | 180 | +0.3% | +0.3% | -0.3% |
| 2p3d-dep4-tep8-b64 | trtllm | 270 | +0.7% | +0.1% | -0.1% |
| 1p3d-dep4-tep8-b32 | trtllm | 144 | +0.4% | -0.1% | +0.1% |
| 1p4d-dep4-tep8-b8 | trtllm | 44 | -0.3% | -0.4% | +0.4% |
| 1p4d-dep4-tep8-b2 | trtllm | 12 | +0.3% | +0.6% | -0.6% |

**8k mtp0 Δ** — tps/gpu: med +0.32%, |max| 0.7% · tps/user: med +0.04%, |max| 0.7% · TPOT: med -0.04%, |max| 0.7%

### 8k · mtp (MTP) — best-of frontier (10 pts)

| config | backend | conc | GPUs | tps/user | tps/gpu | total_tps | median TPOT | median TTFT |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 12p1d-dep4-dep8-b512 | **mm** | 4301 | 56 | 18.4 | 11,059 | 619,327 | 54.46 | 3,118 |
| 10p1d-dep4-dep8-b256 | **mm** | 2253 | 48 | 29.4 | 10,383 | 498,377 | 34.05 | 3,316 |
| 7p1d-dep4-dep8-b128 | **mm** | 1229 | 36 | 42.2 | 9,958 | 358,484 | 23.72 | 4,456 |
| 5p1d-dep4-dep8-b64 | **trtllm** | 666 | 28 | 53.4 | 8,151 | 228,218 | 18.73 | 5,254 |
| 3p1d-dep4-dep8-b32 | **mm** | 333 | 20 | 67.3 | 7,172 | 143,449 | 14.86 | 4,276 |
| 2p1d-dep4-dep8-b16 | **trtllm** | 180 | 16 | 92.1 | 6,165 | 98,646 | 10.86 | 4,193 |
| 2p1d-dep4-dep8-b8 | **trtllm** | 90 | 16 | 104.8 | 3,543 | 56,691 | 9.54 | 3,101 |
| 2p3d-dep4-tep8-b32 | **trtllm** | 144 | 32 | 120.3 | 3,032 | 97,013 | 8.31 | 3,927 |
| 1p2d-dep4-tep8-b16 | **trtllm** | 48 | 20 | 141.1 | 1,894 | 37,890 | 7.09 | 3,381 |
| 1p5d-dep4-tep4-b1 | **trtllm** | 10 | 24 | 209.1 | 382 | 9,178 | 4.78 | 4,518 |

*vs bench (same backend) Δ%:*

| config | backend | conc | Δ tps/gpu | Δ tps/user | Δ TPOT |
|---|---|--:|--:|--:|--:|
| 12p1d-dep4-dep8-b512 | mm | 4301 | +0.3% | +0.6% | -0.6% |
| 10p1d-dep4-dep8-b256 | mm | 2253 | -0.6% | -0.8% | +0.8% |
| 7p1d-dep4-dep8-b128 | mm | 1229 | -0.1% | -0.1% | +0.1% |
| 5p1d-dep4-dep8-b64 | trtllm | 666 | -0.1% | -0.2% | +0.2% |
| 3p1d-dep4-dep8-b32 | mm | 333 | -0.0% | +0.0% | -0.0% |
| 2p1d-dep4-dep8-b16 | trtllm | 180 | +0.1% | -0.4% | +0.4% |
| 2p1d-dep4-dep8-b8 | trtllm | 90 | +0.3% | +0.3% | -0.3% |
| 2p3d-dep4-tep8-b32 | trtllm | 144 | +0.2% | -0.1% | +0.1% |
| 1p2d-dep4-tep8-b16 | trtllm | 48 | -0.2% | -0.0% | +0.0% |
| 1p5d-dep4-tep4-b1 | trtllm | 10 | +1.3% | -0.2% | +0.2% |

**8k mtp Δ** — tps/gpu: med +0.03%, |max| 2.0% · tps/user: med -0.10%, |max| 0.8% · TPOT: med +0.10%, |max| 0.8%

## Notes
- **Pruned (2026-07-14):** selected 8k frontier points were dropped and all non-frontier
  scatter recipes removed from `recipes/`. The `(N pts)` headers reflect the pruning, but the
  per-section **Δ** summary lines (med/|max|) predate it and are no longer exact.
- 8k megamoe points reproduce bench megamoe; trtllm points reproduce bench trtllm (Δ per section).
- megamoe = decode moe_config.backend MEGAMOE_DEEPGEMM (same trtllm_serve stack); recipes tagged `-mm`.
- Full per-point data: E2E-8k_scatter.csv. Bad nodes excluded: bia0005,0054,0058,0060,0103,0106.
