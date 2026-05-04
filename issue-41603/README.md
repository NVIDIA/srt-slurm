# vLLM issue 41603 reproduction artifacts

Artifacts for https://github.com/vllm-project/vllm/issues/41603.

Primary srt-slurm reproducer:

- `recipes/vllm/deepseek-v4-pro/GB200/8k1k/disagg-gb200-high-tpt-megamoe-mtp2.yaml`

Recovery patch tested in job 15975:

- `configs/patches/vllm-container-deps-revert-pr41015.sh`
- `configs/patches/vllm_revert_pr41015_fp4_cvt.py`

Each subdirectory contains the `recipe.lock.yaml` captured by srt-slurm plus
the `benchmark-rollup.csv` for the same job.

| Directory | Job | Container / patch | Result summary |
|---|---:|---|---|
| `good-pr-container-15908/` | 15908 | `vLLM 0.20.1rc1.dev38+g61c3a50f4` PR container | Good reference: about 7,489 total tok/s/GPU |
| `nightly-a749-15902/` | 15902 | `vLLM 0.20.1rc1.dev91+ga749a33d8` nightly | Regression: about 5,305 total tok/s/GPU |
| `official-v0201-15963/` | 15963 | `vllm/vllm-openai:v0.20.1-ubuntu2404` | Bad high-throughput result: about 2,669 total tok/s/GPU |
| `nightly-a749-revert-pr41015-15975/` | 15975 | nightly plus local revert of vLLM PR #41015 FP32->FP4 cvt path | Recovers throughput: about 7,326 total tok/s/GPU |
