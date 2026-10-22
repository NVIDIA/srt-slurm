# Qwen3.5 Dynamo engine environment and caches

The engine environment targets TRT-LLM source `45a52f216772297d2361bb55b94c78fcd02c80ba`,
including the Q35 role profiles introduced in `4d09932ca7`. Use an image built
from this implementation before running these recipes.

The existing `model.container` paths still name the original
`q35repro-09607ea47b` image; rebuild that runtime image or replace those paths
with a new build. That older implementation cannot replace the explicit Q35
switches with profiles. `identity.container.image` records the original image
provenance: TRT-LLM release `1.3.0a0.dev202610151219` (base source `1b480b5af9`)
with Python and compiled overlays from `09607ea47b`.

The directory contains ten Pareto configurations. The low-concurrency additions
are TEP16 with concurrency 8/batch 1 and concurrency 16/batch 2, and TEP8 with
concurrency 28/batch 4. Their engine tuning and worker placement follow the
supplied recipes.

## Cluster paths

The Dynamo recipes use the current Hecate configuration: model and runtime image
under `/lustre/fsw/coreai_comparch_inferencex/rihuo/`, a `/lustre` mount, and
`install_dynamo_wheel.sh` with the local `dynamo_wheels` directory mounted at
`/dynamo_wheels`. They inherit Slurm account, partition and walltime from the
cluster configuration. `identity.container.image` retains the original image
provenance; `model.container` selects the image used at runtime.

The benchmark installs AIPerf from the local `agentx-harness` directory, keeps
Hugging Face downloads in `hf_cache` and dataset mmap data in
`aiperf_mmap_cache`, all beneath the same Lustre directory. Real-time AIPerf
metrics remain enabled and OTEL remains disabled.

## Engine environment defaults

Each prefill worker selects `TRTLLM_Q35_PROFILE: ctx`; each decode worker
selects `TRTLLM_Q35_PROFILE: gen`. In `tensorrt_llm/_torch/q35_env.py` at the
source revision above, the generation profile supplies all 89 former decode
settings, including SM103 MoE tactic pins and fused-front row counts. The
context profile supplies all ten former non-DP prefill settings. With attention
DP enabled it supplies no settings, matching the DEP recipes' original prefill
environment. These comparisons include every switch value, not just its name.

Profiles apply only to Qwen3.5 models on SM103 (GB300). They are opt-in:
omitting `TRTLLM_Q35_PROFILE` would leave the optimization switches off.
Explicit switch overrides take precedence over a profile. Model loading
activates the profile before building layers and refreshes registered readers.

Two settings remain explicit at their engine default values:

| Retained variable | Value | Source |
| --- | --- | --- |
| `TRTLLM_ENABLE_PDL` | `1` | Python PDL readers and `cpp/tensorrt_llm/common/envUtils.cpp` |
| `TRTLLM_CTX_SCHED_SRPT_AGE_CAP` | `2048` | `_torch/pyexecutor/scheduler/scheduler_v2.py` |

Other engine switches remain explicit because the profiles do not supply
them and their values differ from engine defaults. For example, KV transfer
uses eight threads instead of one, and SRPT aging uses eight seconds instead
of being disabled. `TLLM_LOG_LEVEL: INFO` and `TLLM_PROFILE_LOG_RANKS: all`
also differ from the defaults (`error` and rank `0`). CUDA, allocator, NCCL,
UCX and Dynamo tuning remains explicit.

Validation: all ten recipes pass local dry-run and mock launches, with the
expected profiles in worker environments. Expanding the profiles
reproduces all 20 original role environments. No cluster runs were
performed for this cleanup.

## Default filesystem caching

The recipes omit the six cache-location overrides below and use the libraries'
defaults. Removing a location override leaves the cache enabled. These defaults
apply when the image or submitting environment does not supply another override.

| Omitted variable | Purpose | Default location |
| --- | --- | --- |
| `CUDA_CACHE_PATH` | CUDA driver JIT results | `~/.nv/ComputeCache` on Linux |
| `CUTE_DSL_CACHE_DIR` | CuTe DSL compiled-kernel file cache | Temporary-directory `<user>/cutlass_python_cache`, with a `cutlass_python_cache` fallback |
| `FLASHINFER_WORKSPACE_BASE` | Filesystem base for FlashInfer generated sources and compiled artifacts | `$HOME`; artifacts beneath `.cache/flashinfer/<version>/<architecture>/` |
| `TORCHINDUCTOR_CACHE_DIR` | PyTorch Inductor compilation artifacts, when Inductor is used | Temporary-directory `torchinductor_<user>` |
| `TRITON_CACHE_DIR` | Triton compiled kernels and metadata | Standalone Triton uses `$TRITON_HOME/.triton/cache`, with `TRITON_HOME` defaulting to `$HOME`; Inductor may set its own Triton subdirectory |
| `XDG_CACHE_HOME` | General library-cache fallback | Usually `$HOME/.cache` on Linux, for libraries that honor XDG |

`FLASHINFER_WORKSPACE_BASE` does not control the size of GPU workspace buffers.
The source requirements at `09607ea47b` specify FlashInfer 0.6.16, Triton 3.6.0
and CUTLASS DSL 4.5.0. The installed image versions remain the authority.

References: [CUDA cache defaults](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-cache-path),
[FlashInfer 0.6.16 paths](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16/flashinfer/jit/env.py),
[CuTe DSL 4.5.0 paths](https://github.com/NVIDIA/cutlass/blob/v4.5.0/python/CuTeDSL/cutlass/base_dsl/cache_helpers.py),
[Triton 3.6.0 configuration](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/knobs.py),
[PyTorch cache configuration](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html).

## Expected performance impact

Removing the path overrides does not directly change the kernels or disable
compilation caches. With equivalent artifacts and warmed kernel shapes,
steady-state throughput, TTFT and ITL should be similar. This is an expectation
from source behavior, not a measured result for these recipes.

Startup and first-use latency can change: default locations may contain warm
artifacts, require cold compilation, or reside on slower storage. A newly seen
kernel specialization during measurement can affect tail latency. Where a
library persists autotuning results, different cache contents can also change
whether a tactic is reused or remeasured. Default directories must be writable.

The recipes request no dedicated cache mount. Their previous fixed `/tmp`
paths normally received Enroot's fresh tmpfs, while some library defaults are
under `$HOME`. Whether those defaults are private between runs depends on
container and cluster mounts. Omitting path settings does not enforce a cold
cache or disable sharing through a persistent home directory.

## Separate runtime and benchmark caches

- **TRT-LLM autotuner:** `TLLM_AUTOTUNER_CACHE_PATH` is omitted. At
  `09607ea47b`, `_torch/pyexecutor/model_engine.py` passes this optional path
  to `_torch/autotuner.py`; file loading/saving occurs only when it is set.
  With it unset, tactic choices live in process memory. An inherited value can
  enable persistent tuning results independently of the six variables above.
- **Q35 optimizations:** cached sampling parameters, preparation metadata,
  index buffers, stream events and recurrent snapshots live in worker memory.
- **KV offload:** `kv_cache_config.host_cache_size: 137438953472` configures
  128 GiB of host-memory KV capacity. No recipe configures `disk_cache_size` or
  `disk_cache_path`; NIXL transfers KV between live workers.
- **Frontend tokenizer cache:** `DYN_TOKENIZER_CACHE_BYTES: '8000000000'` sets
  a process-memory budget.
- **Benchmark caches:** `HF_HOME`, `HF_HUB_CACHE` and
  `AIPERF_DATASET_MMAP_CACHE_DIR` configure shared dataset/download storage
  for the benchmark client.

The in-memory optimizations and KV offload remain enabled by the recipes.
