# Qwen3.5 Dynamo engine caches

These recipes target the `q35repro-09607ea47b` image: TRT-LLM release
`1.3.0a0.dev202610151219` (base source `1b480b5af9`) with the Python and compiled
overlays from source revision `09607ea47b`. The image records its provenance at
`/opt/qwen35_pareto_repro/`.

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
