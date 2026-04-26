# Accuracy Benchmarks

In srt-slurm, users can run different accuracy benchmarks by setting the benchmark section in the config yaml file. Supported benchmarks include `mmlu`, `gpqa`, `longbenchv2`, and `lm-eval`.

## Table of Contents

- [MMLU](#mmlu)
- [GPQA](#gpqa)
- [LongBench-V2](#longbench-v2)
  - [Configuration](#configuration)
  - [Parameters](#parameters)
  - [Available Categories](#available-categories)
  - [Example: Full Evaluation](#example-full-evaluation)
  - [Example: Quick Validation](#example-quick-validation)
  - [Output](#output)
  - [Important Notes](#important-notes)
- [lm-eval](#lm-eval)

---

**Note**: The `context-length` argument in the config yaml needs to be larger than the `max_tokens` argument of accuracy benchmark.


## MMLU

For MMLU dataset, the benchmark section in yaml file can be modified in the following way:
```bash
benchmark:
  type: "mmlu"
  num_examples: 200 # Number of examples to run
  max_tokens: 2048 # Max number of output tokens
  repeat: 8 # Number of repetition
  num_threads: 512 # Number of parallel threads for running benchmark
```
 
Then launch the script as usual:
```bash
srtctl apply -f config.yaml
```

After finishing benchmarking, the `benchmark.out` will contain the results of accuracy:
```
====================
Repeat: 8, mean: 0.812
Scores: ['0.790', '0.820', '0.800', '0.820', '0.820', '0.790', '0.820', '0.840']
====================
Writing report to /tmp/mmlu_deepseek-ai_DeepSeek-R1.html
{'other': np.float64(0.9), 'other:std': np.float64(0.30000000000000004), 'score:std': np.float64(0.36660605559646725), 'stem': np.float64(0.8095238095238095), 'stem:std': np.float64(0.392676726249301), 'humanities': np.float64(0.7428571428571429), 'humanities:std': np.float64(0.4370588154508102), 'social_sciences': np.float64(0.9583333333333334), 'social_sciences:std': np.float64(0.19982631347136331), 'score': np.float64(0.84)}
Writing results to /tmp/mmlu_deepseek-ai_DeepSeek-R1.json
Total latency: 465.618 s
Score: 0.840
Results saved to: /logs/accuracy/mmlu_deepseek-ai_DeepSeek-R1.json
MMLU evaluation complete
```


## GPQA
For GPQA dataset, the benchmark section in yaml file can be modified in the following way:
```bash
benchmark:
  type: "gpqa"
  num_examples: 198 # Number of examples to run
  max_tokens: 65536 # We need a larger output token number for GPQA
  repeat: 8 # Number of repetition
  num_threads: 128 # Number of parallel threads for running benchmark
```
The `context-length` argument here should be set to a value larger than `max_tokens`.


## LongBench-V2

LongBench-V2 is a long-context evaluation benchmark that tests model performance on extended context tasks. It's particularly useful for validating models with large context windows (128K+ tokens).

### Configuration

```yaml
benchmark:
  type: "longbenchv2"
  max_context_length: 128000  # Maximum context length (default: 128000)
  num_threads: 16             # Concurrent evaluation threads (default: 16)
  max_tokens: 16384           # Maximum output tokens (default: 16384)
  num_examples: 100           # Number of examples to run (default: all)
  categories:                 # Task categories to evaluate (default: all)
    - "single_doc_qa"
    - "multi_doc_qa"
    - "summarization"
    - "few_shot_learning"
    - "code_completion"
    - "synthetic"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_context_length` | int | 128000 | Maximum context length for evaluation. Should not exceed model's trained context window. |
| `num_threads` | int | 16 | Number of concurrent threads for parallel evaluation. Increase for faster throughput on high-capacity endpoints. |
| `max_tokens` | int | 16384 | Maximum tokens for model output. Must be less than `context-length` in sglang_config. |
| `num_examples` | int | all | Limit the number of examples to evaluate. Useful for quick validation runs. |
| `categories` | list | all | Specific task categories to run. Omit to run all categories. |

### Available Categories

LongBench-V2 includes the following task categories:

- **single_doc_qa**: Single document question answering
- **multi_doc_qa**: Multi-document question answering
- **summarization**: Long document summarization
- **few_shot_learning**: Few-shot learning with long context
- **code_completion**: Long-context code completion
- **synthetic**: Synthetic long-context tasks (needle-in-haystack, etc.)

### Example: Full Evaluation

Run complete LongBench-V2 evaluation with all categories:

```yaml
name: "longbench-v2-eval"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 2
  decode_nodes: 4

backend:
  type: sglang
  sglang_config:
    prefill:
      context-length: 131072  # Must exceed max_tokens
      tensor-parallel-size: 4
    decode:
      context-length: 131072
      tensor-parallel-size: 8

benchmark:
  type: "longbenchv2"
  max_context_length: 128000
  max_tokens: 16384
  num_threads: 32
```

### Example: Quick Validation

Run a quick subset for validation:

```yaml
benchmark:
  type: "longbenchv2"
  num_examples: 50           # Limit to 50 examples
  num_threads: 8
  categories:
    - "single_doc_qa"        # Only run single-doc QA
```

### Output

After completion, results are saved to the logs directory:

```bash
/logs/accuracy/longbenchv2_<model_name>.json
```

The output includes per-category scores and aggregate metrics:

```json
{
  "model": "deepseek-ai/DeepSeek-R1",
  "scores": {
    "single_doc_qa": 0.82,
    "multi_doc_qa": 0.78,
    "summarization": 0.85,
    "few_shot_learning": 0.76,
    "code_completion": 0.81,
    "synthetic": 0.92
  },
  "overall_score": 0.82,
  "total_examples": 500,
  "total_latency_s": 1842.5
}
```

### Important Notes

1. **Context Length**: Ensure `context-length` in your sglang_config exceeds `max_tokens` for the benchmark
2. **Memory**: Long-context evaluation requires significant GPU memory. Use appropriate `mem-fraction-static` settings
3. **Throughput**: Increase `num_threads` for faster evaluation, but monitor for OOM errors
4. **Categories**: Running specific categories is useful for targeted validation (e.g., just testing summarization capabilities)


## lm-eval

The `lm-eval` benchmark runner integrates [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) against the deployed OpenAI-compatible endpoint. By default, the runner invokes the `lm_eval` CLI directly (installing the `lm-eval` pip package on demand when it isn't already present in the container). An external eval workspace that exposes a compatible `benchmark_lib.sh` can be plugged in instead — see [External eval harness](#external-eval-harness) below.

### How it works

1. `do_sweep.py` starts infrastructure, workers, and the frontend for the normal recipe topology.
2. For `EVAL_ONLY=true`, `do_sweep.py` skips the throughput benchmark stage and runs `_run_post_eval()` directly after frontend startup.
3. `_run_post_eval()` waits for the OpenAI-compatible endpoint on port 8000 and, in eval-only mode, performs the full `wait_for_model()` health check for the configured prefill/decode or aggregated topology.
4. `_run_post_eval()` launches the registered `lm-eval` runner on the head node.
5. The runner script (`benchmarks/scripts/lm-eval/bench.sh`) uses `MODEL_NAME` from `do_sweep.py`, or auto-discovers the served model from `/v1/models` as a fallback.
6. The runner calls `lm_eval --model local-chat-completions --model_args base_url=<endpoint>/v1/chat/completions,model=<MODEL_NAME>,num_concurrent=<EVAL_CONCURRENT_REQUESTS>,tokenized_requests=False --tasks <LM_EVAL_TASKS>` and writes results to `/logs/eval_results/`.

### EVAL_ONLY mode

srt-slurm supports an `EVAL_ONLY` mode for jobs that should only validate accuracy. It is controlled by environment variables:

| Env var | Description |
|---------|-------------|
| `EVAL_ONLY` | Set to `true` to skip the throughput benchmark stage and run eval only |
| `RUN_EVAL` | Set to `true` to run eval after the throughput benchmark completes |
| `EVAL_CONC` | Concurrent requests for lm-eval; falls back to max of the recipe benchmark concurrency list |
| `MODEL_NAME` | Served model alias for OpenAI-compatible requests; set by `do_sweep.py` from `config.served_model_name` |
| `LM_EVAL_TASKS` | Comma-separated lm-evaluation-harness tasks (default: `gsm8k`) |
| `LM_EVAL_MODEL` | lm-evaluation-harness model type (default: `local-chat-completions`) |
| `LM_EVAL_EXTRA_ARGS` | Extra flags appended to the `lm_eval` command line |
| `EVAL_OUTPUT_DIR` | Where eval artifacts are written inside the container (default: `/logs/eval_results`) |

When `EVAL_ONLY=true`:
- Stage 4 skips the throughput benchmark entirely. No throughput result JSON is expected from srt-slurm.
- The eval path uses the full `wait_for_model()` health check before starting lm-eval.
- `_run_post_eval()` launches the `lm-eval` runner and returns its exit code.
- Eval failure is fatal because eval is the only purpose of the job.

When `RUN_EVAL=true` (without `EVAL_ONLY`):
- Throughput benchmark runs normally.
- After benchmark completes successfully, eval runs as a post-step.
- Eval failure is non-fatal; the benchmark job still succeeds if throughput passed.

### Topology metadata passthrough

`do_sweep.py` also forwards a set of topology/precision env vars to the eval container so downstream aggregation tooling can record them alongside the eval results:

| Env var | Purpose |
|---------|---------|
| `RUN_EVAL`, `EVAL_ONLY`, `IS_MULTINODE` | Whether eval runs and how artifacts are classified |
| `FRAMEWORK`, `PRECISION`, `MODEL_PREFIX`, `RUNNER_TYPE`, `SPEC_DECODING` | Benchmark identity metadata |
| `ISL`, `OSL`, `RESULT_FILENAME` | Sequence length and result-file metadata |
| `MODEL`, `MODEL_PATH`, `MODEL_NAME` | Model metadata and the served model alias |
| `MAX_MODEL_LEN`, `EVAL_MAX_MODEL_LEN` | Context-length metadata |
| `PREFILL_TP`, `PREFILL_EP`, `PREFILL_NUM_WORKERS`, `PREFILL_DP_ATTN` | Prefill-side topology metadata |
| `DECODE_TP`, `DECODE_EP`, `DECODE_NUM_WORKERS`, `DECODE_DP_ATTN` | Decode-side topology metadata |
| `EVAL_CONC`, `EVAL_CONCURRENT_REQUESTS` | Eval concurrency controls |

These variables are optional: they are passed through only when the launcher sets them, and the default lm-eval path works without any of them.

### Concurrency

The runner exports `EVAL_CONCURRENT_REQUESTS`, preferring `EVAL_CONC` when set and falling back to `256`:

```bash
export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-256}}"
```

When `EVAL_CONC` is not set, `do_sweep.py` defaults it to the max of the recipe benchmark concurrency list.

### Output

Eval artifacts are written to `/logs/eval_results/` inside the container (override with `EVAL_OUTPUT_DIR`):
- `results*.json` - lm-eval scores per task
- `sample*.jsonl` - per-sample outputs (when the harness emits them)

### External eval harness

If the launcher needs to drive lm-eval through an existing `benchmark_lib.sh` (for example, an internal evaluation library that also handles summary generation), mount that workspace into the container and point the runner at it with one of:

- `LM_EVAL_LIB` - absolute container path to the `benchmark_lib.sh` file.
- `LM_EVAL_WORKSPACE` - host path to a workspace; it is mounted at `/lm-eval-workspace` inside the container, and the runner sources `/lm-eval-workspace/benchmarks/benchmark_lib.sh`.

When a `benchmark_lib.sh` is found, the runner sources it, calls `run_eval --framework lm-eval` and (if exported) `append_lm_eval_summary`, and copies `meta_env.json`, `results*.json`, and `sample*.jsonl` from the CWD into `EVAL_OUTPUT_DIR`. Otherwise, the default in-container `lm_eval` CLI path is used.
