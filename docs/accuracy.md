# Accuracy Benchmarks

In srt-slurm, users can run different accuracy benchmarks by setting the benchmark section in the config yaml file. Supported benchmarks include `aime`, `mmlu`, `gpqa` and `longbenchv2`.

## Table of Contents

- [How Scoring Works](#how-scoring-works)
- [AIME](#aime)
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

---

**Note**: The `context-length` argument in the config yaml needs to be larger than the `max_tokens` argument of accuracy benchmark.


## How Scoring Works

Accuracy benchmarks send a fixed dataset through the running OpenAI-compatible endpoint and compare each model
response against the benchmark's expected answer. For AIME, NeMo Skills prompts the model to put the final answer in
`\boxed{...}`, extracts that final boxed answer, and grades it with its math evaluator. There is no LLM judge in the
default AIME path; the score is computed from exact/symbolic correctness.

When `repeat` is greater than 1, the benchmark runs multiple sampled generations per problem. NeMo Skills summarizes
metrics across those generations, which is useful for comparing pass@1-style deterministic accuracy and sampled
accuracy on the same serving setup.


## AIME

For AIME, the benchmark section in yaml file can be modified in the following way:
```bash
benchmark:
  type: "aime"
  aime_dataset: "aime25" # One of: aime24, aime25, aime26
  num_examples: null # Number of examples to run; null means all
  max_tokens: 24576 # Max number of output tokens
  repeat: 1 # Number of sampled repetitions
  num_threads: 30 # Number of parallel requests
```

Then launch the script as usual:
```bash
srtctl apply -f config.yaml
```

After finishing benchmarking, AIME outputs are written under `/logs/accuracy/<aime_dataset>/` and summarized metrics
are written to `/logs/accuracy/<aime_dataset>_metrics.json`.


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

