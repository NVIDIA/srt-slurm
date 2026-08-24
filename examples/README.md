# Curated examples

The runnable starting points under `llm/` and `mocker/` are intentionally small rather than a benchmark-results archive. Copy one into your own configuration, set the model path and cluster-specific fields, then use `srtctl dry-run -f <config>` before submitting it.

| Framework | Aggregated | Disaggregated |
| --- | --- | --- |
| vLLM | `llm/vllm/qwen3-32b-aggregated.yaml` | `llm/vllm/qwen3-32b-disaggregated.yaml` |
| SGLang | `llm/sglang/qwen3-32b-aggregated.yaml` | `llm/sglang/qwen3-32b-disaggregated.yaml` |
| TRT-LLM | `llm/trtllm/gpt-oss-120b-aggregated-b200-fp4.yaml` | `llm/trtllm/deepseek-r1-disaggregated-b200-fp4.yaml` |

Additional focused examples:

- `mocker/aggregated.yaml` exercises the end-to-end orchestration path without model weights.
- `llm/sglang/qwen3-32b-ruter-3p2d-direct-host.yaml` is a one-host Dynamo + SGLang route-observability run for `srtctl apply --bash`.

The examples are not performance claims. Their model paths, containers, GPU types, and topology are explicit so they can be adapted safely to a particular cluster.
