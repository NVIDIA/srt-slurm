# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom dataset loader for SA-Bench.

Loads a JSONL dataset file where each line is a JSON object describing one request.
No tokenizer is required — prompt lengths are either read from the dataset or
estimated from the raw text length.

Supported formats:

1. TRT-LLM / OpenAI-style (messages array):
   {"input": {"messages": [{"role": "user", "content": "..."}], "max_tokens": 2048}}
   {"input": {"messages": [{"role": "user", "content": "..."}], "max_tokens": 2048, "num_tokens": 512}}

2. Flat format (prompt string):
   {"prompt": "...", "expected_output_len": 128}
   {"prompt": "...", "expected_output_len": 128, "prompt_len": 64}
"""

from __future__ import annotations

import json
import random


def sample_custom_requests(
    dataset_path: str,
    num_requests: int,
) -> list[tuple[str, int, int, None]]:
    """Load and sample requests from a custom JSONL dataset.

    Args:
        dataset_path: Path to JSONL file.
        num_requests: Maximum number of requests to return.

    Returns:
        List of (prompt, prompt_len, output_len, None) tuples.
    """
    with open(dataset_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.shuffle(data)

    results: list[tuple[str, int, int, None]] = []

    for entry in data:
        if len(results) >= num_requests:
            break

        if "input" in entry:
            messages = entry["input"].get("messages", [])
            prompt = _extract_prompt_from_messages(messages)
            output_len = int(entry["input"].get("max_tokens", 128))
            prompt_len = entry["input"].get("num_tokens")
        elif "prompt" in entry:
            prompt = entry["prompt"]
            output_len = int(entry.get("expected_output_len") or entry.get("max_tokens") or 128)
            prompt_len = entry.get("prompt_len")
        else:
            continue

        if not prompt:
            continue

        if not isinstance(prompt_len, int) or prompt_len <= 0:
            prompt_len = len(prompt) // 4

        results.append((prompt, prompt_len, output_len, None))

    return results


def _extract_prompt_from_messages(messages: list[dict]) -> str:
    """Extract the user prompt from an OpenAI-style messages array."""
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    if messages and messages[-1].get("content"):
        return messages[-1]["content"]
    return ""
