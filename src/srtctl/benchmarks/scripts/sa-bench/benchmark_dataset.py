# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom dataset loader for SA-Bench.

Loads a JSONL dataset file where each line is a JSON object describing one request.

Supported formats:

1. TRT-LLM / OpenAI-style (messages array):
   {"input": {"messages": [{"role": "user", "content": "..."}], "max_tokens": 2048}}

2. Flat format (prompt string):
   {"prompt": "...", "expected_output_len": 128}
"""

from __future__ import annotations

import json
import random


def sample_custom_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    *,
    fixed_output_len: int | None = None,
) -> list[tuple[str, int, int, None]]:
    """Load and sample requests from a custom JSONL dataset.

    Each line of the JSONL file should be one of:
      - {"input": {"messages": [...], "max_tokens": N}}
      - {"input": {"messages": [...], "max_tokens": N, "num_tokens": M}}
      - {"prompt": "...", "expected_output_len": N}

    Args:
        dataset_path: Path to JSONL file.
        num_requests: Maximum number of requests to return.
        tokenizer: HuggingFace tokenizer for computing prompt token lengths.
        fixed_output_len: If set, override output length from the dataset.

    Returns:
        List of (prompt, prompt_len, output_len, None) tuples.
    """
    with open(dataset_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.shuffle(data)

    prompts: list[str] = []
    output_lens: list[int] = []
    precomputed_lengths: list[int | None] = []

    for entry in data:
        if len(prompts) >= num_requests:
            break

        if "input" in entry:
            messages = entry["input"].get("messages", [])
            prompt = _extract_prompt_from_messages(messages)
            max_tokens = entry["input"].get("max_tokens", 128)
            num_tokens = entry["input"].get("num_tokens")
            precomputed = num_tokens if isinstance(num_tokens, int) and num_tokens > 0 else None
        elif "prompt" in entry:
            prompt = entry["prompt"]
            max_tokens = entry.get("expected_output_len") or entry.get("max_tokens") or 128
            precomputed = entry.get("prompt_len")
            precomputed = precomputed if isinstance(precomputed, int) and precomputed > 0 else None
        else:
            continue

        if not prompt:
            continue

        prompts.append(prompt)
        output_lens.append(fixed_output_len if fixed_output_len is not None else int(max_tokens))
        precomputed_lengths.append(precomputed)

    if all(p is not None for p in precomputed_lengths) and precomputed_lengths:
        prompt_lens = [p for p in precomputed_lengths]  # type: ignore[misc]
    else:
        prompt_lens = _batch_tokenize(prompts, tokenizer)

    return [(prompt, plen, olen, None) for prompt, plen, olen in zip(prompts, prompt_lens, output_lens, strict=True)]


def _extract_prompt_from_messages(messages: list[dict]) -> str:
    """Extract the user prompt from an OpenAI-style messages array."""
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    if messages and messages[-1].get("content"):
        return messages[-1]["content"]
    return ""


def _batch_tokenize(prompts: list[str], tokenizer) -> list[int]:
    """Tokenize prompts in batch and return token counts."""
    if not prompts:
        return []
    encoded = tokenizer(prompts)
    return [len(ids) for ids in encoded.input_ids]
