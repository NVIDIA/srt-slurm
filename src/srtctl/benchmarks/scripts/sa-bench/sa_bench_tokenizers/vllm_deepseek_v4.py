# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM DeepSeek-V4 tokenizer adapter for sa-bench.

vLLM's public ``DeepseekV4Tokenizer.from_pretrained()`` returns a cached
tokenizer wrapper. That wrapper is fine for serving, but it hides the
``apply_chat_template`` override from sa-bench's fast-fail check. This adapter
uses vLLM's own DeepSeek-V4 renderer and returns the real HF tokenizer subclass
so sa-bench can both inspect and call the chat-template implementation.
"""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerFast


class VLLMDeepseekV4Tokenizer:
    """Load vLLM's DeepSeek-V4 tokenizer without the cache wrapper."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any):
        try:
            from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
        except ImportError as exc:
            raise ImportError(
                "VLLMDeepseekV4Tokenizer requires the vllm package. "
                "Use this custom_tokenizer from a vLLM benchmark container."
            ) from exc

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        return get_deepseek_v4_tokenizer(tokenizer)
