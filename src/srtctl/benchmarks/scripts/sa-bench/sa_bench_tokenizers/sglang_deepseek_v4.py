# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang-side DeepSeek-V4 tokenizer for sa-bench.

Mirrors what sglang's ``serving_chat._apply_jinja_template`` does
when ``chat_encoding_spec == "dsv4"`` (see
sgl-project/sglang PR #23600), so that the tokens counted on the
sa-bench client side match the tokens the sglang server actually
feeds into the model.

The vllm counterpart lives in ``vllm.tokenizers.deepseek_v4``; sglang
has no equivalent client-side package, so we vendor the rendering
logic from ``encoding_dsv4.py`` in ``_sglang_encoding_dsv4.py``.

Env-var fallback (mirrors sglang ``serving_chat.py``):

- ``SGLANG_ENABLE_THINKING=1`` flips the default ``thinking_mode`` from
  ``"chat"`` to ``"thinking"`` when the caller does not pass ``thinking``
  explicitly. This keeps ISL / TPOT / accept-rate metrics in lock-step
  with the server when users set the env on both sides of a run (e.g.
  in a recipe's ``prefill_environment`` / ``decode_environment``).
- ``SGLANG_REASONING_EFFORT`` provides a default for ``reasoning_effort``
  when the caller does not pass one. Only ``"max"`` / ``"high"`` are
  honored; any other value is filtered out to match sglang.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ._sglang_encoding_dsv4 import encode_messages as _encode_messages


def _env_enable_thinking() -> bool:
    """Parse ``SGLANG_ENABLE_THINKING`` the same way sglang ``EnvBool`` does."""
    return os.environ.get("SGLANG_ENABLE_THINKING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_reasoning_effort() -> Optional[str]:
    """Parse ``SGLANG_REASONING_EFFORT``; only ``max`` / ``high`` are honored."""
    val = os.environ.get("SGLANG_REASONING_EFFORT", "").strip()
    return val if val in ("max", "high") else None


class SGLangDeepseekV4Tokenizer:
    """Client-side DeepSeek-V4 tokenizer matching sglang server behavior.

    The server-side call chain (sglang PR #23600) is:

        # sglang serving_chat.py, chat_encoding_spec == "dsv4" branch:
        thinking_requested = (request.chat_template_kwargs or {}).get(
            "thinking", envs.SGLANG_ENABLE_THINKING.get()
        )
        thinking_mode = "thinking" if thinking_requested else "chat"

        effort_source = request.reasoning_effort
        if effort_source is None:
            env_val = envs.SGLANG_REASONING_EFFORT.get()
            if env_val:
                effort_source = env_val
        reasoning_effort = effort_source if effort_source in ("max", "high") else None

        messages = request.messages                        # OpenAI-style
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        real_input = encoding_dsv4.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        prompt_ids = tokenizer.encode(real_input)

    We reproduce the exact same steps here, including the
    ``SGLANG_ENABLE_THINKING`` / ``SGLANG_REASONING_EFFORT`` env-var
    fallback so that a sa-bench run with these envs set matches the
    server's prompt rendering byte-for-byte.
    """

    def __init__(self, hf_tokenizer):
        self._hf = hf_tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        # DeepSeek-V4 ships a fast tokenizer.json but no tokenization_*.py,
        # and `model_type: deepseek_v4` isn't recognized by stock HF
        # transformers. Load the fast tokenizer directly; fall back to
        # AutoTokenizer (for future versions that register the config).
        kwargs.setdefault("trust_remote_code", True)
        try:
            hf = PreTrainedTokenizerFast.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        except Exception:
            hf = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        return cls(hf)

    def _render_prompt(
        self,
        messages: List[Dict[str, Any]],
        thinking_mode: str = "chat",
        reasoning_effort: Optional[str] = None,
    ) -> str:
        msgs = [dict(m) for m in messages]
        if not msgs or msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": ""})

        if reasoning_effort not in ("max", "high"):
            reasoning_effort = None

        return _encode_messages(
            msgs,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,  # noqa: ARG002  (encoder always adds the <｜Assistant｜>... tail)
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        **_: Any,
    ):
        # Per-caller kwargs win; env is fallback (mirrors sglang serving_chat.py).
        if thinking is None:
            thinking = _env_enable_thinking()
        if reasoning_effort is None:
            reasoning_effort = _env_reasoning_effort()

        msgs = [dict(m) for m in messages]
        if tools:
            if not msgs or msgs[0].get("role") != "system":
                msgs.insert(0, {"role": "system", "content": ""})
            msgs[0]["tools"] = list(tools)

        thinking_mode = "thinking" if thinking else "chat"
        prompt = self._render_prompt(
            msgs,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        if not tokenize:
            return prompt
        return self._hf.encode(prompt, add_special_tokens=False)

    def encode(self, text, **kwargs):
        return self._hf.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self._hf.decode(token_ids, **kwargs)

    def __len__(self):
        return len(self._hf)

    @property
    def vocab_size(self):
        return self._hf.vocab_size

    @property
    def eos_token_id(self):
        return self._hf.eos_token_id

    @property
    def bos_token_id(self):
        return self._hf.bos_token_id

    @property
    def pad_token_id(self):
        return self._hf.pad_token_id

    def __getattr__(self, name):
        return getattr(self._hf, name)
