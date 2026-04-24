
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
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from ._sglang_encoding_dsv4 import encode_messages as _encode_messages


class SGLangDeepseekV4Tokenizer:
    """Client-side DeepSeek-V4 tokenizer matching sglang server behavior.

    The server-side call chain (sglang PR #23600) is:

        messages = request.messages                        # OpenAI-style
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        real_input = encoding_dsv4.encode_messages(
            messages,
            thinking_mode="chat",                          # default
            reasoning_effort=None,                         # "medium" dropped
        )
        prompt_ids = tokenizer.encode(real_input)

    We reproduce the exact same steps here.
    """

    def __init__(self, hf_tokenizer):
        self._hf = hf_tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
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
        thinking: bool = False,
        reasoning_effort: Optional[str] = None,
        **_: Any,
    ):
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
