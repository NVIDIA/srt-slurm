# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SA-Bench chat completions path on the Dynamo backend."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

SA_BENCH_DIR = Path(__file__).resolve().parents[1] / "src" / "srtctl" / "benchmarks" / "scripts" / "sa-bench"

CHAT_URL = "http://localhost:8000/v1/chat/completions"
COMPLETIONS_URL = "http://localhost:8000/v1/completions"


def _import_sa_bench_module(module_name: str):
    sys.path.insert(0, str(SA_BENCH_DIR))
    try:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(SA_BENCH_DIR))


@pytest.fixture
def backend_module():
    return _import_sa_bench_module("backend_request_func")


class FakeContent:
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk


class FakeResponse:
    status = 200
    reason = "OK"

    def __init__(self, chunks: list[bytes]):
        self.content = FakeContent(chunks)


class FakeRequestContext:
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def __aenter__(self):
        return FakeResponse(self.chunks)

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def __init__(self, chunks: list[bytes], **kwargs: Any):
        self.chunks = chunks
        self.kwargs = kwargs
        self.closed = False
        self.close_calls = 0
        self.posts: list[dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()
        return False

    def post(self, **kwargs: Any):
        self.posts.append(kwargs)
        return FakeRequestContext(self.chunks)

    async def close(self):
        if not self.closed:
            self.close_calls += 1
            self.closed = True


# Shaped after the Dynamo frontend: the assistant role rides on the first
# content delta, and only the trailing finish-reason delta has no content.
CHAT_CHUNKS = [
    b'data: {"choices": [{"delta": {"role": "assistant", "content": "hello"}}]}',
    b'data: {"choices": [{"delta": {"content": " world"}}]}',
    b'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}',
    b'data: {"usage": {"completion_tokens": 2}}',
    b"data: [DONE]",
]

COMPLETIONS_CHUNKS = [
    b'data: {"choices": [{"text": "hello"}]}',
    b'data: {"choices": [{"text": " world"}]}',
    b'data: {"usage": {"completion_tokens": 2}}',
    b"data: [DONE]",
]


def _run_dynamo_request(backend_module, api_url, chunks):
    """Drive the registered dynamo adapter and return (output, posted kwargs)."""
    session = FakeSession(chunks)
    request = backend_module.RequestFuncInput(
        prompt="prompt text",
        api_url=api_url,
        prompt_len=8192,
        output_len=1024,
        model="model",
    )
    request_func = backend_module.ASYNC_REQUEST_FUNCS["dynamo"]
    output = asyncio.run(request_func(request, session=session))
    return output, session.posts[0]


def test_dynamo_backend_sends_messages_on_the_chat_api(backend_module):
    output, post = _run_dynamo_request(backend_module, CHAT_URL, CHAT_CHUNKS)

    assert post["url"] == CHAT_URL
    assert post["json"]["messages"] == [{"role": "user", "content": "prompt text"}]
    assert "prompt" not in post["json"]
    assert post["json"]["max_completion_tokens"] == 1024
    assert output.success
    assert output.generated_text == "hello world"
    assert output.output_tokens == 2


def test_dynamo_backend_still_sends_a_prompt_on_the_completions_api(backend_module):
    output, post = _run_dynamo_request(backend_module, COMPLETIONS_URL, COMPLETIONS_CHUNKS)

    assert post["json"]["prompt"] == "prompt text"
    assert "messages" not in post["json"]
    assert post["json"]["max_tokens"] == 1024
    assert output.success
    assert output.generated_text == "hello world"


def test_only_content_deltas_become_latency_samples(backend_module):
    """The role-bearing first delta counts; the finish-reason delta does not."""
    output, _ = _run_dynamo_request(backend_module, CHAT_URL, CHAT_CHUNKS)

    assert output.text_chunks == ["hello", " world"]
    assert len(output.itl) == 1
    assert output.ttft > 0


def test_an_empty_content_delta_still_counts_as_a_token(backend_module):
    """Incremental detokenization emits one per partial UTF-8 sequence."""
    chunks = [
        b'data: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}',
        b'data: {"choices": [{"delta": {"content": "\\u00e4"}}]}',
        b'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}',
        b"data: [DONE]",
    ]

    output, _ = _run_dynamo_request(backend_module, CHAT_URL, chunks)

    assert output.text_chunks == ["", "ä"]
    assert len(output.itl) == 1


def test_chat_requests_borrow_an_injected_session(backend_module):
    session = FakeSession(CHAT_CHUNKS)
    request = backend_module.RequestFuncInput(
        prompt="prompt text",
        api_url=CHAT_URL,
        prompt_len=1,
        output_len=2,
        model="model",
    )

    async def exercise():
        return await asyncio.gather(
            backend_module.async_request_dynamo_chat_completions(request, session=session),
            backend_module.async_request_dynamo_chat_completions(request, session=session),
        )

    outputs = asyncio.run(exercise())

    assert len(session.posts) == 2
    assert session.close_calls == 0
    assert all(output.success for output in outputs)


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        ("http://h:8000/v1/chat/completions", True),
        ("http://h:8000/v1/chat/completions/", True),
        ("http://h:8000/v1/completions", False),
        ("http://h:8000/start_profile", False),
    ],
)
def test_chat_url_detection(backend_module, api_url, expected):
    assert backend_module.is_chat_completions_url(api_url) is expected


class FakeTokenizer:
    """Whitespace tokenizer whose chat template costs three tokens."""

    vocab_size = 1000

    def decode(self, token_ids):
        return " ".join(f"t{int(token_id)}" for token_id in token_ids)

    def encode(self, text, add_special_tokens=True):
        return [self._token_id(piece) for piece in text.split()]

    @staticmethod
    def _token_id(piece):
        return int(piece[1:]) if piece.startswith("t") and piece[1:].isdigit() else 0

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        return f"t900 t901 {messages[0]['content']} t902"


def _sample_one(sa_bench, *, wrap_chat_template):
    tokenizer = FakeTokenizer()
    requests = sa_bench.sample_random_requests(
        prefix_len=0,
        input_len=10,
        output_len=4,
        num_prompts=1,
        range_ratio=1.0,
        tokenizer=tokenizer,
        use_chat_template=True,
        wrap_chat_template=wrap_chat_template,
        num_workers=1,
    )
    prompt, prompt_len, _, _ = requests[0]
    return tokenizer, prompt, prompt_len


def test_chat_api_prompts_reserve_the_template_without_rendering_it():
    """The server renders the template, so ISL must be reserved, not consumed."""
    sa_bench = _import_sa_bench_module("benchmark_serving")

    tokenizer, prompt, prompt_len = _sample_one(sa_bench, wrap_chat_template=False)

    assert "t900" not in prompt and "t902" not in prompt
    # Three tokens are left for the template the server will add.
    assert len(tokenizer.encode(prompt)) == 7
    assert prompt_len == 10


def test_completions_prompts_still_render_the_template_client_side():
    sa_bench = _import_sa_bench_module("benchmark_serving")

    tokenizer, prompt, prompt_len = _sample_one(sa_bench, wrap_chat_template=True)

    assert prompt.startswith("t900 t901 ")
    assert prompt.endswith(" t902")
    assert len(tokenizer.encode(prompt)) == 10
    assert prompt_len == 10
