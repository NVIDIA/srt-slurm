# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SA-Bench Dynamo HTTP connection pool lifecycle."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

SA_BENCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "srtctl"
    / "benchmarks"
    / "scripts"
    / "sa-bench"
)


def _import_sa_bench_module(module_name: str):
    sys.path.insert(0, str(SA_BENCH_DIR))
    try:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(SA_BENCH_DIR))


class FakeConnector:
    def __init__(self, *, limit: int):
        self.limit = limit


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

    def __init__(self):
        self.content = FakeContent(
            [
                b'data: {"choices": [{"text": "hello"}]}',
                b'data: {"choices": [{"text": " world"}]}',
                b'data: {"usage": {"completion_tokens": 2}}',
                b"data: [DONE]",
            ]
        )


class FakeRequestContext:
    async def __aenter__(self):
        return FakeResponse()

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def __init__(self, **kwargs: Any):
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
        return FakeRequestContext()

    async def close(self):
        if not self.closed:
            self.close_calls += 1
            self.closed = True


def test_dynamo_session_factory_has_no_connector_concurrency_limit(monkeypatch):
    module = _import_sa_bench_module("backend_request_func")
    sessions: list[FakeSession] = []

    def make_session(**kwargs: Any):
        session = FakeSession(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(module.aiohttp, "TCPConnector", FakeConnector)
    monkeypatch.setattr(module.aiohttp, "ClientSession", make_session)

    async def exercise():
        session = module.create_dynamo_session()
        await session.close()

    asyncio.run(exercise())

    assert len(sessions) == 1
    assert sessions[0].kwargs["connector"].limit == 0
    assert sessions[0].kwargs["trust_env"] is True


def test_dynamo_requests_reuse_injected_session_without_closing_it():
    module = _import_sa_bench_module("backend_request_func")
    session = FakeSession()
    request = module.RequestFuncInput(
        prompt="prompt",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=2,
        model="model",
    )

    async def exercise():
        return await asyncio.gather(
            module.async_request_dynamo_completions(request, session=session),
            module.async_request_dynamo_completions(request, session=session),
        )

    outputs = asyncio.run(exercise())

    assert len(session.posts) == 2
    assert session.close_calls == 0
    assert all(output.success for output in outputs)
    assert [output.generated_text for output in outputs] == ["hello world", "hello world"]
    assert [output.output_tokens for output in outputs] == [2, 2]


def test_session_is_closed_before_metrics(monkeypatch):
    _import_sa_bench_module("backend_request_func")
    module = _import_sa_bench_module("benchmark_serving")
    sessions: list[FakeSession] = []
    request_sessions: list[FakeSession] = []

    def make_session():
        session = FakeSession()
        sessions.append(session)
        return session

    async def fake_request(request_func_input, pbar=None, *, session):
        request_sessions.append(session)
        return module.RequestFuncOutput(
            success=True,
            output_tokens=1,
            prompt_len=request_func_input.prompt_len,
            start_time=1.0,
            ttft=0.01,
            latency=0.02,
        )

    real_calculate_metrics = module.calculate_metrics

    def checked_calculate_metrics(*args, **kwargs):
        assert sessions[0].closed
        return real_calculate_metrics(*args, **kwargs)

    monkeypatch.setattr(module, "create_dynamo_session", make_session)
    monkeypatch.setitem(module.ASYNC_REQUEST_FUNCS, "dynamo", fake_request)
    monkeypatch.setattr(module, "calculate_metrics", checked_calculate_metrics)

    result = asyncio.run(
        module.run_benchmark_with_cleanup(
            backend="dynamo",
            api_url="http://localhost:8000/v1/completions",
            base_url="http://localhost:8000",
            model_id="model",
            model_name="model",
            tokenizer=object(),
            input_requests=[("prompt", 1, 1, None), ("prompt", 1, 1, None)],
            logprobs=None,
            best_of=1,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            profile=False,
            selected_percentile_metrics=[],
            selected_percentiles=[50.0],
            ignore_eos=True,
            goodput_config_dict={},
            max_concurrency=2,
            lora_modules=None,
        )
    )

    assert result["completed"] == 2
    assert len(sessions) == 1
    assert sessions[0].close_calls == 1
    # Initial probe plus two timed requests all receive the exact same session.
    assert request_sessions == [sessions[0], sessions[0], sessions[0]]


@pytest.mark.parametrize("failure", [RuntimeError("probe failed"), asyncio.CancelledError()])
def test_benchmark_wrapper_closes_session_on_failure(monkeypatch, failure):
    _import_sa_bench_module("backend_request_func")
    module = _import_sa_bench_module("benchmark_serving")
    sessions: list[FakeSession] = []

    def make_session():
        session = FakeSession()
        sessions.append(session)
        return session

    async def fail_benchmark(**kwargs):
        raise failure

    monkeypatch.setattr(module, "create_dynamo_session", make_session)
    monkeypatch.setattr(module, "benchmark", fail_benchmark)

    with pytest.raises(type(failure)):
        asyncio.run(module.run_benchmark_with_cleanup(backend="dynamo"))

    assert len(sessions) == 1
    assert sessions[0].close_calls == 1


def test_separate_event_loops_get_separate_sessions(monkeypatch):
    _import_sa_bench_module("backend_request_func")
    module = _import_sa_bench_module("benchmark_serving")
    sessions: list[FakeSession] = []
    observed: list[FakeSession] = []

    def make_session():
        session = FakeSession()
        sessions.append(session)
        return session

    async def record_session(**kwargs):
        observed.append(kwargs["request_session"])
        return {}

    monkeypatch.setattr(module, "create_dynamo_session", make_session)
    monkeypatch.setattr(module, "benchmark", record_session)

    asyncio.run(module.run_benchmark_with_cleanup(backend="dynamo"))
    asyncio.run(module.run_benchmark_with_cleanup(backend="dynamo"))

    assert len(sessions) == 2
    assert observed == sessions
    assert [session.close_calls for session in sessions] == [1, 1]


def test_request_failure_cancels_pending_tasks_before_session_close(monkeypatch):
    _import_sa_bench_module("backend_request_func")
    module = _import_sa_bench_module("benchmark_serving")
    events: list[str] = []
    blocked_started = asyncio.Event()

    class OrderedSession(FakeSession):
        async def close(self):
            if not self.closed:
                events.append("session_closed")
            await super().close()

    session = OrderedSession()

    async def fake_request(request_func_input, pbar=None, *, session):
        if request_func_input.prompt == "blocked":
            blocked_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                events.append("blocked_cancelled")
                raise
        if request_func_input.prompt == "raise":
            await blocked_started.wait()
            raise RuntimeError("request failed")
        return module.RequestFuncOutput(
            success=True,
            output_tokens=1,
            prompt_len=request_func_input.prompt_len,
        )

    monkeypatch.setattr(module, "create_dynamo_session", lambda: session)
    monkeypatch.setitem(module.ASYNC_REQUEST_FUNCS, "dynamo", fake_request)

    with pytest.raises(RuntimeError, match="request failed"):
        asyncio.run(
            module.run_benchmark_with_cleanup(
                backend="dynamo",
                api_url="http://localhost:8000/v1/completions",
                base_url="http://localhost:8000",
                model_id="model",
                model_name="model",
                tokenizer=object(),
                input_requests=[
                    ("initial", 1, 1, None),
                    ("raise", 1, 1, None),
                    ("blocked", 1, 1, None),
                ],
                logprobs=None,
                best_of=1,
                request_rate=float("inf"),
                burstiness=1.0,
                disable_tqdm=True,
                profile=False,
                selected_percentile_metrics=[],
                selected_percentiles=[50.0],
                ignore_eos=True,
                goodput_config_dict={},
                max_concurrency=3,
                lora_modules=None,
            )
        )

    assert events == ["blocked_cancelled", "session_closed"]
