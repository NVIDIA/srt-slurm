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

SA_BENCH_DIR = Path(__file__).resolve().parents[1] / "src" / "srtctl" / "benchmarks" / "scripts" / "sa-bench"


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


def test_dynamo_requests_use_owned_per_request_sessions_by_default(monkeypatch):
    module = _import_sa_bench_module("backend_request_func")
    sessions: list[FakeSession] = []

    def make_session(**kwargs: Any):
        session = FakeSession(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(module.aiohttp, "ClientSession", make_session)
    request = module.RequestFuncInput(
        prompt="prompt",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=2,
        model="model",
    )

    async def exercise():
        return await asyncio.gather(
            module.async_request_dynamo_completions(request),
            module.async_request_dynamo_completions(request),
        )

    outputs = asyncio.run(exercise())

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert [session.close_calls for session in sessions] == [1, 1]
    assert all(session.kwargs["trust_env"] is True for session in sessions)
    assert all(session.kwargs["timeout"] is module.AIOHTTP_TIMEOUT for session in sessions)
    assert all("connector" not in session.kwargs for session in sessions)
    assert all(output.success for output in outputs)
    assert [output.generated_text for output in outputs] == ["hello world", "hello world"]
    assert [output.output_tokens for output in outputs] == [2, 2]


@pytest.mark.parametrize("borrowed", [False, True])
def test_request_cancellation_respects_session_ownership(monkeypatch, borrowed):
    module = _import_sa_bench_module("backend_request_func")
    started = asyncio.Event()

    class BlockingRequestContext:
        async def __aenter__(self):
            started.set()
            await asyncio.Future()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class BlockingSession(FakeSession):
        def post(self, **kwargs: Any):
            self.posts.append(kwargs)
            return BlockingRequestContext()

    sessions: list[BlockingSession] = []

    def make_session(**kwargs: Any):
        session = BlockingSession(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(module.aiohttp, "ClientSession", make_session)
    request = module.RequestFuncInput(
        prompt="prompt",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=2,
        model="model",
    )
    borrowed_session = BlockingSession() if borrowed else None

    async def exercise():
        task = asyncio.create_task(module.async_request_dynamo_completions(request, session=borrowed_session))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    if borrowed:
        assert sessions == []
        assert borrowed_session is not None
        assert borrowed_session.close_calls == 0
    else:
        assert len(sessions) == 1
        assert sessions[0].close_calls == 1


def test_benchmark_defaults_to_legacy_session_lifecycle(monkeypatch):
    _import_sa_bench_module("backend_request_func")
    module = _import_sa_bench_module("benchmark_serving")
    calls: list[dict[str, Any]] = []

    def unexpected_shared_session():
        raise AssertionError("shared session factory must not run by default")

    async def record_benchmark(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(module, "create_dynamo_session", unexpected_shared_session)
    monkeypatch.setattr(module, "benchmark", record_benchmark)

    asyncio.run(module.run_benchmark_with_cleanup(backend="dynamo"))
    asyncio.run(module.run_benchmark_with_cleanup(backend="dynamo", reuse_http_connections=False))

    assert calls == [{"backend": "dynamo"}, {"backend": "dynamo"}]


def test_http_connection_reuse_rejects_unsupported_backend(monkeypatch):
    _import_sa_bench_module("backend_request_func")
    module = _import_sa_bench_module("benchmark_serving")
    calls: list[dict[str, Any]] = []

    async def record_benchmark(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(module, "benchmark", record_benchmark)

    asyncio.run(module.run_benchmark_with_cleanup(backend="vllm"))
    with pytest.raises(ValueError, match="supported only by the Dynamo backend"):
        asyncio.run(module.run_benchmark_with_cleanup(backend="vllm", reuse_http_connections=True))

    assert calls == [{"backend": "vllm"}]


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
            reuse_http_connections=True,
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
        asyncio.run(module.run_benchmark_with_cleanup(backend="dynamo", reuse_http_connections=True))

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

    asyncio.run(module.run_benchmark_with_cleanup(backend="dynamo", reuse_http_connections=True))
    asyncio.run(module.run_benchmark_with_cleanup(backend="dynamo", reuse_http_connections=True))

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
                reuse_http_connections=True,
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


@pytest.mark.parametrize("reuse_http_connections", [False, True])
def test_result_json_records_effective_http_connection_mode(monkeypatch, tmp_path, reuse_http_connections):
    _import_sa_bench_module("backend_request_func")
    dataset_module = _import_sa_bench_module("benchmark_dataset")
    module = _import_sa_bench_module("benchmark_serving")
    observed: list[bool] = []

    monkeypatch.setattr(module, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        dataset_module,
        "sample_custom_requests",
        lambda **kwargs: [("prompt", 1, 1, None)],
    )
    monkeypatch.setattr(module.gc, "collect", lambda: None)
    monkeypatch.setattr(module.gc, "freeze", lambda: None)
    monkeypatch.setattr(module, "save_to_pytorch_benchmark_format", lambda *args, **kwargs: None)

    async def fake_run_benchmark_with_cleanup(**kwargs):
        observed.append(kwargs.pop("reuse_http_connections"))
        return {"completed": 1, "reuse_http_connections": not reuse_http_connections}

    monkeypatch.setattr(module, "run_benchmark_with_cleanup", fake_run_benchmark_with_cleanup)

    args = module.argparse.Namespace(
        slow_down_servers=None,
        seed=0,
        backend="dynamo",
        model="model",
        served_model_name="model",
        tokenizer="model",
        tokenizer_mode="auto",
        base_url="http://localhost:8000",
        endpoint="/v1/completions",
        host="localhost",
        port=8000,
        trust_remote_code=False,
        custom_tokenizer=None,
        use_chat_template=False,
        dataset_name="custom",
        dataset_path="/data/requests.jsonl",
        num_prompts=1,
        goodput=None,
        logprobs=None,
        best_of=1,
        request_rate=float("inf"),
        burstiness=1.0,
        disable_tqdm=True,
        profile=False,
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="50,90,99",
        ignore_eos=True,
        max_concurrency=1,
        lora_modules=None,
        slow_down_sleep_time=1.0,
        slow_down_wait_time=60.0,
        reuse_http_connections=reuse_http_connections,
        save_result=True,
        metadata=None,
        result_filename="result.json",
        result_dir=str(tmp_path),
    )

    module.main(args)

    result = module.json.loads((tmp_path / "result.json").read_text())
    assert observed == [reuse_http_connections]
    assert result["reuse_http_connections"] is reuse_http_connections
