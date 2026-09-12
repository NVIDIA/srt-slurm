# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request outcomes survive saved diagnostics and SA-Bench shell cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from test_sa_bench_http_session import SA_BENCH_DIR, _import_sa_bench_module


@pytest.mark.parametrize(
    ("requested", "completed", "status"), [(20, 19, "passed"), (20, 18, "failed"), (20, 0, "failed")]
)
def test_request_failure_threshold(requested: int, completed: int, status: str) -> None:
    module = _import_sa_bench_module("benchmark_outcome")
    outcome = module.benchmark_outcome(requested, completed)
    assert outcome["status"] == status
    assert outcome["failed"] == requested - completed
    assert outcome["max_failure_rate"] == 0.05


def _main_args(tmp_path: Path) -> Namespace:
    return Namespace(
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
        dataset_path="requests.jsonl",
        num_prompts=20,
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
        max_concurrency=20,
        lora_modules=None,
        slow_down_sleep_time=1.0,
        slow_down_wait_time=60.0,
        reuse_http_connections=False,
        save_result=True,
        metadata=None,
        result_filename="result.json",
        result_dir=str(tmp_path),
    )


@pytest.mark.parametrize(("sampled", "completed"), [(20, 0), (20, 18), (20, 19), (10, 10)])
def test_main_saves_diagnostics_and_window_before_failure_gate(
    monkeypatch, tmp_path: Path, sampled: int, completed: int
) -> None:
    _import_sa_bench_module("backend_request_func")
    dataset = _import_sa_bench_module("benchmark_dataset")
    module = _import_sa_bench_module("benchmark_serving")
    window_module = sys.modules["measurement_window"]
    window_dir = tmp_path / "power/windows"
    window_dir.mkdir(parents=True)
    monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_DIR", str(window_dir))
    monkeypatch.setattr(window_module, "CONTAINER_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(module, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(dataset, "sample_custom_requests", lambda **kwargs: [("prompt", 8, 1, None)] * sampled)
    monkeypatch.setattr(module.gc, "collect", lambda: None)
    monkeypatch.setattr(module.gc, "freeze", lambda: None)
    monkeypatch.setattr(module, "save_to_pytorch_benchmark_format", lambda *args, **kwargs: None)

    async def fake_benchmark(**kwargs):
        window = kwargs["measurement_window"]
        window.mark_running(100)
        window.record_boundary(start_unix=100, end_unix=102, duration=2)
        return {
            "completed": completed,
            "duration": 2,
            "benchmark_start_time_unix": 100,
            "benchmark_end_time_unix": 102,
            "errors": ["request failed"] * (sampled - completed),
        }

    monkeypatch.setattr(module, "run_benchmark_with_cleanup", fake_benchmark)
    passed = completed >= sampled * 0.95
    if not passed:
        with pytest.raises(SystemExit, match="request failure rate"):
            module.main(_main_args(tmp_path))
    else:
        module.main(_main_args(tmp_path))
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["benchmark_outcome"]["completed"] == completed
    assert result["benchmark_outcome"]["requested"] == result["num_prompts"] == sampled
    assert result["requested_num_prompts"] == 20
    assert len(result["errors"]) == sampled - completed
    window = json.loads((window_dir / "result.json").read_text())
    assert window["status"] == ("completed" if passed else "failed")
    assert window["benchmark_start_time_unix"] == 100
    assert window["benchmark_end_time_unix"] == 102


def test_all_failed_requests_still_return_finite_diagnostics_and_formal_boundary(tmp_path: Path, monkeypatch) -> None:
    module = _import_sa_bench_module("benchmark_serving")
    window = module.MeasurementWindow(str(tmp_path / "window.json"), "result.json", 2)
    requests = 0
    clock = [100.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(module.time, "time", lambda: clock[0])

    async def fake_request(request_func_input, pbar=None):
        nonlocal requests
        if request_func_input.api_url.endswith("_profile"):
            if request_func_input.api_url.endswith("/stop_profile"):
                clock[0] = 500.0
            return module.RequestFuncOutput(success=True)
        requests += 1
        if requests > 1:
            clock[0] = 102.0
        return module.RequestFuncOutput(
            success=requests == 1, output_tokens=1, prompt_len=8, error="" if requests == 1 else "server error"
        )

    monkeypatch.setitem(module.ASYNC_REQUEST_FUNCS, "dynamo", fake_request)
    with pytest.warns(UserWarning, match="All requests failed"):
        result = asyncio.run(
            module.benchmark(
                backend="dynamo",
                api_url="http://localhost/v1/completions",
                base_url="http://localhost",
                model_id="model",
                model_name="model",
                tokenizer=object(),
                input_requests=[("prompt", 8, 1, None)] * 2,
                logprobs=None,
                best_of=1,
                request_rate=float("inf"),
                burstiness=1,
                disable_tqdm=True,
                profile=True,
                selected_percentile_metrics=[],
                selected_percentiles=[50],
                ignore_eos=True,
                goodput_config_dict={},
                max_concurrency=2,
                lora_modules=None,
                measurement_window=window,
            )
        )
    assert result["completed"] == 0
    assert result["total_output_tokens"] == 0
    assert result["errors"] == ["server error", "server error"]
    assert result["benchmark_start_time_unix"] == 100
    assert result["benchmark_end_time_unix"] == 102
    assert result["duration"] == 2  # The profiler stop at t=500 is outside the measurement.
    assert window.fail_at_recorded_boundary("request_gate_failed") is True


@pytest.mark.parametrize("failed_stage", ["warmup", "measured"])
def test_shell_retains_client_exit(tmp_path: Path, failed_stage: str) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls.jsonl"
    python = binary / "python3"
    python.write_text(f"""#!{sys.executable}
import json, sys
if "-c" in sys.argv: sys.exit(0)
with open({str(calls)!r}, "a") as stream: stream.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(7 if {failed_stage!r} == "warmup" or "--save-result" in sys.argv else 0)
""")
    python.chmod(0o755)
    for name in ("curl", "mkdir"):
        stub = binary / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(SA_BENCH_DIR / "bench.sh"),
            "http://localhost:8000",
            "8192",
            "1024",
            "2x4",
            "inf",
            "/model",
            "model",
            "false",
            "1",
            "0",
            "0",
            "0.8",
            "10",
            "2",
            "",
            "false",
            "random",
            "",
            "false",
        ],
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "PROFILE_TYPE": "none"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 7, result.stderr
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(recorded) == (1 if failed_stage == "warmup" else 2)
    assert recorded[0][recorded[0].index("--request-rate") + 1] == "250"
    if failed_stage == "measured":
        assert "--save-result" in recorded[1]
