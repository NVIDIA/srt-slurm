# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentX formal power-measurement window tests."""

import importlib.util
import json
from pathlib import Path

import pytest

AGENTIC_DIR = Path(__file__).resolve().parents[1] / "src/srtctl/benchmarks/scripts/agentic"


def _load_measurement_window():
    spec = importlib.util.spec_from_file_location("agentic_measurement_window", AGENTIC_DIR / "measurement_window.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measurement_window = _load_measurement_window()


def _record(*, start_ns, end_ns, phase="profiling", error=None):
    return {
        "metadata": {
            "benchmark_phase": phase,
            "request_start_ns": start_ns,
            "request_end_ns": end_ns,
        },
        "error": error,
    }


def _fixture(tmp_path):
    logs = tmp_path / "logs"
    artifacts = logs / "agentic" / "conc_160" / "aiperf_artifacts" / "minimax_agentx"
    output = logs / "agentic_agg" / "minimax-agentx_conc160.json"
    windows = logs / "power" / "windows"
    artifacts.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    windows.mkdir(parents=True)
    output.write_text(json.dumps({"conc": 160, "request_metrics": {}}))
    return logs, artifacts, output, windows


def test_publishes_exact_successful_profile_boundary(tmp_path):
    logs, artifacts, output, windows = _fixture(tmp_path)
    profile_start_ns = 1_780_000_000_000_000_000
    profile_end_ns = profile_start_ns + 3_500_000_000
    records = [
        _record(
            start_ns=profile_start_ns - 20_000_000_000,
            end_ns=profile_start_ns - 19_000_000_000,
            phase="warmup",
        ),
        _record(
            start_ns=profile_start_ns - 10_000_000_000,
            end_ns=profile_end_ns + 10_000_000_000,
            error={"message": "failed"},
        ),
        _record(start_ns=profile_start_ns, end_ns=profile_start_ns + 1_000_000_000),
        _record(start_ns=profile_start_ns + 500_000_000, end_ns=profile_end_ns),
    ]
    (artifacts / "profile_export.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

    path = measurement_window.publish_measurement_window(
        result_dir=logs / "agentic" / "conc_160",
        result_file=output,
        concurrency=160,
        benchmark_type="agentic",
        window_dir=windows,
        log_root=logs,
    )

    assert path == windows / "minimax-agentx_conc160.json"
    result = json.loads(output.read_text())
    window = json.loads(path.read_text())
    expected_start = profile_start_ns / 1e9
    expected_end = profile_end_ns / 1e9
    assert result["benchmark_start_time_unix"] == expected_start
    assert result["benchmark_end_time_unix"] == expected_end
    assert result["duration"] == 3.5
    assert window == {
        "schema_version": 1,
        "benchmark_type": "agentic",
        "result_path": "agentic_agg/minimax-agentx_conc160.json",
        "concurrency": 160,
        "benchmark_start_time_unix": expected_start,
        "benchmark_end_time_unix": expected_end,
        "duration": 3.5,
        "clock_source": "head_node_unix_clock",
        "status": "completed",
        "reason": None,
    }


def test_missing_telemetry_directory_is_a_noop(tmp_path):
    logs, artifacts, output, _ = _fixture(tmp_path)
    (artifacts / "profile_export.jsonl").write_text(
        json.dumps(_record(start_ns=1_000_000_000, end_ns=2_000_000_000)) + "\n"
    )

    assert (
        measurement_window.publish_measurement_window(
            result_dir=logs / "agentic" / "conc_160",
            result_file=output,
            concurrency=160,
            window_dir="",
            log_root=logs,
        )
        is None
    )
    assert "duration" not in json.loads(output.read_text())


def test_requested_telemetry_rejects_an_empty_profile(tmp_path):
    logs, artifacts, output, windows = _fixture(tmp_path)
    (artifacts / "profile_export.jsonl").write_text(json.dumps(_record(start_ns=1, end_ns=2, phase="warmup")) + "\n")

    with pytest.raises(ValueError, match="no successful profiling timestamps"):
        measurement_window.publish_measurement_window(
            result_dir=logs / "agentic" / "conc_160",
            result_file=output,
            concurrency=160,
            window_dir=windows,
            log_root=logs,
        )
