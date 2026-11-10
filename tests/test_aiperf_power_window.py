# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""benchmarks/aiperf_power_window.py writes windows the power validator accepts."""

import json
import subprocess
import sys
from pathlib import Path

from srtctl.core.power.windows import WINDOW_STATUS_COMPLETED, WINDOW_STATUS_RUNNING, _check_result, _parse

SCRIPT = Path(__file__).resolve().parent.parent / "benchmarks" / "aiperf_power_window.py"


def _stamp(state: str, result_dir: Path, window_dir: Path | None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin"} | ({"SRT_MEASUREMENT_WINDOW_DIR": str(window_dir)} if window_dir else {})
    argv = [sys.executable, str(SCRIPT), state, "--concurrency", "8", "--result-dir", str(result_dir), "--require"]
    return subprocess.run(argv, env=env, capture_output=True, text=True, check=False)


def _aiperf_artifacts(result_dir: Path) -> None:
    result_dir.mkdir(parents=True)
    aggregate = {"start_time": "2026-11-09T10:00:00+00:00", "end_time": "2026-11-09T11:00:00.5+00:00"}
    (result_dir / "profile_export_aiperf.json").write_text(json.dumps(aggregate))
    records = [
        {"metadata": {"benchmark_phase": "warmup"}, "metrics": {"input_sequence_length": {"value": 5}}},
        {
            "metadata": {"benchmark_phase": "profiling"},
            "metrics": {"input_sequence_length": {"value": 100}, "output_sequence_length": {"value": 20}},
        },
        {"metadata": {"benchmark_phase": "profiling"}, "error": {"type": "timeout"}},
    ]
    (result_dir / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_running_then_completed_window_validates(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    window_dir = log_root / "power" / "windows"
    result_dir = log_root / "agentic" / "concurrency_8"
    _aiperf_artifacts(result_dir)
    window_file = window_dir / "agentic_power_concurrency_8.json"

    assert _stamp("running", result_dir, window_dir).returncode == 0
    running, reasons = _parse(window_file, f"windows/{window_file.name}", log_root)
    assert reasons == () and running is not None and running.status == WINDOW_STATUS_RUNNING

    assert _stamp("completed", result_dir, window_dir).returncode == 0
    window, reasons = _parse(window_file, f"windows/{window_file.name}", log_root)
    assert reasons == () and window is not None
    assert (window.status, window.benchmark_type, window.concurrency) == (WINDOW_STATUS_COMPLETED, "custom", 8)
    assert window.duration == 3600.5
    assert _check_result(window, log_root) == []
    result = json.loads((result_dir / "agentic_power_concurrency_8.json").read_text())
    assert (result["completed"], result["total_input_tokens"], result["total_output_tokens"]) == (1, 100, 20)


def test_noop_without_window_dir(tmp_path: Path) -> None:
    assert _stamp("completed", tmp_path / "missing", None).returncode == 0
    assert not (tmp_path / "missing").exists()


def test_missing_artifacts_fail_only_when_required(tmp_path: Path) -> None:
    window_dir = tmp_path / "logs" / "power" / "windows"
    result_dir = tmp_path / "logs" / "agentic" / "concurrency_8"
    failed = _stamp("completed", result_dir, window_dir)
    assert failed.returncode == 1 and "profile_export_aiperf.json" in failed.stderr
    assert not (window_dir / "agentic_power_concurrency_8.json").exists()
