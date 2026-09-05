# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from srtctl.render import direct_runner
from srtctl.render.direct_runner import DirectRunInterrupted, DirectRunner, _router_counts
from srtctl.render.direct_stages import (
    BenchmarkStageMixin,
    InfrastructureStageMixin,
    PostProcessStageMixin,
    RuntimeSetupStageMixin,
    ServingStageMixin,
    TelemetryStageMixin,
)


def _plan(tmp_path) -> dict[str, object]:
    return {
        "source_dir": str(tmp_path / "source"),
        "output_base": str(tmp_path / "output-base"),
        "sglang_source": str(tmp_path / "sglang-source"),
        "frontend_port": 8000,
        "global_environment": [],
        "benchmark_environment": [],
        "dynamo_source_cache_key": "dynamo-test",
        "tachometer_enabled": False,
        "ruter_enabled": False,
    }


def _runner(tmp_path, monkeypatch) -> DirectRunner:
    output = tmp_path / "output"
    logs = output / "logs"
    artifacts = output / "artifacts"
    logs.mkdir(parents=True)
    artifacts.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output))
    monkeypatch.setenv("LOG_DIR", str(logs))
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.setenv("SRTCTL_PYTHON", sys.executable)
    return DirectRunner(_plan(tmp_path))


def test_router_counts_accepts_dynamo_health_shape() -> None:
    assert _router_counts(
        {
            "instances": [
                {"endpoint": "generate", "component": "prefill"},
                {"endpoint": "generate", "component": "decode"},
                {"endpoint": "generate", "component": "backend"},
                {"endpoint": "metrics", "component": "prefill"},
            ]
        },
    ) == (1, 2)


def test_runner_composes_direct_only_stages() -> None:
    assert DirectRunner.__bases__ == (
        RuntimeSetupStageMixin,
        InfrastructureStageMixin,
        ServingStageMixin,
        TelemetryStageMixin,
        BenchmarkStageMixin,
        PostProcessStageMixin,
    )


def test_runner_executes_as_a_direct_file() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(direct_runner.__file__)), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Rendered direct execution plan" in result.stdout


def test_runner_tracks_subprocess_groups_and_separate_logs(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    worker = runner._launch_shell("worker-0", "worker-0.log", f"{sys.executable} -c 'print(\"worker ready\")'")

    assert worker.process.wait(timeout=10) == 0
    assert (runner.log_dir / "worker-0.log").read_text(encoding="utf-8") == "worker ready\n"
    assert (runner.log_dir / "worker-0.log.command").is_file()
    assert (runner.log_dir / "worker-0.log.pid").read_text(encoding="utf-8") == f"{worker.process.pid}\n"

    runner._cleanup()


def test_dynamo_cache_key_includes_current_python_and_host_shape(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    key = runner._dynamo_source_cache_key()

    assert key.startswith("dynamo-test-")
    assert sys.implementation.cache_tag in key
    assert len(key.rsplit("-", 1)[1]) == 12


def test_runner_converts_termination_to_cleanup_signal(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)

    with pytest.raises(DirectRunInterrupted) as error:
        runner._on_signal(signal.SIGTERM, None)

    assert error.value.signal_number == signal.SIGTERM


def test_runner_keeps_cleanup_best_effort_when_ruter_python_is_missing(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    runner.plan["ruter_enabled"] = True

    runner._normalize_ruter()

    assert "ruter.log" in {path.name for path in runner.log_dir.iterdir()}


def test_dynamo_source_archive_excludes_build_artifacts(tmp_path) -> None:
    source = tmp_path / "build" / "dynamo"
    (source / "src").mkdir(parents=True)
    (source / "target").mkdir()
    (source / ".git").mkdir()
    (source / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "target" / "ignored").write_text("ignored\n", encoding="utf-8")
    (source / ".git" / "ignored").write_text("ignored\n", encoding="utf-8")
    archive = tmp_path / "dynamo-src.tar.gz"

    DirectRunner._write_archive(tmp_path / "build", "dynamo", archive)

    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "dynamo/src/main.py" in names
    assert not any("target" in name or ".git" in name for name in names)
