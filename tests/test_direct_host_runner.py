# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from srtctl.render import direct_host_runner
from srtctl.render.direct_host_runner import DirectHostRunner


class _FakePopen:
    def __init__(self, args: list[str], **_kwargs: object) -> None:
        self.args = args
        self.returncode: int | None = None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def send_signal(self, _signal_number: int) -> None:
        self.returncode = 0


def _host_plan(tmp_path: Path, *, mooncake: bool = False) -> dict[str, object]:
    source = tmp_path / "srt-slurm"
    model = tmp_path / "models" / "model"
    sglang_source = tmp_path / "sglang"
    source.mkdir()
    model.parent.mkdir()
    model.write_text("model", encoding="utf-8")
    sglang_source.mkdir()
    return {
        "name": "direct",
        "source_dir": str(source),
        "output_base": str(tmp_path / "outputs"),
        "model_path": str(model),
        "local_container_image": "lmsysorg/sglang:dev",
        "sglang_source": str(sglang_source),
        "sglang_runtime_key": "runtime-key",
        "mooncake_master_command": ["mooncake_master", "--port=8700"] if mooncake else [],
        "mooncake_container": "nvcr.io/nvidia/mooncake:latest" if mooncake else None,
        "mooncake_environment": [["MOONCAKE_PROTOCOL", "rdma"]] if mooncake else [],
        "direct_plan": {"name": "direct", "worker_processes": []},
    }


def test_host_runner_executes_as_a_direct_file() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(direct_host_runner.__file__)), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Direct Docker host runner" in result.stdout


def test_host_runner_persists_host_and_container_plans(tmp_path: Path) -> None:
    runner = DirectHostRunner(_host_plan(tmp_path))

    runner._prepare_output()

    direct_plan = json.loads(runner.direct_plan_path.read_text(encoding="utf-8"))
    host_plan = json.loads(runner.direct_host_plan_path.read_text(encoding="utf-8"))
    assert direct_plan == {"name": "direct", "worker_processes": []}
    assert host_plan["direct_plan"] == direct_plan
    assert host_plan["output_dir"] == str(runner.output_dir)


def test_host_runner_builds_all_direct_docker_mounts(tmp_path: Path, monkeypatch) -> None:
    runner = DirectHostRunner(_host_plan(tmp_path))
    runner._prepare_output()
    trace = tmp_path / "trace.jsonl"
    trace.write_text("trace", encoding="utf-8")
    monkeypatch.setattr(runner, "_sglang_revision", lambda: "sglang-revision")

    args = runner._main_container_args(trace)

    assert args[:5] == ["run", "--detach", "--name", runner.container_name, "--label"]
    assert "--network" in args and args[args.index("--network") + 1] == "host"
    assert "--ipc" in args and args[args.index("--ipc") + 1] == "host"
    assert "--gpus" in args and args[args.index("--gpus") + 1] == "all"
    assert f"type=bind,src={runner.source_dir},dst={runner.source_dir},readonly" in args
    assert f"type=bind,src={runner.output_base},dst={runner.output_base}" in args
    assert f"type=bind,src={trace},dst={trace},readonly" in args
    assert str(runner.dynamo_cache_root) in " ".join(args)
    assert str(runner.sglang_runtime_dir) == str(
        runner.output_base / ".srtctl-runtime" / "sglang-sglang-revision-runtime-key"
    )


def test_host_runner_mounts_an_explicit_output_outside_the_default_base(tmp_path: Path, monkeypatch) -> None:
    requested_output = tmp_path / "requested-output"
    monkeypatch.setenv("SRTCTL_OUTPUT_DIR", str(requested_output))
    runner = DirectHostRunner(_host_plan(tmp_path))
    runner._prepare_output()
    monkeypatch.setattr(runner, "_sglang_revision", lambda: "sglang-revision")

    args = runner._main_container_args(None)

    assert runner.output_dir == requested_output
    assert f"type=bind,src={requested_output},dst={requested_output}" in args


def test_host_runner_execs_container_runner_with_owned_environment(tmp_path: Path, monkeypatch) -> None:
    runner = DirectHostRunner(_host_plan(tmp_path))
    runner._prepare_output()
    runner.sglang_runtime_dir = tmp_path / "outputs" / ".srtctl-runtime" / "runtime"
    spawned: list[_FakePopen] = []

    def fake_popen(args: list[str], **kwargs: object) -> _FakePopen:
        del kwargs
        process = _FakePopen(args)
        spawned.append(process)
        return process

    monkeypatch.setattr(direct_host_runner.subprocess, "Popen", fake_popen)

    assert runner._run_container(None) == 0
    assert len(spawned) == 1
    command = spawned[0].args
    assert command[:2] == ["docker", "exec"]
    assert "SRTCTL_LOCAL_CONTAINERIZED=1" in command
    assert f"SRTCTL_OUTPUT_DIR={runner.output_dir}" in command
    assert f"OUTPUT_DIR={runner.output_dir}" in command
    assert f"SRTCTL_SGLANG_RUNTIME_DIR={runner.sglang_runtime_dir}" in command
    assert str(runner.direct_plan_path) in command
    assert command[-2:] == ["--plan", str(runner.direct_plan_path)]
    assert command[-3].endswith("/src/srtctl/render/direct_runner.py")


def test_host_runner_starts_mooncake_with_its_own_log(tmp_path: Path, monkeypatch) -> None:
    runner = DirectHostRunner(_host_plan(tmp_path, mooncake=True))
    runner._prepare_output()
    calls: list[tuple[str, ...]] = []
    spawned: list[_FakePopen] = []

    def fake_docker(*args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="container-id\n" if capture_output else None)

    def fake_popen(args: list[str], **kwargs: object) -> _FakePopen:
        del kwargs
        process = _FakePopen(args)
        spawned.append(process)
        return process

    monkeypatch.setattr(runner, "_docker", fake_docker)
    monkeypatch.setattr(direct_host_runner.subprocess, "Popen", fake_popen)

    runner._start_mooncake()

    assert calls == [
        (
            "run",
            "--detach",
            "--name",
            f"{runner.container_name}-mooncake",
            "--label",
            runner.container_label,
            "--network",
            "host",
            "--ipc",
            "host",
            "--gpus",
            "all",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--env",
            "MOONCAKE_PROTOCOL=rdma",
            "nvcr.io/nvidia/mooncake:latest",
            "mooncake_master",
            "--port=8700",
        )
    ]
    assert spawned[0].args == ["docker", "logs", "-f", f"{runner.container_name}-mooncake"]
    runner._stop_mooncake_log_follower()
