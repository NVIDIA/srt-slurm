# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job lifecycle tools of the srtctl MCP: thin layers over srtctl apply, sacct, squeue, scancel."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from srtctl.mcp import job_tools


class _Runner:
    """Records the commands the tools run and answers with canned results."""

    def __init__(self, *responses: subprocess.CompletedProcess) -> None:
        self.responses = list(responses)
        self.commands: list[list[str]] = []

    def __call__(self, command, *, timeout):
        self.commands.append(list(command))
        return self.responses.pop(0)


def _done(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _job_dir(tmp_path: Path, job_id: str = "12807") -> Path:
    job = tmp_path / "outputs" / job_id
    (job / "logs").mkdir(parents=True)
    return job


def test_submit_job_runs_apply_with_overrides_and_parses_the_record(tmp_path: Path, monkeypatch) -> None:
    recipe = tmp_path / "r.yaml"
    recipe.write_text("name: x\n")
    record = {"status": "submitted", "slurm_job_id": "12807", "output_dir": "/o/12807"}
    runner = _Runner(_done("some rich banner\n" + json.dumps(record) + "\n"))
    monkeypatch.setattr(job_tools, "_run", runner)
    monkeypatch.setattr(job_tools, "_srtctl_command", lambda: ["srtctl"])

    result = job_tools.submit_job(
        str(recipe), set_overrides=["resources.gpu_type=b200"], unset=["slurm.time_limit"], tags=["a", "b"]
    )

    assert result["ok"] is True
    assert result["submissions"] == [record]
    assert runner.commands[0] == [
        "srtctl", "apply", "-f", str(recipe), "-y", "--json",
        "--set", "resources.gpu_type=b200",
        "--unset", "slurm.time_limit",
        "--tags", "a,b",
    ]  # fmt: skip


def test_submit_job_reports_failures_with_the_cli_output(tmp_path: Path, monkeypatch) -> None:
    recipe = tmp_path / "r.yaml"
    recipe.write_text("name: x\n")
    monkeypatch.setattr(job_tools, "_run", _Runner(_done("", returncode=1, stderr="Invalid account")))
    monkeypatch.setattr(job_tools, "_srtctl_command", lambda: ["srtctl"])

    result = job_tools.submit_job(str(recipe))

    assert result["ok"] is False
    assert result["exit_code"] == 1
    assert "Invalid account" in result["stderr"]


def test_submit_job_rejects_a_missing_recipe(tmp_path: Path) -> None:
    result = job_tools.submit_job(str(tmp_path / "nope.yaml"))
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_job_status_reads_sacct_metadata_stage_and_errors(tmp_path: Path, monkeypatch) -> None:
    job = _job_dir(tmp_path)
    (job / "12807.json").write_text(json.dumps({"job_id": "12807", "model": {"path": "m"}}))
    (job / "logs" / "sweep_12807.log").write_text(
        "2026-09-08 18:46:56 [INFO] Starting service etcd (etcd) on c012\n"
        "2026-09-08 18:49:59 [INFO] Cleaning up 8 processes (8 running)...\n"
        "2026-09-08 18:50:18 [ERROR] perf dashboard [ingest] KeyError\n"
    )
    (job / "logs" / "benchmark-rollup.json").write_text(json.dumps({"runs": [{"throughput_toks": 42}]}))
    monkeypatch.setattr(job_tools.shutil, "which", lambda name: "/usr/bin/sacct")
    monkeypatch.setattr(
        job_tools,
        "_run",
        _Runner(_done("12807|qwen|COMPLETED|00:03:50|0:0|c[012-013]|2026-09-08T18:46:00|2026-09-08T18:50:18\n")),
    )

    status = job_tools.job_status("12807", output_dir=str(tmp_path / "outputs"))

    assert status["slurm"]["state"] == "COMPLETED"
    assert status["slurm"]["nodelist"] == "c[012-013]"
    assert status["metadata"]["model"]["path"] == "m"
    assert status["stage"].endswith("Cleaning up 8 processes (8 running)...")
    assert status["errors"] == ["2026-09-08 18:50:18 [ERROR] perf dashboard [ingest] KeyError"]
    assert status["benchmark_rollup"]["runs"][0]["throughput_toks"] == 42


def test_job_status_without_slurm_tools_still_reads_the_output_dir(tmp_path: Path, monkeypatch) -> None:
    _job_dir(tmp_path)
    monkeypatch.setattr(job_tools.shutil, "which", lambda name: None)
    status = job_tools.job_status("12807", output_dir=str(tmp_path / "outputs"))
    assert status["slurm"] is None
    assert status["output_dir"].endswith("12807")


def test_job_logs_lists_files_and_tails_one_inside_the_log_dir(tmp_path: Path) -> None:
    job = _job_dir(tmp_path)
    (job / "logs" / "service_etcd.out").write_text("\n".join(f"line {i}" for i in range(10)) + "\n")
    (job / "logs" / "sweep_12807.log").write_text("x\n")
    outputs = str(tmp_path / "outputs")

    listing = job_tools.job_logs("12807", output_dir=outputs)
    assert [f["name"] for f in listing["files"]] == ["service_etcd.out", "sweep_12807.log"]

    tail = job_tools.job_logs("12807", name="service_etcd.out", tail=3, output_dir=outputs)
    assert tail["tail"] == ["line 7", "line 8", "line 9"]

    escape = job_tools.job_logs("12807", name="../12807.json", output_dir=outputs)
    assert escape["ok"] is False
    assert "outside" in escape["error"]


def test_job_id_must_be_numeric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Slurm job id"):
        job_tools.job_status("12807; rm", output_dir=str(tmp_path))
    assert job_tools.cancel_job("abc")["ok"] is False


def test_list_jobs_and_cancel_go_through_squeue_and_scancel(monkeypatch) -> None:
    monkeypatch.setattr(job_tools.shutil, "which", lambda name: f"/usr/bin/{name}")
    runner = _Runner(
        _done("12807|qwen3-0.6b-infra-services|RUNNING|3:39|im-b200-c[012-013]|None\n"),
        _done(""),
    )
    monkeypatch.setattr(job_tools, "_run", runner)

    jobs = job_tools.list_jobs()
    assert jobs["jobs"] == [
        {
            "job_id": "12807",
            "name": "qwen3-0.6b-infra-services",
            "state": "RUNNING",
            "elapsed": "3:39",
            "nodes": "im-b200-c[012-013]",
            "reason": "None",
        }
    ]
    assert runner.commands[0] == ["squeue", "-h", "-o", job_tools.SQUEUE_FORMAT, "--me"]

    cancelled = job_tools.cancel_job("12807")
    assert cancelled["ok"] is True
    assert runner.commands[1] == ["scancel", "12807"]


def test_outputs_dir_prefers_the_explicit_argument(tmp_path: Path) -> None:
    assert job_tools.outputs_dir(str(tmp_path)) == tmp_path.resolve()
