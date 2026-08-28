# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from subprocess import CompletedProcess

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.config import load_config

MINIMAL_DRY_RUN_CONFIG = {
    "name": "stdin-dry-run",
    "model": {
        "path": "hf:fake/mock-model",
        "container": "nvcr.io/fake:latest",
        "precision": "fp8",
    },
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "agg_nodes": 1,
        "agg_workers": 1,
    },
    "backend": {"type": "sglang"},
    "frontend": {
        "type": "sglang",
        "enable_multiple_frontends": False,
    },
    "benchmark": {"type": "custom", "command": "echo stdin-dry-run"},
}


DIRECT_BASH_CONFIG = {
    **MINIMAL_DRY_RUN_CONFIG,
    "backend": {
        "type": "sglang",
        "sglang_config": {"aggregated": {"served-model-name": "fake/mock-model", "tp": 1}},
    },
    "frontend": {
        "type": "dynamo",
        "enable_multiple_frontends": False,
        "args": {"router-mode": "kv"},
    },
    "environment": {
        "SRTCTL_LOCAL_CONTAINER_IMAGE": "lmsysorg/sglang:dev",
        "SRTCTL_SGLANG_SOURCE": "/tmp/sglang-source",
    },
    "dynamo": {"top_of_tree": True},
}


def test_dry_run_accepts_dash_stdin(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "dry-run", "-f", "-"],
    )
    monkeypatch.setattr(sys, "stdin", StringIO(yaml.safe_dump(MINIMAL_DRY_RUN_CONFIG)))

    submit_cli.main()

    output = capsys.readouterr().out
    assert "DRY-RUN" in output
    assert "stdin-dry-run" in output


def test_dry_run_empty_stdin_fails_cleanly(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "dry-run", "-f", "/dev/stdin"],
    )
    monkeypatch.setattr(sys, "stdin", StringIO(""))

    with pytest.raises(SystemExit) as exc_info:
        submit_cli.main()

    assert exc_info.value.code == 1
    error = capsys.readouterr().out
    assert "No YAML received on stdin" in error
    assert "NoneType" not in error


def test_apply_bash_outputs_direct_container_script(monkeypatch, tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(DIRECT_BASH_CONFIG))

    def fail_subprocess_run(*_args, **_kwargs):
        raise AssertionError("--bash must not submit through sbatch")

    monkeypatch.setattr(submit_cli.subprocess, "run", fail_subprocess_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "apply", "-f", str(config_path), "--bash"],
    )

    submit_cli.main()

    captured = capsys.readouterr()
    output = captured.out
    assert captured.err == ""
    assert output.startswith("#!/usr/bin/env bash\n")
    assert "DRY-RUN" not in output
    assert "Direct Docker bootstrap" in output
    assert "direct_host_runner.py" in output
    assert "SRTCTL_DIRECT_HOST_PLAN_" in output
    assert "worker-0.log" in output
    assert "#SBATCH" not in output
    assert "SLURM_" not in output
    assert "srtctl.cli.do_sweep" not in output
    assert "srtctl.cli.run_benchmark" not in output


def test_apply_current_allocation_runs_existing_orchestrator(monkeypatch, tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(MINIMAL_DRY_RUN_CONFIG))
    output_dir = tmp_path / "outputs"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0)

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_NODELIST", "gpu-01")
    monkeypatch.setattr(submit_cli, "validate_setup", lambda _source: None)
    monkeypatch.setattr(submit_cli, "_assert_preflight_passed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(submit_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srtctl",
            "apply",
            "-f",
            str(config_path),
            "-o",
            str(output_dir),
            "--current-allocation",
            "--setup-script",
            "custom.sh",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        submit_cli.main()

    assert exc_info.value.code == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    runtime_config = output_dir / "12345" / "config.yaml"
    assert command == [sys.executable, "-m", "srtctl.cli.do_sweep", str(runtime_config)]
    assert kwargs["check"] is False
    assert kwargs["env"]["SLURM_JOB_ID"] == "12345"
    assert kwargs["env"]["SRTCTL_OUTPUT_DIR"] == str(output_dir / "12345")
    assert kwargs["env"]["SRTCTL_SETUP_SCRIPT"] == "custom.sh"
    assert yaml.safe_load(runtime_config.read_text()) == MINIMAL_DRY_RUN_CONFIG
    assert "Running in SLURM allocation 12345" in capsys.readouterr().out


def test_apply_current_allocation_requires_slurm_job(monkeypatch, tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(MINIMAL_DRY_RUN_CONFIG))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "apply", "-f", str(config_path), "--current-allocation"],
    )

    with pytest.raises(SystemExit) as exc_info:
        submit_cli.main()

    assert exc_info.value.code == 1
    assert "--current-allocation requires SLURM_JOB_ID to be set" in capsys.readouterr().out


def test_load_config_rejects_empty_yaml(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")

    with pytest.raises(ValueError, match="YAML file is empty"):
        load_config(path)
