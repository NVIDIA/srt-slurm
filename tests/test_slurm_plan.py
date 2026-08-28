# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for portable Slurm serving-script rendering."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.config import load_config
from srtctl.core.slurm import build_srun_command
from srtctl.render.slurm_plan import render_slurm_launch_script


def _config() -> dict:
    return {
        "name": "portable-dynamo-pd",
        "model": {
            "path": "hf:fake/model",
            "container": "nvcr.io/fake/sglang:latest",
            "precision": "fp8",
        },
        "resources": {
            "gpu_type": "h200",
            "gpus_per_node": 8,
            "prefill_nodes": 1,
            "decode_nodes": 1,
            "prefill_workers": 1,
            "decode_workers": 1,
        },
        "dynamo": {"install": False},
        "environment": {
            "HF_TOKEN": "must-not-be-rendered",
            "PUBLIC_SETTING": "visible",
        },
        "frontend": {
            "type": "dynamo",
            "enable_multiple_frontends": False,
            "args": {"router-mode": "kv"},
        },
        "backend": {
            "type": "sglang",
            "sglang_config": {
                "prefill": {"tensor-parallel-size": 8},
                "decode": {"tensor-parallel-size": 8},
            },
        },
        "benchmark": {"type": "custom", "command": "echo benchmark-is-not-part-of-serving-launch"},
    }


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(_config()))
    return config_path


def test_build_srun_command_can_inherit_a_future_allocation() -> None:
    portable = build_srun_command(["python3", "-m", "server"], job_id=None, nodelist=["node-a"])
    realized = build_srun_command(["python3", "-m", "server"], job_id="42042", nodelist=["node-a"])

    assert "--jobid" not in portable
    assert realized[:4] == ["srun", "--jobid", "42042", "--overlap"]


def test_rendered_script_contains_portable_serving_lifecycle(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    script = render_slurm_launch_script(config, source_dir=tmp_path, output_base=tmp_path / "outputs")
    script_path = tmp_path / "launch.sh"
    script_path.write_text(script)

    syntax = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
    assert "scontrol show hostnames" in script
    assert "SRTCTL_NODE_IPS" in script
    assert "python3 -m dynamo.sglang" in script
    assert "--disaggregation-mode prefill" in script
    assert "--disaggregation-mode decode" in script
    assert "python3 -m dynamo.frontend" in script
    assert "wait_for_port" in script
    assert "trap cleanup EXIT INT TERM" in script
    assert "--jobid" not in script
    assert "must-not-be-rendered" not in script
    assert ': "${HF_TOKEN:?Set HF_TOKEN before running this script}"' in script
    assert "export PUBLIC_SETTING=visible" in script
    assert "benchmark-is-not-part-of-serving-launch" not in script


def test_rendered_script_hydrates_nodes_and_runs_with_stubbed_slurm(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    script_path = tmp_path / "launch.sh"
    script_path.write_text(render_slurm_launch_script(config, source_dir=tmp_path, output_base=tmp_path / "outputs"))
    script_path.chmod(0o750)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "scontrol").write_text("#!/bin/sh\nprintf 'node-a\\nnode-b\\n'\n")
    (fake_bin / "srun").write_text("#!/bin/sh\nprintf '10.0.0.1\\n'\n")
    (fake_bin / "python3").write_text("#!/bin/sh\ncat >/dev/null\nexit 0\n")
    for executable in fake_bin.iterdir():
        executable.chmod(0o750)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "SLURM_JOB_ID": "9001",
        "SLURM_JOB_NODELIST": "node-[a-b]",
        "HF_TOKEN": "provided-at-runtime",
    }
    result = subprocess.run(
        ["bash", str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Starting infra" in result.stdout
    assert "Starting prefill-0-rank-0" in result.stdout
    assert "Starting decode-0-rank-0" in result.stdout
    assert "Starting frontend-0" in result.stdout
    assert "Dynamo serving stack is ready" in result.stdout


def test_render_launch_cli_prints_only_the_script(monkeypatch, tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["srtctl", "render-launch", "-f", str(config_path)])

    submit_cli.main()

    captured = capsys.readouterr()
    assert captured.out.startswith("#!/usr/bin/env bash\n")
    assert "--jobid" not in captured.out
    assert captured.err == ""
