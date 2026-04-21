# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `srtctl apply --mock`.

The `--mock` flag stubs sbatch in the submit flow, so submit_with_orchestrator
runs end-to-end (config load → script generation → metadata write → JSON
submission record), and then spawns a detached `srtctl.cli.mock_worker` that
drives the real SweepOrchestrator under `srtctl.mock`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from srtctl.cli import submit as submit_cli

MINIMAL_CONFIG = {
    "name": "apply-mock-smoke",
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
    "benchmark": {"type": "custom", "command": "echo apply-mock"},
}


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump(MINIMAL_CONFIG))
    return cfg


def _wait_for(path: Path, *, timeout: float = 30.0, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(interval)
    return False


def test_apply_mock_emits_submission_json_and_spawns_worker(
    monkeypatch, tmp_path: Path, capsys: Any
) -> None:
    cfg = _write_config(tmp_path)
    output_base = tmp_path / "outputs"
    output_base.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srtctl",
            "apply",
            "-f",
            str(cfg),
            "-o",
            str(output_base),
            "--mock",
            "--mock-tick-s",
            "0.1",
            "--json",
        ],
    )
    # get_srtslurm_setting returns cluster info; None is fine for the mock.
    with (
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit._assert_preflight_passed"),
    ):
        submit_cli.main()

    stdout = capsys.readouterr().out.strip()
    assert "\n" not in stdout
    record = json.loads(stdout)
    assert record["status"] == "submitted"
    slurm_job_id = record["slurm_job_id"]
    assert slurm_job_id and slurm_job_id.isdigit()
    output_dir = Path(record["output_dir"])
    assert output_dir.exists()
    assert output_dir.name == slurm_job_id

    # Real submit flow wrote <job_id>.json metadata before the worker even started.
    metadata_path = Path(record["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["job_id"] == slurm_job_id
    assert metadata["orchestrator"] is True
    assert metadata["model"]["path"] == "hf:fake/mock-model"

    # Detached worker produces result.json. Use a longer timeout since the
    # orchestrator walks through every phase for real.
    assert _wait_for(output_dir / "result.json", timeout=20.0), (
        f"mock worker did not produce result.json under {output_dir}. "
        f"worker log:\n{(output_dir / 'mock_worker.log').read_text() if (output_dir / 'mock_worker.log').exists() else '(no worker log)'}"
    )
    result = json.loads((output_dir / "result.json").read_text())
    assert result["job_id"] == slurm_job_id
    assert result["status"] == "completed"
    assert result["exit_code"] == 0

    # Orchestrator artifacts are also present.
    assert (output_dir / "status.json").exists()
    assert (output_dir / "status_events.jsonl").exists()
    assert (output_dir / "recipe.lock.yaml").exists()
    assert (output_dir / "logs" / "benchmark.out").exists()


def test_apply_mock_does_not_call_real_sbatch(
    monkeypatch, tmp_path: Path
) -> None:
    """Ensure the stubbed sbatch prevented an actual subprocess.run(sbatch, ...)."""
    cfg = _write_config(tmp_path)
    output_base = tmp_path / "outputs"
    output_base.mkdir()

    real_run = __import__("subprocess").run

    seen_sbatch: list[list] = []

    def _tracking_run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "sbatch":
            seen_sbatch.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srtctl",
            "apply",
            "-f",
            str(cfg),
            "-o",
            str(output_base),
            "--mock",
            "--mock-tick-s",
            "0.05",
            "--json",
        ],
    )
    # Swap real_run behind the tracker: if the mock patch fails to intercept,
    # this will route the sbatch through to the real system call and surface.
    with (
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit._assert_preflight_passed"),
        patch("subprocess.run", side_effect=_tracking_run),
    ):
        submit_cli.main()

    assert seen_sbatch == [], (
        f"Expected --mock to stub sbatch entirely, but saw real sbatch calls: {seen_sbatch}"
    )
