# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `srtctl apply --json` stdout contract.

`--json` redirects prose output to stderr and emits one JSON line per
submission on stdout so external harnesses can parse submission metadata
without scraping Rich output.
"""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli import submit as submit_cli

MINIMAL_CONFIG = {
    "name": "test-job",
    "model": {
        "path": "/models/test-model",
        "container": "test-container.sqsh",
        "precision": "fp8",
    },
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
    "benchmark": {"type": "manual"},
}


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump(MINIMAL_CONFIG))
    return cfg


def test_apply_json_emits_single_line_on_stdout(monkeypatch, tmp_path: Path, capsys: Any) -> None:
    cfg = _write_config(tmp_path)
    mock_sbatch = MagicMock()
    mock_sbatch.stdout = "Submitted batch job 42042"

    # Unset RUNNER_NAME so get_job_name() falls through to config.name. On
    # GitHub Actions runners RUNNER_NAME is auto-set and would otherwise
    # override the configured job name.
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "apply", "-f", str(cfg), "-o", str(tmp_path), "--json"],
    )
    with (
        patch("subprocess.run", return_value=mock_sbatch),
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit.create_job_record"),
        patch("srtctl.cli.submit._assert_preflight_passed"),
        patch("srtctl.cli.submit.validate_setup"),
    ):
        submit_cli.main()

    captured = capsys.readouterr()
    stdout = captured.out.strip()

    # Exactly one line on stdout, valid JSON.
    assert "\n" not in stdout
    record = json.loads(stdout)
    assert record["status"] == "submitted"
    assert record["slurm_job_id"] == "42042"
    assert record["job_name"] == "test-job"
    assert record["config_path"] == str(cfg)
    assert record["output_dir"].endswith("42042")
    assert record["metadata_path"].endswith("42042.json")


def test_apply_json_emits_error_line_on_failure(monkeypatch, tmp_path: Path, capsys: Any) -> None:
    cfg = _write_config(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("sbatch is on fire")

    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "apply", "-f", str(cfg), "-o", str(tmp_path), "--json"],
    )
    with (
        patch("subprocess.run", side_effect=boom),
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit._assert_preflight_passed"),
        patch("srtctl.cli.submit.validate_setup"),
        pytest.raises(SystemExit) as excinfo,
    ):
        submit_cli.main()

    assert excinfo.value.code == 1
    stdout = capsys.readouterr().out.strip()
    record = json.loads(stdout)
    assert record["status"] == "error"
    assert "on fire" in record["error"]


def test_apply_without_json_emits_prose_only(monkeypatch, tmp_path: Path, capsys: Any) -> None:
    cfg = _write_config(tmp_path)
    mock_sbatch = MagicMock()
    mock_sbatch.stdout = "Submitted batch job 99999"

    monkeypatch.setattr(sys, "argv", ["srtctl", "apply", "-f", str(cfg), "-o", str(tmp_path)])
    with (
        patch("subprocess.run", return_value=mock_sbatch),
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit.create_job_record"),
        patch("srtctl.cli.submit._assert_preflight_passed"),
        patch("srtctl.cli.submit.validate_setup"),
    ):
        submit_cli.main()

    stdout = capsys.readouterr().out
    # Prose path stays as before; should NOT be machine-parseable JSON.
    assert "Job 99999" in stdout
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            pytest.fail(f"Unexpected JSON in non-json stdout: {line!r}")
