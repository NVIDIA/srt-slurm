# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

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
    "benchmark": {"type": "custom", "command": "echo stdin-dry-run"},
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


def test_load_config_rejects_empty_yaml(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")

    with pytest.raises(ValueError, match="YAML file is empty"):
        load_config(path)
