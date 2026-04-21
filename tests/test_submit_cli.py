# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.config import get_cluster_aliases, load_config

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


def test_cluster_aliases_json_reads_srtslurm_yaml(monkeypatch, tmp_path: Path, capsys) -> None:
    (tmp_path / "srtslurm.yaml").write_text(
        yaml.safe_dump(
            {
                "model_paths": {"qwen32b": "/models/qwen32b"},
                "containers": {"dev-0405": "/containers/dev-0405.sqsh"},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "cluster-aliases", "--json"],
    )

    submit_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["model_paths"]["qwen32b"] == "/models/qwen32b"
    assert payload["containers"]["dev-0405"] == "/containers/dev-0405.sqsh"


def test_load_config_rejects_empty_yaml(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")

    with pytest.raises(ValueError, match="YAML file is empty"):
        load_config(path)


def test_get_cluster_aliases_returns_empty_maps_without_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    aliases = get_cluster_aliases()

    assert aliases["model_paths"] == {}
    assert aliases["containers"] == {}
