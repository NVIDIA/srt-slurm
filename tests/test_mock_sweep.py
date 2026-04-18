# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `srtctl.mock.run_mock_sweep`.

The mock runs the full SweepOrchestrator path with external surfaces
(srun, port waits, model health checks, status HTTP) swapped for local
fakes. These tests assert that the real orchestrator code runs to completion
and produces the expected artifact set that external harnesses observe.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from srtctl.mock import MockOptions, run_mock_sweep

MINIMAL_CONFIG = {
    "name": "mock-smoke",
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
    "benchmark": {"type": "custom", "command": "echo fake-benchmark"},
}


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump(MINIMAL_CONFIG))
    return cfg


def test_run_mock_sweep_produces_expected_artifacts(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    output_dir = tmp_path / "outputs" / "42042"

    exit_code = run_mock_sweep(
        config_path=cfg,
        output_dir=output_dir,
        job_id="42042",
        options=MockOptions(child_duration_s=0.15, phase_pause_s=0.05),
    )

    assert exit_code == 0
    # Core artifacts the mock promises.
    assert (output_dir / "status.json").is_file()
    assert (output_dir / "status_events.jsonl").is_file()
    assert (output_dir / "result.json").is_file()
    assert (output_dir / "recipe.lock.yaml").is_file(), "lockfile written by real postprocess stage"
    # Per-component logs emitted by the real orchestrator, via fake srun.
    assert (output_dir / "logs" / "infra.out").is_file()
    assert any((output_dir / "logs").glob("*_agg_w0.out")), "worker log written"
    assert any((output_dir / "logs").glob("*_frontend_*.out")), "frontend log written"
    assert (output_dir / "logs" / "benchmark.out").is_file()


def test_run_mock_sweep_drives_full_status_timeline(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    output_dir = tmp_path / "outputs" / "42043"

    exit_code = run_mock_sweep(
        config_path=cfg,
        output_dir=output_dir,
        job_id="42043",
        options=MockOptions(child_duration_s=0.1, phase_pause_s=0.05),
    )
    assert exit_code == 0

    events_path = output_dir / "status_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]

    # Walk through every orchestrator phase we expect.
    stages = [event.get("stage") for event in events]
    assert "starting" in stages
    assert "head_infrastructure" in stages
    assert "workers" in stages
    assert "frontend" in stages
    assert "benchmark" in stages
    assert "cleanup" in stages

    terminal = events[-1]
    assert terminal["status"] in ("completed", "failed")
    assert terminal["status"] == "completed"


def test_run_mock_sweep_result_json_is_parseable_with_fake_metrics(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    output_dir = tmp_path / "outputs" / "42044"

    run_mock_sweep(
        config_path=cfg,
        output_dir=output_dir,
        job_id="42044",
        options=MockOptions(child_duration_s=0.05, phase_pause_s=0.01),
    )

    result = json.loads((output_dir / "result.json").read_text())
    assert result["job_id"] == "42044"
    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    # Deterministic-ish fake metrics derived from job_id; just confirm shape.
    assert isinstance(result["final_loss"], float)
    assert 0.0 < result["final_loss"] < 1.0
    assert isinstance(result["score"], float)
    assert 0.0 < result["score"] < 1.0
    assert isinstance(result["param_count"], int)
