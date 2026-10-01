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
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from srtctl.cli.do_sweep import SweepOrchestrator
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

TRTLLM_AGGREGATE_CONFIG = {
    "name": "mock-trtllm-aggregate-sa-bench",
    "model": {
        "path": "hf:fake/mock-model",
        "container": "nvcr.io/fake:latest",
        "precision": "fp8",
    },
    "resources": {
        "gpu_type": "gb300",
        "gpus_per_node": 4,
        "agg_nodes": 2,
        "agg_workers": 1,
        "gpus_per_agg": 8,
    },
    "frontend": {
        "type": "trtllm_serve",
        "enable_multiple_frontends": False,
    },
    "backend": {
        "type": "trtllm",
        "trtllm_config": {
            "aggregated": {
                "tensor_parallel_size": 8,
                "moe_expert_parallel_size": 8,
                "pipeline_parallel_size": 1,
            }
        },
    },
    "benchmark": {
        "type": "sa-bench",
        "dataset_name": "random",
        "isl": 128,
        "osl": 32,
        "concurrencies": [1],
        "req_rate": "inf",
        "random_range_ratio": 0.8,
        "num_prompts_mult": 5,
        "num_warmup_mult": 1,
    },
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
    assert (output_dir / "recipe.lock.yaml").is_file(), "lockfile written by real artifact finalization"
    # Per-component logs emitted by the real orchestrator, via fake srun.
    assert (output_dir / "logs" / "service_etcd.out").is_file(), "etcd is a service now"
    assert not (output_dir / "logs" / "service_nats.out").exists(), "tcp request plane: no NATS service runs"
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


def test_trtllm_aggregate_runs_complete_sa_bench_lifecycle_without_router(tmp_path: Path) -> None:
    config_path = tmp_path / "trtllm-aggregate.yaml"
    config_path.write_text(yaml.safe_dump(TRTLLM_AGGREGATE_CONFIG))
    output_dir = tmp_path / "outputs" / "42045"

    exit_code = run_mock_sweep(
        config_path=config_path,
        output_dir=output_dir,
        job_id="42045",
        options=MockOptions(
            child_duration_s=0.05,
            phase_pause_s=0.01,
            nodelist=("mock-node-01", "mock-node-02"),
        ),
    )

    assert exit_code == 0
    result = json.loads((output_dir / "result.json").read_text())
    assert result["status"] == "completed"
    assert any((output_dir / "logs").glob("*_agg_w0.out"))
    assert (output_dir / "logs" / "benchmark.out").is_file()
    assert not (output_dir / "logs" / "service_etcd.out").exists()
    assert not (output_dir / "logs" / "service_nats.out").exists()
    assert not (output_dir / "logs" / "ser.yaml").exists()
    assert not any((output_dir / "logs").glob("*_frontend_*.out"))


@pytest.mark.parametrize("benchmark_exit_code", [0, 7])
def test_sweep_uploads_raw_artifacts_without_automatic_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark_exit_code: int
) -> None:
    from srtctl.analysis import incremental_power, perf_dashboard, power_energy_report
    from srtctl.cli.mixins import postprocess_stage

    config_path = tmp_path / "config.yaml"
    config = {
        **TRTLLM_AGGREGATE_CONFIG,
        "telemetry": {"enabled": True, "cpu_power_exporter": {}},
    }
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "outputs" / "42046"
    output_dir.mkdir(parents=True)
    (output_dir / "config.yaml").write_text(config_path.read_text())
    log_dir = output_dir / "logs"
    fingerprint = {"hostname": "mock-node-01"}
    raw_result = '{"completed": 1, "total_output_tokens": 8}\n'

    def finish_benchmark(orchestrator: SweepOrchestrator, *_args: object) -> int:
        # Worker fingerprints arrive after the initial lockfile was written.
        (log_dir / "fingerprint_agg_w0.json").write_text(json.dumps(fingerprint))
        (log_dir / "benchmark.out").write_text("Successful requests: 1\n")
        results_dir = log_dir / "sa-bench_test"
        results_dir.mkdir()
        (results_dir / "results_concurrency_1.json").write_text(raw_result)
        object.__setattr__(
            orchestrator.config,
            "_lock_data",
            {"slurm": {"job_id": "previous"}, "fingerprints": {"agg_w0": fingerprint}},
        )
        return benchmark_exit_code

    monkeypatch.setattr(SweepOrchestrator, "run_benchmark", finish_benchmark)
    # External power collectors are faked; automatic analysis must stay absent
    # even when telemetry is enabled and a legacy AI config is present.
    for method in ("start_power_telemetry", "start_cpu_power_telemetry", "start_cpu_power_host_telemetry"):
        monkeypatch.setattr(SweepOrchestrator, method, lambda *args: None)
    analysis_calls = [MagicMock(), MagicMock(), MagicMock()]
    monkeypatch.setattr(perf_dashboard, "try_build", analysis_calls[0])
    monkeypatch.setattr(power_energy_report, "build_reports", analysis_calls[1])
    monkeypatch.setattr(incremental_power.IncrementalPowerWatcher, "start", analysis_calls[2])
    monkeypatch.setattr(
        postprocess_stage,
        "load_cluster_config",
        lambda: {
            "reporting": {
                "s3": {"bucket": "test-bucket", "prefix": "runs"},
                "ai_analysis": {"enabled": True, "openrouter_api_key": "test-key"},
            }
        },
    )

    upload_calls: list[dict[str, Any]] = []

    def inspect_launch(call: dict[str, Any]) -> None:
        assert "claude -p" not in " ".join(call["command"])
        if Path(call.get("output") or "").name != "postprocess.log":
            return
        # These artifacts must be finalized before the upload starts.
        assert (log_dir / "config.yaml").read_text() == config_path.read_text()
        lock = yaml.safe_load((output_dir / "recipe.lock.yaml").read_text())["lock"]
        assert lock["fingerprints"] == {"agg_w0": fingerprint}
        assert "Comparing against lockfile from job previous" in (log_dir / "reproduction-report.txt").read_text()
        assert "aws s3 sync /logs s3://test-bucket/runs/" in call["command"][-1]
        upload_calls.append(call)

    exit_code = run_mock_sweep(
        config_path=config_path,
        output_dir=output_dir,
        job_id="42046",
        options=MockOptions(
            child_duration_s=0.01,
            nodelist=("mock-node-01", "mock-node-02"),
            on_srun=inspect_launch,
        ),
    )

    assert exit_code == benchmark_exit_code
    assert len(upload_calls) == 1
    assert (log_dir / "sa-bench_test" / "results_concurrency_1.json").read_text() == raw_result
    for name in ("benchmark-rollup.json", "perf_dashboard.html", "perf_dashboard_bundle", "power_energy_report.json"):
        assert not (log_dir / name).exists()
    for call in analysis_calls:
        call.assert_not_called()

    events = [json.loads(line) for line in (output_dir / "status_events.jsonl").read_text().splitlines()]
    artifact_events = [event for event in events if event.get("logs_url")]
    assert len(artifact_events) == 2, "eager artifact pointer and final completion both report the upload"
    assert artifact_events[0]["status"] == "benchmark"
    assert artifact_events[1]["status"] == ("completed" if benchmark_exit_code == 0 else "failed")
    assert artifact_events[1]["exit_code"] == benchmark_exit_code
    assert artifact_events[0]["logs_url"] == artifact_events[1]["logs_url"]
    assert artifact_events[1]["logs_url"].startswith("s3://test-bucket/runs/")
