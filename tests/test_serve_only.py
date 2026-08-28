# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from srtctl.cli import submit as submit_cli
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.core.config import load_config
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.status import JobStage, JobStatus
from srtctl.core.topology import Process

CONFIG = {
    "name": "serve-only-test",
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
    "frontend": {"type": "sglang", "enable_multiple_frontends": False},
    "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": [1]},
}


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    return config_path


def test_apply_serve_only_reaches_single_job_submission(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    submitted: dict = {}

    def capture_submit(**kwargs):
        submitted.update(kwargs)

    monkeypatch.setattr(submit_cli, "submit_single", capture_submit)
    monkeypatch.setattr(sys, "argv", ["srtctl", "apply", "-f", str(config_path), "--serve-only"])

    submit_cli.main()

    assert submitted["config_path"] == config_path
    assert submitted["serve_only"] is True


def test_serve_only_is_forwarded_to_the_slurm_orchestrator(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    monkeypatch.setattr(submit_cli, "get_srtslurm_setting", lambda *_args, **_kwargs: None)

    normal_script = submit_cli.generate_minimal_sbatch_script(config, config_path)
    serve_script = submit_cli.generate_minimal_sbatch_script(config, config_path, serve_only=True)

    command = 'do_sweep "${OUTPUT_DIR}/config.yaml"'
    assert f"{command} --serve-only" in serve_script
    assert f"{command} --serve-only" not in normal_script
    assert "Serve-Only Orchestrator" in serve_script
    syntax_check = subprocess.run(["bash", "-n"], input=serve_script, text=True, capture_output=True, check=False)
    assert syntax_check.returncode == 0, syntax_check.stderr


class _ServeOnlyHarness(BenchmarkStageMixin):
    def __init__(self, log_dir: Path) -> None:
        self.serve_only = True
        self.config = SrtConfig.Schema().load(CONFIG)
        self.runtime = RuntimeContext(
            job_id="12345",
            run_name="serve-only-test",
            nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0",)),
            head_node_ip="10.0.0.1",
            infra_node_ip="10.0.0.1",
            log_dir=log_dir,
            model_path=Path("/model"),
            container_image=Path("/container.sqsh"),
            gpus_per_node=8,
            network_interface=None,
        )

    @property
    def backend_processes(self) -> list[Process]:
        return []

    def _public_api_node(self) -> str:
        return "frontend-node"


def test_serve_only_waits_for_health_but_never_loads_a_benchmark(tmp_path: Path) -> None:
    harness = _ServeOnlyHarness(tmp_path)
    registry = MagicMock()
    reporter = MagicMock()
    stop_event = threading.Event()
    stop_event.set()

    with (
        patch(
            "srtctl.cli.mixins.benchmark_stage._get_health_expectations",
            return_value=(0, 1, "one aggregate worker", 1),
        ),
        patch("srtctl.cli.mixins.benchmark_stage.wait_for_model", return_value=True) as wait_for_model,
        patch("srtctl.cli.mixins.benchmark_stage.collect_worker_fingerprints", return_value=[]),
        patch("srtctl.benchmarks.get_runner") as get_runner,
    ):
        exit_code = harness.run_benchmark(registry, stop_event, reporter)

    assert exit_code == 0
    wait_for_model.assert_called_once()
    get_runner.assert_not_called()
    reporter.report.assert_called_once_with(JobStatus.FRONTEND, JobStage.FRONTEND, "Inference endpoint ready")


def test_serve_only_takes_precedence_over_eval_only(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    runtime = RuntimeContext(
        job_id="12345",
        run_name="serve-only-test",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0",)),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path / "logs",
        model_path=Path("/model"),
        container_image=Path("/container.sqsh"),
        gpus_per_node=8,
        network_interface=None,
    )
    runtime.log_dir.mkdir()
    orchestrator = SweepOrchestrator(config=config, runtime=runtime, serve_only=True)
    reporter = MagicMock()
    monkeypatch.setenv("EVAL_ONLY", "true")

    with (
        patch("srtctl.cli.do_sweep.record_resource_snapshot", return_value={}),
        patch("srtctl.cli.do_sweep.write_lockfile"),
        patch("srtctl.cli.do_sweep.StatusReporter.from_config", return_value=reporter),
        patch("srtctl.cli.do_sweep.setup_signal_handlers"),
        patch("srtctl.cli.do_sweep.start_process_monitor"),
        patch.object(orchestrator, "start_head_infrastructure", return_value=MagicMock()),
        patch.object(orchestrator, "start_mooncake_master", return_value=None),
        patch.object(orchestrator, "start_all_workers", return_value={}),
        patch.object(orchestrator, "start_frontend", return_value=[]),
        patch.object(orchestrator, "start_tachometer", return_value=[]),
        patch.object(orchestrator, "_print_connection_info"),
        patch.object(orchestrator, "run_benchmark", return_value=0) as run_serve,
        patch.object(orchestrator, "_run_post_eval") as run_eval,
        patch.object(orchestrator, "finalize_power_telemetry", side_effect=lambda exit_code, **_kwargs: exit_code),
        patch.object(orchestrator, "run_postprocess"),
    ):
        exit_code = orchestrator.run()

    assert exit_code == 0
    run_serve.assert_called_once()
    run_eval.assert_not_called()
