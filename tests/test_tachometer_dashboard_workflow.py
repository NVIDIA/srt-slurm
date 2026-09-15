# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-level routing and lifecycle guarantees for the raw Tachometer dashboard."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig


def _config(tmp_path: Path, frontend: str, *, backend: str = "trtllm", benchmark: str = "custom") -> SrtConfig:
    recipe = {
        "schema": 2,
        "name": "raw-dashboard-workflow",
        "model": {"path": "/model/test", "container": "/image.sqsh", "precision": "fp4"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1},
        "backend": {"type": backend},
        "frontend": {"type": frontend, "enable_multiple_frontends": False},
        "benchmark": {"type": benchmark, **({"command": "echo test-load"} if benchmark == "custom" else {})},
        # Analytics is off, but default-on Tachometer still supplies the run UI.
        "observability": {"enabled": False, "tachometer": {"storage_subdir": "capture/custom"}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    config = SrtConfig.from_yaml(config_path)
    assert config.schema_version == 2
    assert config.observability.enabled is False
    assert config.observability.tachometer_enabled is True
    return config


def _runtime(tmp_path: Path) -> RuntimeContext:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return RuntimeContext(
        job_id="12345",
        run_name="raw-dashboard-workflow-12345",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0",)),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=log_dir,
        model_path=Path("/model/test"),
        container_image=Path("/image.sqsh"),
        gpus_per_node=8,
        network_interface=None,
    )


def _write_capture(config: SrtConfig, runtime: RuntimeContext) -> Path:
    path = runtime.log_dir / config.observability.tachometer.storage_subdir / "local" / "final.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "metric_name": 'gpu_util{gpu="0"}',
                    "metric_value": value,
                    "scraper_endpoint": "dcgm_node0",
                    "hostname": "node0",
                    "time_since_start": float(t),
                }
                for t, value in [(0, 20.0), (1, 80.0), (2, 0.0)]
            ]
        ),
        path,
    )
    return path


def _assert_dashboard(runtime: RuntimeContext) -> None:
    html = runtime.log_dir / "dashboard.html"
    assert html.is_file()
    assert 'id="dashboard-catalog"' in html.read_text()
    catalog = json.loads((runtime.log_dir / "dashboard.data" / "catalog.json").read_text())
    assert catalog["row_count"] == 3
    assert catalog["source_policy"] == "Raw Tachometer Parquet/Arrow only"
    assert catalog["start_ns"] is None, "a display title or config must never provide the missing epoch"
    assert not (runtime.log_dir / "perf_dashboard.html").exists()


@pytest.mark.parametrize(
    ("backend", "frontend", "expected"),
    [("trtllm", "dynamo", "raw"), ("trtllm", "trtllm_serve", "raw"), ("sglang", "dynamo", "legacy")],
)
def test_real_v2_config_selects_the_run_ui(tmp_path: Path, monkeypatch, backend: str, frontend: str, expected: str):
    from srtctl.analysis import perf_dashboard
    from srtctl.analysis.tachometer_dashboard import pipeline

    stage = PostProcessStageMixin()
    stage.config = _config(tmp_path, frontend, backend=backend)
    stage.runtime = _runtime(tmp_path)
    raw, legacy = MagicMock(), MagicMock()
    monkeypatch.setattr(pipeline, "try_build", raw)
    monkeypatch.setattr(perf_dashboard, "try_build", legacy)

    stage._build_run_dashboard()

    selected, other = (raw, legacy) if expected == "raw" else (legacy, raw)
    selected.assert_called_once_with(stage.config, stage.runtime)
    other.assert_not_called()


@pytest.mark.parametrize("outcome", [None, RuntimeError("raw renderer failed")])
def test_unavailable_raw_ui_never_falls_back_to_legacy_inputs(tmp_path: Path, monkeypatch, outcome):
    from srtctl.analysis import perf_dashboard
    from srtctl.analysis.tachometer_dashboard import pipeline

    stage = PostProcessStageMixin()
    stage.config = _config(tmp_path, "trtllm_serve")
    stage.runtime = _runtime(tmp_path)
    raw = MagicMock(side_effect=outcome if isinstance(outcome, Exception) else None, return_value=None)
    legacy = MagicMock()
    monkeypatch.setattr(pipeline, "try_build", raw)
    monkeypatch.setattr(perf_dashboard, "try_build", legacy)

    stage._build_run_dashboard()

    raw.assert_called_once()
    legacy.assert_not_called()


@pytest.mark.parametrize("frontend", ["dynamo", "trtllm_serve"])
def test_raw_artifacts_exist_before_upload_without_analytics(tmp_path: Path, monkeypatch, frontend: str):
    stage = PostProcessStageMixin()
    stage.config = _config(tmp_path, frontend)
    stage.runtime = _runtime(tmp_path)
    _write_capture(stage.config, stage.runtime)
    monkeypatch.setattr("srtctl.cli.mixins.postprocess_stage.load_cluster_config", lambda: None)
    upload = MagicMock(side_effect=lambda: (_assert_dashboard(stage.runtime), "s3://example/run")[1])
    monkeypatch.setattr(stage, "_run_postprocess_container", upload)
    reporter = MagicMock()

    stage.run_postprocess(0, reporter=reporter)

    upload.assert_called_once()
    reporter.report_artifacts.assert_called_once_with(logs_url="s3://example/run")
    assert stage._last_logs_url == "s3://example/run"


def test_other_backend_builds_legacy_ui_after_preparing_inputs_and_before_upload(tmp_path: Path, monkeypatch):
    from srtctl.analysis import perf_dashboard
    from srtctl.analysis.tachometer_dashboard import pipeline

    stage = PostProcessStageMixin()
    stage.config = _config(tmp_path, "dynamo", backend="sglang")
    stage.runtime = _runtime(tmp_path)
    rollup = stage.runtime.log_dir / "benchmark-rollup.json"
    normalized = stage.runtime.log_dir / ".ruter" / "manifest.json"
    html = stage.runtime.log_dir / "perf_dashboard.html"

    def generate_rollup() -> None:
        rollup.write_text(json.dumps({"runs": [], "prepared": "rollup"}))

    def normalize_ruter() -> None:
        normalized.parent.mkdir()
        normalized.write_text(json.dumps({"prepared": "normalized"}))

    def build_legacy(config: SrtConfig, runtime: RuntimeContext) -> Path:
        assert config is stage.config and runtime is stage.runtime
        assert json.loads(rollup.read_text())["prepared"] == "rollup"
        assert json.loads(normalized.read_text())["prepared"] == "normalized"
        html.write_text("<html>legacy dashboard with prepared inputs</html>")
        return html

    def upload_artifacts() -> None:
        assert html.read_text() == "<html>legacy dashboard with prepared inputs</html>"

    legacy = MagicMock(side_effect=build_legacy)
    raw = MagicMock()
    upload = MagicMock(side_effect=upload_artifacts)
    monkeypatch.setattr(stage, "_generate_rollup", generate_rollup)
    monkeypatch.setattr(stage, "_normalize_ruter", normalize_ruter)
    monkeypatch.setattr(stage, "_run_postprocess_container", upload)
    monkeypatch.setattr("srtctl.cli.mixins.postprocess_stage.load_cluster_config", lambda: None)
    monkeypatch.setattr(perf_dashboard, "try_build", legacy)
    monkeypatch.setattr(pipeline, "try_build", raw)

    stage.run_postprocess(0)

    legacy.assert_called_once_with(stage.config, stage.runtime)
    upload.assert_called_once()
    raw.assert_not_called()
    assert not (stage.runtime.log_dir / "dashboard.html").exists()


@pytest.mark.parametrize("failing_stage", ["_generate_rollup", "_normalize_ruter"])
def test_legacy_processing_exception_cannot_prevent_raw_artifacts(tmp_path: Path, monkeypatch, failing_stage: str):
    stage = PostProcessStageMixin()
    stage.config = _config(tmp_path, "dynamo")
    stage.runtime = _runtime(tmp_path)
    _write_capture(stage.config, stage.runtime)

    def fail_legacy_processing():
        _assert_dashboard(stage.runtime)
        raise RuntimeError("unrelated legacy processing failed")

    monkeypatch.setattr(stage, failing_stage, fail_legacy_processing)
    with pytest.raises(RuntimeError, match="unrelated legacy processing failed"):
        stage.run_postprocess(1)
    _assert_dashboard(stage.runtime)


class _FlushingScraper:
    """Only the external scraper process is faked; reaping publishes its raw file."""

    pid = 123

    def __init__(self, config: SrtConfig, runtime: RuntimeContext, events: list[str]):
        self.config, self.runtime, self.events = config, runtime, events
        self.returncode: int | None = None
        self.stop_requested = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.stop_requested = True
        self.events.append("scraper_stop_requested")

    def wait(self, timeout: float | None = None) -> int:
        assert self.stop_requested, "the orchestrator must stop capture before trying to reap it"
        if self.returncode is None:
            _write_capture(self.config, self.runtime)
            self.events.append("capture_flushed")
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        pytest.fail("the tiny scraper must be reaped gracefully")


@pytest.mark.parametrize(
    ("frontend", "mode"),
    [
        ("dynamo", "benchmark"),
        ("trtllm_serve", "benchmark"),
        ("dynamo", "serve"),
        ("trtllm_serve", "manual"),
        ("trtllm_serve", "eval"),
        ("dynamo", "failed_benchmark"),
        ("trtllm_serve", "interrupted"),
    ],
)
def test_orchestrator_builds_after_capture_cleanup(tmp_path: Path, monkeypatch, frontend: str, mode: str):
    from srtctl.analysis.tachometer_dashboard import pipeline

    config = _config(tmp_path, frontend, benchmark="manual" if mode == "manual" else "custom")
    runtime = _runtime(tmp_path)
    orchestrator = SweepOrchestrator(config=config, runtime=runtime, serve_only=mode == "serve")
    events: list[str] = []
    scraper = _FlushingScraper(config, runtime, events)
    managed = ManagedProcess(name="tachometer", popen=scraper, critical=False)
    reporter = MagicMock()

    monkeypatch.delenv("RUN_EVAL", raising=False)
    monkeypatch.setenv("EVAL_ONLY", "true" if mode == "eval" else "false")
    monkeypatch.setattr("srtctl.cli.do_sweep.record_resource_snapshot", lambda *_args: {})
    monkeypatch.setattr("srtctl.cli.do_sweep.StatusReporter.from_config", lambda *_args: reporter)
    monkeypatch.setattr("srtctl.cli.do_sweep.setup_signal_handlers", lambda *_args: None)
    monkeypatch.setattr("srtctl.cli.do_sweep.start_process_monitor", lambda *_args: None)
    monkeypatch.setattr("srtctl.cli.mixins.postprocess_stage.load_cluster_config", lambda: None)
    monkeypatch.setattr(orchestrator, "start_head_infrastructure", lambda *_args: None)
    monkeypatch.setattr(orchestrator, "start_services", lambda *_args: None)
    monkeypatch.setattr(orchestrator, "start_all_workers", dict)
    monkeypatch.setattr(orchestrator, "start_frontend", lambda *_args: [])
    monkeypatch.setattr(orchestrator, "_print_connection_info", lambda: None)
    monkeypatch.setattr(orchestrator, "start_tachometer", lambda: [managed])

    def ready(stop_event: threading.Event) -> bool:
        if mode in {"serve", "manual"}:
            stop_event.set()
        return True

    def run_client(*_args) -> int:
        assert scraper.poll() is None
        if mode == "interrupted":
            raise SystemExit(130)
        if mode == "failed_benchmark":
            raise RuntimeError("client failed")
        return 0

    monkeypatch.setattr(orchestrator, "_wait_for_service_ready", ready)
    monkeypatch.setattr(orchestrator, "_run_benchmark_script", run_client)
    monkeypatch.setattr(orchestrator, "_run_post_eval", run_client)
    real_cleanup = ProcessRegistry.cleanup

    def cleanup(registry: ProcessRegistry) -> None:
        real_cleanup(registry)
        events.append("cleanup_finished")

    monkeypatch.setattr(ProcessRegistry, "cleanup", cleanup)
    real_build = pipeline.try_build

    def build_after_flush(config: SrtConfig, runtime: RuntimeContext) -> Path | None:
        assert scraper.poll() == 0
        assert events.count("capture_flushed") == 1
        assert "cleanup_finished" in events
        events.append("dashboard")
        return real_build(config, runtime)

    monkeypatch.setattr(pipeline, "try_build", build_after_flush)

    def upload() -> None:
        _assert_dashboard(runtime)
        events.append("upload")

    monkeypatch.setattr(orchestrator, "_run_postprocess_container", upload)
    if mode == "interrupted":
        with pytest.raises(SystemExit) as caught:
            orchestrator.run()
        assert caught.value.code == 130
    else:
        assert orchestrator.run() == (1 if mode == "failed_benchmark" else 0)

    _assert_dashboard(runtime)
    assert events.count("capture_flushed") == events.count("dashboard") == events.count("upload") == 1
    assert events.index("scraper_stop_requested") < events.index("capture_flushed")
    assert events.index("capture_flushed") < events.index("dashboard")
    assert events.index("cleanup_finished") < events.index("dashboard") < events.index("upload")
    reporter.report_completed.assert_called_once()
