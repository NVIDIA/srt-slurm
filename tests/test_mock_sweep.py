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
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.mock import MockOptions, run_mock_sweep

MINIMAL_CONFIG = {
    "schema": 2,
    "name": "mock-smoke",
    "model": {
        "path": "hf:fake/mock-model",
        "container": "nvcr.io/fake:latest",
        "precision": "fp8",
    },
    "resources": {"gpu_type": "h100", "gpus_per_node": 8},
    "roles": {"agg": {"nodes": 1, "workers": 1}},
    "benchmark": {"type": "custom", "command": "echo fake-benchmark"},
}

TRTLLM_AGGREGATE_CONFIG = {
    "schema": 2,
    "name": "mock-trtllm-aggregate-sa-bench",
    "model": {
        "path": "hf:fake/mock-model",
        "container": "nvcr.io/fake:latest",
        "precision": "fp8",
    },
    "resources": {"gpu_type": "gb300", "gpus_per_node": 4},
    "frontend": {
        "type": "trtllm_serve",
        "enable_multiple_frontends": False,
    },
    "engine": "trtllm",
    "roles": {
        "agg": {
            "nodes": 2,
            "workers": 1,
            "gpus": 8,
            "args": {
                "tensor_parallel_size": 8,
                "moe_expert_parallel_size": 8,
                "pipeline_parallel_size": 1,
            },
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


@pytest.mark.parametrize("critical", [True, False])
def test_mock_worker_restart_preserves_role_launch_settings_and_records_exhaustion(
    tmp_path: Path, critical: bool
) -> None:
    config = deepcopy(MINIMAL_CONFIG)
    config["engine"] = "sglang"
    config["roles"]["agg"].update(
        critical=critical,
        env={"ROLE_SETTING": "kept"},
        srun_options={"mem": "4G"},
        restart={"policy": "on-failure", "max_restarts": 1, "backoff_seconds": 0},
    )
    config_path = tmp_path / "restart.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "outputs" / "42042"
    launches = []

    def kill_and_reconcile(self, registry, stop_event, reporter):
        reconcile = monitor.call_args.kwargs["reconcile"]
        original = registry.get_process("agg_0_mock-node-01")
        assert original is not None and original.supervised
        original.popen.kill()
        reconcile()
        assert not registry.check_failures()
        reconcile()
        replacement = registry.get_process("agg_0_mock-node-01_r1")
        assert replacement is not None and replacement.supervised
        assert replacement.critical is critical
        assert registry.get_process(original.name) is None

        worker_launches = [launch for launch in launches if launch.get("step_name", "").startswith("agg_")]
        assert len(worker_launches) == 2
        first, second = worker_launches
        for key in (
            "command",
            "nodelist",
            "env_to_set",
            "srun_options",
            "output",
            "container_image",
            "container_mounts",
        ):
            assert second[key] == first[key]
        assert second["env_to_set"]["ROLE_SETTING"] == "kept"
        assert second["srun_options"]["mem"] == "4G"

        replacement.popen.kill()
        reconcile()
        assert not replacement.supervised
        assert registry.check_failures() is critical
        return int(critical)

    with (
        patch("srtctl.cli.do_sweep.start_process_monitor") as monitor,
        patch.object(SweepOrchestrator, "run_benchmark", kill_and_reconcile),
    ):
        exit_code = run_mock_sweep(
            config_path=config_path,
            output_dir=output_dir,
            job_id="42042",
            options=MockOptions(child_duration_s=600, phase_pause_s=0, on_srun=launches.append),
        )

    assert exit_code == int(critical)
    restarts = json.loads((output_dir / "logs" / "worker_restarts.json").read_text())
    assert restarts["total_restarts"] == 1
    assert [event["outcome"] for event in restarts["events"]] == ["relaunched", "exhausted"]
    lockfile = yaml.safe_load((output_dir / "recipe.lock.yaml").read_text())
    assert lockfile["lock"]["worker_restarts"] == restarts


@pytest.mark.parametrize("critical", [True, False])
@pytest.mark.parametrize("fail_rank", [0, 1])
def test_mock_failed_relaunch_after_a_clean_exit_tracks_and_stops_partial_workers(
    tmp_path: Path, critical: bool, fail_rank: int
) -> None:
    config = deepcopy(MINIMAL_CONFIG)
    config["engine"] = "sglang"
    config["roles"]["agg"].update(
        nodes=2,
        gpus=16,
        critical=critical,
        restart={"policy": "always", "max_restarts": 1, "backoff_seconds": 0},
    )
    config_path = tmp_path / "restart.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "outputs" / "42042"

    def fail_and_reconcile(self, registry, stop_event, reporter):
        reconcile = monitor.call_args.kwargs["reconcile"]
        for node in ("mock-node-01", "mock-node-02"):
            original = registry.get_process(f"agg_0_{node}")
            assert original is not None and original.supervised
            original.popen._finalize()
            assert original.exit_code == 0
        reconcile()
        assert not registry.check_failures()
        start_worker = self.start_worker

        def failing_start(process, processes, *, attempt=0):
            if process.node_rank == fail_rank:
                if fail_rank:
                    assert registry.get_process("agg_0_mock-node-01_r1") is not None
                raise RuntimeError("srun launch failed")
            return start_worker(process, processes, attempt=attempt)

        with patch.object(self, "start_worker", failing_start):
            reconcile()
        partial = registry.get_process("agg_0_mock-node-01_r1")
        if fail_rank:
            assert partial is not None and not partial.is_running
        else:
            assert partial is None
        assert registry.check_failures() is critical
        return int(critical)

    with (
        patch("srtctl.cli.do_sweep.start_process_monitor") as monitor,
        patch.object(SweepOrchestrator, "run_benchmark", fail_and_reconcile),
    ):
        assert run_mock_sweep(
            config_path=config_path,
            output_dir=output_dir,
            job_id="42042",
            options=MockOptions(nodelist=("mock-node-01", "mock-node-02"), child_duration_s=600, phase_pause_s=0),
        ) == int(critical)
    restarts = json.loads((output_dir / "logs" / "worker_restarts.json").read_text())
    assert restarts["events"][-1]["outcome"] == "launch_failed"
    lockfile = yaml.safe_load((output_dir / "recipe.lock.yaml").read_text())
    assert lockfile["lock"]["worker_restarts"] == restarts


@pytest.mark.parametrize("frontend_type", ["sglang", "vllm", "trtllm_serve"])
def test_mock_direct_replacement_becomes_ready_on_its_public_port(tmp_path: Path, frontend_type: str) -> None:
    config = deepcopy(MINIMAL_CONFIG)
    config["engine"] = "trtllm" if frontend_type == "trtllm_serve" else frontend_type
    config["frontend"] = {"type": frontend_type, "enable_multiple_frontends": False}
    config["roles"]["agg"]["restart"] = {"policy": "on-failure", "backoff_seconds": 0}
    config_path = tmp_path / "restart.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "outputs" / "42042"

    def restart_and_probe(self, registry, stop_event, reporter):
        reconcile = monitor.call_args.kwargs["reconcile"]
        reconcile.__self__.probe_interval = 0
        original = registry.get_process("agg_0_mock-node-01")
        assert original is not None
        original.popen.kill()
        reconcile()
        reconcile()
        reconcile()
        probe.assert_called_once_with("127.0.0.1", self.runtime.frontend_port, "/health", 200, request_timeout=2.0)
        assert self.backend_processes[0].http_port != self.runtime.frontend_port
        assert not registry.check_failures()
        return 0

    with (
        patch("srtctl.cli.do_sweep.start_process_monitor") as monitor,
        patch("srtctl.core.supervisor.probe_http", return_value=True) as probe,
        patch.object(SweepOrchestrator, "run_benchmark", restart_and_probe),
    ):
        assert (
            run_mock_sweep(
                config_path=config_path,
                output_dir=output_dir,
                job_id="42042",
                options=MockOptions(child_duration_s=600, phase_pause_s=0),
            )
            == 0
        )
    restarts = json.loads((output_dir / "logs" / "worker_restarts.json").read_text())
    assert restarts["events"][-1]["outcome"] == "ready"


def test_mock_weight_cache_daemons_survive_an_engine_relaunch(tmp_path: Path) -> None:
    recipe = Path(__file__).resolve().parent.parent / "examples/features/sglang-weight-cache.yaml"
    config = yaml.safe_load(recipe.read_text())
    config["model"].update(path="hf:fake/mock-model", container="nvcr.io/fake:latest")
    config_path = tmp_path / "weight-cache.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "outputs" / "42042"
    launches = []

    def restart_engine(self, registry, stop_event, reporter):
        reconcile = monitor.call_args.kwargs["reconcile"]
        supervisor = reconcile.__self__
        clock = [0.0]
        supervisor._clock = lambda: clock[0]
        supervisor.probe_interval = 0
        daemon_names = [f"service_weight-cache_agg_{i}_mock-node-01" for i in range(2)]
        daemons = [registry.get_process(name) for name in daemon_names]
        assert all(proc is not None and proc.is_running and proc.critical for proc in daemons)
        daemon_launches = [call for call in launches if call.get("step_name", "").startswith("service_weight-cache_")]
        workers = [call for call in launches if call.get("step_name", "").startswith("agg_")]
        assert len(daemon_launches) == len(workers) == 2
        for daemon, worker in zip(daemon_launches, workers, strict=True):
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE",
                "SGLANG_WEIGHT_CACHE_READY_TEMPLATE",
            ):
                assert daemon["env_to_set"][key] == worker["env_to_set"][key]
            assert "{device_uuid}" in daemon["env_to_set"]["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"]
            assert worker["command"][-2:] == ["--weight-cache-mode", "client"]

        original = registry.get_process("agg_0_mock-node-01")
        untouched = registry.get_process("agg_1_mock-node-01")
        assert original is not None and untouched is not None
        original.popen.kill()
        reconcile()
        clock[0] = 5.0
        reconcile()
        reconcile()

        replacement = registry.get_process("agg_0_mock-node-01_r1")
        assert replacement is not None and replacement.is_running and replacement.supervised
        assert registry.get_process(untouched.name) is untouched and untouched.is_running
        for name, daemon in zip(daemon_names, daemons, strict=True):
            assert registry.get_process(name) is daemon and daemon.is_running
        assert len([call for call in launches if call.get("step_name", "").startswith("service_weight-cache_")]) == 2
        for key in ("nodelist", "env_to_set", "container_mounts", "command", "output"):
            assert launches[-1][key] == workers[0][key]
        probe.assert_called_once_with(
            "127.0.0.1", self.backend_processes[0].http_port, "/health", 200, request_timeout=2.0
        )
        assert not registry.check_failures()
        return 0

    with (
        patch("srtctl.cli.do_sweep.start_process_monitor") as monitor,
        patch("srtctl.core.supervisor.probe_http", return_value=True) as probe,
        patch.object(SweepOrchestrator, "run_benchmark", restart_engine),
    ):
        assert (
            run_mock_sweep(
                config_path=config_path,
                output_dir=output_dir,
                job_id="42042",
                options=MockOptions(child_duration_s=600, phase_pause_s=0, on_srun=launches.append),
            )
            == 0
        )
    restarts = json.loads((output_dir / "logs" / "worker_restarts.json").read_text())
    assert restarts["events"][-1]["outcome"] == "ready"
    lockfile = yaml.safe_load((output_dir / "recipe.lock.yaml").read_text())
    assert lockfile["lock"]["worker_restarts"] == restarts


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
    assert (output_dir / "logs" / "service_etcd.out").is_file(), "etcd is a service now"
    assert not (output_dir / "logs" / "service_nats.out").exists(), "tcp request plane: no NATS service runs"
    assert any((output_dir / "logs").glob("*_agg_w0.out")), "worker log written"
    assert any((output_dir / "logs").glob("*_frontend_*.out")), "frontend log written"
    assert (output_dir / "logs" / "benchmark.out").is_file()


@pytest.mark.parametrize("benchmark_exit_code", [0, 7])
def test_mock_sweep_postprocesses_and_propagates_exit_code(tmp_path: Path, benchmark_exit_code: int) -> None:
    cfg = _write_config(tmp_path)
    output_dir = tmp_path / "outputs" / "42046"

    # Inject the benchmark outcome while exercising the real cleanup/postprocess path.
    with patch.object(SweepOrchestrator, "run_benchmark", return_value=benchmark_exit_code):
        exit_code = run_mock_sweep(
            config_path=cfg,
            output_dir=output_dir,
            job_id="42046",
            options=MockOptions(child_duration_s=0.05, phase_pause_s=0.01),
        )

    assert exit_code == benchmark_exit_code
    assert (output_dir / "recipe.lock.yaml").is_file(), "post-processing still runs"
    # Guard against reintroducing dashboard artifacts into automatic post-processing.
    assert not list((output_dir / "logs").glob("perf_dashboard*"))


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
