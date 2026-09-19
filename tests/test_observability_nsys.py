# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic profiler configuration and launch wiring."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError
from test_observability import BASE_CONFIG
from test_slurm import _remap_worker_mixin

from srtctl.core.observability_nsys import wrap_observability_nsys
from srtctl.core.schema import NsysObservabilityConfig, SrtConfig
from srtctl.frontends.dynamo import DynamoFrontend


def config(**overrides):
    data = deepcopy(BASE_CONFIG)
    data.update(backend={"type": "trtllm"}, observability={"enabled": True}, benchmark={"type": "manual"})
    data.update(overrides)
    return SrtConfig.Schema().load(data)


@pytest.mark.parametrize(
    ("observability", "expected"),
    [
        ({}, False),
        ({"enabled": False, "nsys": {"enabled": True}}, False),
        ({"enabled": True}, True),
        ({"enabled": True, "nsys": {"enabled": False}}, False),
    ],
)
def test_preset_requires_observability_and_honors_opt_out(observability, expected):
    cfg = config(observability=observability)
    assert cfg.observability_nsys_enabled is expected
    assert cfg.observability.nsys.capture_window == "measured_workload"


@pytest.mark.parametrize(
    "profiling",
    [
        {"type": "torch", "prefill": {}, "decode": {}},
        {"type": "nsys", "prefill": {}, "decode": {}},
        {"type": "nsys-time", "delay_secs": 1, "duration_secs": 5},
    ],
)
def test_explicit_profiling_takes_precedence(profiling):
    cfg = config(profiling=profiling, backend={"type": "sglang"})
    assert not cfg.observability_nsys_enabled
    assert cfg.profiling.type == profiling["type"]


def test_yaml_round_trip_retains_settings_and_benchmark(tmp_path):
    cfg = config(
        observability={
            "enabled": True,
            "nsys": {
                "capture_window": "including_startup",
                "report_timeout_secs": 45,
                "nvtx_injection_path": "/opt/nsys/libToolsInjection64.so",
            },
        }
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(SrtConfig.Schema().dump(cfg)))
    loaded = SrtConfig.from_yaml(path)
    assert loaded.observability.nsys == cfg.observability.nsys
    assert loaded.benchmark.type == "manual"
    assert loaded.profiling.type == "none"
    assert loaded.profiling.get_env_vars("prefill", str(tmp_path)) == {}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capture_window": "time"}, "capture_window"),
        ({"report_timeout_secs": 0}, "report_timeout_secs"),
        ({"nvtx_injection_path": "relative/library.so"}, "absolute container path"),
    ],
)
def test_invalid_settings_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        NsysObservabilityConfig(**kwargs)


@pytest.mark.parametrize("frontend", [False, True])
@pytest.mark.parametrize("capture_window", ["measured_workload", "including_startup"])
def test_capture_preset_has_fresh_barrier_and_no_benchmark_controls(tmp_path, frontend, capture_window, monkeypatch):
    monkeypatch.setenv("SRTCTL_NSYS_BIN", "/opt/nsys/bin/nsys")
    cfg = config(
        observability={
            "enabled": True,
            "nsys": {"capture_window": capture_window, "nvtx_injection_path": "/opt/nvtx.so"},
        }
    )
    command, env = wrap_observability_nsys(
        ["python3", "-m", "server"],
        config=cfg,
        log_dir=tmp_path,
        report_name="decode/worker_rank%q{SLURM_PROCID}",
        ranks=8,
        frontend=frontend,
    )
    if capture_window == "measured_workload":
        assert command[:3] == ["python3", "/srtctl-runtime/nsys_window.py", "worker"]
        spec = json.loads(command[4])
        assert spec["nsys"] == "/opt/nsys/bin/nsys"
        assert spec["ranks"] == 8
        assert spec["output"].endswith("worker_rank%q{SLURM_PROCID}")
        assert list((tmp_path / "profiles/.control/steps").glob("*.json"))
        script = command[4]
    else:
        script = command[2]
        assert "/opt/nsys/bin/nsys profile" in script
        assert "--trace=nvtx" in script
    assert "--delay" not in script
    assert "--duration" not in script and "cuda,nvtx" not in script
    assert "--sample=process-tree" in script
    assert "--sampling-period=26000000" in script and "--samples-per-backtrace=32" in script
    assert env["SRT_NSYS_REPORT_EXPECTED"] == "8"
    assert env["NVTX_INJECTION64_PATH"] == "/opt/nvtx.so"
    assert env["DYN_ENABLE_RUST_NVTX"] == "1"
    assert "PROFILE_TYPE" not in env and "TLLM_PROFILE_START_STOP" not in env
    if not frontend:
        assert env["TLLM_PROFILE_LOG_RANKS"] == "all"
        assert env["TLLM_LLMAPI_ENABLE_NVTX"] == "1"
    _, retry_env = wrap_observability_nsys(["worker"], config=cfg, log_dir=tmp_path, report_name="retry")
    assert env["SRT_NSYS_REPORT_BARRIER_DIR"] != retry_env["SRT_NSYS_REPORT_BARRIER_DIR"]


@pytest.mark.parametrize("enabled", [False, True])
def test_every_dynamo_frontend_is_wrapped_and_gets_shutdown_budget(tmp_path, enabled):
    cfg = config(observability={"enabled": enabled, "nsys": {"report_timeout_secs": 60}})
    topology = SimpleNamespace(frontend_nodes=["node-a", "node-b"], frontend_port=8180)
    runtime = SimpleNamespace(
        log_dir=tmp_path,
        nodes=SimpleNamespace(infra="head", het_group_for=lambda node: None),
        infra_node_ip="10.0.0.9",
        container_image=Path("/container.sqsh"),
        container_mounts={},
        environment={},
    )
    with patch("srtctl.frontends.dynamo.start_srun_process", return_value=MagicMock()) as launch:
        processes = DynamoFrontend().start_frontends(topology, runtime, cfg, MagicMock(), [])
    assert len(processes) == launch.call_count == 2
    for index, (call, proc) in enumerate(zip(launch.call_args_list, processes, strict=True)):
        command = call.kwargs["command"]
        if enabled:
            spec = json.loads(command[4])
            assert "dynamo.frontend" in command and "--sample=process-tree" in spec["start_args"]
            assert spec["output"].endswith(f"frontend/node-{'ab'[index]}_frontend_{index}")
            assert call.kwargs["env_to_set"]["SRT_NSYS_REPORT_EXPECTED"] == "1"
            assert proc.terminate_timeout == 210
            assert not proc.signal_full
        else:
            assert command[:3] == ["python3", "-m", "dynamo.frontend"]
            assert proc.signal_full


@pytest.mark.parametrize(("mpi", "engine_suffix"), [(False, ""), (True, ""), (False, "_e1")])
def test_worker_launch_profiles_every_task_with_unique_report_names(tmp_path, mpi, engine_suffix):
    cfg = config(backend={"type": "trtllm" if mpi else "vllm"})
    stage, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    # Validate the actual backend schema; stub only command/env construction.
    stage.config = cfg
    backend_class = type(cfg.backend)
    process.engine_suffix = engine_suffix
    process.engine_id = 1 if engine_suffix else 0
    stage.runtime.srun_options = {}
    second = SimpleNamespace(**{**vars(process), "node": "node-b"})
    stage.runtime.nodes.worker.append("node-b")
    with (
        patch.object(backend_class, "build_worker_command", return_value=["python3", "-m", "worker"]),
        patch.object(backend_class, "get_environment_for_mode", return_value={}),
        patch.object(backend_class, "get_process_environment", return_value={}),
        patch.object(
            backend_class,
            "get_srun_config",
            return_value=SimpleNamespace(mpi="pmix", oversubscribe=True, cpu_bind="none"),
        ),
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="10.0.0.2"),
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process", return_value=MagicMock()) as launch,
    ):
        managed = stage.start_endpoint_worker([process, second]) if mpi else stage.start_worker(process, [process])
    args = launch.call_args.kwargs
    spec = json.loads(args["command"][4])
    assert "--sample=process-tree" in spec["start_args"]
    assert args["env_to_set"]["SRT_NSYS_REPORT_EXPECTED"] == ("16" if mpi else "1")
    assert not managed.signal_full
    assert managed.terminate_timeout == cfg.observability.nsys.terminate_timeout
    assert "PROFILE_TYPE" not in args["env_to_set"]
    if mpi:
        assert args["ntasks"] == 16
        assert "rank%q{SLURM_PROCID}" in spec["output"]
    else:
        assert f"_w0{engine_suffix}_profile_gpu0-1-2-3-4-5-6-7" in spec["output"]


@pytest.mark.parametrize("completed", [False, True])
def test_benchmark_success_requires_a_completed_capture(tmp_path, completed):
    import threading

    from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
    from srtctl.runtime_scripts.nsys_window import write_json

    stage = BenchmarkStageMixin()
    stage.config = config(benchmark={"type": "custom", "command": "echo measured"})
    stage.runtime = MagicMock()
    stage.runtime.log_dir = tmp_path
    stage._get_benchmark_env = MagicMock(return_value={})
    stage._benchmark_node = MagicMock(return_value="node")
    runner = MagicMock()
    runner.build_command.return_value = ["true"]
    runner.get_environment.return_value = {}
    proc = MagicMock()
    proc.poll.return_value = proc.returncode = 0
    if completed:
        write_json(tmp_path / "profiles/.control/client.json", {"active": False, "completed": 1})
    with (
        patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=proc),
        patch("srtctl.analysis.host_sampler.try_start_host_sampler", return_value=None),
    ):
        result = stage._run_benchmark_script(runner, tmp_path / "benchmark.out", threading.Event())
    assert result == (0 if completed else 1)


def test_benchmark_hooks_are_enabled_only_for_workload_capture():
    from srtctl.core.observability_nsys import benchmark_nsys_env

    env = benchmark_nsys_env(config())
    assert env["SRT_NSYS_CONTROL_SCRIPT"] == "/srtctl-runtime/nsys_window.py"
    assert env["SRT_NSYS_CONTROL_DIR"] == "/logs/profiles/.control"
    assert not benchmark_nsys_env(
        config(observability={"enabled": True, "nsys": {"capture_window": "including_startup"}})
    )
    assert not benchmark_nsys_env(config(observability={"enabled": False}))


@pytest.mark.parametrize("benchmark_type", ["agentperf", "mmlu", "router"])
def test_benchmarks_without_warmup_hooks_require_an_explicit_capture_choice(benchmark_type):
    with pytest.raises(ValidationError, match="has no warmup hooks"):
        config(benchmark={"type": benchmark_type})
    cfg = config(
        benchmark={"type": benchmark_type},
        observability={"enabled": True, "nsys": {"capture_window": "including_startup"}},
    )
    assert cfg.observability_nsys_enabled
    cfg = config(benchmark={"type": benchmark_type}, observability={"enabled": True, "nsys": {"enabled": False}})
    assert not cfg.observability_nsys_enabled


@pytest.mark.parametrize("flag", ["warmup-request-count", "num-warmup-requests", "warmup-duration"])
def test_aiperf_internal_warmup_cannot_silently_fall_inside_workload_capture(flag):
    with pytest.raises(ValidationError, match="additional aiperf_args warmup"):
        config(benchmark={"type": "trace-replay", "aiperf_args": {flag: 10}})
    config(benchmark={"type": "trace-replay", "aiperf_args": {flag: 0}})
