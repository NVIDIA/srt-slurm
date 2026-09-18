# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic profiler configuration and launch wiring."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.core.observability_nsys import wrap_observability_nsys
from srtctl.core.schema import NsysObservabilityConfig, SrtConfig
from srtctl.frontends.dynamo import DynamoFrontend
from test_observability import BASE_CONFIG
from test_slurm import _remap_worker_mixin


def config(**overrides):
    data = deepcopy(BASE_CONFIG)
    data.update(backend={"type": "trtllm"}, observability={"enabled": True}, benchmark={"type": "manual"})
    data.update(overrides)
    return SrtConfig.Schema().load(data)


@pytest.mark.parametrize(("observability", "expected"), [
    ({}, False), ({"enabled": False, "nsys": {"enabled": True}}, False),
    ({"enabled": True}, True), ({"enabled": True, "nsys": {"enabled": False}}, False),
])
def test_preset_requires_observability_and_honors_opt_out(observability, expected):
    assert config(observability=observability).observability_nsys_enabled is expected


@pytest.mark.parametrize("profiling", [
    {"type": "torch", "prefill": {}, "decode": {}},
    {"type": "nsys", "prefill": {}, "decode": {}},
    {"type": "nsys-time", "delay_secs": 1, "duration_secs": 5},
])
def test_explicit_profiling_takes_precedence(profiling):
    cfg = config(profiling=profiling, backend={"type": "sglang"})
    assert not cfg.observability_nsys_enabled
    assert cfg.profiling.type == profiling["type"]


def test_yaml_round_trip_retains_settings_and_benchmark(tmp_path):
    cfg = config(observability={"enabled": True, "nsys": {
        "delay_secs": 12, "frontend_cpu_sampling": False, "report_timeout_secs": 45,
        "nvtx_injection_path": "/opt/nsys/libToolsInjection64.so",
    }})
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(SrtConfig.Schema().dump(cfg)))
    loaded = SrtConfig.from_yaml(path)
    assert loaded.observability.nsys == cfg.observability.nsys
    assert loaded.benchmark.type == "manual"
    assert loaded.profiling.type == "none"
    assert loaded.profiling.get_env_vars("prefill", str(tmp_path)) == {}


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"delay_secs": -1}, "delay_secs"),
    ({"report_timeout_secs": 0}, "report_timeout_secs"),
    ({"nvtx_injection_path": "relative/library.so"}, "absolute container path"),
])
def test_invalid_settings_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        NsysObservabilityConfig(**kwargs)


@pytest.mark.parametrize("frontend", [False, True])
def test_capture_preset_has_fresh_barrier_and_no_benchmark_controls(tmp_path, frontend, monkeypatch):
    monkeypatch.setenv("SRTCTL_NSYS_BIN", "/opt/nsys/bin/nsys")
    cfg = config(observability={"enabled": True, "nsys": {"delay_secs": 7, "nvtx_injection_path": "/opt/nvtx.so"}})
    command, env = wrap_observability_nsys(
        ["python3", "-m", "server"], config=cfg, log_dir=tmp_path,
        report_name="decode/worker_rank%q{SLURM_PROCID}", ranks=8, frontend=frontend,
    )
    script = command[2]
    assert "/opt/nsys/bin/nsys profile" in script
    assert "--trace=nvtx" in script and "--delay 7" in script
    assert "--duration" not in script and "cuda,nvtx" not in script
    assert ("--sample=system-wide" if frontend else "--sample=none") in script
    if frontend:
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
    cfg = config(observability={"enabled": enabled, "nsys": {"frontend_cpu_sampling": False, "report_timeout_secs": 60}})
    topology = SimpleNamespace(frontend_nodes=["node-a", "node-b"], frontend_port=8180)
    runtime = SimpleNamespace(
        log_dir=tmp_path, nodes=SimpleNamespace(infra="head", het_group_for=lambda node: None),
        container_image=Path("/container.sqsh"), container_mounts={}, environment={},
    )
    with patch("srtctl.frontends.dynamo.start_srun_process", return_value=MagicMock()) as launch:
        processes = DynamoFrontend().start_frontends(topology, runtime, cfg, MagicMock(), [])
    assert len(processes) == launch.call_count == 2
    for index, (call, proc) in enumerate(zip(launch.call_args_list, processes, strict=True)):
        command = call.kwargs["command"]
        if enabled:
            assert "dynamo.frontend" in command[2] and "--sample=none" in command[2]
            assert f"frontend/node-{'ab'[index]}_frontend_{index}" in command[2]
            assert call.kwargs["env_to_set"]["SRT_NSYS_REPORT_EXPECTED"] == "1"
            assert proc.terminate_timeout == 210
            assert not proc.signal_full
        else:
            assert command[:3] == ["python3", "-m", "dynamo.frontend"]
            assert proc.signal_full


@pytest.mark.parametrize("mpi", [False, True])
def test_worker_launch_profiles_every_task_with_unique_report_names(tmp_path, mpi):
    cfg = config()
    stage, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    # Keep the backend launcher stub, but exercise the real schema decision.
    backend = stage.backend
    backend.type = "trtllm" if mpi else "vllm"
    backend.mooncake_kv_store = None
    backend.get_srun_config.return_value = SimpleNamespace(mpi="pmix", oversubscribe=True, cpu_bind="none")
    stage.config = replace(cfg, backend=backend)
    stage.runtime.srun_options = {}
    second = SimpleNamespace(**{**vars(process), "node": "node-b"})
    stage.runtime.nodes.worker.append("node-b")
    with patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="true"), patch(
        "srtctl.cli.mixins.worker_stage.start_srun_process", return_value=MagicMock()
    ) as launch:
        managed = stage.start_endpoint_worker([process, second]) if mpi else stage.start_worker(process, [process])
    args = launch.call_args.kwargs
    assert "--sample=none" in args["command"][2]
    assert args["env_to_set"]["SRT_NSYS_REPORT_EXPECTED"] == ("16" if mpi else "1")
    assert not managed.signal_full
    assert managed.terminate_timeout == cfg.observability.nsys.terminate_timeout
    assert "PROFILE_TYPE" not in args["env_to_set"]
    if mpi:
        assert args["ntasks"] == 16
        assert "rank%q{SLURM_PROCID}" in args["command"][2]
    else:
        assert "profile_gpu0-1-2-3-4-5-6-7" in args["command"][2]
