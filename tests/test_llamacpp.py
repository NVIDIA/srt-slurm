# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp schema, launch construction and the real orchestrator's Docker path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import Backend, LlamaCppBackend
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.docker import DockerLauncher
from srtctl.core.launcher import LaunchSpec
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep


@pytest.fixture
def recipe(tmp_path, monkeypatch) -> dict:
    cluster = tmp_path / "srtslurm.yaml"
    cluster.write_text("launcher: docker\n")
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster))
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.gguf").write_bytes(b"GGUF")
    return {
        "schema": 2,
        "name": "llamacpp-test",
        "model": {"path": str(model), "container": "llama:server-cuda", "precision": "q8_0"},
        "resources": {"gpu_type": "l4", "gpus_per_node": 4},
        "engine": {"type": "llamacpp", "model_file": "weights.gguf"},
        "frontend": {"type": "llamacpp", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 2, "args": {"ctx-size": 4096, "n-gpu-layers": 99}}},
        "benchmark": {"type": "manual"},
    }


def load(recipe: dict) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(recipe, None))


def test_backend_is_frozen_and_round_trips(recipe):
    config = load(recipe)
    backend = config.backend
    assert isinstance(backend, LlamaCppBackend)
    assert isinstance(backend, Backend)
    assert config.served_model_name == "weights"
    assert load(SrtConfig.Schema().dump(config)) == config
    with pytest.raises(FrozenInstanceError):
        backend.metrics = False


@pytest.mark.parametrize("model_file", ["", "../weights.gguf", "/model/weights.gguf", "weights.safetensors"])
def test_invalid_model_file_is_rejected_at_load(recipe, model_file):
    recipe["engine"]["model_file"] = model_file
    with pytest.raises(ValidationError, match="relative .gguf filename inside model.path"):
        load(recipe)


def test_missing_model_file_is_required(recipe):
    del recipe["engine"]["model_file"]
    with pytest.raises(ValidationError, match="model_file"):
        load(recipe)


@pytest.mark.parametrize("flag", ["model", "m", "host", "port", "metrics"])
def test_managed_arguments_cannot_disagree_with_runtime(recipe, flag):
    recipe["roles"]["agg"]["args"][flag] = "conflicting-value"
    with pytest.raises(ValidationError, match="SRT-managed flags"):
        load(recipe)


def test_command_uses_gguf_mount_public_port_alias_and_arguments(recipe, tmp_path):
    recipe["engine"]["model_file"] = "nested/weights.gguf"
    recipe["roles"]["agg"]["args"].update({"alias": "local-qwen", "flash_attn": "on", "threads": 8})
    config = load(recipe)
    runtime = RuntimeContext.from_config(config, "llama-test", log_dir_base=tmp_path / "out")
    endpoints = config.allocate_worker_endpoints(runtime.nodes.worker)
    processes = config.worker_processes(endpoints)
    assert len(processes) == 1
    process = processes[0]
    assert process.gpu_indices == frozenset({0, 1})
    command = config.backend.build_worker_command(
        process, processes, runtime, frontend_type="llamacpp", nsys_prefix=["nsys", "profile"]
    )
    assert command[:3] == ["nsys", "profile", "/app/llama-server"]
    assert command[command.index("--model") + 1] == "/model/nested/weights.gguf"
    assert command[command.index("--port") + 1] == str(runtime.frontend_port)
    assert command[command.index("--alias") + 1] == config.served_model_name == "local-qwen"
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert "--metrics" in command
    assert command[-2:] == ["--threads", "8"]
    spec = LaunchSpec(
        command=command,
        container_image=runtime.container_image,
        container_mounts=runtime.container_mounts,
    )
    docker_command = DockerLauncher().docker_command(spec, "test-llama")
    assert "--network" in docker_command
    assert f"{runtime.model_path}:/model" in docker_command
    assert "llama:server-cuda" in docker_command


def test_custom_binary_and_metrics_opt_out(recipe, tmp_path):
    recipe["engine"].update({"server_binary": "/usr/local/bin/llama-server", "metrics": False})
    config = load(recipe)
    runtime = RuntimeContext.from_config(config, "llama-test", log_dir_base=tmp_path / "out")
    processes = config.worker_processes(config.allocate_worker_endpoints(runtime.nodes.worker))
    command = config.backend.build_worker_command(processes[0], processes, runtime, frontend_type="llamacpp")
    assert command[0] == "/usr/local/bin/llama-server"
    assert "--metrics" not in command


@pytest.mark.parametrize(
    ("alias", "server_aliases", "client_name"),
    [(None, "weights", "weights"), ("", "weights", "weights"), ("primary,secondary", "primary,secondary", "primary")],
)
def test_command_and_client_resolve_the_same_model_identity(recipe, tmp_path, alias, server_aliases, client_name):
    recipe["roles"]["agg"]["args"]["alias"] = alias
    config = load(recipe)
    runtime = RuntimeContext.from_config(config, "llama-alias", log_dir_base=tmp_path / "out")
    processes = config.worker_processes(config.allocate_worker_endpoints(runtime.nodes.worker))
    command = config.backend.build_worker_command(processes[0], processes, runtime, frontend_type="llamacpp")
    assert command[command.index("--alias") + 1] == server_aliases
    assert config.served_model_name == client_name


def test_mock_docker_run_keeps_worker_and_benchmark_images_separate(recipe, tmp_path):
    recipe["benchmark"] = {
        "type": "custom",
        "container_image": "python:3.12-slim",
        "command": 'echo "$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT"',
    }
    # Keep this test about the engine and client; exporter launch behavior is covered elsewhere.
    recipe["observability"] = {"tachometer": {"enabled": False}}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    launches: list[dict] = []
    exit_code = run_mock_sweep(
        config_path=path,
        output_dir=tmp_path / "out",
        job_id="llamacpp-mock",
        options=MockOptions(child_duration_s=0, phase_pause_s=0, on_srun=launches.append),
    )
    assert exit_code == 0
    worker = next(call for call in launches if call["command"][0] == "/app/llama-server")
    assert worker["container_image"] == "llama:server-cuda"
    assert worker["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert worker["step_name"]
    assert not any("dynamo.frontend" in call["command"] for call in launches)
    client = next(call for call in launches if call.get("step_name") == "benchmark")
    assert client["container_image"] == "python:3.12-slim"
    assert client["env_to_set"]["SRT_FRONTEND_PORT"] == "8000"
    assert client["env_to_set"]["SRT_AGG_ENDPOINTS"].endswith(":8000")


def test_manual_run_uses_existing_hold_and_cleanup(recipe, tmp_path):
    recipe["observability"] = {"tachometer": {"enabled": False}}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "out",
            job_id="llamacpp-serve",
            options=MockOptions(child_duration_s=0, phase_pause_s=0, end_manual_hold=True),
        )
        == 0
    )
    assert (tmp_path / "out" / "logs" / "server_ready.json").is_file()


@pytest.mark.parametrize("metrics", [True, False])
def test_metrics_targets_follow_the_engine_setting(recipe, tmp_path, metrics):
    recipe["engine"]["metrics"] = metrics
    config = load(recipe)
    runtime = RuntimeContext.from_config(config, "llama-metrics", log_dir_base=tmp_path / "out")
    orchestrator = SweepOrchestrator(config=config, runtime=runtime)
    capture = orchestrator.tachometer_config_text()
    assert ('name = "backend_agg0_rank0"' in capture) is metrics
    assert 'name = "frontend0"' not in capture


def test_bound_arguments_remain_independent_copies(recipe):
    backend = load(recipe).backend
    args = backend.get_config_for_mode("agg")
    args["ctx-size"] = 8192
    assert backend.get_config_for_mode("agg")["ctx-size"] == 4096
    assert replace(backend, metrics=False).metrics is False
