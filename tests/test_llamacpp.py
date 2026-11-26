# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp schema, launch construction and the real orchestrator's Docker path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from textwrap import indent

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import Backend, LlamaCppBackend
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import load_config, resolve_config_with_defaults
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


@pytest.mark.parametrize(
    "flag",
    [
        "model",
        "host",
        "port",
        "metrics",
        "hf-repo",
        "hf_file",
        "docker-repo",
        "model-url",
        "api-key",
        "api_key_file",
        "api-prefix",
        "ssl-key-file",
        "ssl_cert_file",
    ],
)
def test_managed_arguments_cannot_disagree_with_runtime(recipe, flag):
    recipe["roles"]["agg"]["args"][flag] = "conflicting-value"
    with pytest.raises(ValidationError, match="SRT-managed flags"):
        load(recipe)


def load_yaml_arguments(recipe: dict, tmp_path: Path, arguments: str) -> SrtConfig:
    path = tmp_path / "recipe.yaml"
    path.write_text(
        yaml.safe_dump({key: value for key, value in recipe.items() if key != "roles"})
        + "roles:\n  agg:\n    nodes: 1\n    workers: 1\n    gpus: 1\n    args:\n"
        + indent(arguments, "      ")
        + "\n"
    )
    return load_config(path)


@pytest.mark.parametrize("flag", ["flash-attn", "flash_attn", "fit", "reasoning", "log-colors"])
@pytest.mark.parametrize("value, expected", [("on", "on"), ("off", "off"), ("true", "on"), ("false", "off")])
def test_yaml_value_booleans_emit_explicit_values(recipe, tmp_path, flag, value, expected):
    config = load_yaml_arguments(recipe, tmp_path, f"{flag}: {value}\nn-gpu-layers: 99\nmlock: true\nmmap: false")
    runtime = RuntimeContext.from_config(config, "llama-bools", log_dir_base=tmp_path / "out")
    processes = config.worker_processes(config.allocate_worker_endpoints(runtime.nodes.worker))
    command = config.backend.build_worker_command(processes[0], processes, runtime, frontend_type="llamacpp")
    name = "--" + flag.replace("_", "-")
    assert command[command.index(name) + 1] == expected
    assert command[command.index("--n-gpu-layers") + 1] == "99"
    assert "--mlock" in command
    assert "--mmap" not in command
    # Command rendering must not mutate the bound recipe args.
    assert config.backend.get_config_for_mode("agg")[flag] == yaml.safe_load(value)


@pytest.mark.parametrize(
    "arguments",
    [
        "tensor-split: [0.5, 0.5]",
        "tensor_split: [1, 1]",
        "device: [CUDA0, CUDA1]",
        "override-tensor: ['layer.*=CPU', 'output.*=CUDA0']",
        "lora: [first.gguf, second.gguf]",
        "lora-scaled: ['first.gguf:0.5', 'second.gguf:1.0']",
    ],
)
def test_yaml_lists_fail_at_config_load_with_quoted_string_guidance(recipe, tmp_path, arguments):
    with pytest.raises(ValueError, match="quoted.*string"):
        load_yaml_arguments(recipe, tmp_path, arguments)


def test_csv_strings_and_two_token_layer_range_keep_their_cli_shape(recipe, tmp_path):
    config = load_yaml_arguments(
        recipe,
        tmp_path,
        'tensor-split: "0.5,0.5"\ndevice: "CUDA0,CUDA1"\ncontrol-vector-layer-range: [2, 8]\nflash-attn: auto',
    )
    runtime = RuntimeContext.from_config(config, "llama-lists", log_dir_base=tmp_path / "out")
    processes = config.worker_processes(config.allocate_worker_endpoints(runtime.nodes.worker))
    command = config.backend.build_worker_command(processes[0], processes, runtime, frontend_type="llamacpp")
    assert command[command.index("--tensor-split") + 1] == "0.5,0.5"
    assert command[command.index("--device") + 1] == "CUDA0,CUDA1"
    assert command[command.index("--flash-attn") + 1] == "auto"
    start = command.index("--control-vector-layer-range")
    assert command[start + 1 : start + 3] == ["2", "8"]


@pytest.mark.parametrize("values", ["[]", "[2]", "[2, 8, 10]"])
def test_layer_range_requires_two_values(recipe, tmp_path, values):
    with pytest.raises(ValueError, match="control-vector-layer-range.*two"):
        load_yaml_arguments(recipe, tmp_path, "control-vector-layer-range: " + values)


@pytest.mark.parametrize("scope", ["role", "global"])
@pytest.mark.parametrize(
    "name",
    [
        "LLAMA_API_KEY",
        "LLAMA_ARG_API_KEY_FILE",
        "LLAMA_ARG_API_PREFIX",
        "LLAMA_ARG_SSL_KEY_FILE",
        "LLAMA_ARG_SSL_CERT_FILE",
        "LLAMA_ARG_HF_REPO",
        "LLAMA_ARG_HF_FILE",
        "LLAMA_ARG_DOCKER_REPO",
        "LLAMA_ARG_MODEL_URL",
        "LLAMA_ARG_MODEL",
        "LLAMA_ARG_HOST",
        "LLAMA_ARG_PORT",
        "LLAMA_ARG_ENDPOINT_METRICS",
    ],
)
def test_environment_cannot_change_the_model_or_endpoint_contract(recipe, scope, name):
    environment = {name: "unsupported-value-not-to-echo"}
    if scope == "role":
        recipe["roles"]["agg"]["env"] = environment
    else:
        recipe["environment"] = environment
    with pytest.raises(ValidationError, match=name) as exc:
        load(recipe)
    assert "unsupported-value-not-to-echo" not in str(exc.value)


def test_tuning_environment_is_preserved(recipe):
    recipe["environment"] = {"LLAMA_ARG_FLASH_ATTN": "on"}
    recipe["roles"]["agg"]["env"] = {"CUDA_VISIBLE_DEVICES": "0", "LLAMA_ARG_THREADS": "4"}
    config = load(recipe)
    assert config.environment["LLAMA_ARG_FLASH_ATTN"] == "on"
    assert config.backend.get_environment_for_mode("agg") == {"CUDA_VISIBLE_DEVICES": "0", "LLAMA_ARG_THREADS": "4"}


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
