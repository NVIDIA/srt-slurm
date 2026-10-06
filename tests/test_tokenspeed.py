# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed backend: recipe validation and the dynamo.tokenspeed launch."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import TokenSpeedBackend
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep


def _recipe(**overrides: Any) -> dict[str, Any]:
    recipe: dict[str, Any] = {
        "schema": 2,
        "name": "tokenspeed",
        "model": {"path": "hf:Qwen/Qwen3-0.6B", "container": "tokenspeed", "precision": "bf16"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "dynamo": {"install": False},
        "frontend": {"type": "dynamo", "enable_multiple_frontends": False},
        "engine": "tokenspeed",
        "roles": {
            # One TP16 prefill worker across two nodes, two TP4 decode workers sharing a third.
            "prefill": {"nodes": 2, "workers": 1, "args": {"tensor-parallel-size": 16}},
            "decode": {"nodes": 1, "workers": 2, "gpus": 4, "args": {"served-model-name": "qwen3"}},
        },
        "benchmark": {"type": "custom", "command": "echo done"},
    }
    recipe.update(overrides)
    return recipe


def _load(recipe: dict[str, Any]) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(recipe, None))


@pytest.mark.parametrize("name", ["dynamo-agg", "dynamo-disagg"])
def test_examples_load_as_tokenspeed(name: str) -> None:
    config = SrtConfig.from_yaml(Path("examples/tokenspeed") / f"{name}.yaml")
    assert isinstance(config.backend, TokenSpeedBackend)
    assert config.served_model_name == "Qwen/Qwen3-0.6B"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"frontend": {"type": "sglang-router"}}, "requires backend.type: sglang"),
        ({"dynamo": {"install": False, "sidecar": True}}, "supports sglang, vllm, and trtllm backends only"),
        (
            {"roles": {"agg": {"nodes": 1, "workers": 1, "kv_events": True}}},
            "kv_events is not supported by the tokenspeed engine",
        ),
    ],
)
def test_unsupported_settings_fail_at_load(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _load(_recipe(**change))


def _flag(command: list[str], flag: str) -> str | None:
    return command[command.index(flag) + 1] if flag in command else None


def test_mock_orchestrator_launches_dynamo_tokenspeed_workers(tmp_path: Path) -> None:
    """Multi-node prefill, two decode workers on one node: allocator ports, rendezvous and P/D flags."""
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(_recipe()))
    launches: list[dict[str, Any]] = []

    exit_code = run_mock_sweep(
        config_path=config_path,
        output_dir=tmp_path / "outputs" / "4242",
        job_id="4242",
        options=MockOptions(
            child_duration_s=0.05,
            phase_pause_s=0.01,
            nodelist=("node-01", "node-02", "node-03"),
            on_srun=launches.append,
        ),
    )

    assert exit_code == 0
    workers = {
        launch["step_name"]: shlex.split(" ".join(launch["command"]))
        for launch in launches
        if "dynamo.tokenspeed" in launch["command"]
    }
    assert sorted(workers) == ["decode_0_node-03", "decode_1_node-03", "prefill_0_node-01", "prefill_0_node-02"]

    leader, follower = workers["prefill_0_node-01"], workers["prefill_0_node-02"]
    assert leader[:3] == ["python3", "-m", "dynamo.tokenspeed"]
    assert _flag(leader, "--model") == "Qwen/Qwen3-0.6B"
    assert _flag(leader, "--disaggregation-mode") == "prefill"
    assert _flag(leader, "--disaggregation-bootstrap-port") is not None
    assert _flag(leader, "--tensor-parallel-size") == "16"
    # Both nodes of the prefill worker join the leader's rendezvous.
    assert _flag(leader, "--dist-init-addr") == _flag(follower, "--dist-init-addr")
    assert [(_flag(c, "--nnodes"), _flag(c, "--node-rank")) for c in (leader, follower)] == [("2", "0"), ("2", "1")]

    decode_0, decode_1 = workers["decode_0_node-03"], workers["decode_1_node-03"]
    assert _flag(decode_0, "--disaggregation-mode") == "decode"
    assert _flag(decode_0, "--disaggregation-bootstrap-port") is None
    assert _flag(decode_0, "--served-model-name") == "qwen3"
    # Two engines on one node get their own --port scan base and rendezvous block.
    assert _flag(decode_0, "--port") != _flag(decode_1, "--port")
    assert _flag(decode_0, "--dist-init-addr") != _flag(decode_1, "--dist-init-addr")


def test_smg_runs_grpc_engines_on_their_tokenspeed_port(tmp_path: Path) -> None:
    """Behind SMG each worker is the gRPC engine, advertised as grpc:// on its --port."""
    recipe = _recipe(frontend={"type": "smg", "enable_multiple_frontends": False}, dynamo={"install": False})
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    launches: list[dict[str, Any]] = []

    exit_code = run_mock_sweep(
        config_path=config_path,
        output_dir=tmp_path / "outputs" / "4243",
        job_id="4243",
        options=MockOptions(
            child_duration_s=0.05,
            phase_pause_s=0.01,
            nodelist=("node-01", "node-02", "node-03"),
            on_srun=launches.append,
        ),
    )

    assert exit_code == 0
    commands = [shlex.split(" ".join(launch["command"])) for launch in launches]
    workers = {
        launch["step_name"]: shlex.split(" ".join(launch["command"]))
        for launch in launches
        if "smg_grpc_servicer.tokenspeed" in launch["command"]
    }
    assert sorted(workers) == ["decode_0_node-03", "decode_1_node-03", "prefill_0_node-01", "prefill_0_node-02"]
    assert all(cmd[:3] == ["python3", "-m", "smg_grpc_servicer.tokenspeed"] for cmd in workers.values())

    router = next(cmd for cmd in commands if cmd[:2] == ["smg", "launch"])
    prefill = workers["prefill_0_node-01"]
    prefill_url = f"grpc://{_flag(prefill, '--host')}:{_flag(prefill, '--port')}"
    assert router[router.index("--prefill") + 1 : router.index("--prefill") + 3] == [
        prefill_url,
        _flag(prefill, "--disaggregation-bootstrap-port"),
    ]
    decode_urls = [router[i + 1] for i, token in enumerate(router) if token == "--decode"]
    assert decode_urls == [
        f"grpc://{_flag(workers[step], '--host')}:{_flag(workers[step], '--port')}"
        for step in ("decode_0_node-03", "decode_1_node-03")
    ]
    # The prefill follower node joins the leader's engine and is not routed.
    assert router.count("--prefill") == 1


def test_mooncake_master_service_points_the_l3_store_at_it(tmp_path: Path) -> None:
    """A mooncake-master service gives every worker the MOONCAKE_* environment of its L3 store."""
    recipe = _recipe(
        roles={"agg": {"nodes": 1, "workers": 2, "gpus": 4, "args": {"kvstore-storage-backend": "mooncake"}}},
        services=[{"name": "mooncake-master", "type": "mooncake-master"}],
    )
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    launches: list[dict[str, Any]] = []

    exit_code = run_mock_sweep(
        config_path=config_path,
        output_dir=tmp_path / "outputs" / "4244",
        job_id="4244",
        options=MockOptions(child_duration_s=0.05, phase_pause_s=0.01, nodelist=("node-01",), on_srun=launches.append),
    )

    assert exit_code == 0
    assert any(launch["command"][0] == "mooncake_master" for launch in launches)
    workers = [launch for launch in launches if "dynamo.tokenspeed" in launch["command"]]
    assert len(workers) == 2
    for worker in workers:
        env = worker["env_to_set"]
        assert env["MOONCAKE_MASTER"].endswith(":8700")
        assert env["MOONCAKE_TE_META_DATA_SERVER"].endswith(":8701/metadata")
        assert _flag(shlex.split(" ".join(worker["command"])), "--kvstore-storage-backend") == "mooncake"
