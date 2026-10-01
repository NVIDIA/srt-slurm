# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shepherd Model Gateway frontend (`frontend.type: smg`)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends import SGLangProtocol, TRTLLMProtocol, VLLMProtocol
from srtctl.core.config import load_config
from srtctl.core.health import check_static_router_health
from srtctl.core.schema import FrontendConfig, ResourceConfig, RoleConfig, SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import SMGFrontend, get_frontend
from srtctl.ports import SMG_METRICS_PORT
from tests.launch_snapshots import EXAMPLES_DIR, render_launch_plan


def _config(args: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        frontend=SimpleNamespace(type="smg", args=args, env=None, numa_bind=False, container_image="smg"),
        health_check=None,
    )


def test_registry_resolves_smg_for_every_backend() -> None:
    frontend = get_frontend("smg")
    assert isinstance(frontend, SMGFrontend)
    assert frontend.required_backend is None
    assert frontend.worker_launch == "direct"
    assert frontend.worker_api_port("agg") == "allocated"
    assert frontend.frontend_metrics_port(None) == SMG_METRICS_PORT


@pytest.mark.parametrize("engine", [SGLangProtocol(), VLLMProtocol(), TRTLLMProtocol()])
def test_schema_pairs_smg_with_any_backend(engine) -> None:
    config = SrtConfig(
        name="smg",
        model={"path": "model", "container": "image", "precision": "bf16"},
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
        roles={"agg": RoleConfig(nodes=1, workers=2, gpus=1)},
        frontend=FrontendConfig(type="smg", enable_multiple_frontends=False),
        engine=engine,
    )
    assert config.frontend.type == "smg"


def test_pd_command_advertises_prefill_bootstrap_port() -> None:
    frontend = SMGFrontend()
    processes = [
        Process("node0", frozenset({0}), 7500, 6100, "prefill", 0, bootstrap_port=7200),
        Process("node1", frozenset({0}), 7501, 6100, "decode", 0),
    ]
    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=["10.0.0.1", "10.0.0.2"]):
        workers = frontend.collect_workers(SGLangProtocol(), processes)
    command = frontend.build_router_command(workers, "0.0.0.0", 8000, SGLangProtocol())
    command.extend(frontend.get_managed_frontend_args(_config(), SGLangProtocol(), processes))

    assert command == [
        "smg",
        "launch",
        "--pd-disaggregation",
        "--prefill",
        "http://10.0.0.1:6100",
        "7200",
        "--decode",
        "http://10.0.0.2:6100",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--prometheus-port",
        str(SMG_METRICS_PORT),
    ]


@pytest.mark.parametrize("key", ["prometheus-port", "prometheus_port"])
def test_recipe_cannot_move_the_metrics_listener(key: str) -> None:
    with pytest.raises(ValueError, match="managed by srtctl for smg"):
        SMGFrontend().get_managed_frontend_args(_config({key: 31000}), VLLMProtocol(), [])


def test_readiness_counts_the_workers_registry() -> None:
    """SMG's GET /workers carries stats.{prefill,decode,regular}_count; aggregate workers are regular."""
    frontend = SMGFrontend()
    assert frontend.health_endpoint == "/workers"
    response = {"workers": [], "total": 2, "stats": {"prefill_count": 0, "decode_count": 0, "regular_count": 2}}
    assert frontend.parse_health(response, 0, 2) == check_static_router_health(response, 0, 2)
    assert frontend.parse_health(response, 0, 2).ready


def test_start_frontends_launches_smg_in_its_own_image(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        network_interface=None,
        log_dir=tmp_path,
        container_image="model.sqsh",
        container_mounts={},
        environment={},
        srun_options={},
        nodes=SimpleNamespace(het_group_for=lambda node: None),
    )
    processes = [Process("node1", frozenset({0}), 7500, 6100, "agg", 0)]
    frontend = SMGFrontend()
    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"),
        patch.object(SMGFrontend, "start_process", return_value=MagicMock()) as start,
    ):
        managed = frontend.start_frontends(
            SimpleNamespace(frontend_nodes=["node0"], frontend_port=8000),
            runtime,
            _config({"policy": "cache_aware"}),
            VLLMProtocol(),
            processes,
        )

    assert [process.name for process in managed] == ["smg_0"]
    kwargs = start.call_args.kwargs
    assert kwargs["container_image"] == "smg"
    assert kwargs["step_name"] == "smg_0"
    assert kwargs["command"][:4] == ["smg", "launch", "--worker-urls", "http://10.0.0.1:6100"]
    assert kwargs["command"][-2:] == ["--policy", "cache_aware"]


@pytest.mark.parametrize("recipe", ["vllm/smg-agg.yaml", "sglang/smg-disagg.yaml"])
def test_examples_launch_smg_through_the_orchestrator(recipe: str) -> None:
    """The mock orchestrator starts the workers, then one SMG router fronting them."""
    path = EXAMPLES_DIR / recipe
    assert load_config(path).frontend.type == "smg"
    plan = render_launch_plan(path)

    assert "# exit_code: 0" in plan
    assert "## smg_0" in plan
    launch = next(line.strip() for line in plan.splitlines() if line.strip().startswith("smg launch"))
    expected = "--pd-disaggregation --prefill" if "disagg" in recipe else "--worker-urls"
    assert expected in launch
    assert f"--prometheus-port {SMG_METRICS_PORT}" in launch
