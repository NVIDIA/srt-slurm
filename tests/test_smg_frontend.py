# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shepherd Model Gateway frontend (`frontend.type: smg`)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import SGLangBackend, TRTLLMBackend, VLLMBackend
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


@pytest.mark.parametrize("engine", [SGLangBackend(), VLLMBackend(), TRTLLMBackend()])
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


def test_grpc_workers_are_advertised_as_grpc_urls() -> None:
    """A backend whose mode serves gRPC (vLLM ``grpc: true``) is advertised as ``grpc://``."""
    from srtctl.core.schema import RoleConfig

    backend = VLLMBackend(roles={"prefill": RoleConfig(args={"grpc": True}), "decode": RoleConfig(args={"grpc": True})})
    processes = [
        Process("node0", frozenset({0}), 7500, 6100, "prefill", 0),
        Process("node1", frozenset({0}), 7501, 6100, "decode", 0),
    ]
    frontend = SMGFrontend()
    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=["10.0.0.1", "10.0.0.2"]):
        workers = frontend.collect_workers(backend, processes)
    command = frontend.build_router_command(workers, "0.0.0.0", 8000, backend)
    assert command[:7] == [
        "smg",
        "launch",
        "--pd-disaggregation",
        "--prefill",
        "grpc://10.0.0.1:6100",
        "--decode",
        "grpc://10.0.0.2:6100",
    ]


def test_grpc_workers_are_probed_on_their_port_before_smg_starts(tmp_path: Path) -> None:
    """A gRPC worker has no HTTP /health: SMG starts once its port accepts connections."""
    from srtctl.core.schema import RoleConfig

    backend = VLLMBackend(roles={"agg": RoleConfig(args={"grpc": True})})
    runtime = SimpleNamespace(
        network_interface=None,
        log_dir=tmp_path,
        container_image="model.sqsh",
        container_mounts={},
        environment={},
        srun_options={},
        nodes=SimpleNamespace(het_group_for=lambda node: None),
    )
    config = _config()
    config.health_check = SimpleNamespace(max_attempts=3, interval_seconds=1)
    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"),
        patch("srtctl.frontends.static_router.wait_for_port", return_value=True) as port_probe,
        patch.object(SMGFrontend, "start_process", return_value=MagicMock()) as start,
    ):
        SMGFrontend().start_frontends(
            SimpleNamespace(frontend_nodes=["node0"], frontend_port=8000),
            runtime,
            config,
            backend,
            [Process("node1", frozenset({0}), 7500, 6100, "agg", 0)],
        )

    assert port_probe.call_args.args == ("10.0.0.1", 6100)
    assert start.call_args.kwargs["command"][:4] == ["smg", "launch", "--worker-urls", "grpc://10.0.0.1:6100"]


def test_pd_command_advertises_prefill_bootstrap_port() -> None:
    frontend = SMGFrontend()
    processes = [
        Process("node0", frozenset({0}), 7500, 6100, "prefill", 0, bootstrap_port=7200),
        Process("node1", frozenset({0}), 7501, 6100, "decode", 0),
    ]
    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=["10.0.0.1", "10.0.0.2"]):
        workers = frontend.collect_workers(SGLangBackend(), processes)
    command = frontend.build_router_command(workers, "0.0.0.0", 8000, SGLangBackend())
    command.extend(frontend.get_managed_frontend_args(_config(), SGLangBackend(), processes))

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
        SMGFrontend().get_managed_frontend_args(_config({key: 31000}), VLLMBackend(), [])


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
            VLLMBackend(),
            processes,
        )

    assert [process.name for process in managed] == ["smg_0"]
    kwargs = start.call_args.kwargs
    assert kwargs["container_image"] == "smg"
    assert kwargs["step_name"] == "smg_0"
    assert kwargs["command"][:4] == ["smg", "launch", "--worker-urls", "http://10.0.0.1:6100"]
    assert kwargs["command"][-2:] == ["--policy", "cache_aware"]


def test_setup_script_runs_in_the_smg_container() -> None:
    """A recipe can install SMG into the model image (``pip install smg``) with its setup script."""
    frontend = SMGFrontend()
    assert frontend.build_bash_preamble(SimpleNamespace(setup_script=None)) is None
    preamble = frontend.build_bash_preamble(SimpleNamespace(setup_script="smg-1.11.0.sh"))
    assert preamble is not None
    assert preamble.startswith("setup_script=smg-1.11.0.sh && ")
    assert 'bash "${script_path}"' in preamble


@pytest.mark.parametrize(
    "recipe", ["vllm/smg-agg.yaml", "vllm/smg-dep16.yaml", "vllm/smg-disagg-grpc.yaml", "sglang/smg-disagg.yaml"]
)
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


def test_multi_node_vllm_dp_is_one_smg_worker() -> None:
    """vLLM balances its 16 DP ranks itself; SMG gets the leader's API and nothing from the headless node."""
    plan = render_launch_plan(EXAMPLES_DIR / "vllm/smg-dep16.yaml")
    serves = [line.strip() for line in plan.splitlines() if line.strip().startswith("vllm serve")]
    assert len(serves) == 2
    assert "--headless" in serves[1] and "--port" not in serves[1]
    launch = next(line.strip() for line in plan.splitlines() if line.strip().startswith("smg launch"))
    port = serves[0].split("--port ")[1].split()[0]
    assert launch.startswith(f"smg launch --worker-urls http://127.0.0.1:{port} --host")


def test_vllm_pd_runs_grpc_workers_with_nixl() -> None:
    """vLLM P/D: both roles serve gRPC with NixlConnector, and SMG pairs them by grpc:// URL."""
    plan = render_launch_plan(EXAMPLES_DIR / "vllm/smg-disagg-grpc.yaml")
    serves = [line.strip() for line in plan.splitlines() if line.strip().startswith("vllm serve")]
    assert len(serves) == 2
    assert all("--grpc" in serve and '"kv_connector": "NixlConnector"' in serve for serve in serves)
    launch = next(line.strip() for line in plan.splitlines() if line.strip().startswith("smg launch"))
    assert launch.startswith("smg launch --pd-disaggregation --prefill grpc://")
    assert " --decode grpc://" in launch


@pytest.mark.parametrize(
    ("engine", "path"),
    [(SGLangBackend(), "/metrics"), (VLLMBackend(), "/metrics"), (TRTLLMBackend(), "/prometheus/metrics")],
)
def test_workers_are_scraped_on_their_engine_metrics_route(engine, path: str) -> None:
    """A routed worker is the engine's own server; trtllm-serve's /metrics is JSON iteration stats."""
    assert SMGFrontend().worker_metrics_path(engine, "agg") == path
    assert SMGFrontend.metrics_path == "/metrics"


def test_grpc_workers_are_not_scraped() -> None:
    """vLLM's gRPC server starts no HTTP listener, so a gRPC role has no worker metrics path."""
    backend = load_config(EXAMPLES_DIR / "vllm/smg-disagg-grpc.yaml").backend
    assert SMGFrontend().worker_metrics_path(backend, "prefill") is None
    assert SMGFrontend().worker_metrics_path(backend, "decode") is None


def test_trtllm_example_launches_smg_in_front_of_trtllm_serve() -> None:
    """Each trtllm-serve worker binds its allocated port; SMG owns the public one."""
    path = EXAMPLES_DIR / "trtllm/smg-agg.yaml"
    assert load_config(path).frontend.type == "smg"
    plan = render_launch_plan(path)

    assert "# exit_code: 0" in plan
    workers = [line.strip() for line in plan.splitlines() if "trtllm-serve /model" in line]
    assert len(workers) == 2
    assert all("--port 8000" not in line for line in workers)
    launch = next(line.strip() for line in plan.splitlines() if line.strip().startswith("smg launch"))
    assert launch.startswith("smg launch --worker-urls http://127.0.0.1:6100 http://127.0.0.1:6132 ")


def _pd_config(prefill_engine, decode_engine=None) -> SrtConfig:
    roles = {
        "prefill": RoleConfig(nodes=1, workers=1, gpus=1, engine=prefill_engine),
        "decode": RoleConfig(nodes=1, workers=1, gpus=1, engine=decode_engine or prefill_engine),
    }
    return SrtConfig(
        name="smg",
        model={"path": "model", "container": "image", "precision": "bf16"},
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
        roles=roles,
        frontend=FrontendConfig(type="smg", enable_multiple_frontends=False),
    )


@pytest.mark.parametrize(
    ("prefill", "decode", "match"),
    [
        (VLLMBackend(), None, "only over gRPC"),
        (TRTLLMBackend(), None, "does not support TRT-LLM"),
        (SGLangBackend(), VLLMBackend(), "same engine in both roles"),
    ],
    ids=["vllm-over-http", "trtllm", "mixed-engines"],
)
def test_unsupported_pd_layouts_are_rejected_at_load(prefill, decode, match) -> None:
    with pytest.raises(ValidationError, match=match):
        _pd_config(prefill, decode)


def test_supported_pd_layouts_load() -> None:
    assert _pd_config(SGLangBackend()).frontend.type == "smg"
    for recipe in ("sglang/smg-disagg.yaml", "vllm/smg-disagg-grpc.yaml"):
        assert load_config(EXAMPLES_DIR / recipe).frontend.type == "smg"
