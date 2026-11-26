# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang's Rust router (`frontend.type: sgl-router`)."""

from types import SimpleNamespace

import pytest
from marshmallow import ValidationError

from srtctl.backends import SGLangBackend, VLLMBackend
from srtctl.core.config import load_config
from srtctl.core.schema import FrontendConfig, ResourceConfig, RoleConfig, SrtConfig
from srtctl.frontends import SGLRouterFrontend, get_frontend
from srtctl.frontends.static_router import RouterWorker
from tests.launch_snapshots import EXAMPLES_DIR, render_launch_plan

RECIPE = EXAMPLES_DIR / "sglang/sgl-router-disagg.yaml"


def _worker(mode: str, *, registered: bool = True, healthy: bool = True, models: tuple[str, ...] = ("m",)) -> dict:
    return {
        "id": f"http://{mode}",
        "mode": "plain",  # the seeded mode; runtime.mode is what the engine disclosed
        "runtime": {"registered": registered, "healthy": healthy, "resolved_model_ids": list(models), "mode": mode},
    }


def _srt_config(engine, roles) -> SrtConfig:
    return SrtConfig(
        name="sgl-router",
        model={"path": "model", "container": "image", "precision": "bf16"},
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
        roles=roles,
        frontend=FrontendConfig(type="sgl-router", enable_multiple_frontends=False),
        engine=engine,
    )


def test_registry_resolves_a_static_router_for_sglang() -> None:
    frontend = get_frontend("sgl-router")
    assert isinstance(frontend, SGLRouterFrontend)
    assert frontend.required_backend == "sglang"
    assert frontend.worker_launch == "direct"
    assert frontend.worker_api_port("prefill") == "allocated"
    assert frontend.wait_for_workers_before_start
    # Prometheus is served on the routing port.
    assert frontend.frontend_metrics_port(None) is None


def test_command_is_one_flat_url_list_without_gateway_pd_flags() -> None:
    workers = [RouterWorker("prefill", "http://a:6100", 7200), RouterWorker("decode", "http://b:6100")]
    command = SGLRouterFrontend().build_router_command(workers, "0.0.0.0", 8000, SGLangBackend())
    assert command == [
        "sgl-router",
        "--worker-urls",
        "http://a:6100",
        "http://b:6100",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    with pytest.raises(ValueError, match="no routable workers"):
        SGLRouterFrontend().build_router_command([], "0.0.0.0", 8000, SGLangBackend())


@pytest.mark.parametrize(
    ("hf", "model", "tokenizer"),
    [(False, "/model", "/model/tokenizer.json"), (True, "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B")],
)
def test_model_id_and_tokenizer_are_managed_unless_the_recipe_sets_them(hf: bool, model: str, tokenizer: str) -> None:
    frontend = SGLRouterFrontend()
    runtime = SimpleNamespace(shared_model_arg=model, is_hf_model=hf)
    config = SimpleNamespace(frontend=SimpleNamespace(args={"chat-routing": "reorg"}), served_model_name="served")
    assert frontend.get_managed_frontend_args(config, None, [], runtime) == [
        "--model-id",
        "served",
        "--tokenizer-path",
        tokenizer,
    ]
    config.frontend.args = {"model_id": "other", "no-tokenizer": True}
    assert frontend.get_managed_frontend_args(config, None, [], runtime) == []


def test_readiness_counts_registered_healthy_workers_by_disclosed_role() -> None:
    frontend = SGLRouterFrontend()
    ready = frontend.parse_health([_worker("prefill"), _worker("decode"), _worker("decode")], 1, 2)
    assert ready.ready
    # A worker introspected before its engine answered is registered as plain with no model.
    pending = [_worker("prefill"), _worker("decode"), _worker("plain", models=())]
    assert not frontend.parse_health(pending, 1, 2).ready
    assert not frontend.parse_health([_worker("prefill"), _worker("decode", healthy=False)], 1, 1).ready
    assert frontend.parse_health([_worker("plain"), _worker("plain")], 0, 2).ready
    assert not frontend.parse_health([], 0, 1).ready


def test_launch_plan_starts_the_router_after_the_workers() -> None:
    assert load_config(RECIPE).frontend.type == "sgl-router"
    plan = render_launch_plan(RECIPE)
    assert "# exit_code: 0" in plan
    launch = next(line.strip() for line in plan.splitlines() if line.strip().startswith("sgl-router "))
    assert launch.startswith("sgl-router --worker-urls http://")
    assert "--pd-disaggregation" not in launch and "--prefill" not in launch
    assert "--model-id Qwen/Qwen3-0.6B --tokenizer-path /model/tokenizer.json --chat-routing reorg" in launch
    serves = [line for line in plan.splitlines() if "sglang.launch_server" in line]
    assert len(serves) == 2 and "--disaggregation-bootstrap-port" in serves[0]


def test_preamble_exposes_configs_dir_and_optional_setup_script() -> None:
    frontend = SGLRouterFrontend()
    assert frontend.build_bash_preamble(SimpleNamespace(setup_script=None)) == 'export PATH="$PATH:/configs"'
    with_setup = frontend.build_bash_preamble(SimpleNamespace(setup_script="install.sh"))
    assert with_setup is not None
    assert with_setup.startswith('export PATH="$PATH:/configs" && ')
    assert "install.sh" in with_setup


def test_only_sglang_http_workers_are_accepted() -> None:
    with pytest.raises(ValidationError, match="sgl-router"):
        _srt_config(VLLMBackend(), {"agg": RoleConfig(nodes=1, workers=1, gpus=1)})
    with pytest.raises(ValidationError, match="needs HTTP workers"):
        _srt_config(SGLangBackend(), {"agg": RoleConfig(nodes=1, workers=1, gpus=1, args={"grpc-mode": True})})
