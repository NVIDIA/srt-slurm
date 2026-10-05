# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llm-d router frontend (`frontend.type: llm-d`) and its P/D sidecar service."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import VLLMBackend
from srtctl.core.config import load_config
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Endpoint, Process
from srtctl.frontends import LLMDFrontend, get_frontend
from srtctl.frontends.llm_d import (
    DISCOVERY_PLUGIN,
    ENDPOINTS_FILE,
    ENVOY_CONFIG_FILE,
    EPP_CONFIG_FILE,
    epp_config_document,
    parse_ready_endpoints,
)
from srtctl.ports import LLM_D_ENVOY_ADMIN_PORT, LLM_D_EPP_GRPC_PORT, LLM_D_EPP_METRICS_PORT, WORKER_PROXY_PORT_BASE
from srtctl.services.implicit import effective_services
from srtctl.services.registry import ServiceLaunchContext, get_service_kind
from tests.launch_snapshots import EXAMPLES_DIR, render_launch_plan

DISAGG = EXAMPLES_DIR / "vllm/llm-d-disagg.yaml"
AGG = EXAMPLES_DIR / "vllm/llm-d-agg.yaml"


def _recipe(path: Path = DISAGG) -> dict:
    return yaml.safe_load(path.read_text())


def _load(recipe: dict) -> SrtConfig:
    return SrtConfig.Schema().load(recipe)


def test_registry_resolves_llm_d() -> None:
    frontend = get_frontend("llm-d")
    assert isinstance(frontend, LLMDFrontend)
    assert frontend.required_backend == "vllm"
    assert frontend.worker_api_port("decode") == "allocated"
    assert frontend.frontend_metrics_port(None) == LLM_D_EPP_METRICS_PORT


def test_pd_proxies_decode_workers_and_implies_the_sidecar() -> None:
    config = load_config(DISAGG)
    assert LLMDFrontend().proxied_worker_modes(config) == frozenset({"decode"})
    sidecar = next(entry for entry in effective_services(config) if entry.service.type == "llm-d-sidecar")
    assert sidecar.implicit
    assert sidecar.service.effective_placement == "decode"
    assert sidecar.service.effective_per == "worker"
    assert sidecar.service.effective_start == "before_workers"
    assert sidecar.service.effective_critical


def test_aggregate_job_has_no_proxy_and_no_sidecar() -> None:
    config = load_config(AGG)
    assert LLMDFrontend().proxied_worker_modes(config) == frozenset()
    assert not any(entry.service.type == "llm-d-sidecar" for entry in effective_services(config))


def test_only_routable_decode_workers_get_a_proxy_port() -> None:
    """A multi-node decode worker has one API (its leader); the follower gets neither port."""
    recipe = _recipe()
    recipe["roles"]["decode"] = {**recipe["roles"]["decode"], "nodes": 2, "gpus": 16}
    config = _load(recipe)
    endpoints = [
        Endpoint("prefill", 0, ("node0",), frozenset({0})),
        Endpoint("decode", 0, ("node1", "node2"), frozenset(range(8))),
    ]
    processes = config.worker_processes(endpoints)
    by_role = {(p.endpoint_mode, p.node_rank): p for p in processes}
    assert by_role[("prefill", 0)].proxy_port is None
    assert by_role[("decode", 0)].proxy_port == WORKER_PROXY_PORT_BASE
    assert by_role[("decode", 1)].http_port == 0
    assert by_role[("decode", 1)].proxy_port is None
    kind = get_service_kind("llm-d-sidecar")
    assert [kind.attaches_to(p) for p in processes] == [False, True, False]


def test_sidecar_command_binds_the_proxy_port_in_front_of_the_worker() -> None:
    config = load_config(DISAGG)
    process = Process("node1", frozenset({0}), 7500, 6100, "decode", 0, proxy_port=9600)
    service = next(e.service for e in effective_services(config) if e.service.type == "llm-d-sidecar")
    ctx = ServiceLaunchContext(
        runtime=MagicMock(),
        node="node1",
        node_ip="10.0.0.2",
        node_id=0,
        index=0,
        role="decode",
        process=process,
        config=config,
    )
    kind = get_service_kind("llm-d-sidecar")
    assert kind.build_command(service, ctx) == [
        "pd-sidecar",
        "--port=9600",
        "--model-server-port=6100",
        "--kv-connector=nixlv2",
        "--secure-proxy=false",
    ]
    probe = kind.readiness(service, ctx)
    assert probe is not None and probe.http is not None
    assert (probe.http.port, probe.http.path) == (9600, "/health")
    assert ctx.template_vars()["worker_proxy_port"] == "9600"
    assert ctx.template_vars()["worker_http_port"] == "6100"


def test_declared_sidecar_takes_over_with_its_own_command_and_args() -> None:
    recipe = _recipe()
    recipe["services"] = [
        {
            "name": "llm-d-sidecar",
            "type": "llm-d-sidecar",
            "command": ["/app/pd-sidecar"],
            "args": ["--enable-prefiller-sampling"],
        }
    ]
    config = _load(recipe)
    service = next(e.service for e in effective_services(config) if e.service.type == "llm-d-sidecar")
    command = get_service_kind("llm-d-sidecar").build_command(service, ServiceLaunchContext.preview())
    assert command[0] == "/app/pd-sidecar"
    assert command[-1] == "--enable-prefiller-sampling"


def test_declared_sidecar_must_sit_on_proxied_workers() -> None:
    recipe = _recipe(AGG)
    recipe["services"] = [{"name": "llm-d-sidecar", "type": "llm-d-sidecar"}]
    with pytest.raises(ValidationError, match="fronts the workers the frontend proxies"):
        _load(recipe)


def test_pd_requires_an_epp_scheduler() -> None:
    recipe = _recipe()
    del recipe["frontend"]["epp_config"]
    with pytest.raises(ValidationError, match="needs frontend.epp_config"):
        _load(recipe)


def test_recipe_cannot_configure_discovery() -> None:
    recipe = _recipe()
    recipe["frontend"]["epp_config"]["dataLayer"] = {"discovery": {"pluginRef": "mine"}}
    with pytest.raises(ValidationError, match="must not configure endpoint discovery"):
        _load(recipe)


def test_decode_connector_needs_a_sidecar_protocol() -> None:
    recipe = _recipe()
    recipe["engine"]["connector"] = "lmcache"
    with pytest.raises(ValidationError, match="no protocol for the vllm decode KV connector 'LMCacheConnectorV1'"):
        _load(recipe)


def test_explicit_kv_transfer_config_selects_the_protocol() -> None:
    """A role's own --kv-transfer-config wins over engine.connector, as on the command line."""
    recipe = _recipe()
    recipe["engine"]["connector"] = None
    recipe["roles"]["decode"]["args"]["kv-transfer-config"] = '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
    config = _load(recipe)
    assert isinstance(config.backend, VLLMBackend)
    assert config.backend.kv_connector_class("decode") == "NixlConnector"
    assert config.backend.kv_connector_class("prefill") is None


def test_one_router_replica() -> None:
    recipe = _recipe()
    recipe["frontend"]["enable_multiple_frontends"] = True
    recipe["roles"]["decode"]["nodes"] = 1
    with pytest.raises(ValidationError, match="runs one Endpoint Picker"):
        _load(recipe)


def test_epp_config_gets_srtctl_discovery() -> None:
    user = {"plugins": [{"type": "queue-scorer"}], "dataLayer": {"sources": [{"pluginRef": "m"}]}}
    original = copy.deepcopy(user)
    document = epp_config_document(user, "/logs/endpoints.yaml")
    assert user == original
    assert list(document)[:2] == ["apiVersion", "kind"]
    assert document["plugins"][-1] == {
        "name": DISCOVERY_PLUGIN,
        "type": "file-discovery",
        "parameters": {"path": "/logs/endpoints.yaml", "watchFile": False},
    }
    assert document["dataLayer"] == {"sources": [{"pluginRef": "m"}], "discovery": {"pluginRef": DISCOVERY_PLUGIN}}


def test_start_frontends_writes_the_router_files_and_starts_epp_then_envoy(tmp_path: Path) -> None:
    config = load_config(DISAGG)
    runtime = SimpleNamespace(
        network_interface=None,
        log_dir=tmp_path,
        container_image="model.sqsh",
        container_mounts={},
        environment={},
        srun_options={},
        nodes=SimpleNamespace(het_group_for=lambda node: None),
    )
    processes = [
        Process("node0", frozenset({0}), 7500, 6100, "prefill", 0),
        Process("node1", frozenset({0}), 7501, 6100, "decode", 0, proxy_port=9600),
    ]
    ips = {"node0": "10.0.0.1", "node1": "10.0.0.2"}
    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=lambda node, _: ips[node]),
        patch.object(LLMDFrontend, "wait_for_workers") as wait,
        patch.object(LLMDFrontend, "start_process", return_value=MagicMock()) as start,
    ):
        managed = LLMDFrontend().start_frontends(
            SimpleNamespace(frontend_nodes=["node0"], frontend_port=8000), runtime, config, config.backend, processes
        )

    # The gate probes the workers themselves (vLLM /health), not the sidecar, which answers at once.
    assert [w.url for w in wait.call_args.args[0]] == ["http://10.0.0.1:6100", "http://10.0.0.2:6100"]
    endpoints = yaml.safe_load((tmp_path / ENDPOINTS_FILE).read_text())["endpoints"]
    assert [(e["address"], e["port"], e["labels"]["llm-d.ai/role"]) for e in endpoints] == [
        ("10.0.0.1", "6100", "prefill"),
        ("10.0.0.2", "9600", "decode"),
    ]
    epp = yaml.safe_load((tmp_path / EPP_CONFIG_FILE).read_text())
    assert epp["dataLayer"]["discovery"] == {"pluginRef": DISCOVERY_PLUGIN}
    assert {"name": "prefill", "plugins": epp["schedulingProfiles"][0]["plugins"]} == epp["schedulingProfiles"][0]
    envoy = yaml.safe_load((tmp_path / ENVOY_CONFIG_FILE).read_text())
    assert envoy["admin"]["address"]["socket_address"]["port_value"] == LLM_D_ENVOY_ADMIN_PORT
    listener = envoy["static_resources"]["listeners"][0]["address"]["socket_address"]
    assert listener == {"address": "0.0.0.0", "port_value": 8000}
    routes = envoy["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0]["typed_config"]["route_config"]
    metrics_route = routes["virtual_hosts"][0]["routes"][0]
    assert metrics_route["match"] == {"path": "/metrics"} and metrics_route["direct_response"] == {"status": 404}
    epp_cluster = envoy["static_resources"]["clusters"][1]["load_assignment"]["endpoints"][0]["lb_endpoints"][0]
    assert epp_cluster["endpoint"]["address"]["socket_address"]["port_value"] == LLM_D_EPP_GRPC_PORT

    assert [p.name for p in managed] == ["llm-d-epp_0", "llm-d-envoy_0"]
    assert all(p.critical for p in managed)
    epp_cmd, envoy_cmd = (call.kwargs["command"] for call in start.call_args_list)
    assert epp_cmd[0] == "epp" and "--config-file=/logs/llm-d-epp-config.yaml" in epp_cmd
    assert "--secure-serving=false" in epp_cmd
    assert envoy_cmd == ["envoy", "-c", "/logs/llm-d-envoy.yaml", "--disable-hot-restart"]


def test_frontend_args_cannot_move_managed_epp_flags() -> None:
    config = SimpleNamespace(frontend=SimpleNamespace(args={"metrics_port": 1234}))
    with pytest.raises(ValueError, match="metrics-port, which srtctl manages"):
        LLMDFrontend().epp_command(config, "/logs/c.yaml")


def test_ready_endpoints_gauge_is_parsed() -> None:
    text = (
        "# HELP llm_d_epp_ready_endpoints The number of ready endpoints.\n"
        "# TYPE llm_d_epp_ready_endpoints gauge\n"
        'llm_d_epp_ready_endpoints{name="srtctl"} 2\n'
        "llm_d_epp_ready_endpoints_total 9\n"
    )
    assert parse_ready_endpoints(text) == 2
    assert parse_ready_endpoints("# nothing yet\n") is None


@pytest.mark.parametrize(("ready", "expected"), [(1, False), (2, True)])
def test_probe_needs_envoy_ready_and_every_endpoint_scraped(ready: int, expected: bool) -> None:
    def get(url: str, timeout: float) -> SimpleNamespace:
        if url.endswith(f":{LLM_D_ENVOY_ADMIN_PORT}/ready"):
            return SimpleNamespace(status_code=200, text="LIVE")
        assert url.endswith(f":{LLM_D_EPP_METRICS_PORT}/metrics")
        return SimpleNamespace(status_code=200, text=f'llm_d_epp_ready_endpoints{{name="srtctl"}} {ready}\n')

    with patch("srtctl.frontends.llm_d.requests.get", side_effect=get):
        result = LLMDFrontend().probe_ready("node0", 8000, 1, 1, None)
    assert result.ready is expected
    assert f"{ready}/2" in result.message


def test_probe_waits_for_envoy() -> None:
    with patch("srtctl.frontends.llm_d.requests.get", return_value=SimpleNamespace(status_code=503, text="")):
        assert not LLMDFrontend().probe_ready("node0", 8000, 0, 2, None).ready


def test_example_launches_sidecar_workers_epp_and_envoy() -> None:
    """Through the mock orchestrator: the sidecar starts before the workers, the router after them."""
    plan = render_launch_plan(DISAGG)
    assert "# exit_code: 0" in plan
    steps = [line[3:] for line in plan.splitlines() if line.startswith("## ")]
    assert steps[:5] == [
        "service_llm-d-sidecar_decode_0_node-02",
        "prefill_0_node-01",
        "decode_0_node-02",
        "llm-d-epp_0",
        "llm-d-envoy_0",
    ]
    sidecar = next(line.strip() for line in plan.splitlines() if line.strip().startswith("pd-sidecar"))
    decode = plan.split("## decode_0_node-02")[1].split("## ")[0]
    assert f"--port={WORKER_PROXY_PORT_BASE}" in sidecar
    assert "--model-server-port=6100" in sidecar and "--port 6100" in decode
    assert "--kv-connector=nixlv2" in sidecar


def test_agg_example_launches_epp_and_envoy() -> None:
    plan = render_launch_plan(AGG)
    assert "# exit_code: 0" in plan
    assert "## llm-d-epp_0" in plan and "## llm-d-envoy_0" in plan
    assert "pd-sidecar" not in plan


def test_engine_must_be_vllm() -> None:
    recipe = _recipe(AGG)
    recipe["engine"] = "sglang"
    recipe["roles"]["agg"]["args"] = {"served-model-name": "m"}
    with pytest.raises(ValidationError, match="requires backend"):
        _load(recipe)
