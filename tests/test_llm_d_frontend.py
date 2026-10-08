# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llm-d router frontend (`frontend.type: llm-d`) and its P/D sidecar service."""

from __future__ import annotations

import copy
import json
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
    with_kv_events,
)
from srtctl.ports import (
    LLM_D_ENVOY_ADMIN_PORT,
    LLM_D_EPP_GRPC_PORT,
    LLM_D_EPP_KV_EVENTS_PORT,
    LLM_D_EPP_METRICS_PORT,
    WORKER_PROXY_PORT_BASE,
)
from srtctl.services.implicit import effective_services
from srtctl.services.llm_d_sidecar import sidecar_kv_connector
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


def test_pd_epp_config_needs_scheduling_profiles() -> None:
    recipe = _recipe()
    del recipe["frontend"]["epp_config"]["schedulingProfiles"]
    with pytest.raises(ValidationError, match="needs frontend.epp_config"):
        _load(recipe)


@pytest.mark.parametrize("path", [AGG, DISAGG])
def test_without_epp_config_srtctl_runs_the_guide_scorers(path: Path, tmp_path: Path) -> None:
    """The EPP's own default only applies to a plugin-less file; srtctl's always carries discovery."""
    recipe = _recipe(path)
    del recipe["frontend"]["epp_config"]
    config = _load(recipe)
    runtime = SimpleNamespace(
        network_interface=None,
        log_dir=tmp_path,
        container_image="model.sqsh",
        container_mounts={},
        environment={},
        srun_options={},
        nodes=SimpleNamespace(het_group_for=lambda node: None),
    )
    processes = [Process("node0", frozenset({0}), 7500, 6100, "agg", 0)]
    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"),
        patch.object(LLMDFrontend, "wait_for_workers"),
        patch.object(LLMDFrontend, "start_process", return_value=MagicMock()),
    ):
        LLMDFrontend().start_frontends(
            SimpleNamespace(frontend_nodes=["node0"], frontend_port=8000), runtime, config, config.backend, processes
        )
    epp = yaml.safe_load((tmp_path / EPP_CONFIG_FILE).read_text())
    profiles = {profile["name"]: profile["plugins"] for profile in epp["schedulingProfiles"]}
    guide = [
        {"pluginRef": "queue-scorer", "weight": 2},
        {"pluginRef": "kv-cache-utilization-scorer", "weight": 2},
        {"pluginRef": "prefix-cache-scorer", "weight": 3},
        {"pluginRef": "no-hit-lru-scorer", "weight": 2},
    ]
    if path == AGG:
        assert profiles == {"default": guide}
    else:
        assert profiles == {
            "prefill": [{"pluginRef": "prefill-filter"}, *guide],
            "decode": [{"pluginRef": "decode-filter"}, *guide[:2]],
        }
        assert {"type": "always-disagg-pd-decider"} in epp["plugins"]
    assert epp["plugins"][-1]["type"] == "file-discovery"


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
    assert config.backend.kv_connector_classes("decode") == ("NixlConnector",)
    assert config.backend.kv_connector_classes("prefill") == ()


def test_multi_connector_maps_through_its_transfer_connector() -> None:
    """NIXL next to CPU offloading (llm-d's tiered wide-EP guide) still speaks nixlv2."""
    recipe = _recipe()
    recipe["engine"]["connector"] = None
    recipe["roles"]["decode"]["args"]["kv-transfer-config"] = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "connectors": [
                {"kv_connector": "OffloadingConnector", "kv_role": "kv_both"},
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
            ]
        },
    }
    config = _load(recipe)
    assert config.backend.kv_connector_classes("decode") == ("MultiConnector", "OffloadingConnector", "NixlConnector")
    assert sidecar_kv_connector(config.backend) == "nixlv2"


@pytest.mark.parametrize("additional_frontends", [0, 1, 9])
def test_multinode_nginx_is_rejected_even_with_one_router(additional_frontends: int) -> None:
    recipe = _recipe()
    recipe["frontend"]["enable_multiple_frontends"] = True
    recipe["frontend"]["num_additional_frontends"] = additional_frontends
    recipe["roles"]["decode"]["nodes"] = 1
    with pytest.raises(ValidationError, match="Set frontend.enable_multiple_frontends: false"):
        _load(recipe)


def test_single_node_allows_multiple_frontends_setting_without_nginx() -> None:
    recipe = _recipe(AGG)
    recipe["frontend"]["enable_multiple_frontends"] = True
    config = _load(recipe)
    assert config.engine_node_count == 1


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


DP_RANKS = EXAMPLES_DIR / "vllm/llm-d-dp-ranks.yaml"


def _precise(recipe: dict) -> dict:
    """The disaggregated example with precise prefix-cache routing on its prefill worker."""
    epp = recipe["frontend"]["epp_config"]
    epp["plugins"] = [
        {"type": "token-producer"},
        {"type": "precise-prefix-cache-producer"},
        *(plugin for plugin in epp["plugins"] if plugin["type"] != "approx-prefix-cache-producer"),
    ]
    recipe["roles"]["prefill"]["kv_events"] = True
    return recipe


def test_external_lb_ranks_are_endpoints_with_their_own_ports() -> None:
    """data-parallel-external-lb: one `vllm serve` per DP rank, each an endpoint on its own port."""
    config = load_config(DP_RANKS)
    endpoints = [Endpoint("agg", 0, ("node0", "node1"), frozenset(range(8)))]
    processes = config.worker_processes(endpoints)
    assert [p.dp_rank for p in processes] == list(range(16))
    assert all(p.http_port > 0 for p in processes)
    assert len({(p.node, p.http_port) for p in processes}) == 16
    assert processes[9].engine_suffix == "_dp9"
    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=lambda node, _: node):
        document = LLMDFrontend().endpoints_document(processes, None)
    assert len(document["endpoints"]) == 16
    frontend = LLMDFrontend()
    assert frontend.health_expectations(config, processes) == (0, 16, "16 llm-d endpoints")
    assert frontend.worker_metrics_port(processes[9], MagicMock()) == processes[9].http_port


def test_external_lb_ranks_publish_kv_events_to_the_epp() -> None:
    """Every rank connects to the EPP's one socket (vLLM adds the rank back) under its own endpoint's topic."""
    plan = render_launch_plan(DP_RANKS)
    assert "# exit_code: 0" in plan
    steps = [line[3:] for line in plan.splitlines() if line.startswith("## ")]
    assert len(steps) == len(set(steps))
    assert "agg_0_node-02_dp9" in steps
    rank9 = plan.split("## agg_0_node-02_dp9")[1].split("## ")[0]
    assert "--data-parallel-rank 9 --data-parallel-address" in rank9
    assert "--data-parallel-external-lb" in rank9 and "--headless" not in rank9
    kv_events = json.loads(rank9.split("--kv-events-config '")[1].split("'")[0])
    port = rank9.split("--port ")[1].split()[0]
    assert kv_events["endpoint"] == f"tcp://127.0.0.1:{LLM_D_EPP_KV_EVENTS_PORT - 9}"
    assert kv_events["topic"] == f"kv@127.0.0.1:{port}@Qwen/Qwen3-30B-A3B"
    assert kv_events["publisher"] == "zmq" and kv_events["enable_kv_cache_events"] is True


def test_external_lb_decode_ranks_each_get_a_sidecar() -> None:
    recipe = _precise(_recipe())
    for role in ("prefill", "decode"):
        recipe["roles"][role]["gpus"] = 2
        recipe["roles"][role]["args"].update({"data-parallel-size": 2, "data-parallel-external-lb": True})
    config = _load(recipe)
    endpoints = [
        Endpoint("prefill", 0, ("node0",), frozenset({0, 1})),
        Endpoint("decode", 0, ("node1",), frozenset({0, 1})),
    ]
    processes = config.worker_processes(endpoints)
    decode = [p for p in processes if p.endpoint_mode == "decode"]
    assert [p.proxy_port for p in decode] == [WORKER_PROXY_PORT_BASE, WORKER_PROXY_PORT_BASE + 1]
    kind = get_service_kind("llm-d-sidecar")
    assert [kind.attaches_to(p) for p in processes] == [False, False, True, True]
    # The decode topic names the sidecar, the endpoint the EPP routes to.
    runtime = SimpleNamespace(network_interface=None, head_node_ip="10.0.0.9")
    with patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.2"):
        assert LLMDFrontend().kv_events_subscriber(decode[1], runtime, "m") == (
            "10.0.0.9",
            LLM_D_EPP_KV_EVENTS_PORT,
            f"kv@10.0.0.2:{WORKER_PROXY_PORT_BASE + 1}@m",
        )


def test_kv_events_need_the_precise_producer_and_vice_versa() -> None:
    recipe = _recipe()
    recipe["roles"]["prefill"]["kv_events"] = True
    with pytest.raises(ValidationError, match="add a precise-prefix-cache-producer plugin"):
        _load(recipe)
    recipe = _precise(_recipe())
    del recipe["roles"]["prefill"]["kv_events"]
    with pytest.raises(ValidationError, match="set roles.<role>.kv_events"):
        _load(recipe)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda r: r["frontend"]["epp_config"]["plugins"][1].update(
                {"parameters": {"kvEventsConfig": {"zmqEndpoint": "tcp://*:1"}}}
            ),
            "kvEventsConfig.zmqEndpoint, which srtctl manages",
        ),
        (lambda r: r["frontend"]["epp_config"]["plugins"].pop(0), "add a token-producer plugin"),
        (
            lambda r: r["frontend"]["epp_config"]["plugins"][0].update({"parameters": {"vllm": {"url": "http://x"}}}),
            "sets token-producer vllm.url",
        ),
        (
            lambda r: r["frontend"].update({"placement": {"node": "first_decode"}}),
            "publish to the EPP on the head node",
        ),
        (
            lambda r: r["roles"]["prefill"].update(
                {"gpus": 2, "args": {**r["roles"]["prefill"]["args"], "data-parallel-size": 2}}
            ),
            "set roles.prefill.args.data-parallel-external-lb: true",
        ),
    ],
)
def test_precise_prefix_routing_rules(edit, message: str) -> None:
    recipe = _precise(_recipe())
    edit(recipe)
    with pytest.raises(ValidationError, match=message):
        _load(recipe)


def test_with_kv_events_wires_the_socket_and_the_render_url() -> None:
    document = {
        "plugins": [
            {"type": "token-producer"},
            {"type": "token-producer", "name": "approx", "parameters": {"estimate": {}}},
            {"type": "precise-prefix-cache-producer", "parameters": {"kvEventsConfig": {"concurrency": 8}}},
            {"type": "queue-scorer"},
        ]
    }
    with_kv_events(document, "http://10.0.0.1:6100", "m")
    token, estimate, producer, scorer = document["plugins"]
    assert token["parameters"] == {"modelName": "m", "vllm": {"url": "http://10.0.0.1:6100"}}
    assert estimate["parameters"] == {"estimate": {}}
    assert producer["parameters"]["kvEventsConfig"] == {
        "concurrency": 8,
        "zmqEndpoint": f"tcp://*:{LLM_D_EPP_KV_EVENTS_PORT}",
        "discoverPods": False,
    }
    assert scorer == {"type": "queue-scorer"}


def test_start_frontends_points_the_token_producer_at_the_prefill_worker(tmp_path: Path) -> None:
    config = _load(_precise(_recipe()))
    runtime = SimpleNamespace(
        network_interface=None,
        log_dir=tmp_path,
        container_image="model.sqsh",
        container_mounts={},
        environment={},
        srun_options={},
        model_path=Path("/models/qwen"),
        nodes=SimpleNamespace(het_group_for=lambda node: None),
    )
    processes = [
        Process("node1", frozenset({0}), 7501, 6100, "decode", 0, proxy_port=9600),
        Process("node0", frozenset({0}), 7500, 6101, "prefill", 0),
    ]
    ips = {"node0": "10.0.0.1", "node1": "10.0.0.2"}
    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=lambda node, _: ips[node]),
        patch.object(LLMDFrontend, "wait_for_workers"),
        patch.object(LLMDFrontend, "start_process", return_value=MagicMock()),
    ):
        LLMDFrontend().start_frontends(
            SimpleNamespace(frontend_nodes=["node0"], frontend_port=8000), runtime, config, config.backend, processes
        )
    plugins = {p["type"]: p for p in yaml.safe_load((tmp_path / EPP_CONFIG_FILE).read_text())["plugins"]}
    assert plugins["token-producer"]["parameters"] == {
        "modelName": "Qwen/Qwen3-0.6B",
        "vllm": {"url": "http://10.0.0.1:6101"},
    }
    assert plugins["precise-prefix-cache-producer"]["parameters"]["kvEventsConfig"]["discoverPods"] is False
