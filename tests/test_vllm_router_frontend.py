# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-signal tests for official vLLM Router orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import VLLMBackend
from srtctl.cli.mixins.benchmark_stage import _get_health_expectations
from srtctl.core.schema import FrontendConfig, PlacementConfig, ResourceConfig, RestartPolicy, RoleConfig, SrtConfig
from srtctl.core.topology import Endpoint, NodePortAllocator, Process
from srtctl.frontends import VLLMRouterFrontend, get_frontend
from srtctl.frontends.static_router import RouterWorker, StaticRouterFrontend
from srtctl.frontends.vllm_router import node_local_data_parallel_size, routed_process_dp_size
from srtctl.ports import VLLM_DISCOVERY_PORT


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        model_path=Path("/model"),
        is_hf_model=False,
        worker_model_arg="/model",
        frontend_port=8000,
        network_interface="ib0",
        request_plane="nats",
    )


def test_get_frontend_exposes_official_vllm_router() -> None:
    assert isinstance(get_frontend("vllm-router"), VLLMRouterFrontend)


@pytest.mark.parametrize("frontend_type", ["sglang-router", "vllm-router"])
def test_static_router_aggregate_command_advertises_all_bases(frontend_type: str) -> None:
    frontend = get_frontend(frontend_type)
    command = frontend.build_router_command(
        [
            RouterWorker("agg", "http://10.0.0.1:6100"),
            RouterWorker("agg", "http://10.0.0.2:6100"),
        ],
        "0.0.0.0",
        8000,
        VLLMBackend(),
    )

    assert command[command.index("--worker-urls") + 1 : -4] == [
        "http://10.0.0.1:6100",
        "http://10.0.0.2:6100",
    ]
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8000"]


def test_vllm_router_pd_command_uses_nixl_bootstrap_ports() -> None:
    frontend = VLLMRouterFrontend()
    command = frontend.build_router_command(
        [
            RouterWorker("prefill", "http://10.0.0.1:6100", 5400),
            RouterWorker("decode", "http://10.0.0.2:6100"),
        ],
        "0.0.0.0",
        8000,
        VLLMBackend(),
    )

    assert command[:2] == ["vllm-router", "--vllm-pd-disaggregation"]
    assert command[command.index("--prefill") + 1 : command.index("--decode")] == [
        "http://10.0.0.1:6100",
        "5400",
    ]


def test_discovery_connector_router_registers_workers_instead_of_listing_them() -> None:
    """With MoRI-IO the Router runs in discovery mode: connector and ZMQ address flags, no worker URLs."""
    backend = VLLMBackend(connector="moriio")
    command = VLLMRouterFrontend().build_router_command(
        [
            RouterWorker("prefill", "http://10.0.0.1:6100"),
            RouterWorker("decode", "http://10.0.0.2:6100"),
        ],
        "0.0.0.0",
        8000,
        backend,
    )

    assert command[:2] == ["vllm-router", "--vllm-pd-disaggregation"]
    assert "--prefill" not in command and "--decode" not in command
    assert command[command.index("--kv-connector") + 1] == "moriio"
    assert command[command.index("--vllm-discovery-address") + 1] == f"0.0.0.0:{VLLM_DISCOVERY_PORT}"
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8000"]


def test_discovered_workers_advertise_no_nixl_bootstrap_port() -> None:
    process = Process("p0", frozenset({0}), 7500, 6100, "prefill", 0, nixl_port=5400)
    frontend = VLLMRouterFrontend()

    assert frontend.worker_bootstrap_port(VLLMBackend(), process) == 5400
    assert frontend.worker_bootstrap_port(VLLMBackend(connector="moriio"), process) is None


@pytest.mark.parametrize(
    ("recipe", "message"),
    [
        ({"roles": {"decode": {"connector": "nixl"}}}, "both prefill and decode"),
        ({"frontend": {"enable_multiple_frontends": True}}, "enable_multiple_frontends: false"),
        ({"frontend": {"placement": PlacementConfig(node="last_decode")}}, "placement.node: head"),
        ({"frontend": {"type": "dynamo"}}, "requires frontend.type: vllm-router"),
    ],
)
def test_discovery_connector_recipe_rules(recipe: dict, message: str) -> None:
    """One Router on the head node, both roles on the connector, and only the vLLM Router can run it."""
    frontend = {"type": "vllm-router", "enable_multiple_frontends": False, **recipe.get("frontend", {})}
    decode_args = {"tensor-parallel-size": 1, **recipe.get("roles", {}).get("decode", {})}
    with pytest.raises(ValidationError, match=message):
        SrtConfig(
            name="moriio",
            model={"path": "model", "container": "image", "precision": "bf16"},
            resources=ResourceConfig(gpu_type="mi300x", gpus_per_node=8),
            roles={
                "prefill": RoleConfig(nodes=1, workers=1, args={"tensor-parallel-size": 1}),
                "decode": RoleConfig(nodes=1, workers=1, args=decode_args),
            },
            frontend=FrontendConfig(**frontend),
            engine=VLLMBackend(connector="moriio"),
        )


def _write_mode_args(read_mode: object) -> dict:
    extra = {} if read_mode is None else {"read_mode": read_mode}
    return {
        "tensor-parallel-size": 1,
        "kv-transfer-config": {"kv_connector": "MoRIIOConnector", "kv_connector_extra_config": extra},
    }


def _moriio_config(prefill_read_mode: object, decode_read_mode: object) -> SrtConfig:
    return SrtConfig(
        name="moriio",
        model={"path": "model", "container": "image", "precision": "bf16"},
        resources=ResourceConfig(gpu_type="mi300x", gpus_per_node=8),
        roles={
            "prefill": RoleConfig(nodes=1, workers=1, args=_write_mode_args(prefill_read_mode)),
            "decode": RoleConfig(nodes=1, workers=1, args=_write_mode_args(decode_read_mode)),
        },
        frontend=FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
        engine=VLLMBackend(connector="moriio"),
    )


@pytest.mark.parametrize(("prefill", "decode"), [(False, None), (None, False), (True, False)])
def test_discovery_read_mode_mismatch_fails_before_launch(prefill: bool | None, decode: bool | None) -> None:
    """Prefill and decode must agree on RDMA READ or WRITE; an unset role defaults to READ."""
    with pytest.raises(ValidationError, match="same read_mode"):
        _moriio_config(prefill, decode)


@pytest.mark.parametrize("value", ["false", 0])
def test_non_boolean_read_mode_fails_at_load(value: object) -> None:
    with pytest.raises(ValidationError, match="read_mode must be a JSON boolean"):
        _moriio_config(value, value)


@pytest.mark.parametrize(("prefill", "decode"), [(False, False), (None, True), (None, None)])
def test_discovery_read_mode_agreement_loads(prefill: bool | None, decode: bool | None) -> None:
    config = _moriio_config(prefill, decode)
    assert config.backend.discovery_read_mode("prefill") == config.backend.discovery_read_mode("decode")


def test_discovery_connector_recipe_loads_and_allocates_listeners() -> None:
    """The example MoRI-IO recipe passes validation and every worker gets its own handshake and notify ports."""
    config = SrtConfig.from_yaml(Path("examples/vllm/vllm-router-moriio-disagg.yaml"))
    endpoints = config.backend.allocate_endpoints(
        num_prefill=1,
        num_decode=1,
        num_agg=0,
        gpus_per_prefill=1,
        gpus_per_decode=1,
        gpus_per_agg=0,
        gpus_per_node=8,
        available_nodes=["node0", "node1"],
    )
    processes = config.backend.endpoints_to_processes(
        endpoints, port_allocator=NodePortAllocator(), frontend_type="vllm-router"
    )

    assert config.backend.discovers_workers() is True
    assert len({p.moriio_handshake_port for p in processes}) == 2
    assert len({p.moriio_notify_port for p in processes}) == 2


def test_a_discovering_router_lists_no_worker_urls_and_no_bootstrap_port() -> None:
    """A router whose workers register with it gets the P/D flag but neither --prefill nor --decode."""

    class DiscoveringRouter(VLLMRouterFrontend):
        def discovers_workers(self, backend: object) -> bool:
            return True

    workers = [RouterWorker("prefill", "http://10.0.0.1:6100", 5400), RouterWorker("decode", "http://10.0.0.2:6100")]
    command = DiscoveringRouter().build_router_command(workers, "0.0.0.0", 8000, VLLMBackend())
    process = Process("p0", frozenset({0}), 7500, 6100, "prefill", 0, nixl_port=5400)

    assert command[:2] == ["vllm-router", "--vllm-pd-disaggregation"]
    assert "--prefill" not in command and "--decode" not in command
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8000"]
    assert DiscoveringRouter().worker_bootstrap_port(VLLMBackend(), process) is None
    assert VLLMRouterFrontend().worker_bootstrap_port(VLLMBackend(), process) == 5400


def _pd_processes() -> list[Process]:
    return [
        Process("p0", frozenset({0}), 7500, 6100, "prefill", 0, nixl_port=5400, bootstrap_port=7200),
        Process("d0", frozenset({1}), 7501, 6101, "decode", 0, nixl_port=5401),
    ]


def test_mooncake_prefill_opens_its_bootstrap_server_on_the_advertised_port() -> None:
    """The Router queries each prefill's bootstrap server on the port it is given, so vLLM must listen there."""
    config = SrtConfig.from_yaml(Path("examples/vllm/vllm-router-mooncake-disagg.yaml"))
    backend = config.backend
    endpoints = backend.allocate_endpoints(
        num_prefill=1,
        num_decode=1,
        num_agg=0,
        gpus_per_prefill=1,
        gpus_per_decode=1,
        gpus_per_agg=0,
        gpus_per_node=8,
        available_nodes=["node0", "node1"],
    )
    processes = backend.endpoints_to_processes(
        endpoints, port_allocator=NodePortAllocator(), frontend_type="vllm-router"
    )
    prefill = next(process for process in processes if process.endpoint_mode == "prefill")

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        env = backend.get_process_environment(prefill)
        nixl_env = VLLMBackend().get_process_environment(prefill)

    assert env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] == str(VLLMRouterFrontend().worker_bootstrap_port(backend, prefill))
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT" not in nixl_env
    assert backend.kv_transfer_config("prefill") == '{"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"}'
    assert backend.kv_transfer_config("decode") == '{"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"}'


def test_an_explicit_mooncake_transfer_config_keeps_the_router_mode() -> None:
    """A MultiConnector's Mooncake child selects bootstrap routing and preserves CPU offload."""
    payload = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "connectors": [
                {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"},
                {"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both"},
            ]
        },
    }
    backend = VLLMBackend(connector="mooncake", roles={"prefill": RoleConfig(args={"kv-transfer-config": payload})})

    assert json.loads(backend.kv_transfer_config("prefill")) == payload
    assert backend.bootstrap_port_env("prefill") == "VLLM_MOONCAKE_BOOTSTRAP_PORT"
    assert VLLMRouterFrontend().queries_prefill_bootstrap(backend) is True


@pytest.mark.parametrize("frontend_type", ["vllm-router", "atomesh"])
def test_mooncake_cpu_offload_runs_one_tp4_prefill_and_two_tp4_decodes(
    tmp_path: Path, monkeypatch, frontend_type: str
) -> None:
    """InferenceX #5090's connector/topology contract goes through the real orchestrator without a patch."""
    from srtctl.mock import MockOptions, run_mock_sweep

    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    recipe = yaml.safe_load(Path("examples/vllm/vllm-router-mooncake-disagg.yaml").read_text())
    frontend = get_frontend(frontend_type)
    router_step = f"{frontend.process_name}_0"
    recipe["frontend"] = {"type": frontend_type, "enable_multiple_frontends": False, "container_image": "router-image"}
    recipe["model"] = {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp4"}
    recipe["observability"] = {"enabled": False, "tachometer": {"enabled": False}}
    prefill_transfer = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {"mooncake_protocol": "rdma", "num_workers": 10},
                },
                {
                    "kv_connector": "SimpleCPUOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use": 1349000000000,
                        "lazy_offload": False,
                        "cpu_lru_eviction": True,
                    },
                },
            ]
        },
    }
    decode_transfer = {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {"mooncake_protocol": "rdma", "num_workers": 10},
    }
    recipe["engine"]["set_visible_devices"] = True
    recipe["resources"] = {"gpu_type": "mi355x", "gpus_per_node": 8}
    recipe["roles"] = {
        mode: {
            "nodes": 1,
            "workers": workers,
            "gpus": 4,
            "args": {"tensor-parallel-size": 4, "kv-transfer-config": json.dumps(payload)},
        }
        for mode, workers, payload in (("prefill", 1, prefill_transfer), ("decode", 2, decode_transfer))
    }
    recipe["benchmark"] = {"type": "custom", "command": "echo mooncake-smoke"}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    calls = []
    events = []

    def launched(call: dict) -> None:
        calls.append(call)
        events.append(call.get("step_name"))

    def wait(self, workers, config, stop_event) -> None:
        assert len(workers) == 3
        StaticRouterFrontend.wait_for_workers(self, workers, config, stop_event)
        events.append("workers-healthy")

    with patch.object(type(frontend), "wait_for_workers", new=wait):
        assert (
            run_mock_sweep(
                config_path=path,
                output_dir=tmp_path / "output",
                job_id="42042",
                options=MockOptions(
                    child_duration_s=0,
                    phase_pause_s=0,
                    on_srun=launched,
                    nodelist=("n0", "n1", "n2", "n3"),
                ),
            )
            == 0
        )

    prefills = [call for call in calls if str(call.get("step_name", "")).startswith("prefill_")]
    decodes = [call for call in calls if str(call.get("step_name", "")).startswith("decode_")]
    router = next(call for call in calls if call.get("step_name") == router_step)
    assert len(prefills) == 1 and len(decodes) == 2
    assert events.index("workers-healthy") < events.index(router_step) < events.index("benchmark")
    for call, payload in [(prefills[0], prefill_transfer), *((call, decode_transfer) for call in decodes)]:
        assert events.index(call["step_name"]) < events.index("workers-healthy")
        command = call["command"]
        assert json.loads(command[command.index("--kv-transfer-config") + 1]) == payload
        assert command[command.index("--tensor-parallel-size") + 1] == "4"
    assert all("VLLM_MOONCAKE_BOOTSTRAP_PORT" not in call["env_to_set"] for call in decodes)
    command = router["command"]
    assert command[command.index("--prefill") + 2] == prefills[0]["env_to_set"]["VLLM_MOONCAKE_BOOTSTRAP_PORT"]
    assert router["container_image"] == "router-image"
    if frontend_type == "atomesh":
        assert command[command.index("--backend") + 1] == "vllm"
    else:
        assert command[command.index("--kv-connector") + 1] == "mooncake"


@pytest.mark.parametrize("decode_dp", [1, 2])
def test_multinode_mooncake_prefill_shares_one_allocated_bootstrap_port(decode_dp: int) -> None:
    """TP16 followers register at the leader's port, including the per_node DP conversion path."""
    backend = VLLMBackend(
        connector="mooncake",
        roles={
            "prefill": RoleConfig(args={"tensor-parallel-size": 16, "data-parallel-size": 1}),
            "decode": RoleConfig(args={"tensor-parallel-size": 1, "data-parallel-size": decode_dp}),
        },
    )
    endpoints = [
        Endpoint("prefill", 0, ("n0", "n1")),
        Endpoint("decode", 0, ("n2",), gpu_indices=frozenset(range(decode_dp))),
    ]
    allocator = NodePortAllocator(bases={"bootstrap": 7300})
    processes = backend.endpoints_to_processes(endpoints, port_allocator=allocator, frontend_type="vllm-router")
    prefills = [process for process in processes if process.endpoint_mode == "prefill"]
    assert len(prefills) == 2
    assert {process.bootstrap_port for process in prefills} == {7300}
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        assert {backend.get_process_environment(p)["VLLM_MOONCAKE_BOOTSTRAP_PORT"] for p in prefills} == {"7300"}
        for process in processes:
            assert process.bootstrap_port != process.nixl_port
            if process.endpoint_mode == "decode":
                assert "VLLM_MOONCAKE_BOOTSTRAP_PORT" not in backend.get_process_environment(process)
    frontend = VLLMRouterFrontend()
    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=lambda node, *_: node):
        workers = frontend.collect_workers(backend, processes)
    command = frontend.build_router_command(workers, "0.0.0.0", 8000, backend)
    start = command.index("--prefill")
    assert command[start : start + 3] == ["--prefill", "http://n0:6100", "7300"]
    assert all("n1" not in worker.url for worker in workers)


def test_colocated_mooncake_prefills_get_distinct_bootstrap_ports() -> None:
    backend = VLLMBackend(connector="mooncake")
    endpoints = [Endpoint("prefill", i, ("n0",), gpu_indices=frozenset({i})) for i in range(2)]
    processes = backend.endpoints_to_processes(endpoints, frontend_type="vllm-router")
    assert [process.bootstrap_port for process in processes] == [7200, 7201]


@pytest.mark.parametrize("as_json", [False, True])
def test_raw_mooncake_transfer_configs_select_bootstrap_routing(as_json: bool) -> None:
    roles = {}
    for mode, kv_role in (("prefill", "kv_producer"), ("decode", "kv_consumer")):
        payload = {"kv_connector": "MooncakeConnector", "kv_role": kv_role}
        roles[mode] = RoleConfig(args={"kv-transfer-config": json.dumps(payload) if as_json else payload})
    backend = VLLMBackend(roles=roles)
    assert backend.bootstrap_connector("prefill") == backend.bootstrap_connector("decode") == "mooncake"
    config = SimpleNamespace(
        frontend=SimpleNamespace(args={}),
        backend_for_role=lambda mode: backend,
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
    )
    assert VLLMRouterFrontend().get_managed_frontend_args(config, backend, _pd_processes())[-2:] == [
        "--kv-connector",
        "mooncake",
    ]


def test_transfer_overrides_can_replace_mooncake_on_both_roles() -> None:
    backend = VLLMBackend(
        connector="mooncake",
        roles={
            mode: RoleConfig(args={"kv-transfer-config": {"kv_connector": "NixlConnector"}})
            for mode in ("prefill", "decode")
        },
    )
    assert not VLLMRouterFrontend().queries_prefill_bootstrap(backend)
    assert backend.bootstrap_port_env("prefill") is None


def test_an_unknown_connector_string_fails_validation_by_name() -> None:
    """A misspelled shorthand is neither a table row nor JSON, so loading names it instead of a JSON parse error."""
    with pytest.raises(ValidationError, match="'nixlx' is neither a known connector"):
        SrtConfig(
            name="typo",
            model={"path": "model", "container": "image", "precision": "bf16"},
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
            roles={mode: RoleConfig(nodes=1, workers=1, gpus=1) for mode in ("prefill", "decode")},
            engine=VLLMBackend(connector="nixlx"),
        )


@pytest.mark.parametrize("frontend_type", ["dynamo", "smg", "llm-d"])
def test_mooncake_requires_the_vllm_router(frontend_type: str) -> None:
    with pytest.raises(ValidationError, match="requires frontend.type: vllm-router"):
        SrtConfig(
            name="mooncake",
            model={"path": "model", "container": "image", "precision": "bf16"},
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
            roles={mode: RoleConfig(nodes=1, workers=1, gpus=1) for mode in ("prefill", "decode")},
            frontend=FrontendConfig(type=frontend_type),
            engine=VLLMBackend(connector="mooncake"),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"roles": {"agg": RoleConfig(nodes=1, workers=1, gpus=1)}}, "prefill/decode topology"),
        ({"health_check": None}, "requires health_check"),
        (
            {
                "roles": {
                    "prefill": RoleConfig(nodes=1, workers=1, gpus=1, restart=RestartPolicy(policy="on-failure")),
                    "decode": RoleConfig(nodes=1, workers=1, gpus=1),
                }
            },
            "prefill restart: never",
        ),
        (
            {
                "roles": {
                    mode: RoleConfig(nodes=1, workers=1, gpus=2, args={"data-parallel-size": 2})
                    for mode in ("prefill", "decode")
                }
            },
            "prefill data-parallel-size: 1",
        ),
    ],
)
def test_mooncake_rejects_unsupported_lifecycle_and_topology(overrides: dict, message: str) -> None:
    fields = {
        "name": "mooncake",
        "model": {"path": "model", "container": "image", "precision": "bf16"},
        "resources": ResourceConfig(gpu_type="h100", gpus_per_node=8),
        "roles": {mode: RoleConfig(nodes=1, workers=1, gpus=1) for mode in ("prefill", "decode")},
        "frontend": FrontendConfig(type="vllm-router"),
        "engine": VLLMBackend(connector="mooncake"),
    }
    with pytest.raises(ValidationError, match=message):
        SrtConfig(**(fields | overrides))


@pytest.mark.parametrize(
    ("connector", "frontend_args", "expected"),
    [
        ("mooncake", {}, ["--kv-connector", "mooncake"]),
        ("mooncake", {"kv-connector": "mooncake"}, []),
        ("nixl", {}, []),
    ],
)
def test_mooncake_router_gets_its_kv_connector_once(connector: str, frontend_args: dict, expected: list[str]) -> None:
    backend = VLLMBackend(connector=connector)
    config = SimpleNamespace(
        frontend=SimpleNamespace(args=frontend_args),
        backend_for_role=lambda mode: backend,
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
    )

    managed = VLLMRouterFrontend().get_managed_frontend_args(config, backend, _pd_processes())

    assert managed == ["--worker-startup-timeout-secs", "3600", *expected]


@pytest.mark.parametrize(
    ("recipe", "message"),
    [
        ({"decode": {"connector": "nixl", "tensor-parallel-size": 1}}, "both prefill and decode"),
        ({"frontend_args": {"kv-connector": "nixl"}}, "conflicts with the workers' connector"),
        ({"frontend_args": {"kv-connector": "Mooncake"}}, "conflicts with the workers' connector"),
        (
            {"decode": {"kv-transfer-config": {"kv_connector": "NixlConnector"}, "tensor-parallel-size": 1}},
            "both prefill and decode",
        ),
    ],
)
def test_mooncake_recipe_rules(recipe: dict, message: str) -> None:
    """Both roles run the connector, and a user-set Router connector must be the same one."""
    with pytest.raises(ValidationError, match=message):
        SrtConfig(
            name="mooncake",
            model={"path": "model", "container": "image", "precision": "bf16"},
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
            roles={
                "prefill": RoleConfig(nodes=1, workers=1, args={"tensor-parallel-size": 1}),
                "decode": RoleConfig(nodes=1, workers=1, args=recipe.get("decode", {"tensor-parallel-size": 1})),
            },
            frontend=FrontendConfig(type="vllm-router", args=recipe.get("frontend_args")),
            engine=VLLMBackend(connector="mooncake"),
        )


@pytest.mark.parametrize(("connector", "waits"), [("mooncake", True), ("nixl", False)])
def test_mooncake_router_starts_only_after_workers_are_healthy(connector: str, waits: bool) -> None:
    """The Router keeps the bootstrap answer it reads at startup, so it starts after every worker's /health."""
    frontend = VLLMRouterFrontend()
    backend = VLLMBackend(connector=connector)
    runtime = SimpleNamespace(
        log_dir=Path("/logs"),
        container_image=Path("/worker.sqsh"),
        reuse_container=False,
        container_prepare_timeout_seconds=1800.0,
        container_mounts={},
        environment={},
        network_interface="ib0",
        srun_options={},
        nodes=SimpleNamespace(het_group_for=lambda _node: None),
    )
    config = SimpleNamespace(
        backend=SimpleNamespace(type="vllm"),
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
        frontend=SimpleNamespace(args={}, env={}, container_image=None),
        backend_for_role=lambda mode: backend,
    )
    topology = SimpleNamespace(frontend_nodes=["router0"], frontend_port=8000)
    events: list[object] = []

    def wait(workers: list[RouterWorker], *_args: object) -> None:
        events.append([worker.url for worker in workers])

    def start(**kwargs: object) -> MagicMock:
        events.append("start")
        return MagicMock()

    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1") as resolve,
        patch.object(frontend, "wait_for_workers", side_effect=wait),
        patch.object(frontend, "start_process", side_effect=start),
    ):
        frontend.start_frontends(topology, runtime, config, backend, _pd_processes())

    probed = [["http://10.0.0.1:6100", "http://10.0.0.1:6101"]] if waits else []
    assert events == [*probed, "start"]
    assert resolve.call_count == 2


def test_collect_workers_uses_positive_http_ports_and_configured_interface() -> None:
    frontend = VLLMRouterFrontend()
    processes = [
        SimpleNamespace(endpoint_mode="prefill", node="p0", http_port=6100, nixl_port=5400),
        SimpleNamespace(endpoint_mode="prefill", node="p1", http_port=0, nixl_port=5400),
        SimpleNamespace(endpoint_mode="decode", node="d0", http_port=6100, nixl_port=5500),
    ]

    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=["10.0.0.1", "10.0.0.2"]) as resolve:
        workers = frontend.collect_workers(VLLMBackend(), processes, "ib0")

    assert workers == [
        RouterWorker("prefill", "http://10.0.0.1:6100", 5400),
        RouterWorker("decode", "http://10.0.0.2:6100", 5500),
    ]
    assert [mock_call.args for mock_call in resolve.call_args_list] == [("p0", "ib0"), ("d0", "ib0")]


def test_dep4_expansion_and_health_counts_follow_upstream_per_node_topology() -> None:
    backend = VLLMBackend(
        roles={
            "prefill": RoleConfig(args={"data-parallel-size": 4}),
            "decode": RoleConfig(args={"data-parallel-size": 4}),
        }
    )
    processes = [
        Process("p0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=5400),
        Process("d0", frozenset(range(4)), 7501, 6100, "decode", 0, nixl_port=5500),
        Process("d1", frozenset(range(4)), 7502, 6100, "decode", 1, nixl_port=5504),
    ]
    config = SimpleNamespace(
        frontend=SimpleNamespace(type="vllm-router"),
        backend=backend,
        backend_for_role=lambda mode: backend,
        topology=SimpleNamespace(num_prefill=1, num_decode=2, num_agg=0),
    )

    assert node_local_data_parallel_size(config, processes) == 4
    assert _get_health_expectations(config, processes) == (
        4,
        8,
        "4P + 8D Router workers; logical workers: 1P + 2D",
        12,
    )


def test_multinode_dep8_routes_node_local_hybrid_pools_without_rank_reexpansion() -> None:
    """A later node owns global ranks 4..7, not another local 0..3 namespace."""
    backend = VLLMBackend(
        roles={
            "prefill": RoleConfig(args={"data-parallel-size": 8}),
            "decode": RoleConfig(args={"data-parallel-size": 8}),
        }
    )
    processes = [
        Process("p0", frozenset(range(4)), 7500, 6100, "prefill", 0, node_rank=0),
        Process("p1", frozenset(range(4)), 7501, 6100, "prefill", 0, node_rank=4),
        Process("d0", frozenset(range(4)), 7502, 6100, "decode", 0, node_rank=0),
        Process("d1", frozenset(range(4)), 7503, 6100, "decode", 0, node_rank=4),
    ]
    config = SimpleNamespace(
        frontend=SimpleNamespace(type="vllm-router"),
        backend=backend,
        backend_for_role=lambda mode: backend,
        topology=SimpleNamespace(num_prefill=1, num_decode=1, num_agg=0),
    )

    assert node_local_data_parallel_size(config, processes) == 1
    assert _get_health_expectations(config, processes) == (
        2,
        2,
        "2P + 2D Router workers; logical workers: 1P + 1D",
        4,
    )


@pytest.mark.parametrize("expansion", [1, 4])
def test_multinode_hybrid_pool_schema_matches_router_expansion(expansion: int) -> None:
    """The declared expansion must match the unexpanded node-local pool URLs."""
    data = {
        "name": "hybrid-pools",
        "model": {"path": "/model", "container": "vllm", "precision": "bf16"},
        "resources": {"gpus_per_node": 4},
        "roles": {
            "prefill": {"nodes": 2, "workers": 1, "args": {"data-parallel-size": 8}},
            "decode": {"nodes": 2, "workers": 1, "args": {"data-parallel-size": 8}},
        },
        "engine": "vllm",
        "frontend": {
            "type": "vllm-router",
            "enable_multiple_frontends": False,
            "args": {"intra-node-data-parallel-size": expansion},
        },
    }
    if expansion == 1:
        config = SrtConfig.Schema().load(data)
        assert config.frontend.args["intra-node-data-parallel-size"] == 1
    else:
        with pytest.raises(ValidationError, match="conflicts with the allocated vLLM topology"):
            SrtConfig.Schema().load(data)


def test_cross_node_model_parallel_base_is_not_dp_expanded() -> None:
    backend = VLLMBackend(roles={"agg": RoleConfig(args={"data-parallel-size": 2, "tensor-parallel-size": 8})})
    process = Process("n0", frozenset(range(4)), 7500, 6100, "agg", 0)

    assert routed_process_dp_size(backend, process) == 1


def test_managed_args_derive_dp_and_startup_timeout_without_overriding_user_policy() -> None:
    frontend = VLLMRouterFrontend()
    backend = VLLMBackend(roles={"agg": RoleConfig(args={"data-parallel-size": 4})})
    processes = [Process("n0", frozenset(range(4)), 7500, 6100, "agg", 0)]
    config = SimpleNamespace(
        frontend=SimpleNamespace(args={"policy": "consistent_hash"}),
        backend_for_role=lambda mode: backend,
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
    )

    assert frontend.get_managed_frontend_args(config, backend, processes) == [
        "--intra-node-data-parallel-size",
        "4",
        "--worker-startup-timeout-secs",
        "3600",
    ]


def test_router_launch_uses_router_image_env_setup_and_captured_log() -> None:
    frontend = VLLMRouterFrontend()
    runtime = SimpleNamespace(
        log_dir=Path("/logs"),
        container_image=Path("/worker.sqsh"),
        reuse_container=False,
        container_prepare_timeout_seconds=1800.0,
        container_mounts={"/host": "/container"},
        environment={"GLOBAL": "value", "ROUTER_LOG": "info"},
        network_interface="ib0",
        srun_options={},
        nodes=SimpleNamespace(het_group_for=lambda _node: 1),
    )
    config = SimpleNamespace(
        backend=SimpleNamespace(type="vllm"),
        health_check=SimpleNamespace(max_attempts=360, interval_seconds=10),
        frontend=SimpleNamespace(
            args={"policy": "consistent_hash"},
            env={"ROUTER_LOG": "debug"},
            container_image="docker://router:test",
        ),
        setup_script="router-deps.sh",
    )
    topology = SimpleNamespace(frontend_nodes=["router0"], frontend_port=8000)
    worker = Process("worker0", frozenset(range(4)), 7500, 6100, "agg", 0)
    backend = VLLMBackend()
    config.backend_for_role = lambda mode: backend

    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"),
        patch.object(frontend, "start_process", return_value=MagicMock()) as start,
    ):
        managed = frontend.start_frontends(topology, runtime, config, backend, [worker])

    kwargs = start.call_args.kwargs
    assert kwargs["container_image"] == "docker://router:test"
    assert kwargs["env_to_set"] == {"GLOBAL": "value", "ROUTER_LOG": "debug"}
    assert kwargs["output"] == "/logs/router0_vllm-router_0.out"
    assert "/configs/${setup_script}" in kwargs["bash_preamble"]
    assert managed[0].log_file == Path("/logs/router0_vllm-router_0.out")


def test_vllm_router_setup_preamble_is_adapter_specific() -> None:
    frontend = VLLMRouterFrontend()

    assert frontend.build_bash_preamble(SimpleNamespace()) is None
    assert frontend.build_bash_preamble(SimpleNamespace(setup_script="router deps.sh")).startswith(
        "setup_script='router deps.sh'"
    )
    assert get_frontend("sglang-router").build_bash_preamble(SimpleNamespace(setup_script="router-deps.sh")) is None


def test_schema_rejects_backend_mismatch_and_deprecated_per_gpu_dp() -> None:
    from srtctl.backends import SGLangBackend

    common = {
        "name": "router",
        "model": {"path": "model", "container": "image", "precision": "fp8"},
        "resources": ResourceConfig(gpu_type="h100", gpus_per_node=8),
        "frontend": FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
    }
    with pytest.raises(ValidationError, match="requires backend.type: vllm"):
        SrtConfig(**common, engine=SGLangBackend(), roles={"agg": RoleConfig(nodes=1, workers=1)})
    with pytest.raises(ValidationError, match="requires backend.dp_launch_mode: per_node"):
        SrtConfig(
            **common,
            engine=VLLMBackend(dp_launch_mode="per_gpu"),
            roles={"agg": RoleConfig(nodes=1, workers=1, args={"data-parallel-size": 8})},
        )


def test_vllm_router_preserves_upstream_multinode_tp_serve() -> None:
    backend = VLLMBackend(roles={"agg": RoleConfig(args={"tensor-parallel-size": 8})})
    endpoint = Endpoint("agg", 0, ("n0", "n1"), frozenset(range(4)), gpus_per_node=4)
    processes = backend.endpoints_to_processes(
        [endpoint], port_allocator=NodePortAllocator(), frontend_type="vllm-router"
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        leader = backend.build_worker_command(processes[0], processes, _runtime(), frontend_type="vllm-router")
        follower = backend.build_worker_command(processes[1], processes, _runtime(), frontend_type="vllm-router")

    assert leader[:3] == ["vllm", "serve", "/model"]
    assert leader[leader.index("--port") + 1] == "6100"
    assert leader[leader.index("--nnodes") + 1] == "2"
    assert "--headless" not in leader
    assert "--headless" in follower
    assert "--port" not in follower


def test_vllm_router_per_node_dep4_launches_one_api_per_node_pool() -> None:
    backend = VLLMBackend(roles={"agg": RoleConfig(args={"data-parallel-size": 8})})
    endpoint = Endpoint("agg", 0, ("n0", "n1"), frozenset(range(4)), gpus_per_node=4)
    processes = backend.endpoints_to_processes(
        [endpoint], port_allocator=NodePortAllocator(), frontend_type="vllm-router"
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        commands = [
            backend.build_worker_command(process, processes, _runtime(), frontend_type="vllm-router")
            for process in processes
        ]

    assert [process.http_port for process in processes] == [6100, 6100]
    assert [command[command.index("--data-parallel-start-rank") + 1] for command in commands] == ["0", "4"]
    assert all("--data-parallel-size-local" in command for command in commands)
    assert all("--data-parallel-hybrid-lb" in command for command in commands)
    assert all("--headless" not in command for command in commands)


def test_single_node_dp_preserves_native_vllm_topology() -> None:
    """A single server owns local DP ranks; srt-slurm must not split it."""
    backend = VLLMBackend(roles={"agg": RoleConfig(args={"tensor-parallel-size": 2, "data-parallel-size": 2})})
    endpoint = Endpoint("agg", 0, ("n0",), frozenset(range(4)), gpus_per_node=4)
    processes = backend.endpoints_to_processes(
        [endpoint], port_allocator=NodePortAllocator(), frontend_type="vllm-router"
    )

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(processes[0], processes, _runtime(), frontend_type="vllm-router")

    assert len(processes) == 1
    assert "--data-parallel-size-local" not in command
    assert "--data-parallel-start-rank" not in command
    assert "--data-parallel-hybrid-lb" not in command
    assert command[command.index("--data-parallel-size") + 1] == "2"


def test_dp_size_one_is_not_a_distributed_launch_mode() -> None:
    backend = VLLMBackend(roles={"agg": RoleConfig(args={"data-parallel-size": 1})})

    assert backend.find_dp_modes() == []
    assert backend.is_dp_mode("agg") is False


def test_router_validates_pcp_as_part_of_the_vllm_world_size() -> None:
    common = {
        "name": "router-pcp",
        "model": {"path": "model", "container": "image", "precision": "fp8"},
        "resources": ResourceConfig(gpu_type="h100", gpus_per_node=8),
        "frontend": FrontendConfig(type="vllm-router", enable_multiple_frontends=False),
    }

    with pytest.raises(ValidationError, match=r"DP\*TP\*PP\*PCP=2\*8=16 GPUs"):
        SrtConfig(
            **common,
            engine=VLLMBackend(),
            roles={
                "agg": RoleConfig(
                    nodes=1, workers=1, args={"data-parallel-size": 2, "prefill-context-parallel-size": 8}
                )
            },
        )


def test_vllm_router_pd_worker_uses_direct_vllm_and_nixl_connector() -> None:
    backend = VLLMBackend(roles={"prefill": RoleConfig(args={"tensor-parallel-size": 4})})
    process = Process("p0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=5400)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], _runtime(), frontend_type="vllm-router")

    assert command[:3] == ["vllm", "serve", "/model"]
    assert "--kv-transfer-config" in command
    assert "NixlConnector" in command[command.index("--kv-transfer-config") + 1]


def test_backend_readiness_requires_all_advertised_urls_in_one_pass() -> None:
    from srtctl.core.health import wait_for_http_endpoints

    ready = SimpleNamespace(status_code=200)
    unavailable = SimpleNamespace(status_code=503)
    with (
        patch("srtctl.core.health.requests.get", side_effect=[ready, unavailable, ready, ready]) as get,
        patch("srtctl.core.health.time.sleep"),
    ):
        assert wait_for_http_endpoints(["http://p/health", "http://d/health"], timeout=10)

    assert [mock_call.args[0] for mock_call in get.call_args_list] == [
        "http://p/health",
        "http://d/health",
        "http://p/health",
        "http://d/health",
    ]
