# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infera frontend (`frontend.type: infera`): etcd-registered SGLang workers behind infera-router."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import SGLangBackend, VLLMBackend
from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.core.schema import DynamoConfig, FrontendConfig, ResourceConfig, RoleConfig, SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import DynamicFrontend, InferaFrontend, get_frontend
from srtctl.services.config import ServiceConfig


def _process(node: str, mode: str, *, http_port: int = 6100, kv_events_port: int | None = None) -> Process:
    return Process(
        node=node,
        gpu_indices=frozenset({0, 1, 2, 3}),
        sys_port=7500,
        http_port=http_port,
        endpoint_mode=mode,
        endpoint_index=0,
        node_rank=0,
        bootstrap_port=30001 if mode == "prefill" else None,
        kv_events_port=kv_events_port,
    )


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.model_path = Path("/models/m")
    runtime.is_hf_model = False
    runtime.worker_model_arg = "/model"
    runtime.gpu_type = "mi355x"
    runtime.log_dir = Path("/tmp")
    runtime.network_interface = None
    runtime.infra_node_ip = "10.0.0.9"
    runtime.dynamo = DynamoConfig()
    runtime.request_plane = "tcp"
    runtime.environment = {"GLOBAL": "1"}
    return runtime


def _pd_config(
    engine,
    *,
    frontend_args: dict | None = None,
    services: list | None = None,
    prefill_kv_events: bool = True,
    environment: dict | None = None,
    frontend_env: dict | None = None,
    decode_env: dict | None = None,
) -> SrtConfig:
    return SrtConfig(
        name="infera",
        model={"path": "model", "container": "image", "precision": "fp4"},
        resources=ResourceConfig(gpu_type="mi355x", gpus_per_node=8),
        roles={
            "prefill": RoleConfig(nodes=1, workers=1, gpus=4, kv_events=prefill_kv_events),
            "decode": RoleConfig(nodes=1, workers=1, gpus=4, env=decode_env or {}),
        },
        frontend=FrontendConfig(type="infera", enable_multiple_frontends=False, args=frontend_args, env=frontend_env),
        engine=engine,
        environment=environment or {},
        **({"services": services} if services else {}),
    )


def test_registry_resolves_infera_as_a_dynamic_frontend_with_direct_workers() -> None:
    frontend = get_frontend("infera")
    assert isinstance(frontend, InferaFrontend)
    assert isinstance(frontend, DynamicFrontend)
    assert frontend.required_backend == "sglang"
    # Workers are SGLang's own OpenAI servers on their allocated port; the router fronts them.
    assert frontend.worker_launch == "direct"
    assert frontend.worker_api_port("prefill") == "allocated"
    assert frontend.health_endpoint == "/v1/workers"
    assert frontend.worker_metrics_path(SGLangBackend(), "decode") == "/metrics"


def test_schema_pairs_infera_with_sglang_only() -> None:
    assert _pd_config(SGLangBackend()).frontend.type == "infera"
    with pytest.raises(ValidationError, match="infera"):
        _pd_config(VLLMBackend())


def test_router_args_srtctl_manages_are_rejected() -> None:
    with pytest.raises(ValidationError, match="etcd-endpoint"):
        _pd_config(SGLangBackend(), frontend_args={"etcd-endpoint": "elsewhere:2379", "router-policy": "kv-aware"})


def test_an_external_etcd_is_rejected() -> None:
    external = ServiceConfig(name="etcd", type="etcd", external="http://etcd.example:2379")
    with pytest.raises(ValidationError, match="external etcd"):
        _pd_config(SGLangBackend(), services=[external])


def test_kv_aware_routing_needs_a_role_that_publishes_kv_events() -> None:
    """kv-aware is infera.server's default policy; with no publisher the router silently routes on load."""
    with pytest.raises(ValidationError, match="no role publishes"):
        _pd_config(SGLangBackend(), prefill_kv_events=False)
    with pytest.raises(ValidationError, match="no role publishes"):
        _pd_config(SGLangBackend(), prefill_kv_events=False, frontend_args={"router-policy": "kv-aware"})
    config = _pd_config(SGLangBackend(), prefill_kv_events=False, frontend_args={"router-policy": "round-robin"})
    assert config.frontend.type == "infera"


def test_router_and_workers_must_install_one_infera_release() -> None:
    with pytest.raises(ValidationError, match="environment=0.3.3, roles.decode.env=0.3.4"):
        _pd_config(SGLangBackend(), environment={"INFERA_VERSION": "0.3.3"}, decode_env={"INFERA_VERSION": "0.3.4"})
    with pytest.raises(ValidationError, match="frontend.env=0.3.4"):
        _pd_config(SGLangBackend(), environment={"INFERA_VERSION": "0.3.3"}, frontend_env={"INFERA_VERSION": "0.3.4"})
    config = _pd_config(
        SGLangBackend(), environment={"INFERA_VERSION": "0.3.3"}, decode_env={"INFERA_VERSION": "0.3.3"}
    )
    assert config.frontend.type == "infera"


def test_only_etcd_is_implied() -> None:
    config = _pd_config(SGLangBackend())
    implied = get_frontend("infera").implied_services(config)
    assert [(entry.service.name, entry.service.type) for entry in implied] == [("etcd", "etcd")]


def _registration(worker_id: str, disagg_mode: str, status: str = "active") -> dict:
    """One entry of infera-router v0.3.3's ``/v1/workers``: its ``Worker`` struct, every field serialized."""
    host, port = worker_id.split(":")
    prefill = disagg_mode == "prefill"
    return {
        "worker_id": worker_id,
        "url": f"http://{worker_id}",
        "model_name": "Qwen/Qwen3-0.6B",
        "engine": "sglang",
        "status": status,
        "disagg_mode": disagg_mode,
        "disagg_meta": {
            "protocol": "sglang-bootstrap",
            "params": {"bootstrap_addr": f"{host}:7200"} if prefill else {},
        },
        "kv_events_endpoint": f"tcp://{host}:3000" if prefill else None,
        "kv_block_size": 64 if prefill else None,
        "dp_rank": None,
        "dp_size": None,
        "request_transport": "http",
    }


def test_health_counts_active_registrations_by_role() -> None:
    frontend = InferaFrontend()
    body = {
        "workers": [
            _registration("10.0.0.1:6100", "prefill"),
            _registration("10.0.0.2:6100", "decode"),
            _registration("10.0.0.2:6101", "decode", status="draining"),
        ]
    }
    ready = frontend.parse_health(body, expected_prefill=1, expected_decode=1)
    assert ready.ready and (ready.prefill_ready, ready.decode_ready) == (1, 1)
    waiting = frontend.parse_health(body, expected_prefill=1, expected_decode=2)
    assert not waiting.ready and waiting.decode_ready == 1
    aggregate = frontend.parse_health(
        {"workers": [_registration("10.0.0.1:6100", "mixed")]}, expected_prefill=0, expected_decode=1
    )
    assert aggregate.ready


def test_health_does_not_count_an_entry_whose_fields_it_cannot_read() -> None:
    """A renamed ``status`` or ``disagg_mode`` in a later router must fail readiness, not pass it."""
    renamed_status = {k: v for k, v in _registration("10.0.0.1:6100", "prefill").items() if k != "status"}
    renamed_role = {k: v for k, v in _registration("10.0.0.2:6100", "decode").items() if k != "disagg_mode"}
    result = InferaFrontend().parse_health(
        {"workers": [renamed_status, renamed_role]}, expected_prefill=1, expected_decode=1
    )
    assert not result.ready and (result.prefill_ready, result.decode_ready) == (0, 0)


def _worker_command(backend: SGLangBackend, process: Process) -> list[str]:
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        patch("srtctl.frontends.infera.get_hostname_ip", return_value="10.0.0.1"),
    ):
        return backend.build_worker_command(process, [process], _runtime(), frontend_type="infera")


def test_workers_launch_sglang_through_the_registering_wrapper() -> None:
    backend = SGLangBackend(
        roles={"prefill": RoleConfig(args={"tensor-parallel-size": 1, "pipeline-parallel-size": 4}, kv_events=True)}
    )
    cmd = _worker_command(backend, _process("node0", "prefill", kv_events_port=5560))
    assert cmd[: cmd.index("--model-path")] == [
        "python3",
        "-m",
        "infera.engine.sglang",
        "--discovery-backend",
        "etcd",
        "--etcd-endpoint",
        "http://10.0.0.9:2379",
        "--advertise-host",
        "10.0.0.1",
        "--request-transport",
        "http",
        "--kv-event-transport",
        "zmq",
        "--kv-events",
        "off",
    ]
    # The engine flags are the native server's: allocated port, bootstrap port, allocator's KV-events port.
    assert cmd[cmd.index("--port") + 1] == "6100"
    assert cmd[cmd.index("--disaggregation-bootstrap-port") + 1] == "30001"
    assert '"endpoint": "tcp://*:5560"' in cmd[cmd.index("--kv-events-config") + 1]
    assert "--skip-server-warmup" in cmd
    assert "--request-plane" not in cmd
    assert "--no-enable-kv-events" not in cmd


def test_co_located_workers_open_readiness_on_their_own_allocated_ports() -> None:
    """Two prefill servers on one node (2P1D) must not both bind the launcher's default port."""
    backend = SGLangBackend(roles={"prefill": RoleConfig(args={"tensor-parallel-size": 1})})
    endpoints = backend.allocate_endpoints(
        num_prefill=2,
        num_decode=1,
        num_agg=0,
        gpus_per_prefill=4,
        gpus_per_decode=8,
        gpus_per_agg=0,
        gpus_per_node=8,
        available_nodes=["node0", "node1"],
    )
    processes = backend.endpoints_to_processes(endpoints)
    prefills = [p for p in processes if p.endpoint_mode == "prefill"]
    assert {p.node for p in prefills} == {"node0"}
    frontend = InferaFrontend()
    readiness = {p.endpoint_index: frontend.worker_environment(p, _runtime()) for p in prefills}
    assert readiness == {i: {"INFERA_READINESS_PORT": str(p.sys_port)} for i, p in enumerate(prefills)}
    assert len({p.sys_port for p in processes}) == len(processes)
    # The port is the worker environment's, not an `env` prefix in the command.
    assert "INFERA_READINESS_PORT" not in " ".join(_worker_command(backend, prefills[0]))


def test_the_worker_stage_adds_the_frontend_worker_environment() -> None:
    process = _process("node0", "decode")

    def defaults(frontend_type: str) -> dict[str, str]:
        stage = SimpleNamespace(
            config=SimpleNamespace(frontend=SimpleNamespace(type=frontend_type)),
            runtime=SimpleNamespace(head_node_ip="10.0.0.2"),
        )
        return WorkerStageMixin._worker_environment_defaults(stage, process)

    assert defaults("infera") == {"HEAD_NODE_IP": "10.0.0.2", "INFERA_READINESS_PORT": "7500"}
    assert defaults("sglang-router") == {"HEAD_NODE_IP": "10.0.0.2"}


def test_only_a_role_that_publishes_kv_events_gets_an_endpoint() -> None:
    """Every process gets an allocated KV-events port, so the role's kv_events, not the port, decides."""
    backend = SGLangBackend(
        roles={
            "prefill": RoleConfig(args={"tensor-parallel-size": 1}, kv_events=True),
            "decode": RoleConfig(args={"tensor-parallel-size": 1}),
        }
    )
    endpoints = backend.allocate_endpoints(
        num_prefill=1,
        num_decode=1,
        num_agg=0,
        gpus_per_prefill=4,
        gpus_per_decode=4,
        gpus_per_agg=0,
        gpus_per_node=8,
        available_nodes=["node0"],
    )
    by_mode = {p.endpoint_mode: p for p in backend.endpoints_to_processes(endpoints)}
    assert by_mode["decode"].kv_events_port is not None
    prefill = _worker_command(backend, by_mode["prefill"])
    decode = _worker_command(backend, by_mode["decode"])
    assert "--kv-events-config" in prefill and "--no-enable-kv-events" not in prefill
    assert "--no-enable-kv-events" in decode[: decode.index("--model-path")]
    assert "--kv-events-config" not in decode


def test_other_direct_frontends_keep_the_native_server() -> None:
    backend = SGLangBackend(roles={"agg": RoleConfig(args={"tensor-parallel-size": 1})})
    process = _process("node0", "agg")
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        cmd = backend.build_worker_command(process, [process], _runtime(), frontend_type="sglang-router")
    assert cmd[:3] == ["python3", "-m", "sglang.launch_server"]


def test_router_runs_with_srtctl_discovery_flags_and_the_recipe_args() -> None:
    frontend = InferaFrontend()
    runtime = _runtime()
    config = SimpleNamespace(
        frontend=SimpleNamespace(
            args={"router-policy": "kv-aware", "session-affinity": "prefill"},
            env={"RUST_LOG": "info"},
            numa_bind=False,
            container_image=None,
        ),
        setup_script="install-infera.sh",
    )
    runtime.nodes.het_group_for.return_value = None
    topology = FrontendTopology(nginx_node=None, frontend_nodes=["head"], frontend_port=8000, public_port=8000)
    with patch.object(frontend, "start_process", return_value=MagicMock()) as start:
        processes = frontend.start_frontends(topology, runtime, config, SGLangBackend(), [])

    assert [process.name for process in processes] == ["infera_router_0"]
    kwargs = start.call_args.kwargs
    assert kwargs["command"] == [
        "python3",
        "-m",
        "infera.server",
        "--router-backend",
        "rust",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--discovery-backend",
        "etcd",
        "--etcd-endpoint",
        "http://10.0.0.9:2379",
        "--request-transport",
        "http",
        "--kv-event-transport",
        "zmq",
        "--router-policy",
        "kv-aware",
        "--session-affinity",
        "prefill",
        "--router-tokenizer-path",
        "/model",
    ]
    assert kwargs["env_to_set"] == {"GLOBAL": "1", "RUST_LOG": "info"}
    assert "install-infera.sh" in kwargs["bash_preamble"]


def test_a_recipe_tokenizer_path_wins_over_the_served_checkpoint() -> None:
    config = SimpleNamespace(frontend=SimpleNamespace(args={"router-tokenizer-path": "/tok"}))
    cmd = InferaFrontend().build_router_command(8000, _runtime(), config)
    assert cmd.count("--router-tokenizer-path") == 1
    assert cmd[cmd.index("--router-tokenizer-path") + 1] == "/tok"
