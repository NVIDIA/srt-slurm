# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed backend: recipe validation and the dynamo.tokenspeed launch."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import tomli
import yaml
from marshmallow import ValidationError

from srtctl.backends import TokenSpeedBackend
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.health import wait_for_model
from srtctl.core.schema import RoleConfig, SrtConfig
from srtctl.core.topology import NodePortAllocator, Process
from srtctl.frontends import SMGFrontend
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


@pytest.fixture
def smg_worker() -> tuple[TokenSpeedBackend, Process, SimpleNamespace]:
    config = _load(
        _recipe(
            frontend={"type": "smg", "enable_multiple_frontends": False},
            roles={"agg": {"nodes": 1, "workers": 1}},
        )
    )
    process = config.worker_processes(
        config.allocate_worker_endpoints(["node-01"]), port_allocator=NodePortAllocator()
    )[0]
    runtime = SimpleNamespace(
        network_interface="ib0",
        frontend_node="node-02",
        head_node_ip="10.0.0.1",
        worker_model_arg="Qwen/Qwen3-0.6B",
        model_path=Path("Qwen/Qwen3-0.6B"),
    )
    return config.backend, process, runtime


def test_smg_worker_uses_frontend_listener_on_cluster_interface(smg_worker) -> None:
    backend, process, runtime = smg_worker
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.2") as resolve:
        command = backend.build_worker_command(process, [process], runtime, frontend_type="smg")

    assert _flag(command, "--data-parallel-address") == "10.0.0.2"
    resolve.assert_any_call("node-02", "ib0")


@pytest.mark.parametrize("port", ["tokenspeed_port", "dist_init_port", "tokenspeed_zmq_port"])
def test_worker_command_requires_allocated_ports(smg_worker, port: str) -> None:
    backend, process, runtime = smg_worker
    process = replace(process, **{port: None})
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        pytest.raises(ValueError, match="build the topology with TokenSpeedBackend.endpoints_to_processes"),
    ):
        backend.build_worker_command(process, [process], runtime, frontend_type="smg")


def test_smg_worker_command_requires_resolved_frontend_node(smg_worker) -> None:
    backend, process, runtime = smg_worker
    runtime.frontend_node = None
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        pytest.raises(ValueError, match="require the resolved frontend listener node"),
    ):
        backend.build_worker_command(process, [process], runtime, frontend_type="smg")


def test_smg_worker_urls_require_frontend_listener_node(smg_worker) -> None:
    backend, process, _ = smg_worker
    with pytest.raises(ValueError, match="require the frontend listener node"):
        SMGFrontend().collect_workers(backend, [process])


def test_worker_command_rejects_empty_metrics_reporters(smg_worker) -> None:
    backend, process, runtime = smg_worker
    backend = replace(backend, roles={"agg": RoleConfig(args={"metrics-reporters": []})})
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        pytest.raises(ValueError, match="metrics-reporters must contain at least one reporter"),
    ):
        backend.build_worker_command(process, [process], runtime, frontend_type="smg")


def test_tokenspeed_rejects_dynamo_sidecars_at_allocation() -> None:
    with pytest.raises(ValueError, match="TokenSpeed does not support Dynamo sidecars"):
        TokenSpeedBackend().endpoints_to_processes([], dynamo_sidecar=True)


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


@pytest.mark.parametrize("gateway_node", ["node-01", "node-02"])
def test_smg_runs_headless_engines_on_gateway_zmq_sockets(tmp_path: Path, gateway_node: str) -> None:
    """Remote and colocated P/D workers dial distinct gateway sockets and export local metrics."""
    recipe = _recipe(frontend={"type": "smg", "enable_multiple_frontends": False}, dynamo={"install": False})
    for role in recipe["roles"].values():
        role["args"]["enable-metrics"] = True
        role["args"]["metrics-reporters"] = ["prometheus", "otlp"]
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    launches: list[dict[str, Any]] = []

    # Exercise the resolved topology independently of today's head-only validation rule.
    topology = FrontendTopology(None, [gateway_node], 8000, 8000)
    with patch.object(SweepOrchestrator, "_compute_frontend_topology", return_value=topology):
        exit_code = run_mock_sweep(
            config_path=config_path,
            output_dir=tmp_path / "outputs" / "4243",
            job_id="4243",
            options=MockOptions(
                child_duration_s=0.05,
                phase_pause_s=0.01,
                nodelist=("node-01", "node-02", "node-03"),
                hostname_ips={"node-01": "10.0.0.1", "node-02": "10.0.0.2", "node-03": "10.0.0.3"},
                on_srun=launches.append,
            ),
        )

    assert exit_code == 0
    commands = [shlex.split(" ".join(launch["command"])) for launch in launches]
    workers = {
        launch["step_name"]: shlex.split(" ".join(launch["command"]))
        for launch in launches
        if "tokenspeed.cli" in launch["command"]
    }
    assert sorted(workers) == ["decode_0_node-03", "decode_1_node-03", "prefill_0_node-01", "prefill_0_node-02"]
    assert all(cmd[:5] == ["python3", "-m", "tokenspeed.cli", "serve", "--headless"] for cmd in workers.values())

    router = next(cmd for cmd in commands if cmd[:2] == ["smg", "launch"])
    assert next(launch for launch in launches if launch["command"][:2] == ["smg", "launch"])["nodelist"] == [
        gateway_node
    ]
    prefill = workers["prefill_0_node-01"]
    prefill_url = f"tcp://{_flag(prefill, '--data-parallel-address')}:{_flag(prefill, '--data-parallel-rpc-port')}"
    assert router[router.index("--prefill") + 1 : router.index("--prefill") + 3] == [prefill_url, "--decode"]
    assert _flag(prefill, "--disaggregation-bootstrap-port") is not None
    assert _flag(router, "--backend") == "tokenspeed"
    assert _flag(router, "--model") == "Qwen/Qwen3-0.6B"
    assert _flag(router, "--model-alias") == "qwen3=Qwen/Qwen3-0.6B"
    decode_urls = [router[i + 1] for i, token in enumerate(router) if token == "--decode"]
    assert decode_urls == [
        f"tcp://{_flag(workers[step], '--data-parallel-address')}:{_flag(workers[step], '--data-parallel-rpc-port')}"
        for step in ("decode_0_node-03", "decode_1_node-03")
    ]
    # The prefill follower node joins the leader's engine and is not routed.
    assert router.count("--prefill") == 1
    follower = workers["prefill_0_node-02"]
    assert _flag(follower, "--data-parallel-rpc-port") == _flag(prefill, "--data-parallel-rpc-port")
    assert _flag(follower, "--dist-init-addr") == _flag(prefill, "--dist-init-addr")
    assert _flag(follower, "--node-rank") == "1"
    assert len(set(decode_urls + [prefill_url])) == 3
    assert len({_flag(cmd, "--data-parallel-address") for cmd in workers.values()}) == 1
    assert _flag(workers["decode_0_node-03"], "--host") != _flag(prefill, "--data-parallel-address")
    metrics_urls = set()
    for cmd in workers.values():
        assert _flag(cmd, "--zmq-engine-index") == "0"
        assert _flag(cmd, "--metrics-host") == "0.0.0.0"
        assert [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--metrics-reporters"] == ["prometheus", "otlp"]
        metrics_urls.add(f"http://{_flag(cmd, '--host')}:{_flag(cmd, '--metrics-port')}/metrics")
    assert len(metrics_urls) == len(workers)
    benchmark = next(launch for launch in launches if launch["command"] == ["bash", "-lc", "echo done"])
    assert metrics_urls.issubset(set(benchmark["env_to_set"]["AIPERF_SERVER_METRICS_URLS"].split(",")))
    telemetry = tomli.loads((tmp_path / "outputs/4243/logs/tachometer_config.toml").read_text())
    assert metrics_urls.issubset({endpoint["url"] for endpoint in telemetry["endpoints"]})


def test_smg_connection_addresses_preserve_physical_worker_topology() -> None:
    config = _load(_recipe(frontend={"type": "smg", "enable_multiple_frontends": False}))
    nodes = ["node-01", "node-02", "node-03"]
    processes = config.worker_processes(config.allocate_worker_endpoints(nodes), port_allocator=NodePortAllocator())
    original = [(process.node, process.http_port) for process in processes]
    ips = {"node-01": "10.0.0.1", "node-02": "10.0.0.2", "node-03": "10.0.0.3"}
    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=lambda node, *_: ips[node]):
        workers = SMGFrontend().collect_workers(config.backend, processes, frontend_node=nodes[0])
    assert [worker.url for worker in workers] == [
        f"tcp://10.0.0.1:{process.tokenspeed_zmq_port}" for process in processes if process.is_leader
    ]
    assert len(workers) == 3
    assert all(worker.bootstrap_port is None for worker in workers)
    assert [(process.node, process.http_port) for process in processes] == original
    assert all(process.http_port == 0 for process in processes)
    assert processes[-1].node == "node-03"


def test_direct_launch_does_not_implicitly_select_zmq() -> None:
    """Launch ownership and URL transport are independent parts of the frontend contract."""
    config = _load(_recipe(roles={"agg": {"nodes": 1, "workers": 1}}))
    endpoints = config.allocate_worker_endpoints(["node-01"])
    with patch(
        "srtctl.frontends.get_frontend",
        return_value=SimpleNamespace(worker_launch="direct", worker_scheme=lambda *_: None),
    ):
        processes = config.backend.endpoints_to_processes(endpoints, port_allocator=NodePortAllocator())
    assert all(process.tokenspeed_zmq_port is None for process in processes)


@pytest.mark.parametrize("disaggregated", [False, True])
def test_smg_benchmark_waits_for_pending_and_partially_ready_workers(
    tmp_path: Path, monkeypatch, disaggregated: bool
) -> None:
    from srtctl.cli.mixins import benchmark_stage

    original_wait_for_model = benchmark_stage.wait_for_model
    roles = (
        {"prefill": {"nodes": 1, "workers": 1, "gpus": 1}, "decode": {"nodes": "colocate", "workers": 1, "gpus": 1}}
        if disaggregated
        else {"agg": {"nodes": 1, "workers": 2, "gpus": 1}}
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(
        yaml.safe_dump(
            _recipe(
                frontend={"type": "smg", "enable_multiple_frontends": False},
                roles=roles,
                health_check={"max_attempts": 5, "interval_seconds": 1},
            )
        )
    )
    launches: list[dict[str, Any]] = []
    polls: list[list[dict[str, Any]]] = []
    kinds = ["prefill", "decode"] if disaggregated else ["regular", "regular"]

    def workers_response(url: str, **kwargs: Any) -> SimpleNamespace:
        assert url.endswith("/workers")
        assert not any(launch["command"] == ["bash", "-lc", "echo done"] for launch in launches)
        ready_count = min(len(polls), len(kinds))
        workers = [{"worker_type": kind, "is_healthy": index < ready_count} for index, kind in enumerate(kinds)]
        polls.append(workers)
        body = {"workers": workers, "stats": {f"{kind}_count": kinds.count(kind) for kind in set(kinds)}}
        return SimpleNamespace(status_code=200, json=lambda: body)

    def record_launch(launch: dict[str, Any]) -> None:
        if launch.get("step_name") == "smg_0":
            # Keep fake cluster infrastructure but exercise the real readiness loop.
            # mock_infrastructure restores this attribute when the run exits.
            benchmark_stage.wait_for_model = wait_for_model
        if launch["command"] == ["bash", "-lc", "echo done"]:
            assert len(polls) == 3
            assert all(worker["is_healthy"] for worker in polls[-1])
        launches.append(launch)

    monkeypatch.setattr("srtctl.core.health.requests.get", workers_response)
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "outputs/4247",
            job_id="4247",
            options=MockOptions(
                child_duration_s=0.05, phase_pause_s=0.01, nodelist=("node-01",), on_srun=record_launch
            ),
        )
        == 0
    )
    assert len(polls) == 3
    assert benchmark_stage.wait_for_model is original_wait_for_model
    assert any(launch["command"] == ["bash", "-lc", "echo done"] for launch in launches)


@pytest.mark.parametrize("hf_snapshot", [False, True])
@pytest.mark.parametrize("stage_weights", [False, True])
def test_smg_uses_resolved_shared_model_mount(tmp_path: Path, hf_snapshot: bool, stage_weights: bool) -> None:
    cache_root = tmp_path / "models--org--model"
    model_dir = cache_root / "snapshots/revision" if hf_snapshot else tmp_path / "model"
    model_dir.mkdir(parents=True)
    if hf_snapshot:
        (cache_root / "blobs").mkdir()
        (cache_root / "blobs/config-digest").write_text("{}")
        (model_dir / "config.json").symlink_to("../../blobs/config-digest")
    else:
        (model_dir / "config.json").write_text("{}")
    model = {"path": str(model_dir), "container": "tokenspeed", "precision": "bf16"}
    stage_dir = tmp_path / "node-local/staged"
    if stage_weights:
        stage_dir.parent.mkdir()
        model["stage_dir"] = str(stage_dir)
    path = tmp_path / "recipe.yaml"
    path.write_text(
        yaml.safe_dump(
            _recipe(
                model=model,
                frontend={"type": "smg", "enable_multiple_frontends": False},
                roles={"agg": {"nodes": 1, "workers": 1, "gpus": 1}},
            )
        )
    )
    launches: list[dict[str, Any]] = []
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "outputs/4248",
            job_id="4248",
            options=MockOptions(
                child_duration_s=0.05, phase_pause_s=0.01, nodelist=("node-01",), on_srun=launches.append
            ),
        )
        == 0
    )
    router = next(launch for launch in launches if launch.get("step_name") == "smg_0")
    worker = next(launch for launch in launches if "tokenspeed.cli" in launch["command"])
    shared_path = "/srtctl-model-cache/snapshots/revision" if hf_snapshot else "/model"
    assert _flag(router["command"], "--model") == shared_path
    assert _flag(worker["command"], "--model") == (str(stage_dir / model_dir.name) if stage_weights else shared_path)
    assert router["container_mounts"][cache_root.resolve() if hf_snapshot else model_dir.resolve()] == Path(
        "/srtctl-model-cache" if hf_snapshot else "/model"
    )


@pytest.mark.parametrize("nodes,workers,gpus", [(1, 2, 1), (2, 1, 16)])
def test_smg_aggregate_launch_matches_engine_handshakes(tmp_path: Path, nodes: int, workers: int, gpus: int) -> None:
    recipe = _recipe(
        frontend={"type": "smg", "enable_multiple_frontends": False},
        roles={"agg": {"nodes": nodes, "workers": workers, "gpus": gpus, "args": {"data-parallel-size": gpus}}},
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    launches: list[dict[str, Any]] = []
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "outputs/4246",
            job_id="4246",
            options=MockOptions(
                child_duration_s=0.05,
                phase_pause_s=0.01,
                nodelist=tuple(f"node-0{i + 1}" for i in range(nodes)),
                on_srun=launches.append,
            ),
        )
        == 0
    )
    commands = [shlex.split(" ".join(launch["command"])) for launch in launches]
    engines = [cmd for cmd in commands if cmd[:5] == ["python3", "-m", "tokenspeed.cli", "serve", "--headless"]]
    router = next(cmd for cmd in commands if cmd[:2] == ["smg", "launch"])
    urls = router[router.index("--worker-urls") + 1 : router.index("--host")]
    assert len(engines) == nodes * workers
    assert len(urls) == workers
    assert set(urls) == {
        f"tcp://{_flag(cmd, '--data-parallel-address')}:{_flag(cmd, '--data-parallel-rpc-port')}" for cmd in engines
    }
    assert all(_flag(cmd, "--data-parallel-size") == str(gpus) for cmd in engines)
    assert all("--metrics-port" not in cmd for cmd in engines)


@pytest.mark.parametrize(
    "frontend,role_args,message",
    [
        ({"enable_multiple_frontends": True}, {}, "one SMG frontend on head"),
        ({"placement": {"node": "first_decode"}}, {}, "one SMG frontend on head"),
        ({"args": {"backend": "vllm"}}, {}, "frontend.args.backend: tokenspeed"),
        ({}, {"data_parallel_rpc_port": 1234}, "data-parallel-rpc-port"),
        ({}, {"data-parallel-address": "other-host"}, "data-parallel-address"),
        ({}, {"zmq-engine-index": 1}, "zmq-engine-index"),
        ({}, {"metrics-port": 1234}, "metrics-port"),
    ],
)
def test_smg_rejects_conflicting_zmq_topology(frontend: dict, role_args: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _load(
            _recipe(
                frontend={"type": "smg", "enable_multiple_frontends": False, **frontend},
                roles={"agg": {"nodes": 1, "workers": 1, "args": role_args}},
            )
        )


@pytest.mark.parametrize(
    "args,expected",
    [
        ({}, False),
        ({"enable_metrics": True}, True),
        ({"enable-metrics": True, "metrics-reporters": None}, True),
        ({"enable-metrics": True, "metrics-reporters": ["otlp"]}, False),
    ],
)
def test_smg_allocates_only_enabled_engine_metrics(args: dict, expected: bool) -> None:
    config = _load(
        _recipe(
            frontend={"type": "smg", "enable_multiple_frontends": False},
            roles={"agg": {"nodes": 1, "workers": 2, "gpus": 1, "args": args}},
        )
    )
    processes = config.worker_processes(
        config.allocate_worker_endpoints(["node-01"]), port_allocator=NodePortAllocator()
    )
    ports = [process.tokenspeed_metrics_port for process in processes]
    assert all((port is not None) == expected for port in ports)
    if expected:
        assert len(set(ports)) == 2


@pytest.mark.parametrize("override_tokenizer", [False, True])
def test_smg_preserves_explicit_model_and_tokenizer(override_tokenizer: bool) -> None:
    config = _load(
        _recipe(
            frontend={
                "type": "smg",
                "enable_multiple_frontends": False,
                "args": {
                    "model": "router-model",
                    "backend": "tokenspeed",
                    **({"tokenizer_path": "/custom-tokenizer"} if override_tokenizer else {}),
                },
            },
            roles={
                "agg": {
                    "nodes": 1,
                    "workers": 1,
                    "args": {"served-model-name": "served", "tokenizer": "/engine-tokenizer"},
                }
            },
        )
    )
    frontend = SMGFrontend()
    args = frontend.get_managed_frontend_args(config, config.backend, []) + frontend.get_frontend_args_list(
        config.frontend.args
    )
    assert _flag(args, "--model") == "router-model"
    assert _flag(args, "--tokenizer-path") == ("/custom-tokenizer" if override_tokenizer else "/engine-tokenizer")
    assert _flag(args, "--model-alias") == "served=router-model"
    assert args.count("--model") == args.count("--tokenizer-path") == args.count("--backend") == 1


def test_smg_requires_runtime_to_resolve_default_model() -> None:
    config = _load(_recipe(frontend={"type": "smg", "enable_multiple_frontends": False}))
    with pytest.raises(ValueError, match="require the resolved runtime model path"):
        SMGFrontend().get_managed_frontend_args(config, config.backend, [])


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
        assert "TOKENSPEED_SKIP_GRPC_WARMUP" not in env
        assert env["MOONCAKE_MASTER"].endswith(":8700")
        assert env["MOONCAKE_TE_META_DATA_SERVER"].endswith(":8701/metadata")
        assert _flag(shlex.split(" ".join(worker["command"])), "--kvstore-storage-backend") == "mooncake"


def test_mapping_args_are_passed_as_json(tmp_path: Path) -> None:
    """A mapping-valued arg reaches TokenSpeed as JSON, not a Python dict repr."""
    recipe = _recipe(
        roles={"agg": {"nodes": 1, "workers": 1, "args": {"kv-events-config": {"enable_kv_cache_events": True}}}},
    )
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    launches: list[dict[str, Any]] = []

    exit_code = run_mock_sweep(
        config_path=config_path,
        output_dir=tmp_path / "outputs" / "4245",
        job_id="4245",
        options=MockOptions(child_duration_s=0.05, phase_pause_s=0.01, nodelist=("node-01",), on_srun=launches.append),
    )

    assert exit_code == 0
    worker = next(launch["command"] for launch in launches if "dynamo.tokenspeed" in launch["command"])
    assert json.loads(_flag(worker, "--kv-events-config")) == {"enable_kv_cache_events": True}
