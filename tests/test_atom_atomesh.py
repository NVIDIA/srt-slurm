# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-signal contracts for native ATOM and AToMesh orchestration."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srtctl.backends import AtomProtocol, AtomServerConfig
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import AtomeshFrontend, get_frontend
from srtctl.frontends.static_router import RouterWorker


def _config() -> dict:
    return {
        "schema": 2,
        "engine": "atom",
        "name": "atom-atomesh",
        "model": {
            "path": "hf:Qwen/Qwen3-0.6B",
            "container": "rocm/atom:latest",
            "precision": "bf16",
        },
        "resources": {
            "gpu_type": "mi300x",
            "gpus_per_node": 8,
        },
        "roles": {
            "prefill": {"nodes": 1, "workers": 1},
            "decode": {"nodes": 1, "workers": 1, "args": {"gpu-memory-utilization": 0.9}},
        },
        "frontend": {"type": "atomesh", "enable_multiple_frontends": False},
    }


def test_schema_selects_atom_backend_and_atomesh_frontend() -> None:
    config = SrtConfig.Schema().load(resolve_config_with_defaults(_config(), None))

    assert isinstance(config.backend, AtomProtocol)
    assert isinstance(get_frontend("atomesh"), AtomeshFrontend)


def test_v2_atom_roles_build_prefill_decode_workers(tmp_path) -> None:
    """Load a v2 recipe through the real orchestrator topology path."""
    import yaml

    from srtctl.cli.do_sweep import SweepOrchestrator
    from srtctl.core.config import load_config
    from srtctl.core.runtime import Nodes

    runtime = SimpleNamespace(
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0", "node1")),
        worker_model_arg="Qwen/Qwen3-0.6B",
        network_interface="hsn0",
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(_config()))
    config = load_config(path)
    orchestrator = SweepOrchestrator(config=config, runtime=runtime)
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.20"):
        processes = orchestrator.backend_processes
        launches = [config.backend.build_worker_command(process, [process], runtime) for process in processes]
    assert [(process.node, process.endpoint_mode) for process in processes] == [
        ("node0", "prefill"),
        ("node1", "decode"),
    ]
    prefill, decode = launches
    assert json.loads(prefill[prefill.index("--kv-transfer-config") + 1])["kv_role"] == "kv_producer"
    assert decode[decode.index("--gpu-memory-utilization") + 1] == "0.9"
    assert json.loads(decode[decode.index("--kv-transfer-config") + 1])["kv_role"] == "kv_consumer"


@pytest.mark.parametrize(
    ("model_path", "stage_dir", "expected"),
    [
        ("hf:deepseek-ai/DeepSeek-V4-Pro", None, "deepseek-ai/DeepSeek-V4-Pro"),
        ("/shared/DeepSeek-V4-Pro", None, "/model"),
        ("/shared/DeepSeek-V4-Pro", "/scratch/models", "/scratch/models/DeepSeek-V4-Pro"),
    ],
)
def test_atom_served_name_matches_worker_model_argument(model_path: str, stage_dir: str | None, expected: str) -> None:
    """ATOM advertises its literal --model, not the path basename used by other engines."""
    data = _config()
    data["model"].update(path=model_path, stage_dir=stage_dir)
    config = SrtConfig.Schema().load(resolve_config_with_defaults(data, None))
    process = Process("node0", frozenset(range(8)), 7500, 6100, "prefill", 0, nixl_port=5400)
    runtime = SimpleNamespace(worker_model_arg=expected, network_interface=None)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.20"):
        command = config.backend.build_worker_command(process, [process], runtime)

    assert config.served_model_name == command[command.index("--model") + 1] == expected


def test_atom_builds_native_aggregate_command() -> None:
    backend = AtomProtocol(
        atom_config=AtomServerConfig(
            aggregated={
                "trust-remote-code": True,
                "gpu-memory-utilization": 0.9,
                "kv_cache_dtype": "fp8",
                "no-enable_prefix_caching": True,
            }
        )
    )
    process = Process("node0", frozenset(range(8)), 7500, 6100, "agg", 0, nixl_port=5400)
    runtime = SimpleNamespace(worker_model_arg="/model", network_interface="hsn0")

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.20"):
        command = backend.build_worker_command(process, [process], runtime)

    assert command[:5] == ["env", "ATOM_HOST_IP=10.0.0.20", "python3", "-m", "atom.entrypoints.openai_server"]
    assert command[command.index("--server-port") + 1] == "6100"
    assert command[command.index("-tp") + 1] == "8"
    assert "--kv-transfer-config" not in command
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.9"
    assert "--gpu_memory_utilization" not in command
    assert command[command.index("--kv_cache_dtype") + 1] == "fp8"
    assert "--no-enable_prefix_caching" in command
    assert "--kv-cache-dtype" not in command
    assert "--no-enable-prefix-caching" not in command
    assert "--trust-remote-code" in command


def test_atom_builds_official_mooncake_pd_contract() -> None:
    backend = AtomProtocol()
    process = Process("node0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=6301)
    runtime = SimpleNamespace(worker_model_arg="/model", network_interface=None)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.20"):
        command = backend.build_worker_command(process, [process], runtime)

    payload = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert payload == {
        "kv_role": "kv_producer",
        "kv_connector": "mooncake",
        "proxy_ip": "10.0.0.20",
        "handshake_port": 6301,
    }


def test_atom_rejects_cross_node_model_parallel_endpoint() -> None:
    """ATOM cannot safely coordinate one logical worker across two Slurm nodes."""
    backend = AtomProtocol()
    leader = Process("node0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=6301)
    peer = Process("node1", frozenset(range(4)), 7501, 6101, "prefill", 0, nixl_port=6302)
    runtime = SimpleNamespace(worker_model_arg="/model", network_interface=None)

    with pytest.raises(ValueError, match="fit on one Slurm node"):
        backend.build_worker_command(leader, [leader, peer], runtime)


def test_atom_pd_worker_requires_mooncake_handshake_port() -> None:
    """Fail before launch when the allocated P/D endpoint has no transfer port."""
    backend = AtomProtocol()
    process = Process("node0", frozenset(range(4)), 7500, 6100, "decode", 0)
    runtime = SimpleNamespace(worker_model_arg="/model", network_interface=None)

    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.20"),
        pytest.raises(ValueError, match="missing its Mooncake handshake port"),
    ):
        backend.build_worker_command(process, [process], runtime)


def test_atom_can_select_mooncake_tcp_for_rocm_correctness() -> None:
    backend = AtomProtocol(mooncake_protocol="tcp")
    process = Process("node0", frozenset(range(4)), 7500, 6100, "decode", 0, nixl_port=6301)
    runtime = SimpleNamespace(worker_model_arg="/model", network_interface=None)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.20"):
        command = backend.build_worker_command(process, [process], runtime)

    payload = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert payload["protocol"] == "tcp"


def test_atomesh_builds_native_static_pd_command() -> None:
    frontend = AtomeshFrontend()
    workers = [
        RouterWorker("prefill", "http://10.0.0.20:6100"),
        RouterWorker("decode", "http://10.0.0.21:6101"),
    ]

    command = frontend.build_router_command(workers, "0.0.0.0", 8000)
    command.extend(frontend.get_managed_frontend_args(SimpleNamespace(frontend=SimpleNamespace(args={})), None, []))

    assert command == [
        "atomesh",
        "launch",
        "--pd-disaggregation",
        "--prefill",
        "http://10.0.0.20:6100",
        "--decode",
        "http://10.0.0.21:6101",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--backend",
        "atom",
    ]


def test_atomesh_suppresses_bootstrap_port_for_atom_workers() -> None:
    frontend = AtomeshFrontend()
    process = Process("node0", frozenset({0}), 7500, 6100, "prefill", 0, nixl_port=6301)

    assert frontend.worker_bootstrap_port(AtomProtocol(), process) is None
