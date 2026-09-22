# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ATOM backend and AToMesh frontend contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import AtomProtocol, AtomServerConfig
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import load_config, resolve_config_with_defaults
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import AtomeshFrontend

WORKER_IP = "10.0.0.20"


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


def _load(data: dict) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(data, None))


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(worker_model_arg="/model", network_interface=None)


def _build(backend: AtomProtocol, process: Process, runtime=None, **kwargs) -> list[str]:
    with patch("srtctl.core.slurm.get_hostname_ip", return_value=WORKER_IP):
        return backend.build_worker_command(process, [process], runtime or _runtime(), **kwargs)


def test_atomesh_frontend_requires_atom_backend() -> None:
    data = _config()
    data["engine"] = "sglang"

    with pytest.raises(ValidationError, match="frontend.type: atomesh requires backend.type: atom"):
        _load(data)


def test_atom_worker_requires_atomesh_frontend() -> None:
    process = Process("node0", frozenset(range(8)), 7500, 6100, "agg", 0)

    with pytest.raises(ValueError, match="requires frontend.type: atomesh"):
        AtomProtocol().build_worker_command(process, [process], _runtime(), frontend_type="dynamo")


def test_v2_atom_roles_build_prefill_decode_workers(tmp_path: Path) -> None:
    """Load a v2 recipe through the real orchestrator topology path."""
    runtime = SimpleNamespace(
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0", "node1")),
        worker_model_arg="Qwen/Qwen3-0.6B",
        network_interface="hsn0",
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(_config()))
    config = load_config(path)
    orchestrator = SweepOrchestrator(config=config, runtime=runtime)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value=WORKER_IP):
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


@pytest.mark.parametrize("layout", ["hf", "mounted", "staged"])
def test_atom_served_name_matches_worker_model_argument(tmp_path: Path, layout: str) -> None:
    """ATOM advertises its literal --model, so served_model_name must track the runtime's worker_model_arg."""
    data = _config()
    stage_dir = tmp_path / "scratch" / "models"
    if layout == "hf":
        data["model"]["path"] = "hf:deepseek-ai/DeepSeek-V4-Pro"
    else:
        model_dir = tmp_path / "DeepSeek-V4-Pro"
        model_dir.mkdir()
        data["model"]["path"] = str(model_dir)
        if layout == "staged":
            data["model"]["stage_dir"] = str(stage_dir)
    config = _load(data)
    nodes = Nodes(head="node0", bench="node0", infra="node0", worker=("node0",))

    with (
        patch("srtctl.core.runtime.Nodes.from_slurm", return_value=nodes),
        patch("srtctl.core.runtime.get_srtslurm_setting", side_effect=lambda name, default=None: default),
        patch("srtctl.core.runtime.get_hostname_ip", return_value=WORKER_IP),
    ):
        runtime = RuntimeContext.from_config(config, job_id="42", log_dir_base=tmp_path / "outputs")
    process = Process("node0", frozenset(range(8)), 7500, 6100, "prefill", 0, nixl_port=5400)
    command = _build(config.backend, process, runtime)

    expected = {
        "hf": "deepseek-ai/DeepSeek-V4-Pro",
        "mounted": "/model",
        "staged": str(stage_dir / "DeepSeek-V4-Pro"),
    }[layout]
    assert command[command.index("--model") + 1] == expected
    assert config.served_model_name == expected


def test_atom_builds_native_aggregate_command() -> None:
    """Recipe flags keep ATOM's mixed hyphen/underscore spelling and follow the managed arguments."""
    backend = AtomProtocol(
        atom_config=AtomServerConfig(
            aggregated={
                "trust-remote-code": True,
                "gpu-memory-utilization": 0.9,
                "kv_cache_dtype": "fp8",
                "no-enable_prefix_caching": True,
                "disable-log-stats": False,
            }
        )
    )
    process = Process("node0", frozenset(range(8)), 7500, 6100, "agg", 0, nixl_port=5400)

    command = _build(backend, process, SimpleNamespace(worker_model_arg="/model", network_interface="hsn0"))

    assert command == [
        "env",
        f"ATOM_HOST_IP={WORKER_IP}",
        "python3",
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        "/model",
        "--host",
        "0.0.0.0",
        "--server-port",
        "6100",
        "-tp",
        "8",
        "--gpu-memory-utilization",
        "0.9",
        "--kv_cache_dtype",
        "fp8",
        "--no-enable_prefix_caching",
        "--trust-remote-code",
    ]


@pytest.mark.parametrize("key", ["tensor_parallel_size", "--server-port", "model"])
def test_atom_rejects_recipe_overrides_of_managed_arguments(key: str) -> None:
    """Reserved flags are matched after normalizing dashes and underscores."""
    backend = AtomProtocol(atom_config=AtomServerConfig(aggregated={key: 4}))
    process = Process("node0", frozenset(range(8)), 7500, 6100, "agg", 0)

    with pytest.raises(ValueError, match="srtctl-managed argument"):
        _build(backend, process)


@pytest.mark.parametrize(("protocol", "extra"), [(None, {}), ("tcp", {"protocol": "tcp"})])
def test_atom_pd_worker_emits_mooncake_kv_transfer_config(protocol: str | None, extra: dict) -> None:
    backend = AtomProtocol(mooncake_protocol=protocol)
    process = Process("node0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=6301)

    command = _build(backend, process)

    payload = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert payload == {
        "kv_role": "kv_producer",
        "kv_connector": "mooncake",
        "proxy_ip": WORKER_IP,
        "handshake_port": 6301,
        **extra,
    }


def test_atom_rejects_cross_node_model_parallel_endpoint() -> None:
    """ATOM cannot coordinate one logical worker across two Slurm nodes."""
    leader = Process("node0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=6301)
    peer = Process("node1", frozenset(range(4)), 7501, 6101, "prefill", 0, nixl_port=6302)

    with pytest.raises(ValueError, match="fit on one Slurm node"):
        AtomProtocol().build_worker_command(leader, [leader, peer], _runtime())


def test_atom_pd_worker_requires_mooncake_handshake_port() -> None:
    """Fail before launch when the allocated P/D endpoint has no transfer port."""
    process = Process("node0", frozenset(range(4)), 7500, 6100, "decode", 0)

    with pytest.raises(ValueError, match="missing its Mooncake handshake port"):
        _build(AtomProtocol(), process)


def test_atomesh_builds_static_pd_command_without_bootstrap_ports() -> None:
    """ATOM publishes transfer topology itself, so no handshake port follows --prefill."""
    frontend = AtomeshFrontend()
    backend = AtomProtocol()
    processes = [
        Process("node0", frozenset(range(8)), 7500, 6100, "prefill", 0, nixl_port=6301),
        Process("node1", frozenset(range(8)), 7500, 6101, "decode", 0, nixl_port=6302),
    ]
    config = SimpleNamespace(frontend=SimpleNamespace(args={}))

    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=["10.0.0.20", "10.0.0.21"]):
        workers = frontend.collect_workers(backend, processes, "fabric0")
    command = frontend.build_router_command(workers, "0.0.0.0", 8000, backend)
    command.extend(frontend.get_managed_frontend_args(config, backend, processes))

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


def test_atomesh_rejects_recipe_backend_argument() -> None:
    config = SimpleNamespace(frontend=SimpleNamespace(args={"backend": "vllm"}))

    with pytest.raises(ValueError, match="managed by srtctl"):
        AtomeshFrontend().get_managed_frontend_args(config, AtomProtocol(), [])


def test_atomesh_worker_metrics_are_not_scraped() -> None:
    """The frontend owns scrape targets; ATOM has no worker metrics endpoint."""
    process = Process("node0", frozenset(range(8)), 7500, 6100, "agg", 0)
    assert AtomeshFrontend().worker_metrics_port(process, _runtime()) is None
