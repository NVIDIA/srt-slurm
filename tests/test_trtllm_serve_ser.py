# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trtllm_serve ser.yaml builder (router + server_config_extra)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.core.schema import SrtConfig
from srtctl.frontends import TRTLLMServeFrontend


def _cfg(**frontend_fields):
    frontend_fields.setdefault("ctx_router", None)
    frontend_fields.setdefault("gen_router", None)
    frontend_fields.setdefault("server_config_extra", None)
    return SimpleNamespace(frontend=SimpleNamespace(**frontend_fields))


def test_ser_minimal_no_router():
    ser = TRTLLMServeFrontend._build_ser(_cfg(), ["c0:8000"], ["g0:8001", "g1:8002"], 8000)
    assert ser.get("backend", True)  # tolerate absent backend key
    assert ser["hostname"] == "0.0.0.0"
    assert ser["port"] == 8000
    assert ser["context_servers"] == {"num_instances": 1, "urls": ["c0:8000"]}
    assert ser["generation_servers"] == {"num_instances": 2, "urls": ["g0:8001", "g1:8002"]}
    assert "router" not in ser["context_servers"]
    assert "router" not in ser["generation_servers"]


def test_ser_conversation_router_and_extra():
    ser = TRTLLMServeFrontend._build_ser(
        _cfg(
            ctx_router={"type": "conversation"},
            server_config_extra={"gen_strip_message_history": True, "gen_tokids_ctxbytes": True},
        ),
        ["c0:8000"],
        ["g0:8001"],
        8000,
    )
    assert ser["context_servers"]["router"] == {"type": "conversation"}
    assert "router" not in ser["generation_servers"]
    assert ser["gen_strip_message_history"] is True
    assert ser["gen_tokids_ctxbytes"] is True


def test_ser_extra_server_section_merges_into_generated_section():
    ser = TRTLLMServeFrontend._build_ser(
        _cfg(
            ctx_router={"type": "conversation"},
            server_config_extra={
                "context_servers": {"num_serve_frontends": 8, "serve_frontend_affinity_port_stride": 10},
                "generation_servers": {"num_serve_frontends": 4},
                "return_perf_metrics": True,
            },
        ),
        ["c0:8000", "c0:8032"],
        ["g0:8001"],
        8000,
    )
    assert ser["context_servers"] == {
        "num_instances": 2,
        "urls": ["c0:8000", "c0:8032"],
        "router": {"type": "conversation"},
        "num_serve_frontends": 8,
        "serve_frontend_affinity_port_stride": 10,
    }
    assert ser["generation_servers"] == {"num_instances": 1, "urls": ["g0:8001"], "num_serve_frontends": 4}
    assert ser["return_perf_metrics"] is True


@pytest.mark.parametrize("owned", ["urls", "num_instances", "router"])
def test_ser_extra_server_section_rejects_srtctl_keys(owned: str):
    cfg = _cfg(server_config_extra={"context_servers": {owned: ["<node2_placeholder>:8678"]}})
    with pytest.raises(ValueError, match=f"server_config_extra.context_servers sets \\['{owned}'\\]"):
        TRTLLMServeFrontend._build_ser(cfg, ["c0:8000"], ["g0:8001"], 8000)


def test_ser_gen_router():
    ser = TRTLLMServeFrontend._build_ser(_cfg(gen_router={"type": "default"}), ["c0:8000"], ["g0:8001"], 8000)
    assert ser["generation_servers"]["router"] == {"type": "default"}
    assert "router" not in ser["context_servers"]


def test_frontend_config_accepts_router_fields(tmp_path: Path):
    # The FrontendConfig schema must round-trip the new fields.
    data = {
        "schema": 2,
        "name": "r",
        "model": {"path": "/lustre/m", "container": "trtllm", "precision": "fp4"},
        "resources": {"gpu_type": "gb300", "gpus_per_node": 4},
        "engine": "trtllm",
        "roles": {
            "prefill": {"nodes": 1, "workers": 1},
            "decode": {"nodes": 1, "workers": 1},
        },
        "frontend": {
            "type": "trtllm_serve",
            "enable_multiple_frontends": False,
            "ctx_router": {"type": "conversation"},
            "server_config_extra": {"gen_strip_message_history": True, "gen_tokids_ctxbytes": True},
        },
    }
    config_path = tmp_path / "disaggregated.yaml"
    config_path.write_text(yaml.safe_dump(data))
    config = SrtConfig.from_yaml(config_path)
    assert config.frontend.ctx_router == {"type": "conversation"}
    assert config.frontend.server_config_extra["gen_strip_message_history"] is True


def test_frontend_config_rejects_server_section_urls_at_load(tmp_path: Path):
    """A ser.yaml section copied from another launcher carries its own worker URLs; srtctl writes those."""
    data = {
        "schema": 2,
        "name": "r",
        "model": {"path": "/lustre/m", "container": "trtllm", "precision": "fp4"},
        "resources": {"gpu_type": "gb300", "gpus_per_node": 4},
        "engine": "trtllm",
        "roles": {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}},
        "frontend": {
            "type": "trtllm_serve",
            "enable_multiple_frontends": False,
            "server_config_extra": {
                "context_servers": {"urls": ["<node2_placeholder>:8678"], "num_serve_frontends": 8}
            },
        },
    }
    config_path = tmp_path / "disaggregated.yaml"
    config_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match=r"server_config_extra.context_servers sets \['urls'\]"):
        SrtConfig.from_yaml(config_path)


def _aggregate_config(agg_workers: int) -> dict:
    return {
        "schema": 2,
        "name": "trtllm-raw",
        "model": {"path": "/model", "container": "trtllm", "precision": "fp4"},
        "resources": {"gpu_type": "gb300", "gpus_per_node": 4},
        "engine": "trtllm",
        "roles": {"agg": {"nodes": 2, "workers": agg_workers, "gpus": 8}},
        "frontend": {"type": "trtllm_serve", "enable_multiple_frontends": False},
    }


def test_frontend_config_accepts_single_aggregate_worker(tmp_path: Path):
    config_path = tmp_path / "aggregate.yaml"
    config_path.write_text(yaml.safe_dump(_aggregate_config(agg_workers=1)))
    config = SrtConfig.from_yaml(config_path)

    assert config.topology.num_agg == 1


@pytest.mark.parametrize("agg_workers", [0, 2])
def test_frontend_config_rejects_non_single_aggregate_worker(tmp_path: Path, agg_workers: int):
    config_path = tmp_path / f"aggregate-{agg_workers}.yaml"
    config_path.write_text(yaml.safe_dump(_aggregate_config(agg_workers=agg_workers)))

    with pytest.raises(ValidationError, match="requires exactly one aggregate worker"):
        SrtConfig.from_yaml(config_path)


def test_aggregate_frontend_uses_worker_directly():
    frontend = TRTLLMServeFrontend()
    topology = SimpleNamespace(uses_nginx=False, frontend_nodes=["node0"], public_port=8000)
    config = SimpleNamespace(
        backend=SimpleNamespace(type="trtllm"),
        topology=SimpleNamespace(is_disaggregated=False),
    )
    worker = SimpleNamespace(endpoint_mode="agg", is_leader=True)

    assert frontend.start_frontends(topology, MagicMock(), config, MagicMock(), [worker]) == []


@patch("srtctl.frontends.trtllm_serve.start_srun_process")
@patch("srtctl.frontends.trtllm_serve.wait_for_health", return_value=True)
@patch("srtctl.frontends.trtllm_serve.get_hostname_ip", side_effect=lambda node: f"10.0.0.{node[-1]}")
def test_disaggregated_orchestrator_inherits_runtime_srun_options(_ip, _health, mock_srun, tmp_path: Path):
    """The disaggregated orchestrator gets the same runtime srun options as workers and
    services, so a shared pyxis container-name also covers the frontend node."""
    mock_srun.return_value = MagicMock()
    frontend = TRTLLMServeFrontend()
    topology = SimpleNamespace(uses_nginx=False, frontend_nodes=["node0"], frontend_port=8000, public_port=8000)
    config = SimpleNamespace(
        backend=SimpleNamespace(type="trtllm"),
        topology=SimpleNamespace(is_disaggregated=True),
        frontend=SimpleNamespace(
            ctx_router=None, gen_router=None, server_config_extra=None, args=None, env=None, numa_bind=False
        ),
        health_check=SimpleNamespace(max_attempts=1, interval_seconds=0),
    )
    runtime = SimpleNamespace(
        log_dir=tmp_path,
        container_image="/container.sqsh",
        reuse_container=False,
        container_prepare_timeout_seconds=1800.0,
        container_mounts={},
        srun_options={"container-name": "shared-job-container", "mem": "0"},
        nodes=SimpleNamespace(het_group_for=lambda node: None),
    )
    workers = [
        SimpleNamespace(node="node1", endpoint_mode="prefill", http_port=6100, is_leader=True),
        SimpleNamespace(node="node2", endpoint_mode="decode", http_port=6100, is_leader=True),
    ]

    processes = frontend.start_frontends(topology, runtime, config, MagicMock(), workers)

    assert [p.name for p in processes] == ["trtllm_serve_orchestrator"]
    kwargs = mock_srun.call_args.kwargs
    assert kwargs["srun_options"] == runtime.srun_options
    assert kwargs["nodelist"] == ["node0"]
    assert kwargs["container_image"] == "/container.sqsh"
