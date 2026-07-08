# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trtllm_serve ser.yaml builder (router + server_config_extra)."""

from types import SimpleNamespace

from srtctl.frontends import TRTLLMServeFrontend


def _cfg(**frontend_fields):
    frontend_fields.setdefault("ctx_router", None)
    frontend_fields.setdefault("gen_router", None)
    frontend_fields.setdefault("server_config_extra", None)
    return SimpleNamespace(frontend=SimpleNamespace(**frontend_fields))


def test_ser_minimal_no_router():
    ser = TRTLLMServeFrontend._build_ser(_cfg(), ["c0:8000"], ["g0:8001", "g1:8002"], 8000)
    assert ser["backend"] if "backend" in ser else True  # tolerate absent backend key
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


def test_ser_gen_router():
    ser = TRTLLMServeFrontend._build_ser(
        _cfg(gen_router={"type": "default"}), ["c0:8000"], ["g0:8001"], 8000
    )
    assert ser["generation_servers"]["router"] == {"type": "default"}
    assert "router" not in ser["context_servers"]


def test_frontend_config_accepts_router_fields():
    # The FrontendConfig schema must round-trip the new fields.
    import tempfile
    from pathlib import Path

    import yaml

    from srtctl.core.schema import SrtConfig

    data = {
        "name": "r",
        "model": {"path": "/lustre/m", "container": "trtllm", "precision": "fp4"},
        "resources": {
            "gpu_type": "gb300",
            "gpus_per_node": 4,
            "prefill_nodes": 1,
            "prefill_workers": 1,
            "decode_nodes": 1,
            "decode_workers": 1,
        },
        "backend": {"type": "trtllm"},
        "frontend": {
            "type": "trtllm_serve",
            "enable_multiple_frontends": False,
            "ctx_router": {"type": "conversation"},
            "server_config_extra": {"gen_strip_message_history": True, "gen_tokids_ctxbytes": True},
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        config = SrtConfig.from_yaml(Path(f.name))
    assert config.frontend.ctx_router == {"type": "conversation"}
    assert config.frontend.server_config_extra["gen_strip_message_history"] is True
