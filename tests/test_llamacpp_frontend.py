# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp topology restrictions, readiness and metrics ownership."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from marshmallow import ValidationError

from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Endpoint, endpoints_to_processes
from srtctl.frontends import LlamaCppFrontend, get_frontend


def recipe(**overrides) -> dict:
    return {
        "schema": 2,
        "name": "llamacpp",
        "model": {"path": "/models/qwen", "container": "llama:server-cuda", "precision": "q8_0"},
        "resources": {"gpus_per_node": 4},
        "engine": {"type": "llamacpp", "model_file": "model.gguf"},
        "frontend": {"type": "llamacpp", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 1}},
        **overrides,
    }


def load(data: dict) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(data, None))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frontend": {"type": "llamacpp"}}, "enable_multiple_frontends: false"),
        ({"roles": {"agg": {"nodes": 1, "workers": 2, "gpus": 1}}}, "exactly one aggregate worker"),
        ({"roles": {"agg": {"nodes": 2, "workers": 1, "gpus": 8}}}, "roles.agg.nodes: 1"),
        (
            {"roles": {"prefill": {"nodes": 1, "gpus": 1}, "decode": {"nodes": 1, "gpus": 1}}},
            "exactly one aggregate worker",
        ),
        ({"dynamo": {"sidecar": True}}, "does not support dynamo.sidecar"),
        ({"model": {"path": "hf:org/model", "container": "image", "precision": "q8"}}, "local model.path"),
        ({"engine": "sglang"}, "requires backend.type: llamacpp"),
        ({"frontend": {"type": "dynamo"}}, "requires frontend.type: llamacpp"),
        ({"frontend": {"type": "sglang-router"}}, "requires frontend.type: llamacpp"),
        (
            {"profiling": {"type": "torch", "aggregated": {"start_step": 1, "stop_step": 3}}},
            "only time-based profiling",
        ),
    ],
)
def test_invalid_recipe_fails_before_launch(changes, message):
    with pytest.raises(ValidationError, match=message):
        load(recipe(**changes))


def test_no_extra_frontend_and_correct_port_ownership():
    config = load(recipe())
    frontend = get_frontend("llamacpp")
    assert isinstance(frontend, LlamaCppFrontend)
    processes = endpoints_to_processes([Endpoint("agg", 0, ("node",), frozenset({0}), 4)])
    runtime = SimpleNamespace(frontend_port=8123)
    process = processes[0]
    assert frontend.worker_api_port("agg") == "public"
    assert frontend.worker_ready_port(process) == 8000
    assert frontend.worker_endpoint_port(process, config, runtime) == 8123
    assert frontend.worker_metrics_port(process, runtime) == 8123
    assert frontend.worker_metrics_path(config.backend, "agg") == "/metrics"
    assert frontend.profiling_control_port(process, config, runtime) is None
    assert frontend.direct_endpoint_nodes(processes) == ["node"]
    assert frontend.start_frontends(None, runtime, config, config.backend, processes) == []
    config = load(recipe(engine={"type": "llamacpp", "model_file": "model.gguf", "metrics": False}))
    assert frontend.worker_metrics_path(config.backend, "agg") is None


@pytest.mark.parametrize(
    ("health_status", "models", "ready", "calls"),
    [(503, [], False, 1), (200, [], False, 2), (200, [{"id": "model"}], True, 2)],
)
def test_readiness_waits_for_loaded_model(health_status, models, ready, calls):
    health = Mock(status_code=health_status)
    listing = Mock(status_code=200)
    listing.json.return_value = {"data": models}
    with patch("srtctl.core.health.requests.get", side_effect=[health, listing]) as get:
        result = LlamaCppFrontend().probe_ready("host", 8000, 0, 1, load(recipe()))
    assert result.ready is ready
    assert get.call_count == calls
    assert get.call_args_list[0].args[0] == "http://host:8000/health"
    if calls == 2:
        assert get.call_args_list[1].args[0] == "http://host:8000/v1/models"
