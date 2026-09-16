# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`frontend.type: sglang` is the router-free single worker; `sglang-router` is the Model Gateway."""

from __future__ import annotations

import copy

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.migrate import migrate_recipe_text
from srtctl.core.schema import SrtConfig
from srtctl.frontends import SGLangFrontend, SGLangRouterFrontend, get_frontend


def _recipe(**overrides) -> dict:
    base = {
        "schema": 2,
        "name": "direct",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "bf16"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "engine": "sglang",
        "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 1, "args": {"tensor-parallel-size": 1}}},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }
    base.update(overrides)
    return base


def _load(recipe: dict) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(recipe, None))


def test_direct_single_worker_loads() -> None:
    cfg = _load(_recipe())
    assert cfg.frontend.type == "sglang"
    assert cfg.resources.num_agg == 1
    assert isinstance(get_frontend(cfg.frontend.type), SGLangFrontend)


def test_direct_rejects_replicas_disagg_nginx_and_other_engines() -> None:
    with pytest.raises(ValidationError, match="exactly one aggregate worker.*sglang-router"):
        _load(_recipe(roles={"agg": {"nodes": 1, "workers": 2, "gpus": 1}}))
    with pytest.raises(ValidationError, match="prefill/decode layout.*sglang-router"):
        _load(
            _recipe(
                roles={
                    "prefill": {"nodes": 1, "workers": 1, "gpus": 4},
                    "decode": {"nodes": "colocate", "workers": 1, "gpus": 4},
                }
            )
        )
    with pytest.raises(ValidationError, match="enable_multiple_frontends: false"):
        _load(_recipe(frontend={"type": "sglang", "enable_multiple_frontends": True}))
    with pytest.raises(ValidationError, match="requires engine sglang"):
        _load(_recipe(engine="vllm"))


def test_router_type_still_covers_replicas_and_disagg() -> None:
    cfg = _load(
        _recipe(
            frontend={"type": "sglang-router", "enable_multiple_frontends": False},
            roles={"agg": {"nodes": 1, "workers": 2, "gpus": 1}},
        )
    )
    assert cfg.frontend.type == "sglang-router"
    assert isinstance(get_frontend(cfg.frontend.type), SGLangRouterFrontend)


def test_a_recipe_without_schema_is_rejected_not_reinterpreted() -> None:
    """Schema 1 spelled the router `sglang`; the loader no longer guesses, it points at migrate."""
    v1 = copy.deepcopy(_recipe(roles={"agg": {"nodes": 1, "workers": 2, "gpus": 1}}))
    v1.pop("schema")
    with pytest.raises(ValueError, match="no `schema:` key.*srtctl migrate"):
        _load(v1)
    v1["schema"] = 1
    with pytest.raises(ValueError, match="schema 1 is not supported"):
        _load(v1)


V1_TEXT = """\
name: legacy-router
model:
  path: /m
  container: /c.sqsh
  precision: bf16
resources:
  gpu_type: h100
  gpus_per_node: 8
  agg_nodes: 1
  agg_workers: 2
frontend:
  type: sglang          # the Model Gateway in schema 1
  enable_multiple_frontends: false
backend:
  type: sglang
  sglang_config:
    aggregated:
      tensor-parallel-size: 4
benchmark:
  type: sa-bench
  isl: 128
  osl: 128
  concurrencies: "4"
"""


def test_migrate_renames_the_router_and_the_result_loads_as_the_router() -> None:
    result = migrate_recipe_text(V1_TEXT)
    doc = yaml.safe_load(result.text)
    assert doc["schema"] == 2
    assert doc["frontend"]["type"] == "sglang-router"
    assert any("frontend.type: sglang -> sglang-router" in note for note in result.notes)
    cfg = _load(doc)
    assert cfg.frontend.type == "sglang-router"
    assert cfg.resources.num_agg == 2
    assert isinstance(get_frontend(cfg.frontend.type), SGLangRouterFrontend)


def test_migrate_leaves_a_schema2_direct_recipe_alone() -> None:
    text = "schema: 2\n" + V1_TEXT.replace("  agg_workers: 2\n", "  agg_workers: 1\n")
    doc = yaml.safe_load(migrate_recipe_text(text).text)
    assert doc["frontend"]["type"] == "sglang"
