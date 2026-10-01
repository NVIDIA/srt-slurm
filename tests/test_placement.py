# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`placement:` on the frontend and the benchmark client: one typed block, no per-block booleans."""

from __future__ import annotations

import pytest
from marshmallow import ValidationError

from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import PLACEMENT_DEDICATED, BenchmarkConfig, FrontendConfig, PlacementConfig, SrtConfig


def test_placement_config_resolves_dedicated_to_the_head_location() -> None:
    default = PlacementConfig()
    assert default.node == "head" and not default.dedicated and default.location == "head"
    placed = PlacementConfig(node="first_decode")
    assert placed.location == "first_decode" and not placed.dedicated
    dedicated = PlacementConfig(node=PLACEMENT_DEDICATED)
    assert dedicated.dedicated and dedicated.location == "head"


def test_frontend_and_benchmark_default_to_the_head_node() -> None:
    assert FrontendConfig().placement == PlacementConfig()
    assert BenchmarkConfig().placement == PlacementConfig()


def _recipe(frontend: dict, benchmark: dict | None = None) -> dict:
    return {
        "schema": 2,
        "name": "placement",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "engine": "sglang",
        "roles": {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}},
        "frontend": frontend,
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4", **(benchmark or {})},
    }


def _load(recipe: dict) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(recipe, None))


def test_placement_loads_from_the_recipe() -> None:
    cfg = _load(
        _recipe({"type": "dynamo", "placement": {"node": "first_decode"}}, {"placement": {"node": "dedicated"}})
    )
    assert cfg.frontend.placement == PlacementConfig(node="first_decode")
    assert cfg.frontend.placement.location == "first_decode" and not cfg.frontend.placement.dedicated
    assert cfg.benchmark.placement.dedicated and cfg.benchmark.placement.location == "head"


def test_unknown_placement_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        _load(_recipe({"type": "dynamo", "placement": {"node": "head", "extra": 1}}))


def test_v1_placement_keys_in_a_recipe_are_rejected_at_load() -> None:
    with pytest.raises(ValueError, match=r"pre-2\.0 \(v1\) layout: frontend\.orchestrator_placement"):
        resolve_config_with_defaults(_recipe({"type": "dynamo", "orchestrator_placement": "first_decode"}), None)
    with pytest.raises(ValueError, match=r"benchmark\.client_dedicated_node"):
        resolve_config_with_defaults(_recipe({"type": "dynamo"}, {"client_dedicated_node": True}), None)
