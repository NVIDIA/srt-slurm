# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.placement import BENCHMARK_PLACEMENT_FIELDS, FRONTEND_PLACEMENT_FIELDS, expand_placement
from srtctl.core.schema import SrtConfig


def test_frontend_placement_location_and_dedicated() -> None:
    cfg = {"frontend": {"type": "dynamo", "placement": {"node": "first_decode"}}}
    expand_placement(cfg)
    assert cfg["frontend"]["orchestrator_placement"] == "first_decode"
    assert "placement" not in cfg["frontend"]

    cfg = {"frontend": {"type": "dynamo", "placement": {"node": "dedicated"}}}
    expand_placement(cfg)
    assert cfg["frontend"]["dedicated_node"] is True
    assert cfg["frontend"]["orchestrator_placement"] == "head"


def test_benchmark_placement_location_and_dedicated() -> None:
    cfg = {"benchmark": {"type": "sa-bench", "placement": {"node": "last_decode"}}}
    expand_placement(cfg)
    assert cfg["benchmark"]["client_placement"] == "last_decode"

    cfg = {"benchmark": {"type": "sa-bench", "placement": {"node": "dedicated"}}}
    expand_placement(cfg)
    assert cfg["benchmark"]["client_dedicated_node"] is True
    assert cfg["benchmark"]["client_placement"] == "head"


def test_placement_fields_are_the_internal_names_the_schema_reads() -> None:
    assert FRONTEND_PLACEMENT_FIELDS == ("orchestrator_placement", "dedicated_node")
    assert BENCHMARK_PLACEMENT_FIELDS == ("client_placement", "client_dedicated_node")


def _recipe(frontend: dict) -> dict:
    return {
        "schema": 2,
        "name": "placement",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "engine": "sglang",
        "roles": {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}},
        "frontend": frontend,
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }


def test_placement_loads_into_the_internal_fields() -> None:
    schema = SrtConfig.Schema()
    placed = schema.load(
        resolve_config_with_defaults(_recipe({"type": "dynamo", "placement": {"node": "first_decode"}}), None)
    )
    assert placed.frontend.orchestrator_placement == "first_decode"
    assert placed.frontend.dedicated_node is False
    # The internal layout is what the schema itself reads; the recipe never spells it.
    internal = dict(_recipe({"type": "dynamo", "orchestrator_placement": "first_decode"}))
    internal.pop("schema")
    internal.pop("engine")
    internal.pop("roles")
    internal["backend"] = {"type": "sglang"}
    internal["resources"] = {**internal["resources"], "prefill_nodes": 1, "prefill_workers": 1}
    internal["resources"].update({"decode_nodes": 1, "decode_workers": 1})
    assert schema.dump(placed) == schema.dump(schema.load(internal))


def test_v1_placement_keys_in_a_recipe_are_rejected_at_load() -> None:
    with pytest.raises(ValueError, match=r"pre-2\.0 \(v1\) layout: frontend\.orchestrator_placement"):
        resolve_config_with_defaults(_recipe({"type": "dynamo", "orchestrator_placement": "first_decode"}), None)
    recipe = _recipe({"type": "dynamo"})
    recipe["benchmark"]["client_dedicated_node"] = True
    with pytest.raises(ValueError, match=r"benchmark\.client_dedicated_node"):
        resolve_config_with_defaults(recipe, None)


def test_mixing_placement_with_the_internal_fields_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_placement({"frontend": {"placement": {"node": "head"}, "orchestrator_placement": "head"}})


def test_invalid_placement_values_rejected() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        expand_placement({"frontend": {"placement": {"node": "head", "extra": 1}}})
    with pytest.raises(ValueError, match="requires a 'node'"):
        expand_placement({"frontend": {"placement": {}}})


def test_no_placement_is_a_no_op() -> None:
    cfg = {"frontend": {"type": "dynamo", "orchestrator_placement": "head"}}
    assert expand_placement(dict(cfg)) == cfg
