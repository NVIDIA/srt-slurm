# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest

from srtctl.core.roles import expand_roles, roles_from_legacy
from srtctl.core.schema import SrtConfig


def _legacy_sglang_disagg() -> dict:
    return {
        "schema": 2,
        "name": "roles-test",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {
            "gpu_type": "h100",
            "gpus_per_node": 8,
            "prefill_nodes": 2,
            "prefill_workers": 6,
            "gpus_per_prefill": 2,
            "decode_nodes": 0,
            "decode_workers": 2,
            "gpus_per_decode": 2,
        },
        "backend": {
            "type": "sglang",
            "prefill_environment": {"PYTHONUNBUFFERED": "1"},
            "decode_environment": {"PYTHONUNBUFFERED": "1"},
            "sglang_config": {
                "prefill": {"tensor-parallel-size": 2, "disaggregation-mode": "prefill"},
                "decode": {"tensor-parallel-size": 2, "disaggregation-mode": "decode"},
            },
        },
        "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }


def _roles_sglang_disagg() -> dict:
    return {
        "schema": 2,
        "name": "roles-test",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "backend": {"type": "sglang"},
        "roles": {
            "prefill": {
                "nodes": 2,
                "workers": 6,
                "gpus": 2,
                "env": {"PYTHONUNBUFFERED": "1"},
                "args": {"tensor-parallel-size": 2, "disaggregation-mode": "prefill"},
            },
            "decode": {
                "nodes": 0,
                "workers": 2,
                "gpus": 2,
                "env": {"PYTHONUNBUFFERED": "1"},
                "args": {"tensor-parallel-size": 2, "disaggregation-mode": "decode"},
            },
        },
        "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }


def test_expand_roles_produces_the_legacy_layout() -> None:
    expanded = expand_roles(_roles_sglang_disagg())
    assert "roles" not in expanded
    assert expanded["resources"] == _legacy_sglang_disagg()["resources"]
    assert expanded["backend"] == _legacy_sglang_disagg()["backend"]


def test_roles_and_legacy_load_to_identical_configs() -> None:
    from_roles = SrtConfig.Schema().load(expand_roles(_roles_sglang_disagg()))
    from_legacy = SrtConfig.Schema().load(_legacy_sglang_disagg())
    schema = SrtConfig.Schema()
    assert schema.dump(from_roles) == schema.dump(from_legacy)


def test_roles_round_trips_through_roles_from_legacy() -> None:
    legacy = _legacy_sglang_disagg()
    as_roles = roles_from_legacy(legacy)
    assert as_roles["roles"] == _roles_sglang_disagg()["roles"]
    assert "prefill_workers" not in as_roles.get("resources", {})
    assert "prefill_environment" not in as_roles["backend"]
    assert "sglang_config" not in as_roles["backend"]
    # And expanding it back reproduces the original internal layout.
    assert expand_roles(copy.deepcopy(as_roles))["resources"] == legacy["resources"]
    assert expand_roles(copy.deepcopy(as_roles))["backend"] == legacy["backend"]


def test_agg_role_maps_to_aggregated_env_and_config() -> None:
    config = {
        "backend": {"type": "vllm"},
        "roles": {"agg": {"workers": 2, "gpus": 1, "env": {"X": "1"}, "args": {"tensor-parallel-size": 1}}},
    }
    expand_roles(config)
    assert config["resources"] == {"agg_workers": 2, "gpus_per_agg": 1}
    assert config["backend"]["aggregated_environment"] == {"X": "1"}
    assert config["backend"]["vllm_config"]["aggregated"] == {"tensor-parallel-size": 1}


def test_engine_config_key_follows_backend_type() -> None:
    for btype, key in (("sglang", "sglang_config"), ("vllm", "vllm_config"), ("trtllm", "trtllm_config")):
        config = {"backend": {"type": btype}, "roles": {"agg": {"args": {"a": 1}}}}
        expand_roles(config)
        assert config["backend"][key]["aggregated"] == {"a": 1}


def test_trtllm_extra_args_route_to_mode_extra_args() -> None:
    config = {"backend": {"type": "trtllm"}, "roles": {"prefill": {"extra_args": ["--x"]}}}
    expand_roles(config)
    assert config["backend"]["prefill_extra_args"] == ["--x"]


def test_mixing_roles_with_legacy_fields_is_rejected() -> None:
    config = _roles_sglang_disagg()
    config["resources"]["prefill_workers"] = 1
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_roles(config)


def test_unknown_role_and_unknown_spec_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        expand_roles({"backend": {"type": "sglang"}, "roles": {"warmup": {"workers": 1}}})
    with pytest.raises(ValueError, match="unknown keys"):
        expand_roles({"backend": {"type": "sglang"}, "roles": {"prefill": {"gpu": 1}}})


def test_no_roles_block_is_a_no_op() -> None:
    legacy = _legacy_sglang_disagg()
    assert expand_roles(copy.deepcopy(legacy)) == legacy


def test_decode_nodes_zero_shared_node_survives() -> None:
    config = {"backend": {"type": "sglang"}, "roles": {"decode": {"nodes": 0, "workers": 2}}}
    expand_roles(config)
    assert config["resources"]["decode_nodes"] == 0
    assert config["resources"]["decode_workers"] == 2


def test_preflight_topology_reads_expanded_roles(tmp_path) -> None:
    """A roles: recipe passes topology preflight; the check runs post-expansion."""
    import yaml

    from srtctl.core.validation import preflight_config_variants

    recipe = {
        "schema": 2,
        "name": "roles-preflight",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "backend": {"type": "sglang"},
        "roles": {"agg": {"nodes": 1, "workers": 2, "gpus": 1}},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }
    results = preflight_config_variants(yaml.safe_load(yaml.safe_dump(recipe)), cluster_config=None)
    topo_errors = [issue for result in results for issue in result.errors if issue.field == "resources"]
    assert topo_errors == [], topo_errors
