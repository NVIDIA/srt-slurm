# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest

from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.roles import expand_engine, expand_roles
from srtctl.core.schema import SrtConfig


def _internal_sglang_disagg() -> dict:
    """The internal layout the schema reads (what ``roles:`` expands into)."""
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
        "frontend": {"type": "sglang-router", "enable_multiple_frontends": False},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }


def _roles_sglang_disagg() -> dict:
    return {
        "schema": 2,
        "name": "roles-test",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "engine": "sglang",
        "roles": {
            "prefill": {
                "nodes": 2,
                "workers": 6,
                "gpus": 2,
                "env": {"PYTHONUNBUFFERED": "1"},
                "args": {"tensor-parallel-size": 2, "disaggregation-mode": "prefill"},
            },
            "decode": {
                "nodes": "colocate",
                "workers": 2,
                "gpus": 2,
                "env": {"PYTHONUNBUFFERED": "1"},
                "args": {"tensor-parallel-size": 2, "disaggregation-mode": "decode"},
            },
        },
        "frontend": {"type": "sglang-router", "enable_multiple_frontends": False},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }


def test_expand_roles_produces_the_internal_layout() -> None:
    expanded = expand_roles(_roles_sglang_disagg())
    assert "roles" not in expanded
    assert "engine" not in expanded
    assert expanded["resources"] == _internal_sglang_disagg()["resources"]
    assert expanded["backend"] == _internal_sglang_disagg()["backend"]


def test_roles_load_to_the_same_config_as_the_internal_layout() -> None:
    schema = SrtConfig.Schema()
    from_roles = schema.load(resolve_config_with_defaults(_roles_sglang_disagg(), None))
    from_internal = schema.load(_internal_sglang_disagg())
    assert schema.dump(from_roles) == schema.dump(from_internal)


def test_the_internal_layout_is_not_a_recipe() -> None:
    """A recipe that spells the internal fields itself is the pre-2.0 layout and is rejected."""
    with pytest.raises(ValueError, match=r"pre-2\.0 \(v1\) layout: backend, resources\.prefill_nodes"):
        resolve_config_with_defaults(_internal_sglang_disagg(), None)


def test_agg_role_maps_to_aggregated_env_and_config() -> None:
    config = {
        "backend": {"type": "vllm"},
        "roles": {"agg": {"workers": 2, "gpus": 1, "env": {"X": "1"}, "args": {"tensor-parallel-size": 1}}},
    }
    expand_roles(config)
    assert config["resources"] == {"agg_workers": 2, "gpus_per_agg": 1}
    assert config["backend"]["aggregated_environment"] == {"X": "1"}
    assert config["backend"]["vllm_config"]["aggregated"] == {"tensor-parallel-size": 1}


def test_engine_config_key_follows_engine_type() -> None:
    for btype, key in (("sglang", "sglang_config"), ("vllm", "vllm_config"), ("trtllm", "trtllm_config")):
        config = {"engine": btype, "roles": {"agg": {"args": {"a": 1}}}}
        expand_roles(config)
        assert config["backend"][key]["aggregated"] == {"a": 1}


def test_trtllm_extra_args_route_to_mode_extra_args() -> None:
    config = {"engine": "trtllm", "roles": {"prefill": {"extra_args": ["--x"]}}}
    expand_roles(config)
    assert config["backend"]["prefill_extra_args"] == ["--x"]


def test_roles_cannot_overwrite_internal_fields_already_set() -> None:
    config = _roles_sglang_disagg()
    config["resources"]["prefill_workers"] = 1
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_roles(config)


def test_unknown_role_and_unknown_spec_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        expand_roles({"engine": "sglang", "roles": {"warmup": {"workers": 1}}})
    with pytest.raises(ValueError, match="unknown keys"):
        expand_roles({"engine": "sglang", "roles": {"prefill": {"gpu": 1}}})


def test_no_roles_block_is_a_no_op() -> None:
    internal = _internal_sglang_disagg()
    assert expand_roles(copy.deepcopy(internal)) == internal


def test_decode_colocate_expands_to_the_internal_sentinel() -> None:
    config = {
        "engine": "sglang",
        "roles": {
            "prefill": {"nodes": 1, "workers": 1, "gpus": 4},
            "decode": {"nodes": "colocate", "workers": 2, "gpus": 2},
        },
    }
    expand_roles(config)
    assert config["resources"]["decode_nodes"] == 0
    assert config["resources"]["decode_workers"] == 2
    assert config["resources"]["gpus_per_decode"] == 2


def test_colocate_requires_explicit_gpus_on_both_roles() -> None:
    for prefill, decode, missing in (
        ({"nodes": 1, "workers": 1}, {"nodes": "colocate", "workers": 1, "gpus": 2}, "prefill"),
        ({"nodes": 1, "workers": 1, "gpus": 2}, {"nodes": "colocate", "workers": 1}, "decode"),
        ({"nodes": 1, "workers": 1}, {"nodes": "colocate", "workers": 1}, "prefill, decode"),
    ):
        with pytest.raises(ValueError, match=f"explicit gpus: on both prefill and decode \\(missing on {missing}\\)"):
            expand_roles({"engine": "sglang", "roles": {"prefill": prefill, "decode": decode}})


def test_roles_reject_the_bare_zero_and_colocate_outside_decode() -> None:
    with pytest.raises(ValueError, match="nodes: colocate"):
        expand_roles({"engine": "sglang", "roles": {"decode": {"nodes": 0, "workers": 2}}})
    with pytest.raises(ValueError, match="only the decode role can colocate"):
        expand_roles({"engine": "sglang", "roles": {"prefill": {"nodes": "colocate", "workers": 1}}})
    with pytest.raises(ValueError, match="at least 1"):
        expand_roles({"engine": "sglang", "roles": {"prefill": {"nodes": 0, "workers": 1}}})
    with pytest.raises(ValueError, match="positive integer or 'colocate'"):
        expand_roles({"engine": "sglang", "roles": {"decode": {"nodes": "shared", "workers": 1}}})


def _colocated(
    prefill_nodes: int, prefill_workers: int, prefill_gpus: int, decode_workers: int, decode_gpus: int
) -> dict:
    config = _roles_sglang_disagg()
    config["roles"]["prefill"].update({"nodes": prefill_nodes, "workers": prefill_workers, "gpus": prefill_gpus})
    config["roles"]["decode"].update({"workers": decode_workers, "gpus": decode_gpus})
    return resolve_config_with_defaults(config, None)


def test_colocated_decode_that_fits_loads() -> None:
    # 1 node x 8 GPUs: 1 prefill x 4 + 2 decode x 2 = 8
    cfg = SrtConfig.Schema().load(_colocated(1, 1, 4, 2, 2))
    assert cfg.resources.total_nodes == 1
    assert cfg.resources.decode_nodes == 0


def test_colocated_decode_that_oversubscribes_is_rejected_at_load() -> None:
    from marshmallow import ValidationError

    # 1 node x 8 GPUs: 1 prefill x 6 + 1 decode x 4 = 10 > 8
    with pytest.raises(ValidationError, match="do not fit on the prefill nodes.*10 GPU"):
        SrtConfig.Schema().load(_colocated(1, 1, 6, 1, 4))
    # 2 nodes x 8 GPUs: 2 prefill x 5 leave 3 free per node; a 4-GPU decode worker cannot be packed
    with pytest.raises(ValidationError, match="cannot be packed onto the prefill nodes"):
        SrtConfig.Schema().load(_colocated(2, 2, 5, 1, 4))


def test_preflight_topology_reads_expanded_roles(tmp_path) -> None:
    """A roles: recipe passes topology preflight; the check runs post-expansion."""
    import yaml

    from srtctl.core.validation import preflight_config_variants

    recipe = {
        "schema": 2,
        "name": "roles-preflight",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "engine": "sglang",
        "roles": {"agg": {"nodes": 1, "workers": 2, "gpus": 1}},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }
    results = preflight_config_variants(yaml.safe_load(yaml.safe_dump(recipe)), cluster_config=None)
    topo_errors = [issue for result in results for issue in result.errors if issue.field == "resources"]
    assert topo_errors == [], topo_errors


def test_engine_string_and_mapping_map_onto_backend() -> None:
    assert expand_roles({"engine": "vllm"})["backend"] == {"type": "vllm"}
    expanded = expand_roles({"engine": {"type": "trtllm", "served_model_name": "m"}, "roles": {"agg": {"workers": 1}}})
    assert expanded["backend"]["type"] == "trtllm"
    assert expanded["backend"]["served_model_name"] == "m"
    assert "engine" not in expanded
    # roles.<r>.engine may restate the engine, and must agree.
    assert (
        expand_roles({"engine": "sglang", "roles": {"agg": {"engine": "sglang", "workers": 1}}})["backend"]["type"]
        == "sglang"
    )
    assert expand_roles({"roles": {"agg": {"engine": "vllm", "workers": 1}}})["backend"]["type"] == "vllm"
    with pytest.raises(ValueError, match="conflicts with engine.type"):
        expand_roles({"engine": "sglang", "roles": {"agg": {"engine": "vllm"}}})
    with pytest.raises(ValueError, match="same engine"):
        expand_roles({"roles": {"prefill": {"engine": "vllm"}, "decode": {"engine": "sglang"}}})
    with pytest.raises(ValueError, match="conflicts with backend.type"):
        expand_roles({"engine": "vllm", "backend": {"type": "sglang"}})


def test_engine_mapping_rejects_per_role_settings() -> None:
    """The engine block carries engine-wide knobs only; per-role keys have one spelling, under roles."""
    for key, value in (
        ("sglang_config", {"prefill": {"tp": 1}}),
        ("prefill_environment", {"A": "1"}),
        ("decode_extra_args", ["--x"]),
        ("kv_events_config", True),
        ("mooncake_kv_store", {"env": {}}),
    ):
        with pytest.raises(ValueError, match=f"engine: carries per-role settings \\({key}\\)"):
            expand_engine({"engine": {"type": "sglang", key: value}})
    # engine-wide knobs are fine
    assert expand_engine({"engine": {"type": "vllm", "connector": "nixl"}})["backend"] == {
        "type": "vllm",
        "connector": "nixl",
    }


def test_per_role_kv_events_and_sidecar() -> None:
    config = expand_roles(
        {
            "engine": "sglang",
            "roles": {
                "prefill": {"workers": 1, "kv_events": True, "sidecar": True},
                "decode": {"workers": 1, "kv_events": {"publisher": "zmq", "topic": "kv"}, "sidecar": True},
            },
        }
    )
    assert config["backend"]["kv_events_config"] == {"prefill": True, "decode": {"publisher": "zmq", "topic": "kv"}}
    assert config["dynamo"]["sidecar"] is True

    with pytest.raises(ValueError, match="sidecar must agree"):
        expand_roles({"roles": {"prefill": {"sidecar": True}, "decode": {"sidecar": False}}})
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_roles({"backend": {"kv_events_config": True}, "roles": {"prefill": {"kv_events": True}}})
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_roles({"dynamo": {"sidecar": True}, "roles": {"prefill": {"sidecar": True}}})


def test_per_role_critical_maps_onto_resources_and_the_worker_flag() -> None:
    config = expand_roles(
        {
            "engine": "sglang",
            "roles": {"prefill": {"workers": 1, "critical": False}, "decode": {"workers": 1}},
        }
    )
    assert config["resources"]["prefill_critical"] is False
    assert "decode_critical" not in config["resources"]  # default stays implicit

    recipe = _roles_sglang_disagg()
    recipe["roles"]["prefill"]["critical"] = False
    loaded = SrtConfig.Schema().load(resolve_config_with_defaults(recipe, None))
    assert loaded.resources.worker_critical("prefill") is False
    assert loaded.resources.worker_critical("decode") is True
    assert loaded.resources.worker_critical("agg") is True

    with pytest.raises(TypeError, match="critical must be a boolean"):
        expand_roles({"roles": {"decode": {"critical": "no"}}})
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_roles({"resources": {"decode_critical": False}, "roles": {"decode": {"workers": 1}}})
    # In a recipe the internal spelling is rejected outright, before roles: is even looked at.
    recipe = _roles_sglang_disagg()
    recipe["resources"]["decode_critical"] = False
    with pytest.raises(ValueError, match=r"pre-2\.0 \(v1\) layout: resources\.decode_critical"):
        resolve_config_with_defaults(recipe, None)
