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
        "frontend": {"type": "sglang-router", "enable_multiple_frontends": False},
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
    assert "backend" not in as_roles  # the engine moved to the top level
    assert as_roles["engine"] == "sglang"
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


def test_decode_colocate_expands_to_the_v1_sentinel() -> None:
    config = {
        "backend": {"type": "sglang"},
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
            expand_roles({"backend": {"type": "sglang"}, "roles": {"prefill": prefill, "decode": decode}})


def test_roles_reject_the_bare_zero_and_colocate_outside_decode() -> None:
    with pytest.raises(ValueError, match="nodes: colocate"):
        expand_roles({"backend": {"type": "sglang"}, "roles": {"decode": {"nodes": 0, "workers": 2}}})
    with pytest.raises(ValueError, match="only the decode role can colocate"):
        expand_roles({"backend": {"type": "sglang"}, "roles": {"prefill": {"nodes": "colocate", "workers": 1}}})
    with pytest.raises(ValueError, match="at least 1"):
        expand_roles({"backend": {"type": "sglang"}, "roles": {"prefill": {"nodes": 0, "workers": 1}}})
    with pytest.raises(ValueError, match="positive integer or 'colocate'"):
        expand_roles({"backend": {"type": "sglang"}, "roles": {"decode": {"nodes": "shared", "workers": 1}}})


def test_legacy_decode_nodes_zero_migrates_to_colocate() -> None:
    as_roles = roles_from_legacy(_legacy_sglang_disagg())
    assert as_roles["roles"]["decode"]["nodes"] == "colocate"
    assert as_roles["roles"]["prefill"]["nodes"] == 2


def _colocated(
    prefill_nodes: int, prefill_workers: int, prefill_gpus: int, decode_workers: int, decode_gpus: int
) -> dict:
    config = _roles_sglang_disagg()
    config["roles"]["prefill"].update({"nodes": prefill_nodes, "workers": prefill_workers, "gpus": prefill_gpus})
    config["roles"]["decode"].update({"workers": decode_workers, "gpus": decode_gpus})
    return expand_roles(config)


def test_colocated_decode_that_fits_loads() -> None:
    # 1 node x 8 GPUs: 1 prefill x 4 + 2 decode x 2 = 8
    cfg = SrtConfig.Schema().load(_colocated(1, 1, 4, 2, 2))
    assert cfg.resources.total_nodes == 1
    # legacy spelling keeps working
    legacy = _legacy_sglang_disagg()
    legacy["resources"].update({"prefill_nodes": 1, "prefill_workers": 1, "gpus_per_prefill": 4})
    assert SrtConfig.Schema().load(legacy).resources.decode_nodes == 0


def test_colocated_decode_that_oversubscribes_is_rejected_at_load() -> None:
    from marshmallow import ValidationError

    # 1 node x 8 GPUs: 1 prefill x 6 + 1 decode x 4 = 10 > 8
    with pytest.raises(ValidationError, match="do not fit on the prefill nodes.*10 GPU"):
        SrtConfig.Schema().load(_colocated(1, 1, 6, 1, 4))
    # 2 nodes x 8 GPUs: 2 prefill x 5 leave 3 free per node; a 4-GPU decode worker cannot be packed
    with pytest.raises(ValidationError, match="cannot be packed onto the prefill nodes"):
        SrtConfig.Schema().load(_colocated(2, 2, 5, 1, 4))
    # the same layout with legacy fields is rejected too (it would die in the job otherwise)
    legacy = _legacy_sglang_disagg()
    legacy["resources"].update(
        {"prefill_nodes": 1, "prefill_workers": 1, "gpus_per_prefill": 6, "gpus_per_decode": 4, "decode_workers": 1}
    )
    with pytest.raises(ValidationError, match="do not fit"):
        SrtConfig.Schema().load(legacy)


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


def test_roles_from_legacy_folds_kv_events_and_sidecar() -> None:
    legacy = {
        "backend": {"type": "vllm", "kv_events_config": True, "connector": "nixl"},
        "resources": {"prefill_workers": 1, "decode_workers": 1, "agg_workers": 0},
        "dynamo": {"sidecar": True, "sidecar_port": 50051},
    }
    folded = roles_from_legacy(legacy)
    assert folded["engine"] == {"type": "vllm", "connector": "nixl"}
    assert folded["roles"]["prefill"] == {"workers": 1, "kv_events": True, "sidecar": True}
    assert folded["roles"]["decode"] == {"workers": 1, "kv_events": True, "sidecar": True}
    assert folded["roles"]["agg"] == {"workers": 0, "sidecar": True}  # vLLM's bare `true` never covered agg
    assert folded["dynamo"] == {"sidecar_port": 50051}
    assert "backend" not in folded


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
    loaded = SrtConfig.Schema().load(expand_roles(recipe))
    assert loaded.resources.worker_critical("prefill") is False
    assert loaded.resources.worker_critical("decode") is True
    assert loaded.resources.worker_critical("agg") is True

    with pytest.raises(TypeError, match="critical must be a boolean"):
        expand_roles({"roles": {"decode": {"critical": "no"}}})
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_roles({"resources": {"decode_critical": False}, "roles": {"decode": {"workers": 1}}})


def test_roles_from_legacy_folds_critical() -> None:
    legacy = {
        "backend": {"type": "sglang"},
        "resources": {"prefill_workers": 1, "decode_workers": 1, "decode_critical": False},
    }
    folded = roles_from_legacy(legacy)
    assert folded["roles"]["decode"] == {"workers": 1, "critical": False}
    assert folded["roles"]["prefill"] == {"workers": 1}
    assert "resources" not in folded
    assert expand_roles(copy.deepcopy(folded))["resources"] == legacy["resources"]
