# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``roles:`` loads into ``SrtConfig.roles``; the topology and the engine's per-mode settings derive from it."""

from __future__ import annotations

import pytest
from marshmallow import ValidationError

from srtctl.backends import SGLangBackend
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import RestartPolicy, RoleConfig, SrtConfig


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


def _minimal(**extra) -> dict:
    """The smallest schema document around ``extra`` (no gate, no cluster defaults)."""
    return {
        "name": "roles-test",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        **extra,
    }


def _load(recipe: dict) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(recipe, None))


def test_roles_load_as_role_configs_and_bind_the_engine() -> None:
    cfg = _load(_roles_sglang_disagg())
    assert set(cfg.roles) == {"prefill", "decode"}
    assert cfg.roles["prefill"] == RoleConfig(
        nodes=2,
        workers=6,
        gpus=2,
        env={"PYTHONUNBUFFERED": "1"},
        args={"tensor-parallel-size": 2, "disaggregation-mode": "prefill"},
    )
    assert cfg.roles["decode"].colocated
    assert isinstance(cfg.engine, SGLangBackend)

    topology = cfg.topology
    assert (topology.num_prefill, topology.num_decode) == (6, 2)
    assert (topology.gpus_per_prefill, topology.gpus_per_decode) == (2, 2)
    assert (topology.prefill_nodes, topology.decode_nodes, topology.total_nodes) == (2, 0, 2)
    assert topology.is_disaggregated and topology.colocated_decode

    # The engine's per-mode fields carry each role's env and args.
    assert cfg.backend.get_config_for_mode("prefill") == {"tensor-parallel-size": 2, "disaggregation-mode": "prefill"}
    assert cfg.backend.get_config_for_mode("decode") == {"tensor-parallel-size": 2, "disaggregation-mode": "decode"}
    assert cfg.backend.get_config_for_mode("agg") == {}
    assert cfg.backend.get_environment_for_mode("decode") == {"PYTHONUNBUFFERED": "1"}
    assert cfg.backend.get_environment_for_mode("agg") == {}


def test_a_dump_is_a_recipe_that_loads_back_to_the_same_config() -> None:
    schema = SrtConfig.Schema()
    cfg = _load(_roles_sglang_disagg())
    dumped = schema.dump(cfg)
    assert set(dumped["roles"]) == {"prefill", "decode"}
    assert "backend" not in dumped
    assert "roles" not in dumped["engine"]  # the bound roles are not engine fields; roles: carries them
    reloaded = schema.load(dumped)
    assert schema.dump(reloaded) == dumped
    assert reloaded.backend.get_config_for_mode("prefill") == cfg.backend.get_config_for_mode("prefill")


def test_the_pre_2_0_layout_is_not_a_recipe() -> None:
    """A recipe that spells the pre-2.0 fields is rejected by the gate, before the schema sees it."""
    v1 = _minimal(
        schema=2,
        resources={"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 2, "prefill_workers": 6},
        backend={"type": "sglang", "sglang_config": {"prefill": {"tensor-parallel-size": 2}}},
    )
    with pytest.raises(ValueError, match=r"pre-2\.0 \(v1\) layout: backend, resources\.prefill_nodes"):
        resolve_config_with_defaults(v1, None)
    # And the schema itself has no such fields.
    with pytest.raises(ValidationError, match="Unknown field"):
        SrtConfig.Schema().load(_minimal(resources={"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 2}))


def test_agg_role_binds_to_the_aggregated_mode() -> None:
    cfg = SrtConfig.Schema().load(
        _minimal(
            engine="vllm",
            roles={
                "agg": {"nodes": 1, "workers": 2, "gpus": 1, "env": {"X": "1"}, "args": {"tensor-parallel-size": 1}}
            },
        )
    )
    assert (cfg.topology.num_agg, cfg.topology.gpus_per_agg, cfg.topology.agg_nodes) == (2, 1, 1)
    assert not cfg.topology.is_disaggregated
    assert cfg.backend.get_environment_for_mode("agg") == {"X": "1"}
    assert cfg.backend.get_config_for_mode("agg") == {"tensor-parallel-size": 1}


def test_role_args_reach_every_engine() -> None:
    for engine_type in ("sglang", "vllm", "trtllm", "mocker", "atom", "tilert", "tokenspeed"):
        cfg = SrtConfig.Schema().load(
            _minimal(engine=engine_type, roles={"agg": {"nodes": 1, "workers": 1, "args": {"a": 1}, "env": {"E": "1"}}})
        )
        assert list(cfg.backend.roles) == ["agg"]
        assert cfg.backend.get_config_for_mode("agg") == {"a": 1}
        assert cfg.backend.get_environment_for_mode("agg")["E"] == "1"  # trtllm adds its own EPLB variable
        assert cfg.backend.get_config_for_mode("prefill") == {}


def test_extra_args_are_trtllm_only() -> None:
    cfg = SrtConfig.Schema().load(
        _minimal(engine="trtllm", roles={"prefill": {"nodes": 1, "workers": 1, "extra_args": ["--x"]}})
    )
    assert cfg.backend.get_extra_args_for_mode("prefill") == ["--x"]
    with pytest.raises(ValidationError, match="extra_args is only supported by the trtllm engine"):
        SrtConfig.Schema().load(
            _minimal(engine="sglang", roles={"prefill": {"nodes": 1, "workers": 1, "extra_args": ["--x"]}})
        )


def test_unknown_role_and_unknown_spec_key_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown role"):
        SrtConfig.Schema().load(_minimal(engine="sglang", roles={"warmup": {"workers": 1}}))
    with pytest.raises(ValidationError, match="Unknown field"):
        SrtConfig.Schema().load(_minimal(engine="sglang", roles={"prefill": {"gpu": 1}}))


def test_decode_colocate_reserves_no_nodes() -> None:
    cfg = SrtConfig.Schema().load(
        _minimal(
            engine="sglang",
            roles={
                "prefill": {"nodes": 1, "workers": 1, "gpus": 4},
                "decode": {"nodes": "colocate", "workers": 2, "gpus": 2},
            },
        )
    )
    assert cfg.roles["decode"].node_count == 0
    assert (cfg.topology.decode_nodes, cfg.topology.num_decode, cfg.topology.gpus_per_decode) == (0, 2, 2)
    assert cfg.topology.total_nodes == 1


def test_colocate_requires_explicit_gpus_on_both_roles() -> None:
    for prefill, decode, missing in (
        ({"nodes": 1, "workers": 1}, {"nodes": "colocate", "workers": 1, "gpus": 2}, "prefill"),
        ({"nodes": 1, "workers": 1, "gpus": 2}, {"nodes": "colocate", "workers": 1}, "decode"),
        ({"nodes": 1, "workers": 1}, {"nodes": "colocate", "workers": 1}, "prefill, decode"),
    ):
        with pytest.raises(
            ValidationError, match=f"explicit gpus: on both prefill and decode \\(missing on {missing}\\)"
        ):
            SrtConfig.Schema().load(_minimal(engine="sglang", roles={"prefill": prefill, "decode": decode}))


def test_roles_reject_the_bare_zero_and_colocate_outside_decode() -> None:
    with pytest.raises(ValidationError, match="nodes: colocate"):
        SrtConfig.Schema().load(_minimal(engine="sglang", roles={"decode": {"nodes": 0, "workers": 2}}))
    with pytest.raises(ValidationError, match="only the decode role can colocate"):
        SrtConfig.Schema().load(_minimal(engine="sglang", roles={"prefill": {"nodes": "colocate", "workers": 1}}))
    with pytest.raises(ValidationError, match="at least 1"):
        SrtConfig.Schema().load(_minimal(engine="sglang", roles={"prefill": {"nodes": -1, "workers": 1}}))
    with pytest.raises(ValidationError, match="at least 1"):
        SrtConfig.Schema().load(_minimal(engine="sglang", roles={"prefill": {"nodes": 0, "workers": 1}}))
    with pytest.raises(ValidationError, match="colocate"):
        SrtConfig.Schema().load(_minimal(engine="sglang", roles={"decode": {"nodes": "shared", "workers": 1}}))
    with pytest.raises(ValidationError, match="positive integer or 'colocate'"):
        RoleConfig(nodes="shared")


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
    assert cfg.topology.total_nodes == 1
    assert cfg.topology.decode_nodes == 0


def test_colocated_decode_that_oversubscribes_is_rejected_at_load() -> None:
    # 1 node x 8 GPUs: 1 prefill x 6 + 1 decode x 4 = 10 > 8
    with pytest.raises(ValidationError, match="do not fit on the prefill nodes.*10 GPU"):
        SrtConfig.Schema().load(_colocated(1, 1, 6, 1, 4))
    # 2 nodes x 8 GPUs: 2 prefill x 5 leave 3 free per node; a 4-GPU decode worker cannot be packed
    with pytest.raises(ValidationError, match="cannot be packed onto the prefill nodes"):
        SrtConfig.Schema().load(_colocated(2, 2, 5, 1, 4))


def test_preflight_topology_reads_roles(tmp_path) -> None:
    """A roles: recipe passes topology preflight."""
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
    topo_errors = [issue for result in results for issue in result.errors if issue.field.startswith("roles")]
    assert topo_errors == [], topo_errors


def test_engine_string_and_mapping_and_per_role_engines() -> None:
    assert (
        SrtConfig.Schema().load(_minimal(engine="vllm", roles={"agg": {"nodes": 1, "workers": 1}})).backend.type
        == "vllm"
    )
    cfg = SrtConfig.Schema().load(
        _minimal(engine={"type": "trtllm", "served_model_name": "m"}, roles={"agg": {"nodes": 1, "workers": 1}})
    )
    assert cfg.engine is not None and cfg.engine.type == "trtllm" and cfg.engine.served_model_name == "m"
    assert cfg.backend.served_model_name == "m"

    # No engine at all: SGLang.
    assert SrtConfig.Schema().load(_minimal(roles={"agg": {"nodes": 1, "workers": 1}})).backend.type == "sglang"
    assert not SrtConfig.Schema().load(_minimal(roles={"agg": {"nodes": 1, "workers": 1}})).has_role_backends

    # Every role declares its own engine; the serving role's engine is the job's backend.
    per_role = _minimal(
        roles={
            "prefill": {"engine": "sglang", "nodes": 1, "workers": 1, "gpus": 8, "args": {"tp-size": 8}},
            "decode": {"engine": {"type": "sglang"}, "nodes": 1, "workers": 1, "gpus": 8},
        },
        frontend={"type": "sglang-router", "enable_multiple_frontends": False},
    )
    cfg = SrtConfig.Schema().load(per_role)
    assert cfg.engine is None
    assert set(cfg.role_backends) == {"prefill", "decode"}
    assert cfg.backend is cfg.role_backends["decode"]
    assert cfg.backend_for_role("prefill").get_config_for_mode("prefill") == {"tp-size": 8}
    assert cfg.backend_for_role("decode").get_config_for_mode("prefill") == {}

    with pytest.raises(ValidationError, match="cannot be combined with a top-level engine"):
        SrtConfig.Schema().load(_minimal(engine="vllm", roles={"agg": {"engine": "sglang", "nodes": 1, "workers": 1}}))
    with pytest.raises(ValidationError, match="roles.decode.engine must name a type when no top-level engine is set"):
        SrtConfig.Schema().load(
            _minimal(
                roles={"prefill": {"engine": "sglang", "nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}}
            )
        )


def test_engine_mapping_rejects_per_role_settings() -> None:
    """The engine block carries engine-wide knobs only; per-role keys have one spelling, under roles."""
    for key, value in (
        ("sglang_config", {"prefill": {"tp": 1}}),
        ("prefill_environment", {"A": "1"}),
        ("decode_extra_args", ["--x"]),
        ("kv_events_config", True),
    ):
        with pytest.raises(ValidationError, match=f"per-role settings \\({key}\\)"):
            SrtConfig.Schema().load(
                _minimal(engine={"type": "sglang", key: value}, roles={"agg": {"nodes": 1, "workers": 1}})
            )
        with pytest.raises(ValidationError, match=f"per-role settings \\({key}\\)"):
            SrtConfig.Schema().load(
                _minimal(roles={"agg": {"engine": {"type": "sglang", key: value}, "nodes": 1, "workers": 1}})
            )
    # engine-wide knobs are fine
    cfg = SrtConfig.Schema().load(
        _minimal(engine={"type": "vllm", "connector": "nixl"}, roles={"agg": {"nodes": 1, "workers": 1}})
    )
    assert cfg.engine is not None and cfg.engine.connector == "nixl"


def test_per_role_kv_events_and_sidecar() -> None:
    cfg = SrtConfig.Schema().load(
        _minimal(
            engine="sglang",
            roles={
                "prefill": {"nodes": 1, "workers": 1, "kv_events": True, "sidecar": True},
                "decode": {"nodes": 1, "workers": 1, "kv_events": {"publisher": "zmq", "topic": "kv"}, "sidecar": True},
            },
            frontend={"type": "dynamo"},
        )
    )
    assert cfg.backend.get_kv_events_config_for_mode("prefill") == {"publisher": "zmq", "topic": "kv-events"}
    assert cfg.backend.get_kv_events_config_for_mode("decode") == {"publisher": "zmq", "topic": "kv"}
    assert cfg.backend.get_kv_events_config_for_mode("agg") is None
    assert cfg.dynamo.sidecar is True

    with pytest.raises(ValidationError, match="sidecar must agree"):
        SrtConfig.Schema().load(
            _minimal(
                engine="sglang",
                roles={
                    "prefill": {"nodes": 1, "workers": 1, "sidecar": True},
                    "decode": {"nodes": 1, "workers": 1, "sidecar": False},
                },
                frontend={"type": "dynamo"},
            )
        )
    with pytest.raises(ValidationError, match="disagrees with dynamo.sidecar: true"):
        SrtConfig.Schema().load(
            _minimal(
                engine="sglang",
                roles={"agg": {"nodes": 1, "workers": 1, "sidecar": False}},
                dynamo={"sidecar": True},
                frontend={"type": "dynamo"},
            )
        )
    with pytest.raises(ValidationError, match="roles.agg.kv_events is not supported by the trtllm engine"):
        SrtConfig.Schema().load(_minimal(engine="trtllm", roles={"agg": {"nodes": 1, "workers": 1, "kv_events": True}}))
    # The engine's roles are bound from the recipe; an engine constructed with roles of its own is refused.
    with pytest.raises(ValidationError, match="engine.roles is bound from the recipe's roles block"):
        SrtConfig(
            name="k",
            model={"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
            resources={"gpu_type": "h100", "gpus_per_node": 8},
            engine=SGLangBackend(roles={"agg": RoleConfig(kv_events=True)}),
            roles={"agg": RoleConfig(nodes=1, workers=1)},
        )


def test_per_role_critical_feeds_the_worker_flag() -> None:
    recipe = _roles_sglang_disagg()
    recipe["roles"]["prefill"]["critical"] = False
    loaded = _load(recipe)
    assert loaded.roles["prefill"].critical is False
    assert loaded.topology.worker_critical("prefill") is False
    assert loaded.topology.worker_critical("decode") is True
    assert loaded.topology.worker_critical("agg") is True  # an undeclared role is critical by default

    with pytest.raises(ValidationError, match="Not a valid boolean"):
        SrtConfig.Schema().load(
            _minimal(engine="sglang", roles={"decode": {"nodes": 1, "workers": 1, "critical": "no"}})
        )
    # In a recipe the pre-2.0 spelling is rejected outright, before roles: is even looked at.
    recipe = _roles_sglang_disagg()
    recipe["resources"]["decode_critical"] = False
    with pytest.raises(ValueError, match=r"pre-2\.0 \(v1\) layout: resources\.decode_critical"):
        resolve_config_with_defaults(recipe, None)


@pytest.mark.parametrize("policy", ["never", "on-failure", "always"])
def test_role_restart_shorthand_and_mapping_are_equivalent(policy: str) -> None:
    recipe = _roles_sglang_disagg()
    recipe["roles"]["prefill"]["restart"] = policy
    recipe["roles"]["decode"]["restart"] = {"policy": policy}
    loaded = _load(recipe)
    expected = RestartPolicy(policy=policy)
    assert loaded.roles["prefill"].restart == expected
    assert loaded.roles["decode"].restart == expected
    assert loaded.topology.worker_restart("prefill") == expected
    assert loaded.topology.worker_restart("agg") == RestartPolicy()
    dumped = SrtConfig.Schema().dump(loaded)
    assert dumped["roles"]["prefill"]["restart"]["policy"] == policy
    assert SrtConfig.Schema().load(dumped).roles == loaded.roles


def test_role_restart_options_and_default() -> None:
    recipe = _roles_sglang_disagg()
    recipe["roles"]["decode"]["restart"] = {
        "policy": "on-failure",
        "max_restarts": 5,
        "backoff_seconds": 2,
        "max_backoff_seconds": 20,
    }
    loaded = _load(recipe)
    assert loaded.topology.worker_restart("decode") == RestartPolicy(
        policy="on-failure", max_restarts=5, backoff_seconds=2, max_backoff_seconds=20
    )
    assert loaded.topology.worker_restart("prefill") == RestartPolicy()
    assert "decode_restart" not in SrtConfig.Schema().dump(loaded)["resources"]


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (3, "restart must be a policy name or a mapping"),
        ([], "restart must be a policy name or a mapping"),
        (None, "Field may not be null"),
        ("sometimes", "Must be one of: never, on-failure, always"),
        ({"max_restarts": -1}, "restart.max_restarts"),
        ({"backoff_seconds": -1}, "restart.backoff_seconds"),
        ({"backoff_seconds": 10, "max_backoff_seconds": 5}, "restart.max_backoff_seconds"),
        ({"typo": 1}, "Unknown field"),
    ],
)
def test_role_restart_rejects_invalid_config(value, error: str) -> None:
    recipe = _roles_sglang_disagg()
    recipe["roles"]["decode"]["restart"] = value
    with pytest.raises(ValidationError, match=error):
        _load(recipe)
