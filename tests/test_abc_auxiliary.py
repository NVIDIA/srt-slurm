# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nominal role settings preserve recipe data; log parsers enforce their hooks."""

from dataclasses import FrozenInstanceError

import pytest

from srtctl.backends.base import RoleSettings, role_args, role_env, role_kv_events
from srtctl.core.schema import RoleConfig
from srtctl.dsight.log_metrics import GENERATORS, LogMetricGenerator


def test_role_settings_inheritance_preserves_recipe_round_trip():
    raw = {
        "nodes": 1,
        "workers": 2,
        "env": {"ENGINE_LOG_LEVEL": "debug"},
        "args": {"tensor-parallel-size": 4},
        "extra_args": ["--verbose"],
        "kv_events": {"topic": "worker-events"},
    }
    role = RoleConfig.Schema().load(raw)
    assert isinstance(role, RoleSettings)
    assert RoleConfig.Schema().load(RoleConfig.Schema().dump(role)) == role
    assert role_env({"decode": role}, "decode") == raw["env"]
    assert role_args({"decode": role}, "decode") == raw["args"]
    assert role_kv_events({"decode": role}, "decode", {"publisher": "zmq"}) == {
        "publisher": "zmq",
        "topic": "worker-events",
    }
    assert role.extra_args == raw["extra_args"]
    with pytest.raises(FrozenInstanceError):
        role.nodes = 3


def test_role_settings_defaults_remain_independent():
    first, second = RoleConfig(), RoleConfig()
    first.env["TEST"] = "first"
    first.args["test"] = True
    first.extra_args.append("--test")
    assert second.env == {}
    assert second.args == {}
    assert second.extra_args == []
    assert second.kv_events is None


@pytest.mark.parametrize("missing", ["name", "definitions", "parse_line"])
def test_log_generator_rejects_missing_members(missing):
    members = {"name": "incomplete", "definitions": (), "parse_line": lambda self, line, source: None}
    del members[missing]
    incomplete = type("IncompleteGenerator", (LogMetricGenerator,), members)
    with pytest.raises(TypeError, match=missing):
        incomplete()


def test_registered_log_generators_are_concrete_subclasses():
    assert GENERATORS
    for generator in GENERATORS:
        assert isinstance(generator, LogMetricGenerator)
        assert generator.name
        assert generator.definitions
