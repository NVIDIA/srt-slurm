# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend registration requires concrete implementations of the inherited contract."""

from types import SimpleNamespace
from typing import Any

import pytest

from srtctl.frontends import DynamicFrontend, Frontend, get_frontend, list_frontend_types, register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend


@pytest.mark.parametrize("base_class", [Frontend, DynamicFrontend, StaticRouterFrontend])
def test_frontend_base_classes_cannot_be_instantiated(base_class: Any):
    with pytest.raises(TypeError, match="abstract"):
        base_class()


def test_registry_rejects_a_class_without_frontend_inheritance():
    class DuckFrontend:
        type = "duck-frontend"

    # Exercise runtime validation at an untyped plugin boundary.
    implementation: Any = DuckFrontend
    with pytest.raises(TypeError, match="must inherit from Frontend"):
        register_frontend("duck-frontend")(implementation)
    assert "duck-frontend" not in list_frontend_types()


def test_registry_rejects_incomplete_frontend_before_construction():
    class IncompleteFrontend(Frontend):
        type = "incomplete-frontend"

    with pytest.raises(TypeError, match="must implement all abstract methods"):
        register_frontend("incomplete-frontend")(IncompleteFrontend)
    assert "incomplete-frontend" not in list_frontend_types()


def test_static_router_requires_its_launch_configuration():
    class IncompleteRouter(StaticRouterFrontend):
        type = "incomplete-router"

    implementation: Any = IncompleteRouter
    with pytest.raises(TypeError, match="executable.*pd_flag.*process_name"):
        implementation()
    with pytest.raises(TypeError, match="must implement all abstract methods"):
        register_frontend("incomplete-router")(IncompleteRouter)


@pytest.mark.parametrize("name", [name for name in list_frontend_types() if name != "none"])
def test_every_registered_frontend_is_a_concrete_subclass(name):
    frontend = get_frontend(name)
    assert isinstance(frontend, Frontend)
    assert frontend.type == name


def test_router_inherits_optional_defaults_and_decorator_preserves_class(monkeypatch):
    from srtctl.frontends import base

    monkeypatch.setattr(base, "_FRONTENDS", dict(base._FRONTENDS))

    class ToyRouter(StaticRouterFrontend):
        type = "abc-toy-router"
        executable = ("toy-router",)
        pd_flag = "--disaggregated"
        process_name = "toy-router"

    assert register_frontend("abc-toy-router")(ToyRouter) is ToyRouter
    frontend = get_frontend("abc-toy-router")
    assert isinstance(frontend, ToyRouter)
    config = SimpleNamespace(topology=SimpleNamespace(num_agg=0, num_prefill=2, num_decode=3))
    assert frontend.required_backend is None
    assert frontend.model_name_role is None
    assert frontend.worker_launch == "direct"
    assert frontend.metrics_path == "/metrics"
    assert frontend.expands_node_local_dp is False
    assert frontend.validate(config) is None
    assert frontend.implied_services(config) == []
    assert frontend.frontend_metrics_port(None) is None
    assert frontend.direct_endpoint_nodes([]) == []
    assert frontend.profiling_control_is_leader_only(config) is False
    assert frontend.health_expectations(config, None) == (2, 3, "2P + 3D")


def test_legacy_frontend_name_aliases_the_abstract_base_class():
    from srtctl.frontends import FrontendProtocol
    from srtctl.frontends.base import FrontendProtocol as BaseFrontendProtocol

    assert FrontendProtocol is Frontend
    assert BaseFrontendProtocol is Frontend
