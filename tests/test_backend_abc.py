# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime enforcement and inherited behavior of the backend base class."""

from abc import abstractmethod
from dataclasses import FrozenInstanceError, replace

import pytest

from srtctl.backends import (
    AtomBackend,
    Backend,
    MockerBackend,
    SGLangBackend,
    TileRTBackend,
    TRTLLMBackend,
    VLLMBackend,
)
from srtctl.core.schema import RoleConfig

BACKENDS = (AtomBackend, SGLangBackend, TileRTBackend, TRTLLMBackend, VLLMBackend, MockerBackend)


def test_incomplete_backend_cannot_be_instantiated():
    class IncompleteBackend(Backend):
        type = "incomplete"

    with pytest.raises(TypeError, match="abstract"):
        IncompleteBackend()


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_backend_schema_loads_concrete_frozen_subclass(backend_cls):
    backend = backend_cls.Schema().load({})
    assert isinstance(backend, Backend)
    assert type(backend) is backend_cls
    # Bound roles are supplied by SrtConfig, never deserialized from engine settings.
    dumped = backend_cls.Schema(exclude=("roles",)).dump(backend)
    assert backend_cls.Schema().load(dumped) == backend
    with pytest.raises(FrozenInstanceError):
        backend.type = "changed"


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_new_required_hook_prevents_incomplete_subclass_instantiation(backend_cls):
    class ExtendedBackend(backend_cls):
        @abstractmethod
        def required_hook(self):
            raise NotImplementedError

    with pytest.raises(TypeError, match="required_hook"):
        ExtendedBackend()


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_inherited_role_arguments_are_independent_copies(backend_cls):
    backend = replace(backend_cls(), roles={"agg": RoleConfig(args={"max-tokens": 64}, env={"TEST": "value"})})
    args = backend.get_config_for_mode("agg")
    args["max-tokens"] = 128
    assert backend.get_config_for_mode("agg") == {"max-tokens": 64}
    env = backend.get_environment_for_mode("agg")
    env["TEST"] = "changed"
    assert backend.get_environment_for_mode("agg")["TEST"] == "value"
    assert backend.get_config_for_mode("prefill") == {}
