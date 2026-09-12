# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auxiliary_services: generic sidecar processes declared in a recipe."""

import tempfile
from pathlib import Path

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.core.schema import AuxiliaryServiceConfig, AuxiliaryServiceSourceConfig, SrtConfig

BASE_RECIPE = {
    "name": "auxiliary-services-test",
    "model": {"path": "/models/test", "container": "test.sqsh", "precision": "fp8"},
    "resources": {
        "gpu_type": "gb200",
        "gpus_per_node": 4,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
}

ROUTER_SERVICE = {
    "name": "thunderagent-router",
    "command": ["python3", "-m", "dynamo.thunderagent_router", "--endpoint", "dyn://ns.comp.ep"],
}


def _load_recipe(overrides: dict | None = None) -> SrtConfig:
    data = {**BASE_RECIPE, **(overrides or {})}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        return SrtConfig.from_yaml(Path(f.name))


class TestSchema:
    def test_defaults_to_empty(self):
        config = _load_recipe()
        assert config.auxiliary_services == []

    def test_loads_a_service_from_the_recipe(self):
        config = _load_recipe({"auxiliary_services": [ROUTER_SERVICE]})
        assert len(config.auxiliary_services) == 1
        service = config.auxiliary_services[0]
        assert service.name == "thunderagent-router"
        assert service.command == ROUTER_SERVICE["command"]
        assert service.inherit_discovery_env is True
        assert service.env == {}
        assert service.source is None

    def test_loads_env_and_container_image(self):
        config = _load_recipe(
            {
                "auxiliary_services": [
                    {
                        **ROUTER_SERVICE,
                        "container_image": "my-image.sqsh",
                        "env": {"ROUTER_LOG_LEVEL": "debug"},
                        "inherit_discovery_env": False,
                    }
                ]
            }
        )
        service = config.auxiliary_services[0]
        assert service.container_image == "my-image.sqsh"
        assert service.env == {"ROUTER_LOG_LEVEL": "debug"}
        assert service.inherit_discovery_env is False

    def test_loads_source_and_build_command(self):
        config = _load_recipe(
            {
                "auxiliary_services": [
                    {
                        **ROUTER_SERVICE,
                        "source": {
                            "git": "https://github.com/ai-dynamo/dynamo",
                            "rev": "refs/pull/14000/head",
                        },
                        "build_command": ["bash", "-lc", "maturin develop --uv && pip install -e ."],
                    }
                ]
            }
        )
        service = config.auxiliary_services[0]
        assert service.source.git == "https://github.com/ai-dynamo/dynamo"
        assert service.source.rev == "refs/pull/14000/head"
        assert service.source.path is None
        assert service.build_command == ["bash", "-lc", "maturin develop --uv && pip install -e ."]

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError, match="name"):
            _load_recipe({"auxiliary_services": [{**ROUTER_SERVICE, "name": "  "}]})

    def test_rejects_empty_command(self):
        with pytest.raises(ValidationError, match="command"):
            _load_recipe({"auxiliary_services": [{**ROUTER_SERVICE, "command": []}]})

    def test_rejects_blank_command_argument(self):
        with pytest.raises(ValidationError, match="command"):
            _load_recipe({"auxiliary_services": [{**ROUTER_SERVICE, "command": ["python3", "   "]}]})

    def test_rejects_duplicate_names(self):
        with pytest.raises(ValidationError, match="unique"):
            _load_recipe({"auxiliary_services": [ROUTER_SERVICE, ROUTER_SERVICE]})

    def test_loads_services_in_declared_order(self):
        first = {"name": "svc-a", "command": ["true"]}
        second = {"name": "svc-b", "command": ["true"]}
        config = _load_recipe({"auxiliary_services": [first, second]})
        assert [service.name for service in config.auxiliary_services] == ["svc-a", "svc-b"]

    def test_warns_when_source_set_without_build_command(self, caplog):
        service = {
            **ROUTER_SERVICE,
            "source": {"git": "https://github.com/ai-dynamo/dynamo", "rev": "refs/pull/14000/head"},
        }
        _load_recipe({"auxiliary_services": [service]})
        assert any("build_command" in record.message for record in caplog.records)

    def test_source_rejects_moving_branch_name(self):
        service = {**ROUTER_SERVICE, "source": {"git": "https://github.com/ai-dynamo/dynamo", "rev": "main"}}
        with pytest.raises(ValidationError, match="immutable"):
            _load_recipe({"auxiliary_services": [service]})

    def test_source_rejects_empty_git(self):
        service = {**ROUTER_SERVICE, "source": {"git": "  ", "rev": "abc123"}}
        with pytest.raises(ValidationError, match="git"):
            _load_recipe({"auxiliary_services": [service]})


class TestConstruction:
    """Direct dataclass construction (bypassing YAML) still validates."""

    def test_build_command_must_be_nonempty_when_set(self):
        with pytest.raises(ValidationError, match="build_command"):
            AuxiliaryServiceConfig(name="svc", command=["true"], build_command=[])

    def test_source_config_requires_nonempty_rev(self):
        with pytest.raises(ValidationError, match="rev"):
            AuxiliaryServiceSourceConfig(git="https://example.com/repo", rev="  ")
