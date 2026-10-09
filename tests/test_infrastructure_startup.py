# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure readiness includes native container import latency."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from marshmallow import ValidationError

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core import health
from srtctl.core.schema import InfraConfig


def test_infrastructure_can_become_ready_after_default_import_window(monkeypatch, tmp_path):
    clock = [0.0]
    monkeypatch.setattr(health.time, "time", lambda: clock[0])
    monkeypatch.setattr(health.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def connect(address, timeout):
        if clock[0] < 301:
            raise ConnectionRefusedError
        return MagicMock()

    monkeypatch.setattr(health.socket, "create_connection", connect)
    monkeypatch.setattr("srtctl.cli.do_sweep.start_srun_process", lambda **kwargs: MagicMock())
    config = SimpleNamespace(
        name="cold-image", infra=SimpleNamespace(nats_max_payload_mb=None, startup_timeout_seconds=600)
    )
    runtime = SimpleNamespace(
        nodes=SimpleNamespace(infra="node0", het_group_for=lambda node: None),
        log_dir=tmp_path,
        container_mounts={},
        container_image="registry#backend:sha256:exact-image",
    )
    managed = SweepOrchestrator(config, runtime).start_head_infrastructure(MagicMock())
    assert managed.name == "infra_services"
    assert clock[0] == 301


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_infrastructure_rejects_unbounded_or_nonpositive_wait(timeout):
    with pytest.raises(ValidationError):
        InfraConfig.Schema().load({"startup_timeout_seconds": timeout})


def test_infrastructure_readiness_recipe_round_trip():
    default = InfraConfig.Schema().load({})
    configured = InfraConfig.Schema().load({"startup_timeout_seconds": 9000})
    assert default.startup_timeout_seconds == 300
    assert InfraConfig.Schema().dump(configured)["startup_timeout_seconds"] == 9000
