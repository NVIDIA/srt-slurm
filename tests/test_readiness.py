# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the readiness probe vocabulary and the generic wait loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.core.readiness import ProcessDied, probe_log, run_probe, wait_until_ready
from srtctl.services import HttpProbe, LogProbe, ServiceReadinessConfig, TcpProbe


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


# --- schema ---------------------------------------------------------------------


def test_port_is_shorthand_for_tcp() -> None:
    ready = ServiceReadinessConfig.Schema().load({"port": 9911})
    assert isinstance(ready.probe, TcpProbe) and ready.probe.port == 9911
    assert ready.probe_port == 9911
    assert ready.describe() == "tcp/9911, timeout=120s"


def test_http_and_log_probes() -> None:
    http = ServiceReadinessConfig.Schema().load(
        {"http": {"port": 8000, "path": "/ready", "status": 204}, "timeout_seconds": 30}
    )
    assert isinstance(http.probe, HttpProbe)
    assert http.probe_port == 8000
    assert http.describe() == "http://<node>:8000/ready -> 204, timeout=30s"

    log = ServiceReadinessConfig.Schema().load({"log": {"pattern": r"Uvicorn running on .*:\d+"}})
    assert isinstance(log.probe, LogProbe)
    assert log.probe_port is None
    assert log.describe().startswith("log matches /Uvicorn")


def test_exactly_one_probe() -> None:
    with pytest.raises(ValidationError, match="exactly one probe"):
        ServiceReadinessConfig.Schema().load({})
    with pytest.raises(ValidationError, match="exactly one probe"):
        ServiceReadinessConfig.Schema().load({"tcp": {"port": 1}, "log": {"pattern": "x"}})
    with pytest.raises(ValidationError, match="either port .* or tcp"):
        ServiceReadinessConfig.Schema().load({"port": 1, "tcp": {"port": 2}})
    with pytest.raises(ValidationError, match="valid regular expression"):
        ServiceReadinessConfig.Schema().load({"log": {"pattern": "("}})
    with pytest.raises(ValidationError, match="path must start with"):
        ServiceReadinessConfig.Schema().load({"http": {"port": 80, "path": "health"}})
    with pytest.raises(ValidationError, match="interval_seconds must be positive"):
        ServiceReadinessConfig.Schema().load({"port": 80, "interval_seconds": 0})


# --- probes ------------------------------------------------------------------------


def test_log_probe_reads_the_file(tmp_path: Path) -> None:
    log = tmp_path / "service_x.out"
    assert not probe_log(log, "ready")
    log.write_text("starting\nserver ready on 9000\n")
    assert probe_log(log, r"ready on \d+")
    assert not probe_log(None, "ready")


def test_http_probe_checks_status() -> None:
    response = MagicMock(status_code=200)
    with patch("srtctl.core.readiness.requests.get", return_value=response) as get:
        assert run_probe(HttpProbe(port=8000, path="/ready"), host="node1", log_file=None)
        assert not run_probe(HttpProbe(port=8000, path="/ready", status=204), host="node1", log_file=None)
    assert get.call_args.args[0] == "http://node1:8000/ready"


# --- wait loop ----------------------------------------------------------------------


def test_wait_until_ready_passes_after_retries() -> None:
    clock = FakeClock()
    with patch("srtctl.core.readiness.probe_tcp", side_effect=[False, False, True]):
        assert wait_until_ready(
            TcpProbe(port=1),
            host="n",
            log_file=None,
            timeout=60,
            interval=5,
            is_alive=lambda: True,
            sleep=clock.sleep,
            clock=clock,
        )
    assert clock.slept == [5, 5]


def test_wait_until_ready_fails_fast_when_process_dies() -> None:
    clock = FakeClock()
    with patch("srtctl.core.readiness.probe_tcp", return_value=False), pytest.raises(ProcessDied):
        wait_until_ready(
            TcpProbe(port=1),
            host="n",
            log_file=None,
            timeout=600,
            interval=5,
            is_alive=lambda: False,
            sleep=clock.sleep,
            clock=clock,
        )
    assert clock.slept == []  # no waiting out the 600s budget


def test_wait_until_ready_times_out() -> None:
    clock = FakeClock()
    with patch("srtctl.core.readiness.probe_tcp", return_value=False):
        assert not wait_until_ready(
            TcpProbe(port=1),
            host="n",
            log_file=None,
            timeout=7,
            interval=3,
            is_alive=lambda: True,
            sleep=clock.sleep,
            clock=clock,
        )
    assert clock.slept == [3, 3, 1]  # the last sleep is clipped to the remaining budget
