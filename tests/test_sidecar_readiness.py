# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark readiness includes KV relays absent from frontend worker counts."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srtctl.backends import SGLangProtocol, SGLangServerConfig
from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.core.schema import DynamoConfig
from srtctl.core.topology import Endpoint, Process


class _ReadinessHarness(BenchmarkStageMixin):
    def __init__(
        self,
        *,
        args: dict | None = None,
        kv_events_config: bool | dict | None = True,
        sidecar: bool = True,
        frontend: str = "dynamo",
        node_count: int = 2,
    ) -> None:
        backend = SGLangProtocol(
            kv_events_config=kv_events_config,
            sglang_config=SGLangServerConfig(
                aggregated=args if args is not None else {"tp-size": 8, "dp-size": 2, "enable-dp-attention": True}
            ),
        )
        self.config = SimpleNamespace(
            backend=backend,
            dynamo=DynamoConfig(sidecar=sidecar),
            frontend=SimpleNamespace(type=frontend),
            resources=SimpleNamespace(num_agg=1, num_prefill=0, num_decode=0),
            health_check=SimpleNamespace(interval_seconds=1, max_attempts=2),
        )
        self.runtime = SimpleNamespace(network_interface="ib0")
        endpoint = Endpoint(
            mode="agg",
            index=0,
            nodes=tuple(f"node{i}" for i in range(node_count)),
            gpu_indices=frozenset(range(8 // node_count)),
            gpus_per_node=8 // node_count,
        )
        self._processes = backend.endpoints_to_processes([endpoint], dynamo_sidecar=sidecar)

    @property
    def backend_processes(self) -> list[Process]:
        return self._processes

    def _public_api_node(self) -> str:
        return "frontend-node"


def _node_ip(node: str, interface: str | None = None) -> str:
    assert interface == "ib0"
    return f"10.0.0.{int(node.removeprefix('node')) + 1}"


@pytest.mark.parametrize("outcome", ["healthy", "timeout", "stop"])
def test_ready_frontend_waits_for_sglang_follower_kv_relay(outcome: str) -> None:
    harness = _ReadinessHarness()
    stop_event = threading.Event()
    elapsed = 0.0
    responses = [SimpleNamespace(status_code=503), SimpleNamespace(status_code=200 if outcome == "healthy" else 503)]

    def advance_time(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds
        if outcome == "stop":
            stop_event.set()

    with (
        patch("srtctl.cli.mixins.benchmark_stage.wait_for_model", return_value=True) as frontend_health,
        patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=_node_ip),
        patch("srtctl.core.health.requests.get", side_effect=responses) as get,
        patch("srtctl.core.health.time.time", side_effect=lambda: elapsed),
        patch("srtctl.core.health.time.sleep", side_effect=advance_time) as sleep,
    ):
        assert harness._wait_for_service_ready(stop_event) is (outcome == "healthy")

    # Telemetry followers do not register inference endpoints, so the frontend
    # count remains one even though readiness now has an additional HTTP gate.
    assert frontend_health.call_args.kwargs["n_prefill"] == 0
    assert frontend_health.call_args.kwargs["n_decode"] == 1
    assert [call.args[0] for call in get.call_args_list] == ["http://10.0.0.2:7501/health"] * (
        1 if outcome == "stop" else 2
    )
    assert sleep.call_count == (2 if outcome == "timeout" else 1)


@pytest.mark.parametrize(
    "settings",
    [
        {"args": {"tp-size": 8}},
        {"kv_events_config": False},
        {"kv_events_config": {"aggregated": {"publisher": "null"}}},
        {"sidecar": False},
        {"frontend": "sglang-router"},
    ],
)
def test_readiness_does_not_poll_nonexistent_sglang_relays(settings: dict) -> None:
    harness = _ReadinessHarness(**settings)

    with (
        patch("srtctl.cli.mixins.benchmark_stage.wait_for_model", return_value=True),
        patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=_node_ip),
        patch("srtctl.core.health.requests.get") as get,
    ):
        assert harness._wait_for_service_ready(threading.Event())

    get.assert_not_called()


def test_readiness_only_polls_follower_nodes_owning_kv_publishers() -> None:
    # TP8/DP2 across four nodes has publisher ranks on node0 and node2 only.
    harness = _ReadinessHarness(node_count=4)
    with (
        patch("srtctl.cli.mixins.benchmark_stage.wait_for_model", return_value=True),
        patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=_node_ip),
        patch("srtctl.core.health.requests.get", return_value=SimpleNamespace(status_code=200)) as get,
    ):
        assert harness._wait_for_service_ready(threading.Event())

    get.assert_called_once_with("http://10.0.0.3:7502/health", timeout=5.0)


def test_failed_frontend_health_does_not_start_relay_wait() -> None:
    harness = _ReadinessHarness()
    with (
        patch("srtctl.cli.mixins.benchmark_stage.wait_for_model", return_value=False),
        patch("srtctl.core.health.requests.get") as get,
    ):
        assert not harness._wait_for_service_ready(threading.Event())

    get.assert_not_called()
