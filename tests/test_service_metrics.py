# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Services say where their metrics are; tachometer scrapes them there.

``services[].metrics`` is the scrape annotation. The exporter kinds carry their own,
a generic service writes one, a cluster-shaped service lists several on its head, and
the telemetry stage turns every endpoint into one tachometer target per node.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.cli.submit import show_config_details
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.telemetry import ServiceMetricsTarget
from srtctl.core.topology import Process
from srtctl.services import ServiceConfig, get_service_kind

RECIPE = {
    "schema": 2,
    "name": "service-metrics",
    "model": {"path": "hf:fake/model", "container": "nvcr.io/fake/miles:latest", "precision": "bf16"},
    "resources": {"gpu_type": "b200", "gpus_per_node": 8},
    "frontend": {"type": "none"},
    "services": [
        {"name": "train", "type": "ray", "nodes": 2, "preamble": "cd /root/miles"},
        {
            "name": "engines",
            "type": "generic",
            "command": ["python3", "-m", "exporter", "--port", "9402"],
            "placement": {"pool": "train"},
            "readiness": {"port": 9402},
            "metrics": {"port": 9402},
        },
        {
            "name": "envsrv",
            "type": "generic",
            "command": ["python3", "-m", "server"],
            "placement": {"pool": "train"},
            "metrics": {"port": 8003, "path": "/prom"},
        },
    ],
    "benchmark": {"type": "custom", "command": "echo"},
    "observability": {"tachometer": {"enabled": True}},
}


def _load(**overrides) -> SrtConfig:
    data = yaml.safe_load(yaml.dump(RECIPE))
    data.update(overrides)
    return SrtConfig.Schema().load(data)


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="7",
        run_name="service_metrics_7",
        nodes=Nodes(head="node1", bench="node1", infra="node1", worker=(), pools={"train": ("node1", "node2")}),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/miles.sqsh"),
        gpus_per_node=8,
        network_interface="bond0",
        container_mounts={},
        environment={},
    )


def _targets(orchestrator: SweepOrchestrator) -> dict[tuple[str, str], ServiceMetricsTarget]:
    return {(t.service, t.node): t for t in orchestrator._service_metrics_targets()}


# --- the annotation -------------------------------------------------------------------


def test_metrics_block_is_validated() -> None:
    with pytest.raises(ValidationError, match="metrics.port"):
        _load(services=[{"name": "x", "type": "generic", "command": ["true"], "metrics": {"port": 0}}])
    with pytest.raises(ValidationError, match="metrics.path"):
        _load(
            services=[{"name": "x", "type": "generic", "command": ["true"], "metrics": {"port": 9000, "path": "prom"}}]
        )
    with pytest.raises(ValidationError, match="metrics.nodes must be all or first"):
        _load(
            services=[{"name": "x", "type": "generic", "command": ["true"], "metrics": {"port": 9000, "nodes": "head"}}]
        )
    config = _load(services=[{"name": "x", "type": "generic", "command": ["true"], "metrics": {"port": 9000}}])
    (endpoint,) = config.services[0].metrics
    assert (endpoint.path, endpoint.nodes, endpoint.name) == ("/metrics", "all", None), (
        "a mapping is a one-endpoint list"
    )

    two = [{"port": 9090, "nodes": "first"}, {"port": 31000, "path": "/engine_metrics", "nodes": "first"}]
    with pytest.raises(ValidationError, match="give each a distinct name"):
        _load(services=[{"name": "train", "type": "ray", "nodes": 2, "metrics": two}])
    named = [{**two[0], "name": "miles"}, {**two[1], "name": "engines"}]
    loaded = _load(services=[{"name": "train", "type": "ray", "nodes": 2, "metrics": named}])
    assert [e.name for e in loaded.services[0].metrics] == ["miles", "engines"]
    with pytest.raises(ValidationError, match="distinct name"):
        _load(
            services=[
                {
                    "name": "train",
                    "type": "ray",
                    "nodes": 2,
                    "metrics": [{**two[0], "name": "m"}, {**two[1], "name": "m"}],
                }
            ]
        )


def test_exporter_kinds_carry_their_own_annotation() -> None:
    dcgm = get_service_kind("dcgm-exporter")
    node = get_service_kind("node-exporter")
    process = get_service_kind("process-exporter")
    generic = get_service_kind("generic")

    assert [e.port for e in dcgm.metrics(ServiceConfig(name="dcgm-exporter", type="dcgm-exporter"))] == [9401]
    assert [e.port for e in dcgm.metrics(ServiceConfig(name="d", type="dcgm-exporter", options={"port": 9500}))] == [
        9500
    ]
    assert (dcgm.metrics_filter, dcgm.metrics_endpoint_prefix, dcgm.metrics_gpu_metadata) == ("dcgm", "dcgm", True)
    assert [e.port for e in node.metrics(ServiceConfig(name="node-exporter", type="node-exporter"))] == [9101]
    assert (node.metrics_filter, node.metrics_endpoint_prefix) == ("node_exporter", "node_exporter")
    assert [e.port for e in process.metrics(ServiceConfig(name="process-exporter", type="process-exporter"))] == [9256]
    assert (process.metrics_filter, process.metrics_endpoint_prefix) == ("passthrough", "process_exporter")
    assert generic.metrics(ServiceConfig(name="g", type="generic", command=["true"])) == []
    assert generic.metrics_filter == "passthrough" and generic.metrics_endpoint_prefix is None


# --- targets ----------------------------------------------------------------------------


def test_every_annotated_service_becomes_one_target_per_node(tmp_path: Path) -> None:
    orchestrator = SweepOrchestrator(config=_load(), runtime=_runtime(tmp_path))
    targets = _targets(orchestrator)
    services = sorted({service for service, _node in targets})
    assert services == ["dcgm-exporter", "engines", "envsrv", "node-exporter", "process-exporter"]
    assert ("train", "node1") not in targets, "the ray service publishes no metrics"

    engines = targets[("engines", "node2")]
    assert engines.url == "http://node2:9402/metrics" and engines.filter == "passthrough"
    assert engines.endpoint_name == "engines_node2"
    envsrv = targets[("envsrv", "node1")]
    assert envsrv.url == "http://node1:8003/prom"
    dcgm = targets[("dcgm-exporter", "node1")]
    assert (dcgm.url, dcgm.filter, dcgm.endpoint_name, dcgm.gpu_metadata) == (
        "http://node1:9401/metrics",
        "dcgm",
        "dcgm_node1",
        True,
    )
    assert targets[("process-exporter", "node2")].endpoint_name == "process_exporter_node2"


def test_nodes_first_scrapes_only_the_head_of_a_cluster(tmp_path: Path) -> None:
    """Miles's training collector is a Ray actor pinned to the head: annotate the ray service, first node only."""
    config = _load(
        services=[{"name": "train", "type": "ray", "nodes": 2, "metrics": {"port": 9090, "nodes": "first"}}],
    )
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    targets = _targets(orchestrator)
    assert ("train", "node1") in targets and ("train", "node2") not in targets
    assert targets[("train", "node1")].url == "http://node1:9090/metrics"
    assert targets[("train", "node1")].endpoint_name == "train_node1"


def test_a_service_can_serve_several_endpoints(tmp_path: Path) -> None:
    """The Miles shape: the collector and the router's engine aggregate both live on the Ray head."""
    config = _load(
        services=[
            {
                "name": "train",
                "type": "ray",
                "nodes": 2,
                "metrics": [
                    {"name": "miles", "port": 9090, "nodes": "first"},
                    {"name": "engines", "port": 31000, "path": "/engine_metrics", "nodes": "first"},
                ],
            }
        ],
    )
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    train = [t for t in orchestrator._service_metrics_targets() if t.service == "train"]
    assert [(t.endpoint_name, t.url) for t in train] == [
        ("miles_node1", "http://node1:9090/metrics"),
        ("engines_node1", "http://node1:31000/engine_metrics"),
    ]


def test_external_and_disabled_services_are_not_scraped(tmp_path: Path) -> None:
    config = _load(
        services=[
            {"name": "train", "type": "ray", "nodes": 2},
            {"name": "etcd", "type": "etcd", "external": "http://etcd.example:2379", "metrics": {"port": 2379}},
            {"name": "off", "type": "generic", "command": ["true"], "enabled": False, "metrics": {"port": 9000}},
        ]
    )
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    assert {service for service, _ in _targets(orchestrator)} == {"dcgm-exporter", "node-exporter", "process-exporter"}


def test_power_telemetry_owning_dcgm_replaces_the_implied_exporter(tmp_path: Path) -> None:
    config = _load(
        services=[{"name": "train", "type": "ray", "nodes": 2}],
        telemetry={"enabled": True, "dcgm_exporter": {"container_image": "power-dcgm", "port": 9400}},
        benchmark={"type": "custom", "command": "echo", "concurrencies": [8]},  # power telemetry insists on these
    )
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    assert ("dcgm-exporter", "node1") not in _targets(orchestrator), "no implied dcgm service when power owns one"
    with patch.object(
        SweepOrchestrator, "backend_processes", [Process("node1", frozenset({0}), 8081, 30000, "agg", 0, 0)]
    ):
        power = orchestrator._power_dcgm_targets()
    assert [(t.endpoint_name, t.url, t.gpu_metadata) for t in power] == [
        ("dcgm_node1", "http://node1:9400/metrics", True)
    ]


def test_generated_config_carries_the_targets(tmp_path: Path) -> None:
    orchestrator = SweepOrchestrator(config=_load(), runtime=_runtime(tmp_path))
    with (
        patch("srtctl.cli.mixins.telemetry_stage.start_srun_process"),
        patch.object(TelemetryStageMixin, "_resolve_tachometer_binary", return_value="/bin/tachometer"),
    ):
        orchestrator.start_tachometer()
    toml = (tmp_path / "tachometer_config.toml").read_text()
    assert 'name = "engines_node1"' in toml and 'url = "http://node1:9402/metrics"' in toml
    assert 'name = "envsrv_node2"' in toml and 'url = "http://node2:8003/prom"' in toml
    assert 'name = "dcgm_node1"' in toml and 'name = "node_exporter_node2"' in toml
    assert 'name = "process_exporter_node1"' in toml
    assert '"service" = "engines"' in toml, "rows carry the service name"
    assert "backend_" not in toml and "frontend0" not in toml, "a services-only job has no workers or frontend"


def test_dry_run_shows_the_annotation(capsys) -> None:
    show_config_details(_load())
    out = capsys.readouterr().out
    assert "metrics=:9402/metrics" in out and "metrics=:8003/prom" in out
    assert "metrics=:9401/metrics" in out, "the implied dcgm exporter shows where it is scraped"
    show_config_details(
        _load(
            services=[
                {
                    "name": "train",
                    "type": "ray",
                    "nodes": 2,
                    "metrics": [
                        {"name": "miles", "port": 9090, "nodes": "first"},
                        {"name": "engines", "port": 31000, "path": "/engine_metrics", "nodes": "first"},
                    ],
                }
            ]
        )
    )
    assert "metrics=miles:9090/metrics@first,engines:31000/engine_metrics@first" in capsys.readouterr().out
