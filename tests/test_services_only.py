# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Services-only jobs: ``frontend.type: none``, no engine roles, and a service that owns the nodes.

The job has nothing to serve: no workers, no router, no worker-count health
gate. Services and their readiness probes are the only gate before the
benchmark step runs. This is the shape a Ray cluster driving an RL trainer
needs, and a client run against an endpoint the job does not own.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.submit import show_config_details
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

SERVICES_ONLY = {
    "schema": 2,
    "name": "services-only",
    "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "bf16"},
    "resources": {"gpu_type": "b200", "gpus_per_node": 8},
    "frontend": {"type": "none"},
    "services": [
        {
            "name": "files",
            "command": ["python3", "-m", "http.server", "9911"],
            "placement": {"node": "workers"},
            "nodes": 2,
            "start": "before_workers",
            "readiness": {"port": 9911, "timeout_seconds": 30},
            "inherit_discovery_env": False,
        }
    ],
    "benchmark": {"type": "custom", "command": "echo services-only-client"},
}


def _data(**overrides) -> dict:
    data = yaml.safe_load(yaml.dump(SERVICES_ONLY))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


def _load(**overrides) -> SrtConfig:
    return SrtConfig.Schema().load(_data(**overrides))


# --- schema -------------------------------------------------------------------


def test_services_only_job_takes_its_node_count_from_the_owning_service() -> None:
    config = _load()
    assert config.frontend.type == "none"
    assert config.services[0].nodes == 2
    assert config.services_node_count == 2
    assert config.total_nodes == 2, "the sbatch node count is the service's"
    assert config.resources.has_engine_workers is False
    assert config.resources.num_prefill == config.resources.num_decode == config.resources.num_agg == 0


def test_without_a_node_owner_the_job_is_one_node_and_agg_recipes_are_untouched() -> None:
    data = _data()
    del data["services"][0]["nodes"]
    assert SrtConfig.Schema().load(data).total_nodes == 1
    data = _data(frontend={"type": "sglang-router"})
    data["resources"] = {"gpu_type": "b200", "gpus_per_node": 8, "agg_nodes": 3, "agg_workers": 3}
    del data["services"][0]["nodes"]
    data["backend"] = {"type": "sglang"}
    assert SrtConfig.Schema().load(data).total_nodes == 3


def test_service_nodes_rules() -> None:
    data = _data()
    data["services"][0]["nodes"] = 0
    with pytest.raises(ValidationError, match="at least 1"):
        SrtConfig.Schema().load(data)

    data = _data()
    data["services"][0]["placement"] = {"node": "head"}
    with pytest.raises(ValidationError, match="placement.node must be workers"):
        SrtConfig.Schema().load(data)

    # Two owners are two pools; the job is the sum.
    data = _data()
    data["services"].append({**data["services"][0], "name": "second", "readiness": {"port": 9912}})
    assert SrtConfig.Schema().load(data).total_nodes == 4

    # Owners next to engine roles are pools too (see tests/test_pools.py); frontend none is still engine-free.
    data = _data(frontend={"type": "sglang-router"})
    data["resources"] = {"gpu_type": "b200", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1}
    data["backend"] = {"type": "sglang"}
    assert SrtConfig.Schema().load(data).total_nodes == 3


def test_frontend_none_rejects_engine_workers_and_dedicated_node() -> None:
    data = _data()
    data["resources"] = {"gpu_type": "b200", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1}
    del data["services"][0]["nodes"]
    with pytest.raises(ValidationError, match="only supported without engine roles"):
        SrtConfig.Schema().load(data)
    with pytest.raises(ValidationError, match="dedicated_node is invalid"):
        _load(frontend={"type": "none", "dedicated_node": True})


def test_frontend_none_has_no_implementation() -> None:
    from srtctl.frontends import get_frontend

    with pytest.raises(ValueError, match="services-only"):
        get_frontend("none")


def test_dry_run_renders_services_only_job(capsys) -> None:
    show_config_details(_load())
    out = capsys.readouterr().out
    assert "files" in out
    assert "nodes=2" in out
    assert "custom" in out


def test_preflight_topology_accepts_a_service_node_count() -> None:
    from srtctl.core.validation import validate_topology

    assert validate_topology({"gpu_type": "b200", "gpus_per_node": 8}, service_nodes=2) == []
    assert validate_topology({}, service_nodes=2) == []
    assert validate_topology({"agg_nodes": 1, "agg_workers": 1}, service_nodes=2) == [], "pools next to roles"
    (issue,) = validate_topology({"agg_nodes": 1}, service_nodes=2)
    assert issue.code == "topology-no-workers", "the roles are still validated"
    (issue,) = validate_topology({"gpu_type": "b200"})
    assert issue.code == "topology-missing", "no service node count and no roles is still an error"


def test_preflight_reads_the_service_node_count_from_the_recipe(tmp_path: Path) -> None:
    from srtctl.core.validation import preflight_config_variants

    (result,) = preflight_config_variants(_data())
    assert [issue.code for issue in result.errors if issue.code.startswith("topology")] == []


# --- orchestrator (mock) ---------------------------------------------------------


def test_mock_sweep_runs_services_only_job_end_to_end(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump(SERVICES_ONLY))
    output_dir = tmp_path / "outputs" / "77001"

    exit_code = run_mock_sweep(
        config_path=cfg,
        output_dir=output_dir,
        job_id="77001",
        options=MockOptions(child_duration_s=0.1, phase_pause_s=0.05, nodelist=("mock-node-01", "mock-node-02")),
    )

    assert exit_code == 0
    logs = output_dir / "logs"
    assert (logs / "benchmark.out").is_file(), "the benchmark step ran"
    service_logs = sorted(p.name for p in logs.glob("service_files*.out"))
    assert len(service_logs) == 2, f"one service instance per node, got {service_logs}"
    assert not list(logs.glob("*_frontend_*.out")), "frontend.type none launches no frontend"
    assert not (logs / "service_etcd.out").exists(), "no discovery plane without a Dynamo frontend"
    assert not list(logs.glob("*_agg_w*.out")), "no engine workers"


# --- telemetry -------------------------------------------------------------------


def test_tachometer_targets_follow_exporter_nodes_without_engines_or_frontend() -> None:
    from unittest.mock import MagicMock, patch

    from srtctl.cli.mixins.frontend_stage import FrontendTopology
    from srtctl.core.schema import TachometerConfig, TelemetryExporterConfig
    from srtctl.core.telemetry import generate_tachometer_config

    tachometer = TachometerConfig(
        enabled=True,
        dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
        node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
    )
    runtime = MagicMock()
    runtime.job_id, runtime.run_name, runtime.network_interface = "1", "run_1", "eth0"
    runtime.log_dir = Path("/runs/1/logs")
    topology = FrontendTopology(nginx_node=None, frontend_nodes=["n1"], frontend_port=8000, public_port=8000)
    with patch("srtctl.core.telemetry.get_hostname_ip", return_value="10.0.0.1"):
        text = generate_tachometer_config(
            processes=[],
            frontend_topology=topology,
            runtime=runtime,
            tachometer=tachometer,
            frontend_type="none",
            exporter_nodes=["n1", "n2"],
        )
    assert 'name = "dcgm_n1"' in text and 'name = "dcgm_n2"' in text
    assert 'name = "node_exporter_n1"' in text and 'name = "node_exporter_n2"' in text
    assert "frontend0" not in text, "a services-only job binds no frontend port"
    assert "backend_" not in text


def test_start_tachometer_scrapes_the_exporter_service_nodes(tmp_path: Path) -> None:
    from unittest.mock import patch

    from srtctl.cli.do_sweep import SweepOrchestrator
    from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
    from srtctl.core.runtime import Nodes, RuntimeContext

    config = _load(observability={"tachometer": {"enabled": True}})
    runtime = RuntimeContext(
        job_id="1",
        run_name="run_1",
        nodes=Nodes(head="node1", bench="node1", infra="node1", worker=(), pools={"files": ("node1", "node2")}),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("fake/model"),
        container_image=Path("/img.sqsh"),
        gpus_per_node=8,
        network_interface="eth0",
        container_mounts={},
        environment={},
        is_hf_model=True,
    )
    orchestrator = SweepOrchestrator(config=config, runtime=runtime)
    with (
        patch("srtctl.cli.mixins.telemetry_stage.generate_tachometer_config", return_value="") as gen,
        patch("srtctl.cli.mixins.telemetry_stage.start_srun_process"),
        patch.object(TelemetryStageMixin, "_resolve_tachometer_binary", return_value="/bin/tachometer"),
    ):
        orchestrator.start_tachometer()
    assert gen.call_args.kwargs["exporter_nodes"] == ["node1", "node2"]
    assert gen.call_args.kwargs["frontend_type"] == "none"
    assert gen.call_args.kwargs["processes"] == []


# --- custom benchmark environment ---------------------------------------------------


def test_custom_benchmark_env_locates_the_services(tmp_path: Path) -> None:
    from unittest.mock import patch

    from srtctl.benchmarks.custom import CustomBenchmarkRunner
    from srtctl.cli.do_sweep import SweepOrchestrator
    from srtctl.core.runtime import Nodes, RuntimeContext

    config = _load()
    runtime = RuntimeContext(
        job_id="1",
        run_name="run_1",
        nodes=Nodes(head="node1", bench="node1", infra="node1", worker=(), pools={"files": ("node1", "node2")}),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("fake/model"),
        container_image=Path("/img.sqsh"),
        gpus_per_node=8,
        network_interface="eth0",
        container_mounts={},
        environment={},
        is_hf_model=True,
    )
    orchestrator = SweepOrchestrator(config=config, runtime=runtime)
    ips = {"node1": "10.0.0.1", "node2": "10.0.0.2"}
    with patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=lambda host, iface=None: ips[host]):
        env = orchestrator._get_benchmark_env(CustomBenchmarkRunner())
    assert env["SRT_SERVICE_FILES_NODES"] == "node1,node2"
    assert env["SRT_SERVICE_FILES_IPS"] == "10.0.0.1,10.0.0.2"
    assert env["SRT_SERVICE_FILES_NODE_COUNT"] == "2"
    assert env["SRT_GPUS_PER_NODE"] == "8"
    assert env["SRT_WORKER_NODES"] == "", "no engine roles, so no engine worker nodes"
    assert "SRT_FRONTEND_HOST" not in env, "frontend.type none has no endpoint to point at"
