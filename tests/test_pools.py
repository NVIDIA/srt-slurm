# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pools: services that own nodes, next to engine roles or without them.

The acceptance recipe is a model that does nothing on one node, a two-node pool
for a trainer, and a sleeper on one more: four owners, four disjoint pools, one
recipe. Every reader of the node list has to agree on who owns what.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.submit import show_config_details
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep
from srtctl.services import ServiceConfig, ServicePlacementConfig

TOY = {
    "schema": 2,
    "name": "pools-toy",
    "model": {"path": "hf:fake/model", "container": "nvcr.io/fake/sglang:latest", "precision": "bf16"},
    "resources": {"gpu_type": "b200", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1, "gpus_per_agg": 8},
    "backend": {"type": "sglang"},
    "frontend": {"type": "sglang", "enable_multiple_frontends": False},
    "services": [
        {
            "name": "train",
            "type": "generic",
            "command": ["sleep", "infinity"],
            "nodes": 2,
            "container": "ubuntu:24.04",
            "critical": False,
        },
        {
            "name": "napper",
            "type": "generic",
            "command": ["sleep", "infinity"],
            "nodes": 1,
            "container": "ubuntu:24.04",
            "critical": False,
        },
        {
            "name": "watcher",
            "type": "generic",
            "command": ["sleep", "infinity"],
            "placement": {"pool": "train"},
            "critical": False,
        },
    ],
    "benchmark": {"type": "custom", "command": "echo pools"},
    "observability": {"tachometer": {"enabled": False}},
}

NODES = ("n1", "n2", "n3", "n4", "n5")
IPS = {n: f"10.0.0.{i}" for i, n in enumerate(NODES, start=1)}


def _data(**overrides) -> dict:
    data = yaml.safe_load(yaml.dump(TOY))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


def _load(**overrides) -> SrtConfig:
    return SrtConfig.Schema().load(_data(**overrides))


def _from_slurm(nodelist: tuple[str, ...], **kwargs) -> Nodes:
    with (
        patch("srtctl.core.runtime.get_slurm_nodelist", return_value=list(nodelist)),
        patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=None),
    ):
        return Nodes.from_slurm(**kwargs)


# --- schema -------------------------------------------------------------------


def test_toy_recipe_adds_up_to_four_nodes() -> None:
    config = _load()
    assert config.engine_node_count == 1
    assert [svc.name for svc in config.pool_services] == ["train", "napper"]
    assert config.services_node_count == 3
    assert config.total_nodes == 4
    watcher = next(svc for svc in config.services if svc.name == "watcher")
    assert watcher.effective_pool == "train"
    train = next(svc for svc in config.services if svc.name == "train")
    assert train.effective_pool == "train", "an owner is its own pool"


def test_existing_recipes_are_untouched() -> None:
    data = _data(services=[])
    assert SrtConfig.Schema().load(data).total_nodes == 1
    data["resources"] = {"gpu_type": "b200", "gpus_per_node": 8, "agg_nodes": 3, "agg_workers": 3}
    data["frontend"] = {"type": "sglang-router"}
    assert SrtConfig.Schema().load(data).total_nodes == 3


def test_services_only_pools_still_work_without_roles() -> None:
    data = _data(frontend={"type": "none"})
    data["resources"] = {"gpu_type": "b200", "gpus_per_node": 8}
    data.pop("backend")
    config = SrtConfig.Schema().load(data)
    assert config.engine_node_count == 0
    assert config.total_nodes == 3


def test_pool_rules() -> None:
    with pytest.raises(ValidationError, match="names no service that declares nodes"):
        _load(services=[*TOY["services"][:2], {**TOY["services"][2], "placement": {"pool": "ghost"}}])
    with pytest.raises(ValidationError, match="give either node or pool"):
        _load(
            services=[*TOY["services"][:2], {**TOY["services"][2], "placement": {"node": "workers", "pool": "train"}}]
        )
    with pytest.raises(ValidationError, match="makes the service its own pool"):
        _load(services=[{**TOY["services"][0], "placement": {"pool": "napper"}}, TOY["services"][1]])
    with pytest.raises(ValidationError, match="placement.node must be workers"):
        _load(services=[{**TOY["services"][0], "placement": {"node": "head"}}])
    with pytest.raises(ValidationError, match="only supported without engine roles"):
        _load(frontend={"type": "none"})


def test_pools_are_refused_on_heterogeneous_jobs() -> None:
    data = _data(frontend={"type": "dynamo"})
    data["resources"] = {
        "gpu_type": "b200",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
        "het_jobs": True,
    }
    with pytest.raises(ValidationError, match="not supported together with resources.het_jobs"):
        SrtConfig.Schema().load(data)


# --- carving ------------------------------------------------------------------


def test_carve_engine_nodes_first_then_pools_in_order() -> None:
    nodes = _from_slurm(NODES[:4], engine_nodes=1, pools=[("train", 2), ("napper", 1)])
    assert nodes.head == nodes.bench == nodes.infra == "n1"
    assert nodes.worker == ("n1",)
    assert nodes.pools == {"train": ("n2", "n3"), "napper": ("n4",)}
    assert nodes.compute == ("n1", "n2", "n3", "n4")


def test_carve_with_a_dedicated_infra_node() -> None:
    nodes = _from_slurm(NODES, etcd_nats_dedicated_node=True, engine_nodes=1, pools=[("train", 2), ("napper", 1)])
    assert nodes.infra == "n1"
    assert nodes.head == "n2" and nodes.worker == ("n2",)
    assert nodes.pools == {"train": ("n3", "n4"), "napper": ("n5",)}
    assert "n1" not in nodes.compute


def test_carve_services_only_puts_the_head_on_the_first_pool_node() -> None:
    nodes = _from_slurm(NODES[:3], engine_nodes=0, pools=[("train", 2), ("napper", 1)])
    assert nodes.worker == ()
    assert nodes.head == "n1", "SLURM runs the orchestrator on the first node; it stays head and bench"
    assert nodes.pools == {"train": ("n1", "n2"), "napper": ("n3",)}
    assert nodes.compute == ("n1", "n2", "n3")


def test_carve_legacy_recipes_keep_every_node_as_a_worker() -> None:
    nodes = _from_slurm(NODES[:3])
    assert nodes.worker == NODES[:3] and nodes.pools == {}
    assert nodes.compute == NODES[:3]


def test_carve_rejects_a_short_allocation_and_het_jobs() -> None:
    with pytest.raises(ValueError, match="needs 4"):
        _from_slurm(NODES[:3], engine_nodes=1, pools=[("train", 2), ("napper", 1)])
    with (
        patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=[["a"], ["b"]]),
        pytest.raises(ValueError, match="not supported for heterogeneous"),
    ):
        Nodes.from_slurm(engine_nodes=1, pools=[("train", 1)])


# --- placement resolution --------------------------------------------------------


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="1",
        run_name="pools_1",
        nodes=Nodes(
            head="n1", bench="n1", infra="n1", worker=("n1",), pools={"train": ("n2", "n3"), "napper": ("n4",)}
        ),
        head_node_ip=IPS["n1"],
        infra_node_ip=IPS["n1"],
        log_dir=tmp_path,
        model_path=Path("fake/model"),
        container_image=Path("/img.sqsh"),
        gpus_per_node=8,
        network_interface="eth0",
        container_mounts={},
        environment={},
        is_hf_model=True,
    )


def test_service_nodes_follow_pools(tmp_path: Path) -> None:
    orchestrator = SweepOrchestrator(config=_load(), runtime=_runtime(tmp_path))
    by_name = {svc.name: svc for svc in orchestrator.config.services}
    assert orchestrator.service_nodes(by_name["train"]) == ["n2", "n3"], "an owner runs on its own pool"
    assert orchestrator.service_nodes(by_name["napper"]) == ["n4"]
    assert orchestrator.service_nodes(by_name["watcher"]) == ["n2", "n3"], "a rider runs on the owner's pool"
    assert orchestrator.service_nodes(
        ServiceConfig(name="x", command=["true"], placement=ServicePlacementConfig(node="workers"))
    ) == ["n1"]
    assert orchestrator.service_nodes(
        ServiceConfig(name="x", command=["true"], placement=ServicePlacementConfig(node="compute"))
    ) == [
        "n1",
        "n2",
        "n3",
        "n4",
    ]
    assert orchestrator.service_nodes(
        ServiceConfig(name="x", command=["true"], placement=ServicePlacementConfig(node="all"))
    ) == [
        "n1",
        "n2",
        "n3",
        "n4",
    ]


def test_dry_run_prints_the_node_map(capsys) -> None:
    show_config_details(_load())
    out = capsys.readouterr().out
    assert "engine roles: 1" in out
    assert "pool train (generic): 2" in out
    assert "pool napper (generic): 1" in out
    assert "total: 4" in out


# --- orchestrator (mock) -----------------------------------------------------------


def test_mock_sweep_runs_the_toy_recipe_on_four_nodes(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump(TOY))
    output_dir = tmp_path / "outputs" / "79001"

    exit_code = run_mock_sweep(
        config_path=cfg,
        output_dir=output_dir,
        job_id="79001",
        options=MockOptions(child_duration_s=0.1, phase_pause_s=0.05, nodelist=NODES[:4]),
    )

    assert exit_code == 0
    logs = output_dir / "logs"
    assert any(logs.glob("*_agg_w0.out")), "the model's worker ran on the engine node"
    assert sorted(p.name for p in logs.glob("service_train_*.out")) == ["service_train_n2.out", "service_train_n3.out"]
    assert [p.name for p in logs.glob("service_napper*.out")] == ["service_napper.out"], "one instance, unsuffixed"
    assert sorted(p.name for p in logs.glob("service_watcher_*.out")) == [
        "service_watcher_n2.out",
        "service_watcher_n3.out",
    ], "the rider landed on the train pool"
    assert (logs / "benchmark.out").is_file()
