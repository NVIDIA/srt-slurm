# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime CPU/GPU allocation snapshots and warnings."""

from __future__ import annotations

import json
from types import SimpleNamespace

from srtctl.core.resource_snapshot import (
    RESOURCE_SNAPSHOT_FILENAME,
    collect_resource_snapshot,
    format_resource_summary,
    parse_slurm_cpus_per_node,
    record_resource_snapshot,
)
from srtctl.core.runtime import Nodes
from srtctl.core.schema import SrtConfig


def _config() -> SrtConfig:
    return SrtConfig.Schema().load(
        {
            "name": "cpu-allocation-test",
            "model": {"path": "/model", "container": "/container.sqsh", "precision": "fp8"},
            "resources": {
                "gpu_type": "b300",
                "gpus_per_node": 8,
                "agg_nodes": 1,
                "agg_workers": 1,
                "gpus_per_agg": 4,
            },
        }
    )


def _runtime(tmp_path=None):
    log_dir = tmp_path / "logs" if tmp_path is not None else None
    if log_dir is not None:
        log_dir.mkdir(parents=True)
    return SimpleNamespace(
        job_id="31315",
        nodes=Nodes(head="b300-010", bench="b300-010", infra="b300-010", worker=("b300-010",)),
        log_dir=log_dir,
    )


def test_parse_slurm_cpus_per_node_expands_compressed_counts() -> None:
    assert parse_slurm_cpus_per_node("72(x2),36") == (72, 72, 36)


def test_parse_slurm_cpus_per_node_rejects_unknown_format() -> None:
    assert parse_slurm_cpus_per_node("72,unknown") == ()


def test_two_cpu_kimi_shape_emits_warning() -> None:
    snapshot = collect_resource_snapshot(
        _config(),
        _runtime(),
        environ={"SLURM_JOB_CPUS_PER_NODE": "2", "SLURM_CPUS_ON_NODE": "2"},
        affinity=(2, "0-1"),
    )

    assert snapshot["gpus"]["backend_total"] == 4
    assert snapshot["cpus"]["allocated_total"] == 2
    assert snapshot["cpus"]["allocated_per_node"] == [2]
    assert snapshot["cpus"]["effective_for_check"] == 2
    assert snapshot["cpu_check"]["status"] == "warning"
    assert snapshot["cpu_check"]["minimum_cpu_count"] == 4
    assert "2 CPUs for 4 backend GPUs" in snapshot["cpu_check"]["message"]


def test_exclusive_cpu_allocation_meets_baseline() -> None:
    snapshot = collect_resource_snapshot(
        _config(),
        _runtime(),
        environ={"SLURM_JOB_CPUS_PER_NODE": "256", "SLURM_CPUS_ON_NODE": "256"},
        affinity=(256, "0-255"),
    )

    assert snapshot["cpus"]["allocated_total"] == 256
    assert snapshot["cpu_check"]["minimum_cpu_count"] == 4
    assert snapshot["cpu_check"]["status"] == "ok"


def test_runtime_summary_is_high_signal_for_agents(tmp_path) -> None:
    snapshot = collect_resource_snapshot(
        _config(),
        _runtime(),
        environ={"SLURM_JOB_CPUS_PER_NODE": "2", "SLURM_CPUS_ON_NODE": "2"},
        affinity=(2, "0-1"),
    )

    summary = format_resource_summary(snapshot, tmp_path / RESOURCE_SNAPSHOT_FILENAME)

    assert "Runtime Resource Summary" in summary
    assert "CPU hardware:" in summary
    assert "GPUs: 4 backend / 8 configured" in summary
    assert "CPUs: 2 total; per node: 2" in summary
    assert "CPU affinity: 2 [0-1]" in summary
    assert "CPU/GPU: 0.50 effective vs 1 minimum — WARNING" in summary
    assert "resource_snapshot.json" in summary


def test_single_node_warning_uses_stricter_process_affinity() -> None:
    snapshot = collect_resource_snapshot(
        _config(),
        _runtime(),
        environ={"SLURM_JOB_CPUS_PER_NODE": "256", "SLURM_CPUS_ON_NODE": "256"},
        affinity=(2, "0-1"),
    )

    assert snapshot["cpus"]["allocated_total"] == 256
    assert snapshot["cpus"]["process_affinity_count"] == 2
    assert snapshot["cpus"]["effective_for_check"] == 2
    assert snapshot["cpu_check"]["available_cpu_source"] == "minimum_of_available_local_signals"
    assert snapshot["cpu_check"]["status"] == "warning"


def test_het_component_cpu_counts_are_combined() -> None:
    runtime = SimpleNamespace(
        job_id="42",
        nodes=Nodes(
            head="p0",
            bench="p0",
            infra="p0",
            worker=("p0", "p1", "d0"),
            het=True,
            prefill_group=("p0", "p1"),
            decode_group=("d0",),
        ),
        log_dir=None,
    )
    snapshot = collect_resource_snapshot(
        _config(),
        runtime,
        environ={
            "SLURM_JOB_CPUS_PER_NODE": "1",
            "SLURM_JOB_CPUS_PER_NODE_HET_GROUP_0": "72(x2)",
            "SLURM_JOB_CPUS_PER_NODE_HET_GROUP_1": "36",
        },
        affinity=(72, "0-71"),
    )

    assert snapshot["cpus"]["allocated_per_node"] == [72, 72, 36]
    assert snapshot["cpus"]["allocated_total"] == 180


def test_record_resource_snapshot_writes_artifact_and_updates_metadata(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    metadata_path = tmp_path / "31315.json"
    metadata_path.write_text(json.dumps({"job_id": "31315", "resources": {"gpu_type": "b300"}}))
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "2")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "2")

    snapshot = record_resource_snapshot(_config(), runtime)

    assert snapshot is not None
    artifact = json.loads((runtime.log_dir / RESOURCE_SNAPSHOT_FILENAME).read_text())
    metadata = json.loads(metadata_path.read_text())
    assert artifact["cpus"]["allocated_total"] == 2
    assert metadata["resources"]["cpu_allocation"]["allocated_total"] == 2
    assert metadata["resources"]["cpu_check"]["status"] == "warning"


def test_job_script_logs_slurm_cpu_allocation(monkeypatch, tmp_path) -> None:
    from srtctl.cli.submit import generate_minimal_sbatch_script

    monkeypatch.setattr("srtctl.cli.submit.get_srtslurm_setting", lambda _key, default=None: default)
    script = generate_minimal_sbatch_script(_config(), tmp_path / "config.yaml")

    assert 'echo "CPUs per node: ${SLURM_JOB_CPUS_PER_NODE:-unknown}"' in script
    assert 'echo "CPUs on orchestrator node: ${SLURM_CPUS_ON_NODE:-unknown}"' in script
