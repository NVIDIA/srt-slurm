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

ARM_CPUINFO = """\
processor       : 0
BogoMIPS        : 2000.00
Features        : fp asimd evtstrm
CPU implementer : 0x41
CPU architecture: 8
CPU variant     : 0x1
CPU part        : 0xd49
CPU revision    : 1
"""

ARM_CPU_MODEL = "ARM CPU implementer 0x41 part 0xd49 (architecture 8, variant 0x1, revision 1)"


def _config(**resource_overrides) -> SrtConfig:
    resources = {
        "gpu_type": "b300",
        "gpus_per_node": 8,
        "agg_nodes": 1,
        "agg_workers": 1,
        "gpus_per_agg": 4,
    }
    resources.update(resource_overrides)
    return SrtConfig.Schema().load(
        {
            "name": "cpu-allocation-test",
            "model": {"path": "/model", "container": "/container.sqsh", "precision": "fp8"},
            "resources": resources,
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


def test_resource_snapshot_cpu_model_uses_arm_cpuinfo_fallback(monkeypatch) -> None:
    from srtctl.core import resource_snapshot

    monkeypatch.setattr(resource_snapshot.Path, "read_text", lambda _self, errors=None: ARM_CPUINFO)

    assert resource_snapshot._cpu_model() == ARM_CPU_MODEL


def test_runtime_summary_uses_arm_cpuinfo_fallback(monkeypatch, tmp_path) -> None:
    from srtctl.core import resource_snapshot

    monkeypatch.setattr(resource_snapshot, "_cpu_model", lambda: ARM_CPU_MODEL)
    snapshot = collect_resource_snapshot(
        _config(),
        _runtime(),
        environ={"SLURM_JOB_CPUS_PER_NODE": "144", "SLURM_CPUS_ON_NODE": "144"},
        affinity=(144, "0-143"),
    )

    summary = format_resource_summary(snapshot, tmp_path / RESOURCE_SNAPSHOT_FILENAME)

    assert "CPU hardware: ARM CPU implementer 0x41 part 0xd49" in summary
    assert "CPU hardware: unknown" not in summary


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


def test_dedicated_infra_cpus_do_not_mask_starved_worker_node() -> None:
    runtime = SimpleNamespace(
        job_id="42",
        nodes=Nodes(head="worker0", bench="worker0", infra="infra0", worker=("worker0",)),
        log_dir=None,
    )

    snapshot = collect_resource_snapshot(
        _config(),
        runtime,
        environ={"SLURM_JOB_CPUS_PER_NODE": "144,2", "SLURM_CPUS_ON_NODE": "144"},
        affinity=(144, "0-143"),
    )

    assert snapshot["cpus"]["allocated_total"] == 146
    assert snapshot["cpus"]["allocated_per_node"] == [144, 2]
    assert snapshot["cpus"]["backend_node_allocations"] == [{"node": "worker0", "allocated_cpus": 2, "backend_gpus": 4}]
    assert snapshot["cpus"]["effective_for_check"] == 2
    assert snapshot["cpu_check"]["status"] == "warning"
    assert snapshot["cpu_check"]["gpu_count_basis"] == 4
    assert snapshot["cpu_check"]["minimum_cpu_count"] == 4
    assert snapshot["cpu_check"]["available_cpu_source"] == "minimum_backend_node_allocation"
    assert snapshot["cpu_check"]["node"] == "worker0"
    assert "backend node worker0" in snapshot["cpu_check"]["message"]
    assert "2 CPUs for 4 backend GPUs" in snapshot["cpu_check"]["message"]


def test_multi_worker_check_uses_most_constrained_backend_node() -> None:
    runtime = SimpleNamespace(
        job_id="42",
        nodes=Nodes(head="worker0", bench="worker0", infra="worker0", worker=("worker0", "worker1")),
        log_dir=None,
    )

    snapshot = collect_resource_snapshot(
        _config(gpus_per_node=4, agg_nodes=2, agg_workers=2, gpus_per_agg=4),
        runtime,
        environ={"SLURM_JOB_CPUS_PER_NODE": "2,64", "SLURM_CPUS_ON_NODE": "2"},
        affinity=(2, "0-1"),
    )

    assert snapshot["cpus"]["allocated_total"] == 66
    assert snapshot["cpus"]["backend_node_allocations"] == [
        {"node": "worker0", "allocated_cpus": 2, "backend_gpus": 4},
        {"node": "worker1", "allocated_cpus": 64, "backend_gpus": 4},
    ]
    assert snapshot["cpus"]["effective_for_check"] == 2
    assert snapshot["cpu_check"]["status"] == "warning"
    assert snapshot["cpu_check"]["node"] == "worker0"


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
