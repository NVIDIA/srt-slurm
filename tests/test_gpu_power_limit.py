# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from unittest.mock import call, patch

import pytest

from srtctl.core.gpu_power_limit import GpuPowerLimitManager, build_power_assignments
from srtctl.core.schema import GpuPowerLimitConfig
from srtctl.core.topology import Process, WorkerMode


def _process(node: str, mode: WorkerMode, gpu_indices: set[int]) -> Process:
    return Process(
        node=node,
        gpu_indices=frozenset(gpu_indices),
        sys_port=8081,
        http_port=30000,
        endpoint_mode=mode,
        endpoint_index=0,
    )


def test_build_power_assignments_uses_worker_roles():
    config = GpuPowerLimitConfig(prefill_watts=1000, decode_watts=1000)
    processes = [
        _process("node0", "prefill", {0, 1}),
        _process("node1", "decode", {2, 3}),
    ]
    assert build_power_assignments(config, processes) == {
        ("node0", 0): 1000.0,
        ("node0", 1): 1000.0,
        ("node1", 2): 1000.0,
        ("node1", 3): 1000.0,
    }


def test_build_power_assignments_rejects_conflicting_shared_gpu():
    config = GpuPowerLimitConfig(prefill_watts=1000, decode_watts=900)
    processes = [
        _process("node0", "prefill", {0}),
        _process("node0", "decode", {0}),
    ]
    with pytest.raises(ValueError, match="conflicting power limits"):
        build_power_assignments(config, processes)


def test_apply_verifies_and_restore_reinstates_snapshot(tmp_path):
    requested = {("node0", 0): 1000.0, ("node1", 0): 1000.0}
    original = {("node0", 0): 1400.0, ("node1", 0): 1400.0}
    manager = GpuPowerLimitManager(
        job_id="123",
        assignments=requested,
        log_dir=tmp_path,
        setter="dcgmi",
    )
    with (
        patch.object(manager, "_query_limits", side_effect=[original, requested, original]),
        patch.object(manager, "_set_limits") as set_limits,
    ):
        manager.apply()
        manager.restore()

    assert set_limits.call_args_list == [call(requested), call(original, force_setter=None)]
    assert (tmp_path / "gpu_power_limits_before.csv").read_text().splitlines() == [
        "node,gpu_index,power_limit_w",
        "node0,0,1400",
        "node1,0,1400",
    ]
    assert (tmp_path / "gpu_power_limits_applied.csv").exists()
    assert (tmp_path / "gpu_power_limits_restored.csv").exists()


def test_apply_restores_after_failed_verification(tmp_path):
    requested = {("node0", 0): 1000.0}
    original = {("node0", 0): 1400.0}
    wrong = {("node0", 0): 1100.0}
    manager = GpuPowerLimitManager(job_id="123", assignments=requested, log_dir=tmp_path, setter="dcgmi")
    with (
        patch.object(manager, "_query_limits", side_effect=[original, wrong, original]),
        patch.object(manager, "_set_limits") as set_limits,
        pytest.raises(RuntimeError, match="verification failed"),
    ):
        manager.apply()
    assert set_limits.call_args_list == [call(requested), call(original, force_setter=None)]


def test_dcgmi_set_uses_temporary_group_and_cleanup(tmp_path):
    manager = GpuPowerLimitManager(job_id="123", assignments={}, log_dir=tmp_path, setter="dcgmi")
    assignments = {
        ("node0", 0): 1000.0,
        ("node0", 1): 1000.0,
        ("node1", 0): 1000.0,
        ("node1", 1): 1000.0,
    }
    with patch.object(manager, "_run_on_nodes") as run_on_nodes:
        manager._set_limits_dcgmi(assignments)

    nodes, command = run_on_nodes.call_args.args
    assert nodes == ("node0", "node1")
    assert "dcgmi group -c srtslurm-123-1000_0w -a 0,1" in command
    assert 'dcgmi config -g "$gid" --set -P 1000' in command
    assert "dcgmi group -d" in command


def test_query_limits_parses_node_output(tmp_path):
    manager = GpuPowerLimitManager(job_id="123", assignments={}, log_dir=tmp_path)
    result = subprocess.CompletedProcess(args=[], returncode=0, stdout="node0,0,1400.00\nnode1,1,1000.00\n", stderr="")
    with patch.object(manager, "_run_on_nodes", return_value=result):
        assert manager._query_limits(("node0", "node1")) == {
            ("node0", 0): 1400.0,
            ("node1", 1): 1000.0,
        }
