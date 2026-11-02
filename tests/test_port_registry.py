# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs sharing nodes allocate around each other's ports through ``port_registry_dir``.

Every job is its own orchestrator process, and a POSIX record lock never conflicts
with another lock of the same process, so each other job here runs in a forked
process of its own.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from srtctl.core.port_registry import PortLease, reserve_ports
from srtctl.core.runtime import parse_port_ranges
from srtctl.core.topology import NodePortAllocator
from srtctl.mock import MockOptions, run_mock_sweep
from srtctl.ports import HTTP_PORTS, NCCL_PORTS

FORK = multiprocessing.get_context("fork")


def _job(directory: Path, job_id: str, nodes: list[str], results, release) -> None:
    """One job's orchestrator: allocate around the registry, publish, hold until released."""
    with reserve_ports(directory, job_id, nodes) as (lease, occupied):
        allocator = NodePortAllocator(occupied=occupied)
        ports = {node: allocator.next(HTTP_PORTS, node) for node in nodes}
        ports["nccl"] = allocator.next(NCCL_PORTS)
        lease.publish(allocator.ports_by_node(nodes))
    results.put((job_id, ports))
    release.wait(30)
    lease.release()


def _start_jobs(directory: Path, jobs: dict[str, list[str]]):
    results, release = FORK.Queue(), FORK.Event()
    processes = [
        FORK.Process(target=_job, args=(directory, job_id, nodes, results, release)) for job_id, nodes in jobs.items()
    ]
    for process in processes:
        process.start()
    allocated = dict(results.get(timeout=30) for _ in processes)
    return allocated, release, processes


def test_jobs_starting_together_on_one_node_get_disjoint_ports(tmp_path: Path) -> None:
    allocated, release, processes = _start_jobs(tmp_path, {"101": ["n0"], "102": ["n0", "n1"], "103": ["n1"]})
    try:
        assert sorted(ports["n0"] for ports in allocated.values() if "n0" in ports) == [6100, 6132]
        assert sorted(ports["n1"] for ports in allocated.values() if "n1" in ports) == [6100, 6132]
        # 102 shares a node with each of the others; 101 and 103 share none, so they may match.
        assert allocated["102"]["nccl"] not in {allocated["101"]["nccl"], allocated["103"]["nccl"]}
    finally:
        release.set()
        for process in processes:
            process.join(10)
    assert not list(tmp_path.glob("job-*.json")), "every job removes its entry when it releases"


def test_jobs_on_other_nodes_do_not_count(tmp_path: Path) -> None:
    allocated, release, processes = _start_jobs(tmp_path, {"201": ["a"]})
    try:
        with reserve_ports(tmp_path, "202", ["b"]) as (_lease, occupied):
            assert occupied == {}
    finally:
        release.set()
        for process in processes:
            process.join(10)
    assert allocated["201"]["a"] == 6100


def test_a_job_that_died_frees_its_ports(tmp_path: Path) -> None:
    def publish_and_exit() -> None:
        with reserve_ports(tmp_path, "301", ["n0"]) as (lease, _occupied):
            lease.publish({"n0": frozenset({6100})})

    process = FORK.Process(target=publish_and_exit)
    process.start()
    process.join(10)
    assert (tmp_path / "job-301.json").exists(), "the file outlives the process that wrote it"

    with reserve_ports(tmp_path, "302", ["n0"]) as (_lease, occupied):
        assert occupied == {}
    assert not (tmp_path / "job-301.json").exists()


def test_a_held_registry_lock_times_out(tmp_path: Path) -> None:
    held, release = FORK.Event(), FORK.Event()

    def hold_lock() -> None:
        with reserve_ports(tmp_path, "401", ["n0"]):
            held.set()
            release.wait(30)

    process = FORK.Process(target=hold_lock)
    process.start()
    try:
        assert held.wait(10)
        with pytest.raises(TimeoutError, match="stayed locked"), reserve_ports(tmp_path, "402", ["n0"], timeout_s=0.2):
            pass
    finally:
        release.set()
        process.join(10)


@pytest.mark.parametrize(
    ("entries", "parsed"),
    [([], ()), (["32768-60999"], ((32768, 60999),)), (["8080", "9000-9010"], ((8080, 8080), (9000, 9010)))],
)
def test_reserved_port_ranges_parse(entries: list[str], parsed: tuple) -> None:
    assert parse_port_ranges(entries) == parsed


@pytest.mark.parametrize("entry", ["high-low", "70000", "9010-9000", "0-10"])
def test_reserved_port_ranges_reject_bad_entries(entry: str) -> None:
    with pytest.raises(ValueError, match="reserved_port_ranges"):
        parse_port_ranges([entry])


def test_orchestrator_allocates_around_another_job_and_releases_its_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep on a node another live job holds 6100 on serves elsewhere, then frees what it took."""
    registry = tmp_path / "registry"
    cluster = tmp_path / "srtslurm.yaml"
    cluster.write_text(yaml.safe_dump({"port_registry_dir": str(registry), "reserved_port_ranges": ["32768-60999"]}))
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "schema": 2,
                "name": "port-registry",
                "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
                "resources": {"gpu_type": "h100", "gpus_per_node": 8},
                "roles": {"agg": {"nodes": 1, "workers": 1}},
                "benchmark": {"type": "custom", "command": "echo fake-benchmark"},
            }
        )
    )

    held, release = FORK.Event(), FORK.Event()

    def other_job() -> None:
        with reserve_ports(registry, "900", ["mock-node-01"]) as (lease, _occupied):
            lease.publish({"mock-node-01": frozenset(HTTP_PORTS.span(HTTP_PORTS.base))})
        held.set()
        release.wait(60)
        lease.release()

    process = FORK.Process(target=other_job)
    process.start()
    published: list[dict[str, frozenset[int]]] = []
    real_publish = PortLease.publish

    def recording_publish(self: PortLease, ports_by_node):
        published.append(dict(ports_by_node))
        real_publish(self, ports_by_node)

    try:
        assert held.wait(10)
        with patch.object(PortLease, "publish", recording_publish):
            exit_code = run_mock_sweep(
                config_path=recipe,
                output_dir=tmp_path / "outputs" / "901",
                job_id="901",
                options=MockOptions(child_duration_s=0.15, phase_pause_s=0.05),
            )
        assert exit_code == 0
        [ours] = published
        assert HTTP_PORTS.base + HTTP_PORTS.stride in ours["mock-node-01"], "the second HTTP slot, past the other job's"
        assert not ours["mock-node-01"] & frozenset(HTTP_PORTS.span(HTTP_PORTS.base))
        assert not (registry / "job-901.json").exists(), "the finished job released its entry"
        assert (registry / "job-900.json").exists(), "the other job still holds its ports"
    finally:
        release.set()
        process.join(10)
