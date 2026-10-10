# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check diagnostic evidence handling without GPUs, MPI, or cluster access."""

import argparse
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "configs/cluster-perf-diagnostics.py"))


def test_counters_distinguish_missing_reset_and_zero() -> None:
    before = {
        "ports/1/counters/errors": "10",
        "ports/1/counters/wait": "4",
        "ports/1/counters/ok": "3",
        "ports/1/counters/denied": "UNAVAILABLE",
        "device/numa_node": "1",
    }
    after = {**before, "ports/1/counters/errors": "12", "ports/1/counters/wait": "1"}
    assert MODULE["counter_deltas"](before, after) == {
        "ports/1/counters/errors": 2,
        "ports/1/counters/wait": "RESET_OR_WRAP",
        "ports/1/counters/ok": 0,
    }


def test_process_records_all_threads_and_only_allowed_environment(tmp_path: Path) -> None:
    root = tmp_path / "42"
    for tid, cpus in (("42", "0-3"), ("43", "4-7")):
        task = root / "task" / tid
        task.mkdir(parents=True)
        (task / "status").write_text(f"Name:\tworker\nCpus_allowed_list:\t{cpus}\nMems_allowed_list:\t0-1\n")
    (root / "environ").write_bytes(b"MPI_UCX_NET_DEVICES=mlx5_0:1\0OPENAI_API_KEY=do-not-record\0")
    result = MODULE["process"](42, tmp_path)
    assert result["threads"]["42"]["Cpus_allowed_list"] == "0-3"
    assert result["threads"]["43"]["Cpus_allowed_list"] == "4-7"
    assert result["environment"] == {"MPI_UCX_NET_DEVICES": "mlx5_0:1"}
    assert "UNAVAILABLE" in result["numa_maps"]


def test_latency_percentiles_preserve_tail() -> None:
    values = [1.0] * 98 + [100.0, 1000.0]
    assert MODULE["summarize"](values) == {"p50_us": 1.0, "p99_us": 100.0, "max_us": 1000.0}
    with pytest.raises(ValueError, match="No measured"):
        MODULE["summarize"]([])


def test_model_manifest_reports_metadata_and_sizes(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"glm"}')
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    result = MODULE["model_manifest"](tmp_path)
    assert len(result["metadata_sha256"]["config.json"]) == 64
    assert result["weight_sizes"] == {"model.safetensors": 7}
    assert "do not prove" in result["note"]


def test_nested_comparison_reports_changed_and_missing_fields() -> None:
    assert MODULE["differences"]({"code": {"a": "same", "b": "old"}}, {"code": {"a": "same", "b": "new"}}) == [
        {"field": "code.b", "left": "old", "right": "new"}
    ]


def test_collect_records_unavailable_counters_without_assuming_no_errors(monkeypatch) -> None:
    collect = MODULE["collect"]
    monkeypatch.setitem(collect.__globals__, "command", lambda argv: {"error": "unavailable"})
    monkeypatch.setitem(collect.__globals__, "nic_snapshot", dict)
    monkeypatch.setitem(collect.__globals__, "code_hashes", dict)
    result = collect(argparse.Namespace(pid=[], model=None, duration=0, interval=1))
    assert result["readable_counter_count"] == 0
    assert result["counter_deltas"] == {}
    assert result["commands"]["gpu"] == {"error": "unavailable"}


def test_communication_reports_buffer_and_object_cases(monkeypatch) -> None:
    import numpy as np

    class Comm:
        def Get_rank(self) -> int:
            return 0

        def Get_size(self) -> int:
            return 2

        def Allgather(self, send, recv) -> None:
            recv[:] = np.arange(2)[:, None]

        def allgather(self, value) -> list[dict]:
            return [{**value, "rank": rank} for rank in range(2)]

        def Barrier(self) -> None:
            pass

        def gather(self, value, root) -> list:
            return [value, value]

    mpi = SimpleNamespace(COMM_WORLD=Comm(), Get_library_version=lambda: "mock MPI")
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=mpi))
    communication = MODULE["communication"]
    monkeypatch.setitem(communication.__globals__, "process", lambda pid: {})
    result = communication(argparse.Namespace(warmup=1, iterations=2, nccl=False))
    assert result["world_size"] == 2
    assert [row.get("bytes_per_rank") for row in result["results"]] == [64, 1024, 8192, None]
    assert result["results"][-1]["operation"] == "MPI_object_allgather"
    assert all("p99_us" in row["slowest_rank_per_iteration"] for row in result["results"])


def test_rendezvous_address_uses_remote_peer_route(monkeypatch) -> None:
    resolve = MODULE["routed_address"]
    sock = resolve.__globals__["socket"]
    monkeypatch.setattr(sock, "gethostname", lambda: "node-a")
    lookup = MagicMock(return_value=[(None, None, None, None, ("192.0.2.2", 9))])
    monkeypatch.setattr(sock, "getaddrinfo", lookup)
    route = MagicMock()
    route.__enter__.return_value = route
    route.getsockname.return_value = ("192.0.2.1", 40000)
    monkeypatch.setattr(sock, "socket", lambda *args: route)
    assert resolve(["node-a", "node-a", "node-b"]) == "192.0.2.1"
    assert lookup.call_args.args[0] == "node-b"
    route.connect.assert_called_once_with(("192.0.2.2", 9))
    route.send.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("manual_endpoint", [False, True])
def test_nccl_store_publishes_bound_port_without_waiting_for_clients(
    monkeypatch, rank: int, manual_endpoint: bool
) -> None:
    setup = MODULE["nccl_store"]
    monkeypatch.setitem(setup.__globals__, "routed_address", lambda hosts: "192.0.2.1")
    endpoint = {"address": "192.0.2.1", "port": 23456, "host": "node-a"}
    comm = SimpleNamespace(
        allgather=lambda host: ["node-a", "node-b"],
        Get_rank=lambda: rank,
        Get_size=lambda: 2,
        bcast=lambda value, root: value if rank == 0 else endpoint,
    )
    store = SimpleNamespace(port=23456)
    factory = MagicMock(return_value=store)
    result, published = setup(
        comm,
        SimpleNamespace(TCPStore=factory),
        "192.0.2.1" if manual_endpoint else None,
        23456 if manual_endpoint else None,
    )
    assert result is store
    assert published == endpoint
    assert factory.call_args.args == ("192.0.2.1", 23456 if rank or manual_endpoint else 0, 2, rank == 0)
    if rank == 0:
        assert factory.call_args.kwargs["wait_for_workers"] is False


def test_rendezvous_setup_error_is_broadcast_to_every_rank(monkeypatch) -> None:
    setup = MODULE["nccl_store"]
    monkeypatch.setitem(setup.__globals__, "routed_address", lambda hosts: "192.0.2.1")
    published = []
    comm = SimpleNamespace(
        allgather=lambda host: ["node-a", "node-b"],
        Get_rank=lambda: 0,
        Get_size=lambda: 2,
        bcast=lambda value, root: published.append(value) or value,
    )
    with pytest.raises(RuntimeError, match="port unavailable"):
        setup(comm, SimpleNamespace(TCPStore=MagicMock(side_effect=RuntimeError("port unavailable"))), None, None)
    assert published == [{"error": "port unavailable"}]
