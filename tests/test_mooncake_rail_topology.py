# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep optional fabric evidence separate from the required RDMA preflight."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import mooncake_rail_worker as worker


@pytest.mark.parametrize("optional_failure", ["", "missing", "permission", "timeout"])
def test_topology_evidence_preserves_sysfs_details_and_optional_failures_are_nonfatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, optional_failure: str
) -> None:
    sysfs = tmp_path / "sys/class/infiniband/mlx5_0"
    entries = {
        "device/numa_node": "1",
        "ports/1/state": "4: ACTIVE",
        "ports/1/phys_state": "5: LinkUp",
        "ports/1/link_layer": "InfiniBand",
        "ports/1/rate": "400 Gb/sec (4X NDR)",
        "ports/1/lid": "0x0042",
        "ports/1/sm_lid": "0x0001",
        "ports/1/active_mtu": "4096",
        "ports/1/gids/0": "fe80:0000:0000:0000:0200:0000:0000:0042",
        "ports/1/gids/1": "fe80:0000:0000:0000:0200:0000:0000:0043",
        "ports/1/pkeys/0": "0xffff",
        "node_guid": "0200:0000:0000:0042",
        "sys_image_guid": "0200:0000:0000:0040",
        "device/current_link_speed": "32.0 GT/s PCIe",
        "device/current_link_width": "16",
        "device/max_link_speed": "32.0 GT/s PCIe",
        "device/max_link_width": "16",
    }
    for relative, value in entries.items():
        path = sysfs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")
    read_text, sysfs_table = worker.read_text, worker.sysfs_table

    def read(path: Path) -> str | None:
        if str(path).startswith("/sys/class/infiniband/"):
            return read_text(tmp_path / str(path).lstrip("/"))
        if str(path) == "/proc/self/status":
            return "Cpus_allowed_list: 0-3\nMems_allowed_list: 0-1\n"
        if str(path) == "/proc/meminfo":
            return "MemTotal: 16000000 kB\nMemAvailable: 14000000 kB\n"
        if path.is_relative_to(tmp_path):
            return read_text(path)
        return None

    optional_tools = {"ibstat", "ibv_devinfo", "iblinkinfo", "rdma"}
    captures = []

    def capture(argv: list[str], timeout: float = 3) -> dict[str, Any]:
        captures.append((argv, timeout))
        if Path(argv[0]).name in optional_tools:
            if optional_failure == "timeout":
                return {"argv": argv, "error": "Command timed out"}
            if optional_failure == "permission":
                return {"argv": argv, "returncode": 1, "stdout": "", "stderr": "UMAD permission denied"}
        return {"argv": argv, "returncode": 0, "stdout": "synthetic evidence", "stderr": ""}

    monkeypatch.setenv("MC_RAIL_NICS", "mlx5_0")
    monkeypatch.setattr(worker, "read_text", read)
    monkeypatch.setattr(worker, "sysfs_table", lambda path: sysfs_table(tmp_path / str(path).lstrip("/")))
    monkeypatch.setattr(worker, "capture", capture)
    monkeypatch.setattr(worker, "cgroup_evidence", lambda: {})
    monkeypatch.setattr(worker.socket, "getaddrinfo", lambda *args: [(2, 1, 6, "", ("10.20.30.40", 0))])
    monkeypatch.setattr(
        worker.shutil,
        "which",
        lambda name: None if optional_failure == "missing" and name in optional_tools else f"/usr/bin/{name}",
    )

    info = worker.inventory("selected-node")

    assert info["ready"] and not info["errors"]
    nic = info["nics"]["mlx5_0"]
    assert nic["rate"] == "400 Gb/sec (4X NDR)"
    assert nic["lid"] == "0x0042" and nic["sm_lid"] == "0x0001"
    assert nic["node_guid"] == "0200:0000:0000:0042"
    assert nic["sys_image_guid"] == "0200:0000:0000:0040"
    assert nic["gid_prefix"] == "fe80:0000:0000:0000"
    assert nic["gids"] == {"0": entries["ports/1/gids/0"], "1": entries["ports/1/gids/1"]}
    assert nic["pkeys"] == {"0": "0xffff"}
    assert nic["pci_link"]["current_link_width"] == "16"
    assert nic["numa_node"] == 1
    hardware = info["host_hardware"]
    assert hardware["lscpu"]["argv"] == ["lscpu", "--json"]
    assert hardware["numactl_hardware"]["argv"] == ["/usr/bin/numactl", "--hardware"]
    assert "MemAvailable: 14000000 kB" in hardware["meminfo"]
    diagnostics = info["optional_diagnostics"]
    first_hop = diagnostics["iblinkinfo"]["mlx5_0"]
    assert first_hop["argv"][1:] == ["-C", "mlx5_0", "-P", "1", "-D", "0", "--hops", "1", "--all", "--line"]
    if optional_failure == "missing":
        assert "unavailable" in first_hop
    else:
        assert (first_hop["argv"], 5) in captures
        if optional_failure == "permission":
            assert first_hop["returncode"] == 1 and "permission denied" in first_hop["stderr"]
        elif optional_failure == "timeout":
            assert "timed out" in first_hop["error"]
    assert not any("ibnetdiscover" in argv[0] for argv, _ in captures)


def test_capture_bounds_subprocess_runtime_and_preserves_timeout_as_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 5
        assert kwargs["capture_output"] and kwargs["text"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(worker.subprocess, "run", run)
    argv = ["iblinkinfo", "-C", "mlx5_0", "-D", "0", "--hops", "1"]

    result = worker.capture(argv, timeout=5)

    assert result["argv"] == argv
    assert "timed out" in result["error"]
    assert "returncode" not in result


def test_optional_sysfs_tables_handle_missing_or_inaccessible_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    assert worker.sysfs_table(tmp_path / "missing") == {}
    assert worker.optional_diagnostics([])["iblinkinfo"] == {}
