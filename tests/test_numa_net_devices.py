# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise per-worker NIC discovery and exports with a synthetic sysfs tree."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("node", ["0", "1"])
@pytest.mark.parametrize("mode", ["filters", "defaults", "excluded", "no-match", "unknown-nic", "explicit-ethernet"])
def test_worker_network_affinity(tmp_path: Path, node: str, mode: str) -> None:
    sysfs = tmp_path / "sys"

    def write(path: str, value: str) -> None:
        target = sysfs / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)

    write("bus/pci/devices/0000:ab:00.0/numa_node", node)
    write(f"devices/system/node/node{node}/cpulist", "0-3")
    for name, affinity in (("mlx5_0", "0"), ("mlx5_1", "1"), ("mlx5_11", "1"), ("mlx5_9", "-1")):
        write(f"class/infiniband/{name}/device/numa_node", affinity)
        (sysfs / "class/infiniband" / name / "ports/1").mkdir(parents=True)
    for name, affinity in (("eth0", "0"), ("eth1", "1"), ("eth9", "-1")):
        write(f"class/net/{name}/device/numa_node", affinity)
        write(f"class/net/{name}/type", "1")
    if mode == "explicit-ethernet":
        # An unlisted local HCA sharing the mlx5_1 prefix must stay excluded.
        write("class/infiniband/mlx5_10/device/numa_node", node)
        (sysfs / "class/infiniband/mlx5_10/ports/1").mkdir(parents=True)
        write("class/net/eth0/device/numa_node", "-1")
    if mode == "unknown-nic":
        for path in (sysfs / "class").glob("*/*/device/numa_node"):
            path.write_text("-1")

    configs = Path(__file__).resolve().parents[1] / "configs"
    script = tmp_path / "numa_cpu_bind.sh"
    script.write_text((configs / script.name).read_text().replace("/sys/", f"{sysfs}/"))
    helper = tmp_path / "numa_net_devices.py"
    helper.write_text((configs / helper.name).read_text().replace('Path("/sys")', f"Path({str(sysfs)!r})"))
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf '%s\\n' '00000000:AB:00.0'\n")
    smi.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "SLURM_LOCALID": "0"}
    keys = ("MPI_UCX_NET_DEVICES", "UCX_NET_DEVICES", "NCCL_IB_HCA")
    for key in (*keys, "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    if mode != "defaults":
        env.update(
            MPI_UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_11:1",
            UCX_NET_DEVICES="^mlx5_9:1",
            NCCL_IB_HCA="mlx5_0::0:0,mlx5_1::1:0,mlx5_11::3:1",
        )
    if mode == "excluded":
        env["MPI_UCX_NET_DEVICES"] = "^mlx5_9:1"
        env["NCCL_IB_HCA"] = "^=mlx5_9"
    if mode == "no-match":
        env["NCCL_IB_HCA"] = "=mlx5_9"
    if mode == "explicit-ethernet":
        env["UCX_NET_DEVICES"] = "mlx5_0:1,mlx5_1:1,mlx5_11:1,eth0"
        env["NCCL_IB_HCA"] = "=" + env["NCCL_IB_HCA"]
    worker = "import json, os; print(json.dumps({key: os.getenv(key) for key in " + repr(keys) + "}))"
    result = subprocess.run(
        ["bash", str(script), "--no-bind-cpu", sys.executable, "-c", worker],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if mode in ("no-match", "unknown-nic"):
        assert result.returncode == 2
        assert result.stdout == ""
        assert "no permitted devices on NUMA node" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    rdma = "mlx5_0:1" if node == "0" else "mlx5_1:1,mlx5_11:1"
    ucx = f"{rdma},eth{node}"
    nccl = "=mlx5_0:1:0:0" if node == "0" else "=mlx5_1:1:1:0,mlx5_11:1:3:1"
    assert json.loads(result.stdout) == {
        "MPI_UCX_NET_DEVICES": ucx if mode in ("defaults", "excluded") else rdma,
        "UCX_NET_DEVICES": f"{rdma},eth0" if mode == "explicit-ethernet" else ucx,
        "NCCL_IB_HCA": f"={rdma}" if mode in ("defaults", "excluded") else nccl,
    }
    assert all(f"{key}=" in result.stderr for key in keys)
