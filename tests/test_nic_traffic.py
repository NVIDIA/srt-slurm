# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The RDMA NIC traffic sampler against fake sysfs trees."""

from __future__ import annotations

import csv
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from srtctl.core import nic_traffic
from srtctl.core.nic_traffic import SAMPLES_HEADER, discover_ports


def _rdma_device(
    sysfs: Path, name: str, *, driver: str, netdev: str, counters: dict[str, int], port: str = "1"
) -> Path:
    """``/sys/class/infiniband/<name>`` with a driver symlink, a netdev and port counter files."""
    driver_dir = sysfs / "bus" / "pci" / "drivers" / driver
    driver_dir.mkdir(parents=True, exist_ok=True)
    device_dir = sysfs / "class" / "infiniband" / name
    (device_dir / "device" / "net" / netdev).mkdir(parents=True)
    (device_dir / "device" / "driver").symlink_to(driver_dir)
    port_dir = device_dir / "ports" / port
    for relative, value in counters.items():
        path = port_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{value}\n")
    return port_dir


def test_ionic_reads_unicast_hw_counters_as_bytes(tmp_path: Path) -> None:
    _rdma_device(
        tmp_path,
        "rdma0",
        driver="ionic",
        netdev="enp1s0",
        counters={
            "hw_counters/tx_rdma_ucast_bytes": 490_015_000_000,
            "hw_counters/rx_rdma_ucast_bytes": 7,
            "hw_counters/tx_rdma_mcast_bytes": 111,
            "hw_counters/tx_rdma_retx_bytes": 222,
            "hw_counters/tx_rdma_ccl_cts_bytes": 333,
        },
    )

    ports, unsupported = discover_ports(tmp_path)

    assert unsupported == []
    [port] = ports
    assert (port.device, port.port, port.netdev, port.driver) == ("rdma0", "1", "enp1s0", "ionic")
    assert port.read_bytes() == (490_015_000_000, 7)
    assert port.describe()["counter_source"] == "hw_counters/rdma_ucast_bytes"


def test_standard_driver_reads_port_data_in_four_octet_units(tmp_path: Path) -> None:
    _rdma_device(
        tmp_path,
        "mlx5_0",
        driver="mlx5_core",
        netdev="ib0",
        counters={"counters/port_xmit_data": 250, "counters/port_rcv_data": 100},
    )

    [port], unsupported = discover_ports(tmp_path)

    assert unsupported == []
    assert port.driver == "mlx5_core"
    assert port.read_bytes() == (1000, 400)
    assert port.describe()["bytes_per_unit"] == 4


def test_port_without_its_drivers_counters_is_unsupported_not_sampled(tmp_path: Path) -> None:
    _rdma_device(
        tmp_path,
        "bnxt_re0",
        driver="bnxt_en",
        netdev="eth0",
        counters={"hw_counters/tx_bytes": 5, "hw_counters/rx_bytes": 5},
    )
    _rdma_device(
        tmp_path,
        "mlx5_1",
        driver="mlx5_core",
        netdev="ib1",
        counters={"counters/port_xmit_data": 1, "counters/port_rcv_data": 1},
    )

    ports, unsupported = discover_ports(tmp_path)

    assert [port.device for port in ports] == ["mlx5_1"]
    [entry] = unsupported
    assert (entry.device, entry.port, entry.driver) == ("bnxt_re0", "1", "bnxt_en")
    assert "counters/port_xmit_data" in entry.reason


def test_host_without_rdma_has_nothing_to_sample(tmp_path: Path) -> None:
    assert discover_ports(tmp_path) == ([], [])


def _wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def _run_standalone(sysfs: Path, out: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    # -I -S: no site-packages, so srtctl is not importable and the file must run on the stdlib alone.
    return subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(Path(nic_traffic.__file__)),
            "--output-dir",
            str(out),
            "--interval-seconds",
            "0.05",
            "--sysfs-root",
            str(sysfs),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_standalone_file_samples_raw_counters_until_sigterm(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    port_dir = _rdma_device(
        sysfs,
        "rdma0",
        driver="ionic",
        netdev="enp1s0",
        counters={"hw_counters/tx_rdma_ucast_bytes": 100, "hw_counters/rx_rdma_ucast_bytes": 200},
    )
    out = tmp_path / "out"
    proc = _run_standalone(sysfs, out, {"SLURMD_NODENAME": "node-a"})
    try:
        _wait_for(out / "node-a.ready.json")
        (port_dir / "hw_counters/tx_rdma_ucast_bytes").write_text("900\n")
        deadline = time.monotonic() + 10
        while "900" not in (out / "node-a.csv").read_text():
            assert time.monotonic() < deadline, "the sampler never re-read the counter"
            time.sleep(0.02)
    finally:
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0, stderr
    with (out / "node-a.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == SAMPLES_HEADER
    assert rows[1][2:] == ["node-a", "rdma0", "1", "enp1s0", "ionic", "100", "200"]
    assert rows[-1][2:] == ["node-a", "rdma0", "1", "enp1s0", "ionic", "900", "200"]
    timestamps = [float(row[0]) for row in rows[1:]]
    assert timestamps == sorted(timestamps)
    assert [int(row[1]) for row in rows[1:]] == list(range(len(rows) - 1))
    metadata = json.loads((out / "node-a.metadata.json").read_text())
    assert metadata["sample_row_count"] == len(rows) - 1
    assert [device["device"] for device in metadata["devices"]] == ["rdma0"]


class _SlowCounterClock:
    """Stands in for the sampler's ``time`` module; every counter read advances it by 0.1 s."""

    def __init__(self, passes: int) -> None:
        self.now = 1000.0
        self.passes_left = passes
        self.signal_handlers = {}

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, _seconds: float) -> None:
        self.passes_left -= 1
        if self.passes_left == 0:
            self.signal_handlers[signal.SIGTERM](signal.SIGTERM, None)


def test_each_row_is_stamped_at_its_own_port_reads_and_shares_the_pass_seq(tmp_path: Path, monkeypatch) -> None:
    sysfs = tmp_path / "sys"
    for name in ("rdma0", "rdma1"):
        _rdma_device(
            sysfs,
            name,
            driver="ionic",
            netdev=f"net-{name}",
            counters={"hw_counters/tx_rdma_ucast_bytes": 0, "hw_counters/rx_rdma_ucast_bytes": 0},
        )
    clock = _SlowCounterClock(passes=2)
    read_counter = nic_traffic._read_counter

    def slow_read(path: Path) -> int:
        clock.now += 0.1
        return read_counter(path)

    monkeypatch.setattr(nic_traffic, "time", clock)
    monkeypatch.setattr(nic_traffic, "_read_counter", slow_read)
    monkeypatch.setattr(nic_traffic.signal, "signal", clock.signal_handlers.__setitem__)
    monkeypatch.setenv("SLURMD_NODENAME", "node-a")
    out = tmp_path / "out"

    nic_traffic.collect(output_dir=out, ready_dir=out, interval_seconds=1.0, sysfs_root=sysfs)

    with (out / "node-a.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    assert [(row[1], row[3]) for row in rows] == [("0", "rdma0"), ("0", "rdma1"), ("1", "rdma0"), ("1", "rdma1")]
    # Each port's tx and rx reads take 0.2 s, so rdma1 is read 0.2 s after rdma0 in the same pass;
    # each row carries the midpoint of its own two reads.
    ready_at = json.loads((out / "node-a.ready.json").read_text())["ready_at_unix"]
    assert [round(float(row[0]) - ready_at, 6) for row in rows] == [0.1, 0.3, 0.5, 0.7]


def test_node_without_counters_writes_an_error_marker_and_exits_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = _run_standalone(tmp_path / "sys", out, {"SLURMD_NODENAME": "node-b"})
    _, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0, stderr
    assert "no RDMA port" in json.loads((out / "node-b.error.json").read_text())["error"]
    assert not (out / "node-b.csv").exists()
    assert not (out / "node-b.ready.json").exists()
