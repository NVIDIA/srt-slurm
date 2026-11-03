# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-node RDMA NIC traffic sampler.

Reads every RDMA port's cumulative transmit and receive byte counters from
sysfs once per interval and appends them, as raw cumulative bytes, to a
per-node CSV. Rates are computed later, per benchmark window, by
``srtctl.analysis.nic_traffic_report``.

It runs on the bare host and imports only the standard library, so it also
runs as a copied file::

    timeout -s TERM 30 python3 nic_traffic.py --output-dir /tmp/nic

Which sysfs counter carries RDMA payload bytes depends on the driver; the
``COUNTER_SOURCES`` table names it per driver.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
# timestamp_unix is the midpoint of that port's own counter reads; sample_seq
# numbers the sampler passes from 0 and is shared by every port read in a pass.
SAMPLES_HEADER = (
    "timestamp_unix",
    "sample_seq",
    "hostname",
    "device",
    "port",
    "netdev",
    "driver",
    "tx_bytes",
    "rx_bytes",
)


@dataclass(frozen=True)
class CounterSource:
    """Where a driver exposes a port's cumulative RDMA byte counters.

    Paths are relative to ``/sys/class/infiniband/<device>/ports/<port>/``;
    a reading times ``bytes_per_unit`` is bytes.
    """

    name: str
    tx_path: str
    rx_path: str
    bytes_per_unit: int


# The IB core's PMA counters, which the kernel creates for every driver that
# implements process_mad (mlx5 and the other InfiniBand and RoCE drivers). The
# kernel ABI defines port_xmit_data and port_rcv_data as data octets divided by 4.
PORT_DATA_COUNTERS = CounterSource(
    name="counters/port_data",
    tx_path="counters/port_xmit_data",
    rx_path="counters/port_rcv_data",
    bytes_per_unit=4,
)

# Keyed by the driver bound to /sys/class/infiniband/<device>/device. A driver
# without a row reads PORT_DATA_COUNTERS.
COUNTER_SOURCES: dict[str, CounterSource] = {
    # AMD Pollara: ionic implements no process_mad, so the port has no counters/
    # directory; its firmware-named hw_counters carry the RDMA payload. Unicast
    # bytes matched an ib_write_bw payload exactly. The multicast, ccl_cts and
    # retransmit byte counters are left out: their overlap with unicast is
    # undocumented, and retransmits would count the same payload twice.
    "ionic": CounterSource(
        name="hw_counters/rdma_ucast_bytes",
        tx_path="hw_counters/tx_rdma_ucast_bytes",
        rx_path="hw_counters/rx_rdma_ucast_bytes",
        bytes_per_unit=1,
    ),
}


@dataclass(frozen=True)
class RdmaPort:
    """One RDMA port whose byte counters were readable at startup."""

    device: str
    port: str
    netdev: str
    driver: str
    source: CounterSource
    port_dir: Path

    def read_bytes(self) -> tuple[int, int]:
        """Cumulative (tx, rx) bytes."""
        scale = self.source.bytes_per_unit
        return (
            _read_counter(self.port_dir / self.source.tx_path) * scale,
            _read_counter(self.port_dir / self.source.rx_path) * scale,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "port": self.port,
            "netdev": self.netdev,
            "driver": self.driver,
            "counter_source": self.source.name,
            "tx_counter": self.source.tx_path,
            "rx_counter": self.source.rx_path,
            "bytes_per_unit": self.source.bytes_per_unit,
        }


@dataclass(frozen=True)
class UnsupportedPort:
    """A device or port without readable byte counters, recorded once at startup."""

    device: str
    port: str
    driver: str
    reason: str

    def describe(self) -> dict[str, Any]:
        return {"device": self.device, "port": self.port, "driver": self.driver, "reason": self.reason}


def _read_counter(path: Path) -> int:
    return int(path.read_text().strip())


def _driver_name(device_dir: Path) -> str:
    try:
        return os.path.basename(os.readlink(device_dir / "device" / "driver"))
    except OSError:
        return ""


def _netdev_name(device_dir: Path) -> str:
    try:
        return "+".join(sorted(entry.name for entry in (device_dir / "device" / "net").iterdir()))
    except OSError:
        return ""


def _port_order(port_dir: Path) -> tuple[int, str]:
    return (int(port_dir.name), "") if port_dir.name.isdigit() else (0, port_dir.name)


def discover_ports(sysfs_root: Path) -> tuple[list[RdmaPort], list[UnsupportedPort]]:
    """Every RDMA port under ``<sysfs_root>/class/infiniband``, split by whether its counters read."""
    try:
        device_dirs = sorted((sysfs_root / "class" / "infiniband").iterdir())
    except OSError:
        return [], []
    ports: list[RdmaPort] = []
    unsupported: list[UnsupportedPort] = []
    for device_dir in device_dirs:
        driver = _driver_name(device_dir)
        source = COUNTER_SOURCES.get(driver, PORT_DATA_COUNTERS)
        try:
            port_dirs = sorted((device_dir / "ports").iterdir(), key=_port_order)
        except OSError as exc:
            unsupported.append(UnsupportedPort(device_dir.name, "", driver, f"no ports directory: {exc}"))
            continue
        for port_dir in port_dirs:
            try:
                _read_counter(port_dir / source.tx_path)
                _read_counter(port_dir / source.rx_path)
            except (OSError, ValueError) as exc:
                unsupported.append(UnsupportedPort(device_dir.name, port_dir.name, driver, f"{source.name}: {exc}"))
                continue
            ports.append(RdmaPort(device_dir.name, port_dir.name, _netdev_name(device_dir), driver, source, port_dir))
    return ports, unsupported


def node_hostname() -> str:
    """The Slurm node name, which is what the head node expects in every row."""
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".", 1)[0]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def collect(*, output_dir: Path, ready_dir: Path, interval_seconds: float, sysfs_root: Path) -> int:
    """Sample until SIGTERM/SIGINT into ``<output_dir>/<host>.csv`` and mark readiness in ``ready_dir``."""
    os.umask(0o002)
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_dir.mkdir(parents=True, exist_ok=True)
    hostname = node_hostname()
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    ports, unsupported = discover_ports(sysfs_root)
    metadata_path = output_dir / f"{hostname}.metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hostname": hostname,
        "sysfs_root": str(sysfs_root),
        "sample_interval_seconds": interval_seconds,
        "started_at_unix": time.time(),
        "devices": [port.describe() for port in ports],
        "unsupported": [entry.describe() for entry in unsupported],
    }
    _atomic_json(metadata_path, metadata)
    if not ports:
        _atomic_json(
            ready_dir / f"{hostname}.error.json",
            {"hostname": hostname, "error": "no RDMA port with readable byte counters"},
        )
        # Exit 0: a node without RDMA counters is a recorded outcome, not a
        # sampler failure, and a nonzero task exit can end the other nodes'
        # samplers in the same step on clusters that set KillOnBadExit.
        return 0

    row_count = 0
    read_failures = 0
    with (output_dir / f"{hostname}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLES_HEADER)
        handle.flush()
        _atomic_json(
            ready_dir / f"{hostname}.ready.json",
            {"hostname": hostname, "port_count": len(ports), "ready_at_unix": time.time()},
        )
        next_sample = time.monotonic()
        sample_seq = 0
        while not stop:
            for port in ports:
                # One ionic counter read can take ~100 ms, so a pass over many
                # ports can last hundreds of ms; stamp each row at its own reads.
                read_start = time.time()
                try:
                    tx_bytes, rx_bytes = port.read_bytes()
                except (OSError, ValueError):
                    read_failures += 1
                    continue
                timestamp = (read_start + time.time()) / 2
                writer.writerow(
                    (
                        repr(timestamp),
                        sample_seq,
                        hostname,
                        port.device,
                        port.port,
                        port.netdev,
                        port.driver,
                        tx_bytes,
                        rx_bytes,
                    )
                )
                row_count += 1
            handle.flush()
            sample_seq += 1
            next_sample += interval_seconds
            time.sleep(max(0.0, next_sample - time.monotonic()))
    metadata.update({"ended_at_unix": time.time(), "sample_row_count": row_count, "read_failure_count": read_failures})
    _atomic_json(metadata_path, metadata)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample RDMA NIC byte counters from sysfs until SIGTERM.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for <host>.csv and metadata")
    parser.add_argument("--ready-dir", type=Path, help="Directory for the ready/error marker (default: output dir)")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0:
        parser.error("--interval-seconds must be finite and positive")
    return collect(
        output_dir=args.output_dir,
        ready_dir=args.ready_dir or args.output_dir,
        interval_seconds=args.interval_seconds,
        sysfs_root=args.sysfs_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
