# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU power reader and artifact lifecycle tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from srtctl.core.cpu_power import SAMPLES_HEADER, AcpiPowerMeterReader, CpuPowerSourceUnavailable
from srtctl.core.cpu_power_session import CpuPowerSessionSettings, CpuPowerTelemetrySession


def _make_acpi_sensor(root: Path, *, socket_id: int, microwatts: int) -> None:
    hwmon = root / f"hwmon{socket_id}"
    device = hwmon / "device"
    device.mkdir(parents=True)
    (hwmon / "name").write_text("power_meter\n")
    (device / "power1_average").write_text(f"{microwatts}\n")
    (device / "power1_oem_info").write_text(f"CPU Power Socket {socket_id}\n")
    (device / "power1_accuracy").write_text("1\n")
    (device / "power1_average_interval").write_text("100\n")


def test_acpi_reader_reports_only_cpu_socket_rails(tmp_path: Path) -> None:
    _make_acpi_sensor(tmp_path, socket_id=0, microwatts=125_500_000)
    reader = AcpiPowerMeterReader(tmp_path)

    assert reader.read_watts() == {"CPU0:cpuPowerUsageW": 125.5}
    metadata = reader.metadata()
    assert metadata["source"] == "acpi"
    assert metadata["sensors"][0]["socket_id"] == 0
    assert metadata["sensors"][0]["average_interval_ms"] == 100


def test_acpi_reader_rejects_missing_cpu_domains(tmp_path: Path) -> None:
    with pytest.raises(CpuPowerSourceUnavailable, match="no ACPI"):
        AcpiPowerMeterReader(tmp_path)


class _FakeProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self.running = True

    @property
    def is_running(self) -> bool:
        return self.running

    def terminate(self) -> None:
        self.running = False


def _write_node_csv(path: Path, hostname: str, timestamp: float, watts: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLES_HEADER)
        writer.writerow((1, timestamp, hostname, "acpi", "CPU0:cpuPowerUsageW", 0, watts, watts))


def test_session_aggregates_every_expected_node(tmp_path: Path) -> None:
    settings = CpuPowerSessionSettings(
        cpu_dir=tmp_path / "cpu",
        job_id="123",
        run_name="run",
        nodes=("node-a", "node-b"),
        source="auto",
        sample_interval_seconds=0.1,
        startup_timeout_seconds=1.0,
        required=True,
    )
    session = CpuPowerTelemetrySession(settings)
    session.initialize()
    session.add_process(_FakeProcess("cpu"))  # type: ignore[arg-type]
    _write_node_csv(session.samples_dir / "node-a.csv", "node-a", 2.0, 100.0)
    _write_node_csv(session.samples_dir / "node-b.csv", "node-b", 1.0, 110.0)

    outcome = session.stop_and_finalize()

    assert outcome.publication_valid is True
    with session.samples_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(SAMPLES_HEADER)
    assert [row[2] for row in rows[1:]] == ["node-b", "node-a"]
    manifest = json.loads(session.manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["sample_row_count"] == 2


def test_required_session_fails_when_a_node_has_no_samples(tmp_path: Path) -> None:
    settings = CpuPowerSessionSettings(
        cpu_dir=tmp_path / "cpu",
        job_id="123",
        run_name="run",
        nodes=("node-a", "node-b"),
        source="auto",
        sample_interval_seconds=0.1,
        startup_timeout_seconds=1.0,
        required=True,
    )
    session = CpuPowerTelemetrySession(settings)
    session.initialize()
    _write_node_csv(session.samples_dir / "node-a.csv", "node-a", 1.0, 100.0)

    outcome = session.stop_and_finalize()

    assert outcome.publication_valid is False
    assert outcome.exit_nonzero is True
    assert "cpu_node_samples_missing" in outcome.reason_codes
