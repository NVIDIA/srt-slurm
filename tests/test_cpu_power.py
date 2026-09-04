# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU power reader and artifact lifecycle tests."""

from __future__ import annotations

import csv
import json
import types
from datetime import datetime
from pathlib import Path

import pytest

from srtctl.core import cpu_power
from srtctl.core.cpu_power import (
    SAMPLES_HEADER,
    AcpiPowerMeterReader,
    CpuPowerSourceUnavailable,
    _add_standard_dcgm_binding_path,
    format_local_timestamp,
)
from srtctl.core.cpu_power_session import CpuPowerSessionSettings, CpuPowerTelemetrySession


def _make_acpi_sensor(
    root: Path,
    *,
    hwmon_id: int,
    socket_id: int,
    microwatts: int,
    domain: str | None = None,
    value_attribute: str = "average",
) -> None:
    hwmon = root / f"hwmon{hwmon_id}"
    device = hwmon / "device"
    device.mkdir(parents=True)
    (hwmon / "name").write_text("power_meter\n")
    (device / f"power1_{value_attribute}").write_text(f"{microwatts}\n")
    (device / "power1_oem_info").write_text(f"{domain or f'CPU Power Socket {socket_id}'}\n")
    (device / "power1_accuracy").write_text("1\n")
    (device / "power1_average_interval").write_text("100\n")


def test_acpi_reader_preserves_legacy_cpu_socket_total_name(tmp_path: Path) -> None:
    _make_acpi_sensor(tmp_path, hwmon_id=0, socket_id=0, microwatts=125_500_000)
    reader = AcpiPowerMeterReader(tmp_path)

    readings = reader.read_watts()
    assert readings == {"CPU0:cpuPowerUsageW": 125.5}
    assert reader.aggregate_watts(readings) == 125.5
    metadata = reader.metadata()
    assert metadata["source"] == "acpi"
    assert metadata["sensors"][0]["socket_id"] == 0
    assert metadata["sensors"][0]["average_interval_ms"] == 100
    assert metadata["aggregate_scope"] == "cpu_side_socket_total"


def test_acpi_reader_collects_breakdowns_without_double_counting_total(tmp_path: Path) -> None:
    domains = (
        (0, "Total Power in uW socket 0", 150_000_000),
        (1, "CPU Rail Power in uW socket 0", 70_000_000),
        (2, "SOC Rail Power in uW socket 0", 6_000_000),
        (3, "DRAM Power in uW socket 0", 8_000_000),
        (4, "CPU Rail Output Power in uW socket 0", 55_000_000),
        (5, "Total CPU Energy In uJ socket 0", 1_000_000),
        (6, "Chipthrot DDR Throttle (samples x1000) socket 0", 2_000),
        (7, "Total Power in uW socket 1", 160_000_000),
        (8, "CPU Rail Power in uW socket 1", 75_000_000),
        (9, "SOC Rail Power in uW socket 1", 7_000_000),
        (10, "DRAM Power in uW socket 1", 9_000_000),
    )
    for hwmon_id, domain, microwatts in domains:
        socket_id = 1 if domain.endswith("socket 1") else 0
        _make_acpi_sensor(
            tmp_path,
            hwmon_id=hwmon_id,
            socket_id=socket_id,
            microwatts=microwatts,
            domain=domain,
        )

    reader = AcpiPowerMeterReader(tmp_path)
    readings = reader.read_watts()

    assert readings == {
        "CPU0:cpuSidePowerUsageW": 150.0,
        "CPU0:cpuRailPowerUsageW": 70.0,
        "CPU0:socPowerUsageW": 6.0,
        "CPU0:dramPowerUsageW": 8.0,
        "CPU1:cpuSidePowerUsageW": 160.0,
        "CPU1:cpuRailPowerUsageW": 75.0,
        "CPU1:socPowerUsageW": 7.0,
        "CPU1:dramPowerUsageW": 9.0,
    }
    assert reader.aggregate_watts(readings) == 310.0
    assert {sensor["domain_kind"] for sensor in reader.metadata()["sensors"]} == {
        "total",
        "cpu_rail",
        "soc",
        "dram",
    }


def test_acpi_reader_accepts_input_only_hwmon_channel(tmp_path: Path) -> None:
    _make_acpi_sensor(
        tmp_path,
        hwmon_id=0,
        socket_id=0,
        microwatts=141_250_000,
        domain="Total Power in uW socket 0",
        value_attribute="input",
    )

    reader = AcpiPowerMeterReader(tmp_path)

    assert reader.read_watts() == {"CPU0:cpuSidePowerUsageW": 141.25}


def test_acpi_reader_does_not_publish_partial_socket_total(tmp_path: Path) -> None:
    _make_acpi_sensor(
        tmp_path,
        hwmon_id=0,
        socket_id=0,
        microwatts=150_000_000,
        domain="Total Power in uW socket 0",
    )
    _make_acpi_sensor(
        tmp_path,
        hwmon_id=1,
        socket_id=1,
        microwatts=160_000_000,
        domain="Total Power in uW socket 1",
    )
    reader = AcpiPowerMeterReader(tmp_path)

    assert reader.aggregate_watts({"CPU0:cpuSidePowerUsageW": 150.0}) is None


def test_acpi_reader_rejects_missing_cpu_domains(tmp_path: Path) -> None:
    with pytest.raises(CpuPowerSourceUnavailable, match="no ACPI"):
        AcpiPowerMeterReader(tmp_path)


def test_format_local_timestamp_round_trips_to_the_same_unix_time() -> None:
    timestamp = 1_788_310_143.627448

    local = format_local_timestamp(timestamp)

    assert datetime.fromisoformat(local).timestamp() == pytest.approx(timestamp)
    assert datetime.fromisoformat(local).utcoffset() is not None


def test_standard_dcgm_binding_path_is_discovered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding_dir = tmp_path / "dcgm-bindings"
    binding_dir.mkdir()
    (binding_dir / "dcgm_agent.py").write_text("# probe\n")
    monkeypatch.setattr(cpu_power, "DCGM_PYTHON_BINDING_DIRS", (binding_dir,))
    monkeypatch.setattr(cpu_power.sys, "path", [path for path in cpu_power.sys.path if path != str(binding_dir)])

    discovered = _add_standard_dcgm_binding_path()

    assert discovered == binding_dir
    assert cpu_power.sys.path[0] == str(binding_dir)


def test_dcgm_reader_watches_cpu_power_before_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []

    class FakeHandle:
        handle = object()

        def Shutdown(self) -> None:
            events.append("shutdown")

    class FakeSamples:
        def WatchFields(self, field_group: object, frequency: int, age: float, samples: int) -> None:
            events.append(("watch", frequency, age, samples))

        def UnwatchFields(self, field_group: object) -> None:
            events.append("unwatch")

    class FakeGroup:
        def __init__(self, handle: object, **kwargs: object) -> None:
            self.samples = FakeSamples()

        def AddEntity(self, entity_group: int, entity_id: int) -> None:
            events.append(("entity", entity_group, entity_id))

        def Delete(self) -> None:
            events.append("group_delete")

    class FakeFieldGroup:
        def __init__(self, handle: object, **kwargs: object) -> None:
            events.append(("field_group", kwargs["fieldIds"]))

        def Delete(self) -> None:
            events.append("field_group_delete")

    class FakeEntity:
        entityGroupId = 0
        entityId = 0

    fake_structs = types.SimpleNamespace(
        DCGM_GEGE_FLAG_ONLY_SUPPORTED=1,
        DCGM_GROUP_EMPTY=0,
        DCGM_FV_FLAG_LIVE_DATA=1,
        DCGM_ST_OK=0,
        c_dcgmGroupEntityPair_t=FakeEntity,
    )
    fake_fields = types.SimpleNamespace(DCGM_FE_CPU=7)

    def fake_latest_values(_handle: object, _entities: object, _fields: object, flags: int) -> list[object]:
        events.append(("latest", flags))
        return []

    fake_agent = types.SimpleNamespace(
        dcgmGetEntityGroupEntities=lambda *_args: [0, 1],
        dcgmEntitiesGetLatestValues=fake_latest_values,
        dcgmUpdateAllFields=lambda _handle, wait: events.append(("update", wait)),
    )
    fake_pydcgm = types.SimpleNamespace(
        DcgmHandle=lambda **_kwargs: FakeHandle(),
        DcgmGroup=FakeGroup,
        DcgmFieldGroup=FakeFieldGroup,
    )
    modules = {
        "dcgm_agent": fake_agent,
        "dcgm_fields": fake_fields,
        "dcgm_structs": fake_structs,
        "pydcgm": fake_pydcgm,
    }
    monkeypatch.setattr(cpu_power, "_add_standard_dcgm_binding_path", lambda: None)
    monkeypatch.setattr(cpu_power.importlib, "import_module", modules.__getitem__)

    reader = cpu_power.DcgmCpuPowerReader()
    reader.read_watts()
    reader.close()

    assert ("entity", 7, 0) in events
    assert ("entity", 7, 1) in events
    assert ("field_group", [cpu_power.CPU_POWER_FIELD_ID]) in events
    assert ("watch", 100_000, 60.0, 600) in events
    assert ("update", True) in events
    assert ("latest", 0) in events
    assert events[-4:] == ["unwatch", "field_group_delete", "group_delete", "shutdown"]


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
    timestamp_local = format_local_timestamp(timestamp)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLES_HEADER)
        writer.writerow((2, timestamp, timestamp_local, hostname, "acpi", "CPU0:cpuPowerUsageW", 0, watts, watts))


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
    assert [row[3] for row in rows[1:]] == ["node-b", "node-a"]
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
