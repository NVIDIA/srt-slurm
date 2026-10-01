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
    CPU_UTILIZATION_FIELDS,
    SAMPLES_HEADER,
    SAMPLES_HEADER_V2,
    SAMPLES_SCHEMA_VERSION,
    UTILIZATION_COLUMNS,
    AcpiPowerMeterReader,
    CpuPowerSourceUnavailable,
    _add_standard_dcgm_binding_path,
    format_local_timestamp,
)
from srtctl.core.cpu_power_exporter import _build_metrics, _find_power_meter_sensors
from srtctl.core.cpu_power_session import CpuPowerSessionSettings, CpuPowerTelemetrySession
from srtctl.core.power.cpu_rails import RAIL_COLUMN_NAMES
from srtctl.core.power.cpu_sample import CpuSample, RailReading


def _make_acpi_sensor(
    root: Path,
    *,
    hwmon_id: int,
    socket_id: int,
    microwatts: int,
    domain: str | None = None,
    value_attribute: str = "average",
    legacy_layout: bool = False,
) -> None:
    hwmon = root / f"hwmon{hwmon_id}"
    device = hwmon / "device"
    device.mkdir(parents=True)
    # Legacy hwmon_device_register() publishes name only on the parent device.
    (device / "name" if legacy_layout else hwmon / "name").write_text("power_meter\n")
    (device / f"power1_{value_attribute}").write_text(f"{microwatts}\n")
    (device / "power1_oem_info").write_text(f"{domain or f'CPU Power Socket {socket_id}'}\n")
    (device / "power1_accuracy").write_text("1\n")
    (device / "power1_average_interval").write_text("100\n")


def test_acpi_reader_maps_grace_socket_total_and_component_rails(tmp_path: Path) -> None:
    _make_acpi_sensor(
        tmp_path,
        hwmon_id=0,
        socket_id=0,
        microwatts=150_000_000,
        domain="Grace Power Socket 0",
    )
    _make_acpi_sensor(tmp_path, hwmon_id=1, socket_id=0, microwatts=125_500_000)
    _make_acpi_sensor(
        tmp_path,
        hwmon_id=2,
        socket_id=0,
        microwatts=9_500_000,
        domain="SysIO Power Socket 0",
    )
    reader = AcpiPowerMeterReader(tmp_path)

    readings = reader.read_watts()
    assert readings == {
        "CPU0:cpuSidePowerUsageW": 150.0,
        "CPU0:cpuRailPowerUsageW": 125.5,
        "CPU0:socPowerUsageW": 9.5,
    }
    assert reader.aggregate_watts(readings) == 150.0
    metadata = reader.metadata()
    assert metadata["source"] == "acpi"
    assert metadata["sensors"][0]["socket_id"] == 0
    assert metadata["sensors"][0]["average_interval_ms"] == 100
    assert metadata["aggregate_scope"] == "cpu_side_socket_total"
    assert {sensor["domain_kind"] for sensor in metadata["sensors"]} == {
        "total",
        "cpu_rail",
        "soc",
    }


def test_acpi_reader_does_not_treat_grace_cpu_rail_as_socket_total(tmp_path: Path) -> None:
    _make_acpi_sensor(tmp_path, hwmon_id=0, socket_id=0, microwatts=125_500_000)

    with pytest.raises(CpuPowerSourceUnavailable, match="no ACPI socket-total"):
        AcpiPowerMeterReader(tmp_path)


def test_acpi_reader_accepts_legacy_hwmon_without_class_name(tmp_path: Path) -> None:
    # Legacy registration exposes attributes only on the parent ACPI device.
    _make_acpi_sensor(
        tmp_path,
        hwmon_id=11,
        socket_id=0,
        microwatts=98_029_000,
        domain="Grace Power Socket 0",
        legacy_layout=True,
    )
    _make_acpi_sensor(tmp_path, hwmon_id=12, socket_id=0, microwatts=46_046_000, legacy_layout=True)
    (tmp_path / "hwmon1" / "device").mkdir(parents=True)
    (tmp_path / "hwmon1" / "device" / "name").write_text("nvme\n")
    (tmp_path / "hwmon1" / "device" / "power1_average").write_text("1000000\n")

    reader = AcpiPowerMeterReader(tmp_path)

    assert reader.read_watts() == {
        "CPU0:cpuSidePowerUsageW": 98.029,
        "CPU0:cpuRailPowerUsageW": 46.046,
    }


def test_exporter_discovers_legacy_hwmon_without_class_name(tmp_path: Path) -> None:
    _make_acpi_sensor(
        tmp_path,
        hwmon_id=11,
        socket_id=1,
        microwatts=88_706_000,
        domain="Grace Power Socket 1",
        legacy_layout=True,
    )
    (tmp_path / "hwmon0").mkdir()
    (tmp_path / "hwmon0" / "name").write_text("acpitz\n")
    hwmon = tmp_path / "hwmon11"
    # Not a real legacy layout (the class node has no attributes there); the alias
    # makes the same channel visible twice to exercise canonical-path dedup.
    (hwmon / "power1_average").symlink_to(hwmon / "device" / "power1_average")

    sensors = _find_power_meter_sensors(tmp_path)

    assert [(s["oem"], s["socket"]) for s in sensors] == [("Grace Power Socket 1", "1")]
    assert 'source="acpi"} 88.706000' in _build_metrics(sensors)


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
    # The reader only classifies; the shared pivot builds one CpuSample per
    # socket whose power_w is the total envelope and whose component rails
    # ride along without ever standing in for it.
    assert reader.classify_readings(readings)[:2] == [
        RailReading(0, "total", "CPU0:cpuSidePowerUsageW", 150.0),
        RailReading(0, "cpu_rail", "CPU0:cpuRailPowerUsageW", 70.0),
    ]
    samples = reader.socket_samples(readings)
    assert [(s.source, s.socket_id, s.sensor, s.power_w, s.rails) for s in samples] == [
        ("acpi", 0, "CPU0:cpuSidePowerUsageW", 150.0, {"cpu_rail": 70.0, "soc": 6.0, "dram": 8.0}),
        ("acpi", 1, "CPU1:cpuSidePowerUsageW", 160.0, {"cpu_rail": 75.0, "soc": 7.0, "dram": 9.0}),
    ]
    assert all(isinstance(s, CpuSample) for s in samples)


def test_acpi_socket_samples_drop_a_socket_whose_total_failed_to_read(tmp_path: Path) -> None:
    """A component rail must never be published as a socket's power_w."""
    _make_acpi_sensor(tmp_path, hwmon_id=0, socket_id=0, microwatts=150_000_000, domain="Grace Power Socket 0")
    _make_acpi_sensor(tmp_path, hwmon_id=1, socket_id=0, microwatts=70_000_000, domain="CPU Power Socket 0")
    reader = AcpiPowerMeterReader(tmp_path)

    readings = reader.read_watts()
    readings["CPU0:cpuSidePowerUsageW"] = None  # simulate a failed sysfs read of the envelope

    assert reader.socket_samples(readings) == ()
    assert reader.aggregate_watts(readings) is None  # and no partial node total either


def test_acpi_reader_collects_input_power_naming_variants(tmp_path: Path) -> None:
    domains = (
        (0, "Total Input Power in uW socket 0", 150_000_000),
        (1, "CPU Rail Input Power in uW socket 0", 70_000_000),
        (2, "SoC Rail Input Power in uW socket 0", 6_000_000),
        (3, "DRAM Input Power in uW socket 0", 8_000_000),
        (4, "CPU Rail Output Power in uW socket 0", 55_000_000),
    )
    for hwmon_id, domain, microwatts in domains:
        _make_acpi_sensor(
            tmp_path,
            hwmon_id=hwmon_id,
            socket_id=0,
            microwatts=microwatts,
            domain=domain,
        )

    reader = AcpiPowerMeterReader(tmp_path)

    assert reader.read_watts() == {
        "CPU0:cpuSidePowerUsageW": 150.0,
        "CPU0:cpuRailPowerUsageW": 70.0,
        "CPU0:socPowerUsageW": 6.0,
        "CPU0:dramPowerUsageW": 8.0,
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


def _install_fake_dcgm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reject_watch_containing: int | None = None,
) -> list[object]:
    """Install a fake pydcgm/dcgm_agent; returns the event log.

    ``reject_watch_containing`` makes ``WatchFields`` raise for any field group
    whose ids include that field -- the shape of a libdcgm that does not
    support 1132 for CPU entities.
    """
    events: list[object] = []
    field_ids_by_group: dict[int, list[int]] = {}

    class FakeHandle:
        handle = object()

        def Shutdown(self) -> None:
            events.append("shutdown")

    class FakeSamples:
        def WatchFields(self, field_group: object, frequency: int, age: float, samples: int) -> None:
            ids = field_ids_by_group[id(field_group)]
            if reject_watch_containing is not None and reject_watch_containing in ids:
                events.append(("watch_rejected", ids))
                raise RuntimeError(f"field {reject_watch_containing} not supported for CPU entities")
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
            field_ids_by_group[id(self)] = list(kwargs["fieldIds"])  # type: ignore[arg-type]
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

    def fake_latest_values(_handle: object, _entities: object, fields: object, flags: int) -> list[object]:
        events.append(("latest", flags, list(fields)))
        return [
            _fake_value(0, cpu_power.CPU_POWER_FIELD_ID, 120.5),
            _fake_value(1, cpu_power.CPU_POWER_FIELD_ID, 130.0),
            _fake_value(0, 1132, 6.25),  # SysIO rail for socket 0 only
            _fake_value(9, cpu_power.CPU_POWER_FIELD_ID, 999.0),  # entity we never enumerated: dropped
            _fake_value(0, 1100, 0.42),
            _fake_value(0, 1101, 0.30),
            _fake_value(0, 1103, 0.10),
            _fake_value(1, 1100, 0.05),
            _fake_value(1, 1104, float("nan")),  # non-finite: dropped
        ]

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
    return events


def test_dcgm_reader_watches_cpu_power_before_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _install_fake_dcgm(monkeypatch)

    reader = cpu_power.DcgmCpuPowerReader()
    assert reader.power_field_ids == (1130, 1132)
    watts = reader.read_watts()
    utilization = reader.read_utilization()
    reader.close()

    expected_fields = [1130, 1132, *(field.field_id for field in CPU_UTILIZATION_FIELDS)]
    assert ("entity", 7, 0) in events
    assert ("entity", 7, 1) in events
    assert ("field_group", expected_fields) in events
    assert ("watch", 100_000, 60.0, 600) in events
    assert ("update", True) in events
    assert ("latest", 0, expected_fields) in events
    assert events[-4:] == ["unwatch", "field_group_delete", "group_delete", "shutdown"]
    assert watts == {
        "CPU0:cpuPowerUsageW": 120.5,
        "CPU0:cpuRailPowerUsageW": 120.5,  # 1130 is the CPU rail read through DCGM
        "CPU0:socPowerUsageW": 6.25,
        "CPU1:cpuPowerUsageW": 130.0,
        "CPU1:cpuRailPowerUsageW": 130.0,
        "CPU1:socPowerUsageW": None,  # no 1132 sample for socket 1
    }
    samples = reader.socket_samples(watts)
    assert [(s.socket_id, s.power_w, s.rails) for s in samples] == [
        (0, 120.5, {"cpu_rail": 120.5, "soc": 6.25}),
        (1, 130.0, {"cpu_rail": 130.0}),
    ]
    assert reader.aggregate_watts(watts) == 250.5  # sum of 1130 only; SysIO never joins the total
    assert utilization == {
        0: {"cpu_util_total": 0.42, "cpu_util_user": 0.30, "cpu_util_sys": 0.10},
        1: {"cpu_util_total": 0.05},
    }
    metadata = reader.metadata()
    assert [field["column"] for field in metadata["utilization_fields"]] == list(UTILIZATION_COLUMNS)
    assert metadata["utilization_fields"][0]["field_id"] == 1100
    assert metadata["aggregate_scope"] == "cpu_rail_only"
    assert [(f["field_id"], f["rail_kind"], f["column"], f["watched"]) for f in metadata["power_fields"]] == [
        (1130, "cpu_rail", "power_w", True),
        (1132, "soc", "soc_w", True),
    ]


def test_dcgm_reader_falls_back_to_the_cpu_rail_alone_when_sysio_is_refused(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A libdcgm that rejects 1132 must still come up on 1130 -- in auto, DCGM is the last resort."""
    events = _install_fake_dcgm(monkeypatch, reject_watch_containing=1132)
    utilization_ids = [field.field_id for field in CPU_UTILIZATION_FIELDS]

    with caplog.at_level("WARNING", logger="srtctl.core.cpu_power"):
        reader = cpu_power.DcgmCpuPowerReader()
    watts = reader.read_watts()
    reader.close()

    assert reader.power_field_ids == (1130,)
    assert ("watch_rejected", [1130, 1132, *utilization_ids]) in events
    assert ("field_group", [1130, *utilization_ids]) in events
    assert ("watch", 100_000, 60.0, 600) in events
    # The rejected attempt's field group is cleaned up; the surviving one at close.
    assert events.count("field_group_delete") == 2
    assert any("watching the CPU rail only" in rec.message for rec in caplog.records)
    # No SysIO kind is expected of a reader that never watched 1132: no blank soc column entries.
    assert watts == {
        "CPU0:cpuPowerUsageW": 120.5,
        "CPU0:cpuRailPowerUsageW": 120.5,
        "CPU1:cpuPowerUsageW": 130.0,
        "CPU1:cpuRailPowerUsageW": 130.0,
    }
    assert [(s.socket_id, s.power_w, s.rails) for s in reader.socket_samples(watts)] == [
        (0, 120.5, {"cpu_rail": 120.5}),
        (1, 130.0, {"cpu_rail": 130.0}),
    ]
    assert [(f["field_id"], f["watched"]) for f in reader.metadata()["power_fields"]] == [(1130, True), (1132, False)]


def test_dcgm_reader_fails_when_every_field_set_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_dcgm(monkeypatch, reject_watch_containing=1130)
    with pytest.raises(CpuPowerSourceUnavailable, match="cannot watch DCGM CPU power fields"):
        cpu_power.DcgmCpuPowerReader()


def _acpi_root_reading(tmp_path: Path, *watts: float) -> Path:
    # One Grace socket envelope sensor per value, under a fresh hwmon root.
    root = tmp_path / "hwmon"
    for socket_id, value in enumerate(watts):
        _make_acpi_sensor(
            root,
            hwmon_id=socket_id,
            socket_id=socket_id,
            microwatts=round(value * 1_000_000),
            domain=f"Grace Power Socket {socket_id}",
        )
    return root


def test_acpi_probe_is_dead_when_every_sensor_reads_zero(tmp_path: Path) -> None:
    reader = AcpiPowerMeterReader(_acpi_root_reading(tmp_path, 0.0, 0.0))
    assert reader.probe_live(retries=1, delay_seconds=0.0) == 0


def test_acpi_probe_is_live_when_any_sensor_reads_positive(tmp_path: Path) -> None:
    reader = AcpiPowerMeterReader(_acpi_root_reading(tmp_path, 0.0, 97.5))
    assert reader.probe_live(retries=0, delay_seconds=0.0) == 1


def _stuck_total_live_rail_root(tmp_path: Path) -> Path:
    # Socket 0 healthy; socket 1's Grace envelope reads 0 while its CPU rail is live.
    root = tmp_path / "hwmon"
    _make_acpi_sensor(root, hwmon_id=0, socket_id=0, microwatts=100_000_000, domain="Grace Power Socket 0")
    _make_acpi_sensor(root, hwmon_id=1, socket_id=0, microwatts=50_000_000, domain="CPU Power Socket 0")
    _make_acpi_sensor(root, hwmon_id=2, socket_id=1, microwatts=0, domain="Grace Power Socket 1")
    _make_acpi_sensor(root, hwmon_id=3, socket_id=1, microwatts=52_000_000, domain="CPU Power Socket 1")
    return root


def test_acpi_zero_total_is_missing_not_zero_watts(tmp_path: Path) -> None:
    """0 from power1_average means the sensor is not reporting; the socket gets no row, never power_w=0."""
    reader = AcpiPowerMeterReader(_stuck_total_live_rail_root(tmp_path))
    readings = reader.read_watts()
    assert readings["CPU1:cpuSidePowerUsageW"] is None
    assert readings["CPU1:cpuRailPowerUsageW"] == 52.0
    assert [(s.socket_id, s.power_w, s.rails) for s in reader.socket_samples(readings)] == [
        (0, 100.0, {"cpu_rail": 50.0}),
    ]
    # A stuck envelope also voids the node total: a partial sum must not pass for the node's power.
    assert reader.aggregate_watts(readings) is None


def test_acpi_probe_ignores_live_component_rails_when_every_total_is_zero(tmp_path: Path) -> None:
    """A live CPU rail must not vouch for ACPI when no socket envelope reads positive."""
    root = tmp_path / "hwmon"
    _make_acpi_sensor(root, hwmon_id=0, socket_id=0, microwatts=0, domain="Grace Power Socket 0")
    _make_acpi_sensor(root, hwmon_id=1, socket_id=0, microwatts=50_000_000, domain="CPU Power Socket 0")
    assert AcpiPowerMeterReader(root).probe_live(retries=0, delay_seconds=0.0) == 0


def test_acpi_probe_counts_only_live_totals(tmp_path: Path) -> None:
    # Two live rails, one live total: the probe reports the one total.
    assert AcpiPowerMeterReader(_stuck_total_live_rail_root(tmp_path)).probe_live(retries=0, delay_seconds=0.0) == 1


def test_acpi_probe_retries_once_when_the_first_pass_is_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hwmon averages can read 0 on the first poll after boot; one retry must rescue a good node."""
    root = _acpi_root_reading(tmp_path, 0.0)
    reader = AcpiPowerMeterReader(root)
    sensor_file = next(root.rglob("power1_average"))
    sleeps: list[float] = []

    def sleep_then_wake(seconds: float) -> None:
        sleeps.append(seconds)
        sensor_file.write_text("98000000")  # the sensor comes alive between passes

    monkeypatch.setattr(cpu_power.time, "sleep", sleep_then_wake)
    assert reader.probe_live(retries=1, delay_seconds=1.0) == 1
    assert sleeps == [1.0]


def test_auto_falls_through_dead_acpi_to_dcgm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Discovered-but-all-zero ACPI sensors must not be published as 0 W; auto steps down to DCGM."""
    root = _acpi_root_reading(tmp_path, 0.0, 0.0)
    monkeypatch.setattr(cpu_power, "AcpiPowerMeterReader", lambda *_a, **_k: AcpiPowerMeterReader(root))
    monkeypatch.setattr(cpu_power, "ACPI_PROBE_RETRY_DELAY_SECONDS", 0.0)
    _install_fake_dcgm(monkeypatch)

    with caplog.at_level("INFO", logger="srtctl.core.cpu_power"):
        reader = cpu_power.create_reader("auto")
    reader.close()

    assert isinstance(reader, cpu_power.DcgmCpuPowerReader)
    messages = [rec.message for rec in caplog.records]
    assert any("no ACPI socket-total power_meter sensor read positive across 2 probe(s)" in m for m in messages)


def test_auto_commits_to_live_acpi_without_touching_dcgm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _acpi_root_reading(tmp_path, 101.0, 99.0)
    monkeypatch.setattr(cpu_power, "AcpiPowerMeterReader", lambda *_a, **_k: AcpiPowerMeterReader(root))

    def no_dcgm(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("DCGM must not be constructed when ACPI is live")

    monkeypatch.setattr(cpu_power, "DcgmCpuPowerReader", no_dcgm)
    reader = cpu_power.create_reader("auto")
    assert isinstance(reader, AcpiPowerMeterReader)


def test_explicit_acpi_source_fails_on_dead_sensors_instead_of_falling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _acpi_root_reading(tmp_path, 0.0)
    monkeypatch.setattr(cpu_power, "AcpiPowerMeterReader", lambda *_a, **_k: AcpiPowerMeterReader(root))
    monkeypatch.setattr(cpu_power, "ACPI_PROBE_RETRY_DELAY_SECONDS", 0.0)
    with pytest.raises(CpuPowerSourceUnavailable, match="no ACPI socket-total power_meter sensor read positive"):
        cpu_power.create_reader("acpi")


def _fake_value(entity_id: int, field_id: int, dbl: float) -> object:
    return types.SimpleNamespace(entityId=entity_id, fieldId=field_id, status=0, value=types.SimpleNamespace(dbl=dbl))


def test_samples_header_pins_wide_socket_layout() -> None:
    """v4: one row per socket; rails as columns between power_w and total_power_w."""
    assert SAMPLES_SCHEMA_VERSION == 4
    assert RAIL_COLUMN_NAMES == ("cpu_rail_w", "soc_w", "dram_w")
    expected = (
        *SAMPLES_HEADER_V2[:8],  # ... through power_w
        *RAIL_COLUMN_NAMES,
        "total_power_w",
        *UTILIZATION_COLUMNS,
    )
    assert expected == SAMPLES_HEADER
    assert UTILIZATION_COLUMNS == ("cpu_util_total", "cpu_util_user", "cpu_util_nice", "cpu_util_sys", "cpu_util_irq")
    assert [field.field_id for field in CPU_UTILIZATION_FIELDS] == [1100, 1101, 1102, 1103, 1104]


class _FakeReader(cpu_power.CpuPowerReader):
    source_name = "acpi"  # must be a real origin: it decides which rail kind is the socket's power

    def __init__(self, utilization: dict[int, dict[str, float]] | None = None) -> None:
        self._utilization = utilization or {}

    def read_watts(self) -> dict[str, float | None]:
        return {"CPU0:cpuSidePowerUsageW": 100.0, "CPU1:cpuSidePowerUsageW": 110.0}

    def read_utilization(self) -> dict[int, dict[str, float]]:
        return self._utilization

    def classify_readings(self, readings: dict[str, float | None]) -> list[RailReading]:
        return [
            RailReading(0, "total", "CPU0:cpuSidePowerUsageW", 100.0),
            RailReading(0, "cpu_rail", "CPU0:cpuRailPowerUsageW", 60.0),
            RailReading(1, "total", "CPU1:cpuSidePowerUsageW", 110.0),
        ]

    def metadata(self) -> dict[str, object]:
        return {"source": self.source_name}

    def close(self) -> None:
        pass


def _run_collect_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reader: cpu_power.CpuPowerReader) -> Path:
    handlers: dict[int, object] = {}
    monkeypatch.setattr(cpu_power.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    monkeypatch.setattr(cpu_power, "create_reader", lambda _source: reader)
    monkeypatch.setattr(cpu_power.os, "umask", lambda _mask: 0)
    monkeypatch.setenv("SLURMD_NODENAME", "node-a")

    def stop_after_first_sample(_seconds: float) -> None:
        handlers[cpu_power.signal.SIGTERM](cpu_power.signal.SIGTERM, None)

    monkeypatch.setattr(cpu_power.time, "sleep", stop_after_first_sample)
    output_dir = tmp_path / "nodes"
    rc = cpu_power.collect(output_dir=output_dir, ready_dir=tmp_path / "ready", source="auto", interval_seconds=0.1)
    assert rc == 0
    return output_dir / "node-a.csv"


def test_collect_writes_socket_utilization_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reader = _FakeReader({0: {"cpu_util_total": 0.5, "cpu_util_sys": 0.1}})

    csv_path = _run_collect_once(monkeypatch, tmp_path, reader)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(SAMPLES_HEADER)
    assert [row["socket_id"] for row in rows] == ["0", "1"]
    assert rows[0]["schema_version"] == "4"
    assert [row["power_w"] for row in rows] == ["100.0", "110.0"]
    assert [row["total_power_w"] for row in rows] == ["210.0", "210.0"]
    assert rows[0]["cpu_rail_w"] == "60.0"
    assert rows[0]["soc_w"] == "" and rows[0]["dram_w"] == ""
    assert all(rows[1][column] == "" for column in RAIL_COLUMN_NAMES)
    assert rows[0]["cpu_util_total"] == "0.5"
    assert rows[0]["cpu_util_sys"] == "0.1"
    assert rows[0]["cpu_util_user"] == ""
    assert all(rows[1][column] == "" for column in UTILIZATION_COLUMNS)
    metadata = json.loads(csv_path.with_name("node-a.metadata.json").read_text())
    assert metadata["schema_version"] == 4


def test_collect_leaves_utilization_blank_without_a_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    csv_path = _run_collect_once(monkeypatch, tmp_path, _FakeReader())

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row[column] == "" for row in rows for column in UTILIZATION_COLUMNS)


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
        rail_blanks = ("",) * len(RAIL_COLUMN_NAMES)
        util_blanks = ("",) * len(UTILIZATION_COLUMNS)
        writer.writerow(
            (
                SAMPLES_SCHEMA_VERSION,
                timestamp,
                timestamp_local,
                hostname,
                "acpi",
                "CPU0:cpuSidePowerUsageW",
                0,
                watts,
            )
            + rail_blanks
            + (watts,)
            + util_blanks
        )


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
