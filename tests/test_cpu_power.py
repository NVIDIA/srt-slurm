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
) -> None:
    hwmon = root / f"hwmon{hwmon_id}"
    device = hwmon / "device"
    device.mkdir(parents=True)
    (hwmon / "name").write_text("power_meter\n")
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

    reader = cpu_power.DcgmCpuPowerReader()
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
    assert [(f["field_id"], f["rail_kind"], f["column"]) for f in metadata["power_fields"]] == [
        (1130, "cpu_rail", "power_w"),
        (1132, "soc", "soc_w"),
    ]


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
