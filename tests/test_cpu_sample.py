# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared per-socket pivot: origin picks the primary rail; the aggregate is derived, never stored."""

import pytest

from srtctl.core.power.cpu_sample import (
    CpuSample,
    RailReading,
    node_total_watts,
    pivot_socket_samples,
    primary_kind,
)


def _acpi_readings(socket_id: int, total: float, cpu_rail: float, soc: float | None = None) -> list[RailReading]:
    readings = [
        RailReading(socket_id, "total", f"CPU{socket_id}:cpuSidePowerUsageW", total),
        RailReading(socket_id, "cpu_rail", f"CPU{socket_id}:cpuRailPowerUsageW", cpu_rail),
    ]
    if soc is not None:
        readings.append(RailReading(socket_id, "soc", f"CPU{socket_id}:socPowerUsageW", soc))
    return readings


def test_primary_kind_follows_the_origin():
    assert primary_kind("acpi") == "total"
    assert primary_kind("dcgm") == "dcgm"
    with pytest.raises(ValueError, match="unknown CPU power source"):
        primary_kind("fake")


def test_acpi_sample_power_is_the_total_envelope_and_rails_are_the_components():
    sample = CpuSample("acpi", 0, tuple(_acpi_readings(0, total=94.29, cpu_rail=49.0, soc=5.1)))

    assert sample.power_w == 94.29
    assert sample.sensor == "CPU0:cpuSidePowerUsageW"
    assert sample.rails == {"cpu_rail": 49.0, "soc": 5.1}  # primary excluded; canonical order
    assert sample.primary.kind == "total"
    assert sample.reading("soc").watts == 5.1
    assert sample.reading("dram") is None


def test_dcgm_sample_has_no_rails():
    sample = CpuSample("dcgm", 1, (RailReading(1, "dcgm", "CPU1:cpuPowerUsageW", 52.35),))

    assert sample.power_w == 52.35
    assert sample.rails == {}


def test_sample_refuses_to_exist_without_exactly_one_primary():
    with pytest.raises(ValueError, match="exactly one 'total'"):
        CpuSample("acpi", 0, (RailReading(0, "cpu_rail", "CPU0:cpuRailPowerUsageW", 49.0),))
    with pytest.raises(ValueError, match="exactly one 'total'"):
        CpuSample("acpi", 0, tuple(_acpi_readings(0, 90.0, 40.0) + [RailReading(0, "total", "dup", 91.0)]))


def test_sample_refuses_readings_from_another_socket():
    with pytest.raises(ValueError, match="another socket"):
        CpuSample("acpi", 0, tuple(_acpi_readings(1, 90.0, 40.0)))


def test_pivot_groups_by_socket_and_drops_sockets_without_their_primary():
    readings = [
        *_acpi_readings(1, total=110.0, cpu_rail=45.0),
        *_acpi_readings(0, total=100.0, cpu_rail=40.0, soc=15.0),
        RailReading(2, "cpu_rail", "CPU2:cpuRailPowerUsageW", 39.0),  # envelope failed to read
    ]

    samples = pivot_socket_samples("acpi", readings)

    assert [s.socket_id for s in samples] == [0, 1]  # sorted; socket 2 dropped
    assert samples[0].rails == {"cpu_rail": 40.0, "soc": 15.0}
    assert samples[1].rails == {"cpu_rail": 45.0}
    assert node_total_watts(samples) == 210.0  # envelopes only, never cpu_rail


def test_pivot_keeps_the_first_reading_for_a_duplicated_rail():
    readings = _acpi_readings(0, total=100.0, cpu_rail=40.0) + [RailReading(0, "total", "dup", 999.0)]

    (sample,) = pivot_socket_samples("acpi", readings)

    assert sample.power_w == 100.0


def test_node_total_is_none_with_no_sockets():
    assert node_total_watts(()) is None
    assert node_total_watts(pivot_socket_samples("acpi", [])) is None


def test_from_columns_round_trips_the_wide_csv_shape():
    sample = CpuSample.from_columns(source="acpi", socket_id=0, power_w=94.29, rails={"cpu_rail": 49.0, "dram": 8.0})

    assert sample.power_w == 94.29
    assert sample.sensor == "CPU0:cpuSidePowerUsageW"  # canonical name when the CSV has none
    assert sample.rails == {"cpu_rail": 49.0, "dram": 8.0}
    assert CpuSample.from_columns(source="dcgm", socket_id=3, power_w=50.0).sensor == "CPU3:cpuPowerUsageW"
