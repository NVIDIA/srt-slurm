# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one place CPU rail names live; every producer/consumer derives from it."""

import pytest

from srtctl.core.power import cpu_rails
from srtctl.core.power.cpu_rails import (
    ACPI_RAIL_KINDS,
    COMPONENT_RAIL_KINDS,
    RAIL_COLUMN_NAMES,
    SENSOR_SUFFIXES,
    classify_acpi_label,
    classify_sensor,
    legacy_rail_rank,
    normalize_kind,
    sensor_name,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        # Grace firmware labels (GB200/GB300)
        ("Grace Power Socket 0", ("total", 0)),
        ("CPU Power Socket 1", ("cpu_rail", 1)),
        ("SysIO Power Socket 0", ("soc", 0)),
        # Generic / "input" variants seen on other Grace platforms
        ("Total Power in uW socket 0", ("total", 0)),
        ("Total Input Power in uW socket 1", ("total", 1)),
        ("CPU Rail Power in uW socket 0", ("cpu_rail", 0)),
        ("CPU Rail Input Power in uW socket 0", ("cpu_rail", 0)),
        ("SoC Rail Input Power in uW socket 0", ("soc", 0)),
        ("SOC Rail Power in uW socket 1", ("soc", 1)),
        ("DRAM Power socket 0", ("dram", 0)),
        ("DRAM Input Power in uW socket 1", ("dram", 1)),
        # Not power rails we publish
        ("CPU Rail Output Power in uW socket 0", None),
        ("Total CPU Energy In uJ socket 0", None),
        ("Module Socket A", None),
        ("Chipthrot DDR Throttle (samples x1000) socket 0", None),
    ],
)
def test_classify_acpi_label(label, expected):
    assert classify_acpi_label(label) == expected


def test_grace_cpu_power_is_a_component_rail_not_the_total():
    """'CPU Power Socket N' is the Grace CPU rail; only 'Grace Power' is the envelope."""
    assert classify_acpi_label("CPU Power Socket 0") == ("cpu_rail", 0)
    assert classify_acpi_label("Grace Power Socket 0") == ("total", 0)


def test_sensor_name_and_classify_sensor_round_trip():
    for kind in SENSOR_SUFFIXES:
        assert classify_sensor(sensor_name(kind, 3)) == kind
    assert sensor_name("total", 0) == "CPU0:cpuSidePowerUsageW"
    assert sensor_name("dcgm", 1) == "CPU1:cpuPowerUsageW"


def test_classify_sensor_handles_legacy_scraper_oem_labels():
    """v1 scraper CSVs stored the raw firmware label in the sensor cell."""
    assert classify_sensor("Grace Power Socket 0") == "total"
    assert classify_sensor("CPU Power Socket 0") == "cpu_rail"
    assert classify_sensor("SysIO Power Socket 1") == "soc"
    assert classify_sensor("Module Socket A") == "other"


def test_legacy_rail_rank_prefers_total_then_dcgm_then_components():
    ranked = sorted(
        ["CPU0:socPowerUsageW", "CPU0:cpuPowerUsageW", "Grace Power Socket 0", "CPU0:dramPowerUsageW", "weird"],
        key=legacy_rail_rank,
    )
    assert ranked == [
        "Grace Power Socket 0",
        "CPU0:cpuPowerUsageW",
        "CPU0:socPowerUsageW",
        "CPU0:dramPowerUsageW",
        "weird",
    ]


def test_normalize_kind_accepts_canonical_and_legacy_exporter_types():
    assert normalize_kind("total") == "total"
    assert normalize_kind("cpu_rail") == "cpu_rail"
    assert normalize_kind("grace") == "total"
    assert normalize_kind("cpu") == "cpu_rail"
    assert normalize_kind("sysio") == "soc"
    assert normalize_kind("other") is None
    assert normalize_kind("") is None
    assert normalize_kind(None) is None


def test_rail_columns_follow_component_kind_order():
    assert COMPONENT_RAIL_KINDS == ("cpu_rail", "soc", "dram")
    assert RAIL_COLUMN_NAMES == ("cpu_rail_w", "soc_w", "dram_w")
    assert frozenset({"total", "cpu_rail", "soc", "dram"}) == ACPI_RAIL_KINDS


def test_every_acpi_kind_has_a_sensor_suffix():
    assert set(SENSOR_SUFFIXES) == ACPI_RAIL_KINDS | {cpu_rails.DCGM_KIND}
