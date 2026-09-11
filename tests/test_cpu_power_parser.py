# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from srtctl.core.power.cpu_parser import parse_cpu_scrape


def _dcgm_body():
    return (
        "# HELP cpu_power_dcgm_watts x\n"
        "# TYPE cpu_power_dcgm_watts gauge\n"
        'cpu_power_dcgm_watts{socket="0",source="dcgm"} 43.878000\n'
        'cpu_power_dcgm_watts{socket="1",source="dcgm"} 52.350000\n'
    )


def _acpi_body(*, include_grace=True):
    lines = [
        "# HELP cpu_power_acpi_watts x\n",
        "# TYPE cpu_power_acpi_watts gauge\n",
        'cpu_power_acpi_watts{sensor="a/0",type="cpu_rail",socket="0",oem_info="CPU Power Socket 0"} 48.1\n',
        'cpu_power_acpi_watts{sensor="a/1",type="soc",socket="0",oem_info="SysIO Power Socket 0"} 5.2\n',
    ]
    if include_grace:
        lines.append(
            'cpu_power_acpi_watts{sensor="a/2",type="total",socket="0",oem_info="Grace Power Socket 0"} 93.4\n'
        )
    lines.append('cpu_power_acpi_watts{sensor="a/3",type="other",socket="",oem_info="Module Socket A"} 1.0\n')
    return "".join(lines)


def test_dcgm_mode_sums_all_sockets_into_the_total():
    scrape = parse_cpu_scrape(_dcgm_body())

    assert scrape.mode == "dcgm"
    assert [r.socket_id for r in scrape.readings] == [0, 1]
    assert scrape.readings[0].sensor == "CPU0:cpuPowerUsageW"
    assert scrape.readings[0].source == "dcgm"
    assert scrape.total_power_w == 43.878 + 52.35


def test_acpi_mode_totals_only_total_channels():
    scrape = parse_cpu_scrape(_acpi_body())

    assert scrape.mode == "acpi"
    kinds = {r.kind for r in scrape.readings}
    assert kinds == {"cpu_rail", "soc", "total"}
    assert scrape.total_power_w == 93.4


def test_acpi_mode_leaves_total_blank_without_a_total_channel():
    scrape = parse_cpu_scrape(_acpi_body(include_grace=False))

    assert scrape.mode == "acpi"
    assert scrape.total_power_w is None


def test_acpi_mode_totals_a_generic_total_power_label_too():
    # Platforms that don't say "Grace" still report a socket-total rail
    # under a generic "Total Power" label; it must count the same as grace.
    body = (
        "# HELP cpu_power_acpi_watts x\n"
        "# TYPE cpu_power_acpi_watts gauge\n"
        'cpu_power_acpi_watts{sensor="a/0",type="total",socket="0",oem_info="Total Power in uW socket 0"} 88.0\n'
        'cpu_power_acpi_watts{sensor="a/1",type="dram",socket="0",oem_info="DRAM Power socket 0"} 8.0\n'
    )
    scrape = parse_cpu_scrape(body)

    assert scrape.mode == "acpi"
    assert scrape.total_power_w == 88.0


def test_acpi_mode_drops_unclassified_rails():
    scrape = parse_cpu_scrape(_acpi_body())

    assert all(r.sensor != "Module Socket A" for r in scrape.readings)
    assert len(scrape.readings) == 3


def test_acpi_wins_when_both_families_are_present():
    scrape = parse_cpu_scrape(_acpi_body() + _dcgm_body())

    assert scrape.mode == "acpi"


def test_malformed_body_yields_no_readings():
    scrape = parse_cpu_scrape("malformed")

    assert scrape.readings == ()
    assert scrape.mode == "unknown"
    assert scrape.total_power_w is None


def test_empty_body_yields_no_readings():
    scrape = parse_cpu_scrape("")

    assert scrape.readings == ()
    assert scrape.mode == "unknown"
