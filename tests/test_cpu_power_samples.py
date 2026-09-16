# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from srtctl.core.power.cpu_sample import CpuSample
from srtctl.core.power.cpu_samples import CpuSampleRow, CpuSampleWriter, LegacyCpuRailRow, read_cpu_samples


def _row(
    *,
    timestamp_unix=1788310143.461,
    hostname="node-a",
    source="dcgm",
    sensor="CPU0:cpuPowerUsageW",
    socket_id=0,
    power_w=43.878,
    rails=None,
    total_power_w=96.228,
):
    sample = CpuSample.from_columns(source=source, socket_id=socket_id, power_w=power_w, sensor=sensor, rails=rails)
    return CpuSampleRow(timestamp_unix=timestamp_unix, hostname=hostname, sample=sample, total_power_w=total_power_w)


def test_writer_round_trips_through_reader(tmp_path):
    path = tmp_path / "cpu" / "samples.csv"
    writer = CpuSampleWriter(path)
    writer.append([_row(), _row(socket_id=1, sensor="CPU1:cpuPowerUsageW", power_w=52.35)])
    writer.close()

    rows, reasons = read_cpu_samples(path)

    assert reasons == ()
    assert len(rows) == 2
    assert rows[0].sensor == "CPU0:cpuPowerUsageW"
    assert rows[1].power_w == 52.35
    assert rows[0].rails == {}  # DCGM: no component rails
    assert writer.closed


def test_writer_round_trips_acpi_component_rails(tmp_path):
    path = tmp_path / "cpu" / "samples.csv"
    writer = CpuSampleWriter(path)
    writer.append(
        [
            _row(
                source="acpi",
                sensor="CPU0:cpuSidePowerUsageW",
                power_w=94.29,
                rails={"cpu_rail": 49.023, "soc": 5.1},
                total_power_w=189.002,
            )
        ]
    )
    writer.close()

    rows, reasons = read_cpu_samples(path)

    assert reasons == ()
    assert rows[0].power_w == 94.29
    assert rows[0].rails == {"cpu_rail": 49.023, "soc": 5.1}  # dram absent -> not in the dict
    with open(path) as handle:
        header, data = handle.read().splitlines()[:2]
    assert (
        header
        == "schema_version,timestamp_unix,hostname,source,sensor,socket_id,power_w,cpu_rail_w,soc_w,dram_w,total_power_w"
    )
    assert data == "2,1788310143.461,node-a,acpi,CPU0:cpuSidePowerUsageW,0,94.29,49.023,5.1,,189.002"


def test_reader_accepts_the_legacy_v1_long_layout(tmp_path):
    """Runs collected before v2 still load; each rail is its own row with empty rails."""
    path = tmp_path / "cpu" / "samples.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "schema_version,timestamp_unix,hostname,source,sensor,socket_id,power_w,total_power_w\n"
        "1,1789416029.2655346,theia0202,acpi,Grace Power Socket 0,0,94.29,189.002\n"
        "1,1789416029.2655346,theia0202,acpi,CPU Power Socket 0,0,49.023,189.002\n"
    )

    rows, reasons = read_cpu_samples(path)

    assert reasons == ()
    assert all(isinstance(r, LegacyCpuRailRow) for r in rows)  # a v1 row is one rail, not one socket
    assert [(r.sensor, r.power_w, r.rails, r.schema_version) for r in rows] == [
        ("Grace Power Socket 0", 94.29, {}, 1),
        ("CPU Power Socket 0", 49.023, {}, 1),
    ]


def test_v2_rows_carry_a_cpu_sample_with_origin_and_rails_kept_apart(tmp_path):
    path = tmp_path / "cpu" / "samples.csv"
    writer = CpuSampleWriter(path)
    writer.append([_row(source="acpi", sensor="CPU0:cpuSidePowerUsageW", power_w=94.29, rails={"cpu_rail": 49.0})])
    writer.close()

    (row,), _ = read_cpu_samples(path)

    assert isinstance(row, CpuSampleRow)
    assert row.sample.source == "acpi"
    assert row.sample.primary.kind == "total"
    assert row.sample.reading("cpu_rail").watts == 49.0
    assert row.sample.reading("dram") is None
    assert row.power_w == 94.29  # column view == sample's primary rail


def test_writer_serializes_a_missing_total_as_empty(tmp_path):
    path = tmp_path / "cpu" / "samples.csv"
    writer = CpuSampleWriter(path)
    writer.append([_row(total_power_w=None)])
    writer.close()

    rows, reasons = read_cpu_samples(path)

    assert reasons == ()
    assert rows[0].total_power_w is None


def test_reader_reports_a_missing_file(tmp_path):
    rows, reasons = read_cpu_samples(tmp_path / "cpu" / "samples.csv")

    assert rows == ()
    assert reasons == ("cpu_samples_csv_missing",)


def test_reader_rejects_a_header_mismatch(tmp_path):
    path = tmp_path / "cpu" / "samples.csv"
    path.parent.mkdir(parents=True)
    path.write_text("not,the,right,header\n")

    rows, reasons = read_cpu_samples(path)

    assert rows == ()
    assert reasons == ("cpu_samples_csv_header_mismatch",)


def test_reader_skips_a_malformed_row_but_keeps_the_rest(tmp_path):
    path = tmp_path / "cpu" / "samples.csv"
    writer = CpuSampleWriter(path)
    writer.append([_row()])
    writer.close()
    with open(path, "a") as handle:
        handle.write("2,not-a-float,node-a,dcgm,CPU0:cpuPowerUsageW,0,1.0,,,,1.0\n")

    rows, reasons = read_cpu_samples(path)

    assert len(rows) == 1
    assert reasons == ("cpu_samples_csv_malformed",)


def test_append_after_close_raises(tmp_path):
    path = tmp_path / "cpu" / "samples.csv"
    writer = CpuSampleWriter(path)
    writer.close()

    try:
        writer.append([_row()])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
