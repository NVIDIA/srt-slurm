# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from srtctl.core.power.cpu_samples import CpuSampleRow, CpuSampleWriter, read_cpu_samples


def _row(**overrides):
    fields = dict(
        timestamp_unix=1788310143.461,
        hostname="node-a",
        source="dcgm",
        sensor="CPU0:cpuPowerUsageW",
        socket_id=0,
        power_w=43.878,
        total_power_w=96.228,
    )
    fields.update(overrides)
    return CpuSampleRow(**fields)


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
    assert writer.closed


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
        handle.write("1,not-a-float,node-a,dcgm,CPU0:cpuPowerUsageW,0,1.0,1.0\n")

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
