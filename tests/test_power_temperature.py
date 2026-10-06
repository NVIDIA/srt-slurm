# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from srtctl.core.power.contract import SAMPLES_HEADER, Reason
from srtctl.core.power.parser import parse_power_scrape
from srtctl.core.power.samples import SampleRow, SampleWriter, read_samples

POWER = 'DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-a"} 400\n'


@pytest.mark.parametrize("value", [-273.15, 0, 42.5, 85, 200])
def test_temperature_follows_matching_device_without_changing_power(value):
    parsed = parse_power_scrape(POWER + f'DCGM_FI_DEV_GPU_TEMP{{gpu="0",UUID="GPU-a"}} {value}\n')
    assert parsed.reason_codes == ()
    assert [(r.power_w, r.temperature_c) for r in parsed.readings] == [(400, value)]


@pytest.mark.parametrize(
    "extra",
    [
        "",
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-other"} 60\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 60\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a",GPU_I_ID="1"} 60\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} NaN\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} +Inf\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 200.5\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 2147483632\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 9223372036854775794\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} -273.16\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} -300\n',
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 60\nDCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 61\n',
    ],
)
def test_unavailable_temperature_does_not_invalidate_power(extra):
    parsed = parse_power_scrape(POWER + extra)
    assert parsed.reason_codes == ()
    assert [(r.power_w, r.temperature_c) for r in parsed.readings] == [(400, None)]


def test_temperature_round_trip_keeps_missing_distinct_from_zero(tmp_path):
    path = tmp_path / "samples.csv"
    writer = SampleWriter(path)
    writer.append(
        [
            SampleRow(1000, 0, "node-a", 0, "GPU-a", 400, temperature_c=0),
            SampleRow(1001, 1, "node-a", 0, "GPU-a", 420, temperature_c=65.5),
            SampleRow(1002, 2, "node-a", 0, "GPU-a", 430),
        ]
    )
    writer.close()
    rows, reasons = read_samples(path)
    assert reasons == ()
    assert [r.schema_version for r in rows] == [3, 3, 3]
    assert [r.temperature_c for r in rows] == [0, 65.5, None]


@pytest.mark.parametrize("cell", ["NaN", "+Inf", "-Inf", "-273.16", "200.5", "2147483632", "not-a-number"])
def test_invalid_persisted_temperature_rejects_the_row(tmp_path, cell):
    # The CSV boundary rejects corruption; the exporter boundary only omits temperature.
    path = tmp_path / "samples.csv"
    path.write_text(",".join(SAMPLES_HEADER) + f"\n3,1000,0,node-a,0,GPU-a,400,,,{cell}\n")

    rows, reasons = read_samples(path)

    assert rows == ()
    assert reasons == (Reason.SAMPLES_CSV_MALFORMED,)


@pytest.mark.parametrize("value", [-273.15, 200.0])
def test_persisted_temperature_accepts_the_valid_range_endpoints(tmp_path, value):
    path = tmp_path / "samples.csv"
    path.write_text(",".join(SAMPLES_HEADER) + f"\n3,1000,0,node-a,0,GPU-a,400,,,{value}\n")

    rows, reasons = read_samples(path)

    assert reasons == ()
    assert [row.temperature_c for row in rows] == [value]


@pytest.mark.parametrize(
    ("version", "columns", "cells"),
    [(1, "", ""), (2, ",gpu_util_pct,sm_active", ",25,0.5")],
)
def test_older_samples_have_no_temperature(tmp_path, version, columns, cells):
    path = tmp_path / "samples.csv"
    path.write_text(
        "schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w"
        f"{columns}\n{version},1000,0,node-a,0,GPU-a,400{cells}\n"
    )
    rows, reasons = read_samples(path)
    assert reasons == ()
    assert [(r.power_w, r.temperature_c) for r in rows] == [(400, None)]
