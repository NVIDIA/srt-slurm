# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io

import pytest

from srtctl.analysis.power_energy_report import (
    load_cpu_samples,
    load_cpu_samples_from,
    load_gpu_samples,
    load_gpu_samples_from,
)

GPU_CSV = (
    "schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w\n"
    "1,1000.0,0,node-a,0,GPU-aaa,100.0\n"
    "1,1001.0,1,node-a,0,GPU-aaa,110.0\n"
)

CPU_CSV = (
    "schema_version,timestamp_unix,hostname,source,sensor,socket_id,power_w,total_power_w\n"
    "1,1000.0,node-a,dcgm,CPU0:cpuPowerUsageW,0,40.0,80.0\n"
    "1,1001.0,node-a,dcgm,CPU0:cpuPowerUsageW,0,42.0,82.0\n"
)


def test_gpu_handle_loader_matches_the_path_loader(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(GPU_CSV)

    from_path = load_gpu_samples(path, None)
    from_handle = load_gpu_samples_from(io.StringIO(GPU_CSV), None)

    assert from_path.per_device.keys() == from_handle.per_device.keys()
    assert list(from_path.per_device[("node-a", 0)][1]) == list(from_handle.per_device[("node-a", 0)][1])


ACPI_THEN_DCGM_V2_CSV = (
    "schema_version,timestamp_unix,hostname,source,sensor,socket_id,power_w,cpu_rail_w,soc_w,dram_w,total_power_w\n"
    "2,1000.0,node-a,acpi,CPU0:cpuSidePowerUsageW,0,94.0,40.0,5.0,,94.0\n"
    "2,1001.0,node-a,dcgm,CPU0:cpuPowerUsageW,0,50.0,,,,50.0\n"
)
ACPI_THEN_DCGM_V3_CSV = (
    "schema_version,timestamp_unix,hostname,source,sensor,socket_id,power_w,cpu_rail_w,soc_w,dram_w,module_w,total_power_w\n"
    "3,1000.0,node-a,acpi,CPU0:cpuSidePowerUsageW,0,94.0,40.0,5.0,,480.0,94.0\n"
    "3,1001.0,node-a,dcgm,CPU0:cpuPowerUsageW,0,50.0,,,,470.0,50.0\n"
)


@pytest.mark.parametrize("csv_text", [ACPI_THEN_DCGM_V2_CSV, ACPI_THEN_DCGM_V3_CSV], ids=["v2", "v3"])
def test_wide_cpu_csv_loads_as_wide_with_or_without_module_w(csv_text):
    samples = load_cpu_samples_from(io.StringIO(csv_text))

    times, watts = samples.per_socket[("node-a", 0)]
    assert list(times) == [1000.0, 1001.0]
    assert list(watts) == [94.0, 50.0]
    assert list(samples.per_node["node-a"][1]) == [94.0, 50.0]


def test_cpu_handle_loader_matches_the_path_loader(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(CPU_CSV)

    from_path = load_cpu_samples(path)
    from_handle = load_cpu_samples_from(io.StringIO(CPU_CSV))

    assert from_path.per_socket.keys() == from_handle.per_socket.keys()
    assert list(from_path.per_node["node-a"][1]) == list(from_handle.per_node["node-a"][1])
