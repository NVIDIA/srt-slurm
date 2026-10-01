# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from srtctl.core.power.contract import (
    CPU_MANIFEST_FILENAME,
    CPU_SAMPLES_FILENAME,
    CPU_SAMPLES_HEADER,
    CPU_SAMPLES_HEADER_V1,
    CPU_SCHEMA_VERSION,
    CPU_SCHEMA_VERSION_V1,
)


def test_cpu_power_contract_constants():
    assert CPU_SCHEMA_VERSION_V1 == 1
    assert CPU_SCHEMA_VERSION == 2
    assert CPU_SAMPLES_FILENAME == "samples.csv"
    assert CPU_MANIFEST_FILENAME == "cpu_manifest.json"
    assert CPU_SAMPLES_HEADER_V1 == (
        "schema_version",
        "timestamp_unix",
        "hostname",
        "source",
        "sensor",
        "socket_id",
        "power_w",
        "total_power_w",
    )
    # v2: one row per socket; power_w is the socket envelope and the ACPI
    # component rails are columns, so no reader has to decide which rows
    # belong to the same socket.
    assert CPU_SAMPLES_HEADER == (
        "schema_version",
        "timestamp_unix",
        "hostname",
        "source",
        "sensor",
        "socket_id",
        "power_w",
        "cpu_rail_w",
        "soc_w",
        "dram_w",
        "total_power_w",
    )
