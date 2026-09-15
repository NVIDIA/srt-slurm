# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from srtctl.core.power.contract import (
    CPU_MANIFEST_FILENAME,
    CPU_SAMPLES_FILENAME,
    CPU_SAMPLES_HEADER,
    CPU_SCHEMA_VERSION,
)


def test_cpu_power_contract_constants():
    assert CPU_SCHEMA_VERSION == 1
    assert CPU_SAMPLES_FILENAME == "samples.csv"
    assert CPU_MANIFEST_FILENAME == "cpu_manifest.json"
    assert CPU_SAMPLES_HEADER == (
        "schema_version",
        "timestamp_unix",
        "hostname",
        "source",
        "sensor",
        "socket_id",
        "power_w",
        "total_power_w",
    )
