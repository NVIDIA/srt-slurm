# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for sweep orchestration helpers."""

from srtctl.cli.do_sweep import _build_mooncake_master_command
from srtctl.ports import MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_MASTER_PORT, MOONCAKE_METRICS_PORT


def test_mooncake_master_command_is_compatible_with_pinned_image() -> None:
    command = _build_mooncake_master_command()

    assert command == [
        "mooncake_master",
        f"--port={MOONCAKE_MASTER_PORT}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_port={MOONCAKE_HTTP_METADATA_PORT}",
        "--eviction_high_watermark_ratio=0.9",
        "--default_kv_lease_ttl=10000",
        "--rpc_thread_num=16",
        "--enable_metric_reporting=true",
        f"--metrics_port={MOONCAKE_METRICS_PORT}",
    ]
    assert not any("nof_eviction_high_watermark_ratio" in arg for arg in command)


def test_mooncake_master_command_installs_pinned_distribution() -> None:
    spec = "mooncake-transfer-engine-cuda13==0.3.12.post1"
    command = _build_mooncake_master_command(spec)

    assert command[:2] == ["/bin/bash", "-lc"]
    assert "pip install --no-cache-dir --no-deps --force-reinstall" in command[2]
    assert spec in command[2]
    assert "&& exec mooncake_master" in command[2]
    assert "nof_eviction_high_watermark_ratio" not in command[2]
