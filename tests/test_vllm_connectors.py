# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM KV connector table and the one resolver for engine.connector and its per-role override.

Every worker command path asks ``kv_transfer_config(mode)``; nothing reads the
raw ``connector`` field or compares a connector name outside ``_CONNECTOR_MAP``.
"""

import json

import pytest

from srtctl.backends import VLLMBackend
from srtctl.backends.vllm import _CONNECTOR_MAP, KVConnector, kv_connector_row
from srtctl.core.schema import RoleConfig


def test_table_presets_serialize_exactly_as_before():
    """The JSON handed to --kv-transfer-config is byte-identical to the former dict presets."""
    assert VLLMBackend(connector="nixl").kv_transfer_config("prefill") == json.dumps(
        {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    )
    assert VLLMBackend(connector="LMCache").kv_transfer_config("decode") == json.dumps(
        {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    )
    assert VLLMBackend(connector="lmcache-mp").kv_transfer_config("prefill") == json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"lmcache.mp.host": "tcp://localhost", "lmcache.mp.port": 8750},
        }
    )
    assert VLLMBackend(connector="kvbm").kv_transfer_config("decode") == json.dumps(
        {
            "kv_connector": "DynamoConnector",
            "kv_connector_module_path": "kvbm.vllm_integration.connector",
            "kv_role": "kv_both",
        }
    )


def test_role_override_wins_for_that_role_only():
    backend = VLLMBackend(connector="nixl", roles={"decode": RoleConfig(args={"connector": "lmcache"})})

    assert backend.connector_for_mode("prefill") == "nixl"
    assert backend.connector_for_mode("decode") == "lmcache"
    assert backend.kv_connector_for_mode("decode") is _CONNECTOR_MAP["lmcache"]


@pytest.mark.parametrize("connector", [None, "none", "null", "NONE"])
def test_no_connector_means_no_transfer_config(connector):
    assert VLLMBackend(connector=connector).kv_transfer_config("prefill") is None
    assert kv_connector_row(connector) is None


def test_raw_json_passes_through_and_has_no_table_row():
    raw = json.dumps({"kv_connector": "MyConnector", "kv_role": "kv_both"})
    backend = VLLMBackend(connector=raw)

    assert backend.kv_transfer_config("prefill") == raw
    assert backend.kv_connector_for_mode("prefill") is None


def test_only_the_discovery_row_discovers_workers():
    """Rows list their workers on the router command line unless they say otherwise; today that is only MoRI-IO."""
    assert {name for name, row in _CONNECTOR_MAP.items() if row.discovery} == {"moriio"}
    assert VLLMBackend().discovers_workers() is False
    assert VLLMBackend(connector="moriio").discovers_workers() is True


def test_a_mode_dependent_role_follows_the_worker_mode():
    row = KVConnector("SomeConnector", kv_role=None)

    assert row.transfer_config("prefill")["kv_role"] == "kv_producer"
    assert row.transfer_config("decode")["kv_role"] == "kv_consumer"
    assert KVConnector("SomeConnector").transfer_config("prefill")["kv_role"] == "kv_both"


@pytest.mark.parametrize(
    ("aggregated", "expected"),
    [
        ({}, None),
        ({"connector": "none"}, None),
        ({"connector": "lmcache-mp"}, _CONNECTOR_MAP["lmcache-mp"].transfer_config("agg")),
    ],
)
def test_direct_aggregate_worker_runs_only_its_role_connector(aggregated, expected):
    """A direct `vllm serve` aggregate worker skips the P/D default connector but keeps the one its role names."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from srtctl.core.topology import Process
    from srtctl.core.schema import RoleConfig

    backend = VLLMBackend(connector="nixl", roles={"agg": RoleConfig(args=aggregated)})
    process = Process(
        node="node0",
        gpu_indices=frozenset(range(8)),
        sys_port=8081,
        http_port=0,
        endpoint_mode="agg",
        endpoint_index=0,
        node_rank=0,
    )
    runtime = MagicMock(model_path=Path("/model"), is_hf_model=False, frontend_port=9000)

    cmd = backend.build_worker_command(
        process=process, endpoint_processes=[process], runtime=runtime, frontend_type="vllm"
    )

    assert "--connector" not in cmd
    kv_config = json.loads(cmd[cmd.index("--kv-transfer-config") + 1]) if "--kv-transfer-config" in cmd else None
    assert kv_config == expected
