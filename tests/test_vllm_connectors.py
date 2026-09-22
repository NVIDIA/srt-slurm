# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM KV connector table and the one resolver for engine.connector and its per-role override.

Every worker command path asks ``kv_transfer_config(mode)``; nothing reads the
raw ``connector`` field or compares a connector name outside ``_CONNECTOR_MAP``.
"""

import json

import pytest

from srtctl.backends import VLLMProtocol, VLLMServerConfig
from srtctl.backends.vllm import _CONNECTOR_MAP, KVConnector, kv_connector_row


def test_table_presets_serialize_exactly_as_before():
    """The JSON handed to --kv-transfer-config is byte-identical to the former dict presets."""
    assert VLLMProtocol(connector="nixl").kv_transfer_config("prefill") == json.dumps(
        {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    )
    assert VLLMProtocol(connector="LMCache").kv_transfer_config("decode") == json.dumps(
        {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    )
    assert VLLMProtocol(connector="kvbm").kv_transfer_config("decode") == json.dumps(
        {
            "kv_connector": "DynamoConnector",
            "kv_connector_module_path": "kvbm.vllm_integration.connector",
            "kv_role": "kv_both",
        }
    )


def test_role_override_wins_for_that_role_only():
    backend = VLLMProtocol(connector="nixl", vllm_config=VLLMServerConfig(decode={"connector": "lmcache"}))

    assert backend.connector_for_mode("prefill") == "nixl"
    assert backend.connector_for_mode("decode") == "lmcache"
    assert backend.kv_connector_for_mode("decode") is _CONNECTOR_MAP["lmcache"]


@pytest.mark.parametrize("connector", [None, "none", "null", "NONE"])
def test_no_connector_means_no_transfer_config(connector):
    assert VLLMProtocol(connector=connector).kv_transfer_config("prefill") is None
    assert kv_connector_row(connector) is None


def test_raw_json_passes_through_and_has_no_table_row():
    raw = json.dumps({"kv_connector": "MyConnector", "kv_role": "kv_both"})
    backend = VLLMProtocol(connector=raw)

    assert backend.kv_transfer_config("prefill") == raw
    assert backend.kv_connector_for_mode("prefill") is None


def test_only_the_discovery_row_discovers_workers():
    """Rows list their workers on the router command line unless they say otherwise; today that is only MoRI-IO."""
    assert {name for name, row in _CONNECTOR_MAP.items() if row.discovery} == {"moriio"}
    assert VLLMProtocol().discovers_workers() is False
    assert VLLMProtocol(connector="moriio").discovers_workers() is True


def test_a_mode_dependent_role_follows_the_worker_mode():
    row = KVConnector("SomeConnector", kv_role=None)

    assert row.transfer_config("prefill")["kv_role"] == "kv_producer"
    assert row.transfer_config("decode")["kv_role"] == "kv_consumer"
    assert KVConnector("SomeConnector").transfer_config("prefill")["kv_role"] == "kv_both"
