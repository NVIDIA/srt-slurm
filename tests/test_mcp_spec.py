# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from srtctl.mcp.spec_tools import (
    explain_field,
    get_config_reference,
    preflight_config,
    resolve_config,
    schema_summary,
    validate_config,
)


def test_schema_summary_lists_top_level_fields() -> None:
    summary = schema_summary()
    names = [field["name"] for field in summary["top_level_fields"]]
    assert "model" in names
    assert "resources" in names
    assert "reporting" in names


def test_get_config_reference_finds_reporting() -> None:
    result = get_config_reference(query="reporting", max_matches=2)
    assert result["matches"]
    assert any(
        "reporting" in match["snippet"].lower() or "reporting" in match["heading"].lower()
        for match in result["matches"]
    )


def test_explain_field_returns_schema_and_docs() -> None:
    result = explain_field("reporting")
    assert result["resolved"] is True
    assert result["schema"]["leaf"]["name"] == "reporting"
    assert "docs" in result


def test_explain_field_resolves_nested_reporting_endpoint() -> None:
    result = explain_field("reporting.status.endpoint")
    assert result["resolved"] is True
    assert result["schema"]["leaf"]["name"] == "endpoint"
    assert result["schema"]["leaf"]["type"] == "UnionType[str, NoneType]"


def test_validate_config_accepts_minimal_recipe() -> None:
    result = validate_config(
        config={
            "name": "mcp-test",
            "model": {
                "path": "/tmp/model",
                "container": "/tmp/container.sqsh",
                "precision": "bf16",
            },
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "prefill_workers": 1,
                "decode_workers": 1,
            },
        },
    )
    assert result["valid"] is True
    assert result["normalized"][0]["config"]["name"] == "mcp-test"
    assert result["cluster_defaults_source"] == "not-used-by-mcp"
    assert "Host-side srtslurm.yaml is not used" in result["operator_boundary"]


def test_preflight_config_reports_missing_container(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    result = preflight_config(
        config={
            "name": "mcp-test",
            "model": {
                "path": str(model_dir),
                "container": "missing-container",
                "precision": "bf16",
            },
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 1,
                "decode_nodes": 1,
            },
        },
    )

    assert result["ok"] is False
    assert result["scope"] == "explicit-local-paths"
    assert result["cluster_defaults_source"] == "not-used-by-mcp"
    assert "compute side" in result["operator_hint"]
    assert result["variants"][0]["errors"][0]["field"] == "model.container"


def test_validate_config_rejects_disagg_with_zero_prefill_workers() -> None:
    """Reproduces the reported bad config: disagg-style block that should be aggregated."""
    result = validate_config(
        config={
            "name": "mcp-test",
            "model": {
                "path": "/tmp/model",
                "container": "/tmp/container.sqsh",
                "precision": "bf16",
            },
            "resources": {
                "gpu_type": "gb200",
                "prefill_nodes": 0,
                "decode_nodes": 1,
                "prefill_workers": 0,
                "decode_workers": 1,
                "gpus_per_node": 4,
            },
        },
    )
    assert result["valid"] is False
    assert len(result["errors"]) == 1
    message = result["errors"][0]
    assert "prefill_workers" in message
    assert "agg_nodes: 1" in message
    assert "agg_workers: 1" in message


def test_validate_config_accepts_correct_aggregated_form() -> None:
    result = validate_config(
        config={
            "name": "mcp-test",
            "model": {
                "path": "/tmp/model",
                "container": "/tmp/container.sqsh",
                "precision": "bf16",
            },
            "resources": {
                "gpu_type": "gb200",
                "gpus_per_node": 4,
                "agg_nodes": 1,
                "agg_workers": 1,
            },
        },
    )
    assert result["valid"] is True


def test_validate_config_rejects_mixed_disagg_and_agg() -> None:
    result = validate_config(
        config={
            "name": "mcp-test",
            "model": {
                "path": "/tmp/model",
                "container": "/tmp/container.sqsh",
                "precision": "bf16",
            },
            "resources": {
                "gpu_type": "gb200",
                "gpus_per_node": 4,
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "agg_nodes": 1,
                "agg_workers": 1,
            },
        },
    )
    assert result["valid"] is False
    assert "Mixes disaggregated fields" in result["errors"][0]


def test_resolve_config_returns_variants() -> None:
    result = resolve_config(
        config={
            "base": {
                "name": "base",
                "model": {
                    "path": "/tmp/model",
                    "container": "/tmp/container.sqsh",
                    "precision": "bf16",
                },
                "resources": {
                    "gpu_type": "h100",
                    "gpus_per_node": 8,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                },
            },
            "override_alt": {
                "benchmark": {
                    "type": "sa-bench",
                },
            },
        },
    )
    assert result["variant_count"] == 1
    assert result["scope"] == "schema-only"
    assert result["cluster_defaults_source"] == "not-used-by-mcp"
    assert result["variants"][0]["variant"] == "alt"


def test_mcp_tools_reject_host_side_cluster_defaults() -> None:
    config = {
        "name": "mcp-test",
        "model": {
            "path": "model-alias",
            "container": "container-alias",
            "precision": "bf16",
        },
        "resources": {
            "gpu_type": "h100",
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 1,
        },
    }

    for tool in (validate_config, preflight_config, resolve_config):
        with pytest.raises(ValueError, match="Host-side srtslurm.yaml is not used"):
            tool(config=config, apply_cluster_defaults=True)
