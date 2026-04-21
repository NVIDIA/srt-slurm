# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
            },
        },
    )
    assert result["valid"] is True
    assert result["normalized"][0]["config"]["name"] == "mcp-test"
    assert result["cluster_defaults_source"] == "disabled"


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
    assert result["scope"] == "local"
    assert result["cluster_defaults_source"] == "disabled"
    assert "IBAR remote preflight" in result["operator_hint"]
    assert result["variants"][0]["errors"][0]["field"] == "model.container"


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
    assert result["scope"] == "local"
    assert result["cluster_defaults_source"] == "disabled"
    assert result["variants"][0]["variant"] == "alt"
