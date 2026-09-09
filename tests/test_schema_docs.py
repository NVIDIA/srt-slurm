# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from srtctl.cli import submit as submit_cli
from srtctl.core.schema import ObservabilityConfig, ResourceConfig, SrtConfig
from srtctl.core.schema_docs import (
    DEFAULT_OUTPUT,
    field_docs,
    render_schema_reference,
    schema_reference_is_current,
    write_schema_reference,
)


def test_checked_in_schema_reference_is_current() -> None:
    """docs/schema-reference.md must be regenerated whenever the schema changes.

    Fix with: uv run srtctl schema-docs
    """
    assert DEFAULT_OUTPUT.exists(), f"{DEFAULT_OUTPUT} is missing; run `srtctl schema-docs`"
    assert schema_reference_is_current(), (
        f"{DEFAULT_OUTPUT.name} is stale relative to srtctl.core.schema; run `srtctl schema-docs` and commit the result"
    )


def test_render_is_deterministic() -> None:
    assert render_schema_reference() == render_schema_reference()


def test_top_level_recipe_keys_are_documented() -> None:
    rows = {row.key for row in field_docs(SrtConfig)}
    for key in ("name", "model", "resources", "backend", "frontend", "benchmark", "observability", "host_setup"):
        assert key in rows, key


def test_marshmallow_data_key_wins_over_private_attribute_name() -> None:
    rows = {row.key: row for row in field_docs(ResourceConfig)}
    assert "gpus_per_prefill" in rows
    assert "gpus_per_decode" in rows
    assert "_explicit_gpus_per_prefill" not in rows
    assert rows["gpus_per_node"].default == "`4`"
    assert rows["gpu_type"].default == "`None`"


def test_required_field_renders_as_required() -> None:
    rows = {row.key: row for row in field_docs(SrtConfig)}
    assert rows["name"].default == "required"


def test_docstring_attributes_become_descriptions() -> None:
    rows = {row.key: row for row in field_docs(ObservabilityConfig)}
    assert "Master analytics knob" in rows["enabled"].description


def test_field_comments_become_descriptions() -> None:
    rows = {row.key: row for row in field_docs(SrtConfig)}
    assert "Custom setup script" in rows["setup_script"].description


def test_backend_types_and_cluster_config_are_rendered() -> None:
    text = render_schema_reference()
    for heading in (
        "## Recipe",
        "## Backend types",
        "### SGLangProtocol",
        "### TRTLLMProtocol",
        "### VLLMProtocol",
        "### MockerProtocol",
        "## Cluster config",
    ):
        assert heading in text, heading
    assert "`backend.type: sglang`" in text
    assert "`sglang_config`" in text
    assert "`default_account`" in text
    assert "<!-- GENERATED FILE" in text


def test_cli_check_passes_on_a_fresh_file(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "schema-reference.md"
    write_schema_reference(output)
    monkeypatch.setattr(sys, "argv", ["srtctl", "schema-docs", "--check", "--output", str(output)])
    submit_cli.main()
    assert "up to date" in capsys.readouterr().out


def test_cli_check_fails_on_a_stale_file(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "schema-reference.md"
    output.write_text("# stale\n")
    monkeypatch.setattr(sys, "argv", ["srtctl", "schema-docs", "--check", "--output", str(output)])
    with pytest.raises(SystemExit) as exc_info:
        submit_cli.main()
    assert exc_info.value.code == 1
    assert "stale" in capsys.readouterr().out


def test_cli_writes_the_file(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "nested" / "schema-reference.md"
    monkeypatch.setattr(sys, "argv", ["srtctl", "schema-docs", "--output", str(output)])
    submit_cli.main()
    assert output.read_text() == render_schema_reference()
