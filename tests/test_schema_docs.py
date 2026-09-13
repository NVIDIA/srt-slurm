# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from srtctl.cli import submit as submit_cli
from srtctl.core.migrate import migrate_recipe_text
from srtctl.core.schema import BenchmarkConfig, FrontendConfig, ObservabilityConfig, ResourceConfig, SrtConfig
from srtctl.core.schema_docs import (
    BACKEND_TYPES,
    DEFAULT_LEGACY_OUTPUT,
    DEFAULT_OUTPUT,
    LEGACY_CLASSES,
    LEGACY_FIELDS,
    LEGACY_TOP_LEVEL,
    field_docs,
    render_legacy_reference,
    render_schema_reference,
    schema_reference_is_current,
    write_schema_reference,
)


def test_checked_in_schema_reference_is_current() -> None:
    """docs/schema-reference.md and docs/legacy-v1.md must be regenerated whenever the schema changes.

    Fix with: uv run srtctl schema-docs
    """
    assert DEFAULT_OUTPUT.exists(), f"{DEFAULT_OUTPUT} is missing; run `srtctl schema-docs`"
    assert DEFAULT_LEGACY_OUTPUT.exists(), f"{DEFAULT_LEGACY_OUTPUT} is missing; run `srtctl schema-docs`"
    assert schema_reference_is_current(), (
        f"{DEFAULT_OUTPUT.name} or {DEFAULT_LEGACY_OUTPUT.name} is stale relative to the code; "
        "run `srtctl schema-docs` and commit the result"
    )


def test_render_is_deterministic() -> None:
    assert render_schema_reference() == render_schema_reference()
    assert render_legacy_reference() == render_legacy_reference()


def _legacy_keys() -> set[str]:
    keys = set(LEGACY_TOP_LEVEL)
    for mapping in LEGACY_FIELDS.values():
        keys.update(mapping)
    return keys


def _section(text: str, heading: str) -> str:
    """The body of one `### Heading` section of a rendered document."""
    start = text.index(f"\n### {heading}\n")
    rest = text[start + 1 :]
    end = rest.find("\n### ", 1)
    return rest if end == -1 else rest[:end]


def test_schema_reference_documents_only_the_2_0_layout() -> None:
    text = render_schema_reference()
    recipe_table = text[text.index("## Recipe\n") : text.index("## Authoring surface")]
    for key in LEGACY_TOP_LEVEL:
        assert f"| `{key}` |" not in recipe_table, f"legacy top-level key {key} leaked into schema-reference.md"
    for cls, mapping in LEGACY_FIELDS.items():
        section = _section(text, cls.__name__)
        for key in mapping:
            assert f"| `{key}` |" not in section, f"legacy key {cls.__name__}.{key} leaked into schema-reference.md"
    for cls in LEGACY_CLASSES:
        assert f"### {cls.__name__}" not in text, f"legacy class {cls.__name__} leaked into schema-reference.md"
    for needle in (
        "## Authoring surface",
        "### engine",
        "### roles",
        "### placement",
        "`colocate`",
        "## Engine types",
        "`engine.type: sglang`",
        "## Cluster config",
        "[legacy-v1.md](legacy-v1.md)",
    ):
        assert needle in text, needle
    assert "`backend.type:" not in text
    assert "## Backend types" not in text


def test_legacy_reference_documents_every_v1_key() -> None:
    text = render_legacy_reference()
    for key in LEGACY_TOP_LEVEL:
        assert f"| `{key}` (top level) |" in text, key
    for cls, mapping in LEGACY_FIELDS.items():
        section = _section(text, cls.__name__) if cls in LEGACY_CLASSES or cls.__name__.endswith("Protocol") else None
        for key in mapping:
            assert f".{key}` |" in text, f"legacy key {cls.__name__}.{key} missing from the mapping table"
            if section is not None:
                assert f"| `{key}` |" in section, f"legacy key {cls.__name__}.{key} missing from its table"
    for cls in LEGACY_CLASSES:
        assert f"### {cls.__name__}" in text, cls.__name__
    for needle in ("## v1 keys and what replaced them", "## backend", "## infra", "srtctl migrate"):
        assert needle in text, needle
    for type_name, _ in BACKEND_TYPES:
        assert f"`backend.type: {type_name}`" in text


def test_every_documented_legacy_key_is_rewritten_by_migrate() -> None:
    """LEGACY_FIELDS is the doc partition; the migrator is the behavior. They must agree."""
    v1 = """
name: legacy-all
model: {path: /m, container: /c.sqsh, precision: bf16}
resources:
  gpu_type: h100
  gpus_per_node: 8
  prefill_nodes: 1
  prefill_workers: 1
  gpus_per_prefill: 4
  decode_nodes: 0
  decode_workers: 1
  gpus_per_decode: 4
frontend:
  type: dynamo
  orchestrator_placement: head
  dedicated_node: false
dynamo:
  install: true
  hash: "abc1234"
  cargo_patches: ['x = 1']
infra:
  etcd_nats_dedicated_node: false
  nats_max_payload_mb: 16
backend:
  type: sglang
  prefill_environment: {A: "1"}
  decode_environment: {B: "2"}
  sglang_config:
    prefill: {tensor-parallel-size: 4}
    decode: {tensor-parallel-size: 4}
  kv_events_config:
    prefill: true
benchmark:
  type: sa-bench
  isl: 128
  osl: 128
  concurrencies: "4"
  client_placement: head
  client_dedicated_node: false
"""
    migrated = migrate_recipe_text(v1).text
    import yaml

    doc = yaml.safe_load(migrated)
    assert "backend" not in doc and "infra" not in doc
    for key in LEGACY_FIELDS[ResourceConfig]:
        assert key not in doc.get("resources", {}), key
    for key in LEGACY_FIELDS[FrontendConfig]:
        assert key not in doc.get("frontend", {}), key
    for key in LEGACY_FIELDS[BenchmarkConfig]:
        assert key not in doc.get("benchmark", {}), key
    for key in ("hash", "cargo_patches", "version", "wheel", "top_of_tree"):
        assert key not in doc.get("dynamo", {}), key
    assert doc["roles"]["decode"]["nodes"] == "colocate"


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


def test_engine_types_and_cluster_config_are_rendered() -> None:
    text = render_schema_reference()
    for heading in (
        "## Recipe",
        "## Engine types",
        "### SGLangProtocol",
        "### TRTLLMProtocol",
        "### VLLMProtocol",
        "### MockerProtocol",
        "## Cluster config",
    ):
        assert heading in text, heading
    assert "`engine.type: sglang`" in text
    assert "`sglang_config`" not in text  # per-mode config is a v1 spelling; roles.<role>.args in 2.0
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


def test_cli_writes_both_files(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "nested" / "schema-reference.md"
    monkeypatch.setattr(sys, "argv", ["srtctl", "schema-docs", "--output", str(output)])
    submit_cli.main()
    assert output.read_text() == render_schema_reference()
    assert (output.parent / "legacy-v1.md").read_text() == render_legacy_reference()


def test_cli_check_fails_when_only_the_legacy_file_is_stale(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "schema-reference.md"
    write_schema_reference(output)
    (tmp_path / "legacy-v1.md").write_text("# stale\n")
    monkeypatch.setattr(sys, "argv", ["srtctl", "schema-docs", "--check", "--output", str(output)])
    with pytest.raises(SystemExit) as exc_info:
        submit_cli.main()
    assert exc_info.value.code == 1
