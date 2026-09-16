# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.config import (
    LEGACY_SECTION_KEYS,
    LEGACY_TOP_LEVEL_KEYS,
    legacy_keys_present,
    resolve_config_with_defaults,
)
from srtctl.core.migrate import migrate_recipe_text
from srtctl.core.schema import ObservabilityConfig, ResourceConfig, SrtConfig
from srtctl.core.schema_docs import (
    BACKEND_TYPES,
    DEFAULT_OUTPUT,
    INTERNAL_CLASSES,
    INTERNAL_FIELDS,
    INTERNAL_TOP_LEVEL,
    field_docs,
    render_schema_reference,
    schema_reference_is_current,
    write_schema_reference,
)

LEGACY_DOC = Path(__file__).parent.parent / "docs" / "legacy-v1.md"


def test_checked_in_schema_reference_is_current() -> None:
    """docs/schema-reference.md must be regenerated whenever the schema changes.

    Fix with: uv run srtctl schema-docs
    """
    assert DEFAULT_OUTPUT.exists(), f"{DEFAULT_OUTPUT} is missing; run `srtctl schema-docs`"
    assert schema_reference_is_current(), (
        f"{DEFAULT_OUTPUT.name} is stale relative to the code; run `srtctl schema-docs` and commit the result"
    )


def test_render_is_deterministic() -> None:
    assert render_schema_reference() == render_schema_reference()


def _section(text: str, heading: str) -> str:
    """The body of one `### Heading` section of a rendered document."""
    start = text.index(f"\n### {heading}\n")
    rest = text[start + 1 :]
    end = rest.find("\n### ", 1)
    return rest if end == -1 else rest[:end]


def test_schema_reference_documents_only_the_recipe_layout() -> None:
    text = render_schema_reference()
    recipe_table = text[text.index("## Recipe\n") : text.index("## Authoring surface")]
    for key in INTERNAL_TOP_LEVEL:
        assert f"| `{key}` |" not in recipe_table, f"internal top-level key {key} leaked into schema-reference.md"
    for cls, keys in INTERNAL_FIELDS.items():
        section = _section(text, cls.__name__)
        for key in keys:
            assert f"| `{key}` |" not in section, f"internal key {cls.__name__}.{key} leaked into schema-reference.md"
    for cls in INTERNAL_CLASSES:
        assert f"### {cls.__name__}" not in text, f"internal class {cls.__name__} leaked into schema-reference.md"
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
        "| `top_of_tree` |",  # still a recipe key: no dynamo.source equivalent
    ):
        assert needle in text, needle
    assert "`backend.type:" not in text
    assert "## Backend types" not in text
    assert "`sglang_config`" not in text  # per-mode config is internal; roles.<role>.args in a recipe
    assert "| `schema` | int | required |" in text


def test_internal_fields_are_exactly_the_keys_the_loader_rejects() -> None:
    """The docs partition and the loader gate must name the same recipe keys."""
    assert frozenset(LEGACY_TOP_LEVEL_KEYS) == INTERNAL_TOP_LEVEL
    for cls, section in ((ResourceConfig, "resources"),):
        assert frozenset(LEGACY_SECTION_KEYS[section]) == INTERNAL_FIELDS[cls]
    for _, cls in BACKEND_TYPES:
        assert INTERNAL_FIELDS[cls] <= {row.key for row in field_docs(cls)}
        assert {"prefill_environment", "decode_environment", "aggregated_environment"} <= INTERNAL_FIELDS[cls]


def test_legacy_doc_is_static_and_lists_every_rejected_key() -> None:
    """docs/legacy-v1.md is hand-written now; it must still cover every key the gate rejects."""
    text = LEGACY_DOC.read_text(encoding="utf-8")
    assert "GENERATED FILE" not in text
    assert "no longer loads" in text
    for key in LEGACY_TOP_LEVEL_KEYS:
        assert f"| `{key}` (top level) |" in text, key
    for section, keys in LEGACY_SECTION_KEYS.items():
        for key in keys:
            assert f"| `{section}.{key}` |" in text, f"{section}.{key} missing from the mapping table"
    for needle in (
        "## v1 keys and what replaced them",
        "## backend",
        "## infra",
        "srtctl migrate",
        "`dynamo.top_of_tree` is not in this table",
    ):
        assert needle in text, needle
    for type_name, _ in BACKEND_TYPES:
        assert f"`backend.type: {type_name}`" in text


V1_EVERYTHING = """
name: legacy-all
model: {path: /m, container: /c.sqsh, precision: bf16}
resources:
  gpu_type: h100
  gpus_per_node: 8
  prefill_nodes: 1
  prefill_workers: 1
  gpus_per_prefill: 4
  prefill_critical: false
  decode_nodes: 0
  decode_workers: 1
  gpus_per_decode: 4
  decode_critical: false
frontend:
  type: dynamo
  orchestrator_placement: head
  dedicated_node: false
dynamo:
  install: true
  version: "0.8.0"
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


def test_every_rejected_key_is_rewritten_by_migrate_into_something_that_loads() -> None:
    """The loader gate's key list and the migrator are two views of one contract."""
    original = yaml.safe_load(V1_EVERYTHING)
    assert set(legacy_keys_present(original)) >= {"backend", "infra", "resources.prefill_nodes", "dynamo.hash"}
    with pytest.raises(ValueError, match="no `schema:` key"):
        resolve_config_with_defaults(original, None)

    migrated = yaml.safe_load(migrate_recipe_text(V1_EVERYTHING).text)
    assert migrated["schema"] == 2
    assert legacy_keys_present(migrated) == [], legacy_keys_present(migrated)
    assert migrated["roles"]["decode"]["nodes"] == "colocate"
    config = SrtConfig.Schema().load(resolve_config_with_defaults(migrated, None))
    assert config.resources.decode_nodes == 0
    assert config.resources.worker_critical("prefill") is False
    assert config.dynamo.hash == "abc1234"
    assert config.dynamo.cargo_patches == ["x = 1"]
    assert config.infra.nats_max_payload_mb == 16
    assert config.backend.get_kv_events_config_for_mode("prefill")

    # A wheel install migrates too (it cannot share a recipe with hash).
    wheel = yaml.safe_load(
        migrate_recipe_text(
            "name: w\nmodel: {path: /m, container: /c, precision: bf16}\ndynamo:\n  wheel: '1.4.0'\n"
        ).text
    )
    assert legacy_keys_present(wheel) == []
    assert wheel["dynamo"] == {"source": {"wheel": "1.4.0"}}


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


def test_cli_writes_only_the_schema_reference(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "nested" / "schema-reference.md"
    monkeypatch.setattr(sys, "argv", ["srtctl", "schema-docs", "--output", str(output)])
    submit_cli.main()
    assert output.read_text() == render_schema_reference()
    assert sorted(p.name for p in output.parent.iterdir()) == ["schema-reference.md"]
