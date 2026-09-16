# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recipe schema gate: only ``schema: 2`` loads, and only in the 2.0 layout.

The pre-2.0 (v1) layout is rejected at every entry point with a pointer to
``srtctl migrate``, which is the one part of srtctl that still reads it.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.config import (
    LEGACY_SECTION_KEYS,
    LEGACY_TOP_LEVEL_KEYS,
    generate_override_configs,
    legacy_keys_present,
    load_config,
    require_current_schema,
    resolve_config_with_defaults,
    validate_config_file,
)
from srtctl.core.migrate import MIGRATABLE_SCHEMA_VERSIONS, migrate_recipe_text
from srtctl.core.schema import CURRENT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, SrtConfig

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

PLAIN = {
    "schema": 2,
    "name": "schema-version-test",
    "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8},
    "engine": "sglang",
    "roles": {"agg": {"nodes": 1, "workers": 1}},
    "frontend": {"type": "sglang-router", "enable_multiple_frontends": False},
    "benchmark": {"type": "custom", "command": "echo hi"},
}

# The same job in the pre-2.0 layout: no schema key, backend:, resources.agg_*.
PLAIN_V1 = {
    "name": "schema-version-test",
    "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1},
    "backend": {"type": "sglang"},
    "frontend": {"type": "sglang-router", "enable_multiple_frontends": False},
    "benchmark": {"type": "custom", "command": "echo hi"},
}


def _write(tmp_path: Path, data: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _without_schema(recipe: dict) -> dict:
    stripped = copy.deepcopy(recipe)
    stripped.pop("schema", None)
    return stripped


# --- the gate ---------------------------------------------------------------------------


def test_only_the_current_schema_is_supported() -> None:
    assert CURRENT_SCHEMA_VERSION == 2
    assert SUPPORTED_SCHEMA_VERSIONS == (2,)
    assert MIGRATABLE_SCHEMA_VERSIONS == (1, 2)


def test_schema_2_is_accepted(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, PLAIN))
    assert config.schema_version == 2
    assert config.resources.num_agg == 1


def test_absent_schema_key_is_rejected_as_pre_2_0(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no `schema:` key.*srtctl migrate") as exc:
        load_config(_write(tmp_path, _without_schema(PLAIN)))
    assert "legacy-v1.md" in str(exc.value)


def test_explicit_schema_1_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema 1 is not supported.*srtctl migrate"):
        resolve_config_with_defaults({**PLAIN, "schema": 1}, None)


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema 3 is not supported"):
        load_config(_write(tmp_path, {**PLAIN, "schema": 3}))
    with pytest.raises(ValueError, match="not supported"):
        resolve_config_with_defaults({**PLAIN, "schema": "2"}, None)
    with pytest.raises(ValueError, match="not supported"):
        resolve_config_with_defaults({**PLAIN, "schema": True}, None)


def test_v1_keys_are_rejected_even_under_schema_2(tmp_path: Path) -> None:
    """Declaring schema: 2 on an unmigrated recipe is not a migration; the error names the keys."""
    with pytest.raises(
        ValueError, match=r"pre-2\.0 \(v1\) layout: backend, resources\.agg_nodes, resources\.agg_workers"
    ):
        load_config(_write(tmp_path, {"schema": 2, **PLAIN_V1}))


@pytest.mark.parametrize("key", LEGACY_TOP_LEVEL_KEYS)
def test_every_v1_top_level_key_is_rejected(key: str) -> None:
    recipe = {**PLAIN, key: {}}
    assert legacy_keys_present(recipe) == [key]
    with pytest.raises(ValueError, match=rf"pre-2\.0 \(v1\) layout: {key}\b"):
        require_current_schema(recipe)


@pytest.mark.parametrize(
    ("section", "key"),
    [(section, key) for section, keys in LEGACY_SECTION_KEYS.items() for key in keys],
)
def test_every_v1_section_key_is_rejected(section: str, key: str) -> None:
    recipe = copy.deepcopy(PLAIN)
    recipe.setdefault(section, {})[key] = 1
    assert legacy_keys_present(recipe) == [f"{section}.{key}"]
    with pytest.raises(ValueError, match=rf"pre-2\.0 \(v1\) layout: {section}\.{key}"):
        require_current_schema(recipe)


def test_top_of_tree_is_still_a_recipe_key() -> None:
    """It has no dynamo.source equivalent, so the migrator leaves it and the loader accepts it."""
    recipe = {**PLAIN, "frontend": {"type": "dynamo"}, "dynamo": {"top_of_tree": True}}
    config = SrtConfig.Schema().load(resolve_config_with_defaults(recipe, None))
    assert config.dynamo.top_of_tree is True
    assert config.dynamo.needs_source_install


def test_engine_mapping_rejects_per_role_settings() -> None:
    """The v1 per-mode keys cannot be smuggled back in through the engine mapping."""
    recipe = {**PLAIN, "engine": {"type": "sglang", "sglang_config": {"aggregated": {"tp": 1}}}}
    with pytest.raises(ValueError, match="engine: carries per-role settings \\(sglang_config\\)"):
        resolve_config_with_defaults(recipe, None)
    recipe = {**PLAIN, "engine": {"type": "sglang", "aggregated_environment": {"A": "1"}}}
    with pytest.raises(ValueError, match="per-role settings \\(aggregated_environment\\)"):
        resolve_config_with_defaults(recipe, None)


def test_from_yaml_is_gated_like_load_config(tmp_path: Path) -> None:
    assert SrtConfig.from_yaml(_write(tmp_path, PLAIN)).schema_version == 2
    with pytest.raises(ValueError, match="no `schema:` key"):
        SrtConfig.from_yaml(_write(tmp_path, PLAIN_V1, "v1.yaml"))


def test_validate_config_file_reports_a_v1_recipe(tmp_path: Path) -> None:
    errors = validate_config_file(_write(tmp_path, PLAIN_V1))
    assert len(errors) == 1
    assert "srtctl migrate" in errors[0]


# --- schema key placement in multi-recipe files ----------------------------------------------


def test_schema_key_beside_base_propagates_to_every_override_variant() -> None:
    raw = {
        "schema": 2,
        "base": _without_schema(PLAIN),
        "override_small": {"roles": {"agg": {"workers": 1}}},
        "zip_override_names": {"name": ["a", "b"], "benchmark": {"command": ["echo a", "echo b"]}},
    }
    variants = generate_override_configs(raw)
    assert len(variants) == 3
    assert all(cfg["schema"] == 2 for _, cfg in variants)
    assert generate_override_configs(raw, selector="base")[0][1]["schema"] == 2


def test_validate_config_file_accepts_schema_on_override_and_sweep_files(tmp_path: Path) -> None:
    override = _write(
        tmp_path,
        {"schema": 2, "base": _without_schema(PLAIN), "override_x": {"benchmark": {"command": "echo x"}}},
        "override.yaml",
    )
    assert validate_config_file(override) == []

    sweep_config = {**PLAIN, "sweep": {"cmd": ["echo 1", "echo 2"]}}
    sweep_config["benchmark"] = {"type": "custom", "command": "{cmd}"}
    sweep = _write(tmp_path, sweep_config, "sweep.yaml")
    assert validate_config_file(sweep) == []


def test_override_file_without_schema_is_rejected(tmp_path: Path) -> None:
    override = _write(tmp_path, {"base": _without_schema(PLAIN), "override_x": {}}, "override.yaml")
    errors = validate_config_file(override)
    assert errors and "srtctl migrate" in errors[0]


# --- srtctl migrate still reads schema 1 -------------------------------------------------------


def test_migrate_inserts_schema_first_and_preserves_comments() -> None:
    text = '# my recipe\nname: "x"  # keep quotes\nmodel:\n  path: m\n  container: c\n  precision: fp8\n'
    result = migrate_recipe_text(text)
    assert result.changed
    assert result.from_version == 1
    assert result.to_version == CURRENT_SCHEMA_VERSION
    assert result.text.startswith('# my recipe\nschema: 2\nname: "x"  # keep quotes\n')
    assert "set schema: 2" in result.notes


def test_migrate_is_idempotent() -> None:
    once = migrate_recipe_text("name: x\nmodel: {path: m, container: c, precision: fp8}\n")
    twice = migrate_recipe_text(once.text)
    assert not twice.changed
    assert twice.notes == ()


def test_migrate_upgrades_an_explicit_schema_1() -> None:
    result = migrate_recipe_text("schema: 1\nname: x\n")
    assert result.text.startswith("schema: 2\nname: x\n")


def test_migrate_rejects_unknown_versions() -> None:
    with pytest.raises(ValueError, match="not supported"):
        migrate_recipe_text("schema: 9\nname: x\n")


def test_migrate_keeps_override_and_lock_sections_top_level() -> None:
    text = "base:\n  name: x\noverride_big:\n  name: y\nlock:\n  integrity: abc\n"
    result = migrate_recipe_text(text)
    loaded = yaml.safe_load(result.text)
    assert list(loaded) == ["schema", "base", "override_big", "lock"]
    assert "schema" not in loaded["base"]


def test_cli_migrate_prints_to_stdout_by_default(tmp_path: Path, monkeypatch, capsys) -> None:
    path = _write(tmp_path, PLAIN_V1)
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "-f", str(path)])
    submit_cli.main()
    out = capsys.readouterr().out
    assert out.startswith("schema: 2\n")
    assert yaml.safe_load(path.read_text()).get("schema") is None, "stdout mode must not touch the file"


def test_cli_migrate_in_place_rewrites_the_file_into_something_that_loads(tmp_path: Path, monkeypatch, capsys) -> None:
    path = _write(tmp_path, PLAIN_V1)
    with pytest.raises(ValueError, match="srtctl migrate"):
        load_config(path)
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "-f", str(path), "--in-place"])
    submit_cli.main()
    doc = yaml.safe_load(path.read_text())
    assert doc["schema"] == 2
    assert legacy_keys_present(doc) == []
    assert "schema 1 -> 2" in capsys.readouterr().out
    config = load_config(path)
    assert config.schema_version == 2
    assert config.resources.num_agg == 1


def test_cli_migrate_output_writes_a_new_file(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, PLAIN_V1)
    output = tmp_path / "out" / "migrated.yaml"
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "-f", str(path), "--output", str(output)])
    submit_cli.main()
    assert yaml.safe_load(output.read_text())["schema"] == 2
    assert load_config(output).name == PLAIN_V1["name"]


def test_cli_migrate_no_longer_offers_verify(tmp_path: Path, monkeypatch) -> None:
    """--verify compared the v1 and v2 loaders; there is no v1 loader left to compare against."""
    path = _write(tmp_path, PLAIN_V1)
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "-f", str(path), "--verify"])
    with pytest.raises(SystemExit) as exc:
        submit_cli.main()
    assert exc.value.code == 2


def test_every_example_declares_the_current_schema() -> None:
    for path in sorted(EXAMPLES_DIR.rglob("*.yaml")):
        declared = yaml.safe_load(path.read_text()).get("schema")
        assert declared == CURRENT_SCHEMA_VERSION, f"{path} declares schema {declared!r}"
