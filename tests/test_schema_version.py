# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.config import generate_override_configs, load_config, validate_config_file
from srtctl.core.migrate import migrate_recipe_text
from srtctl.core.schema import CURRENT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

PLAIN = {
    "name": "schema-version-test",
    "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1},
    "backend": {"type": "sglang"},
    "frontend": {"type": "sglang", "enable_multiple_frontends": False},
    "benchmark": {"type": "custom", "command": "echo hi"},
}


def _write(tmp_path: Path, data: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_absent_schema_key_means_version_1(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, PLAIN))
    assert config.schema_version == 1


def test_schema_2_is_accepted(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, {"schema": 2, **PLAIN}))
    assert config.schema_version == 2
    assert CURRENT_SCHEMA_VERSION == 2
    assert SUPPORTED_SCHEMA_VERSIONS == (1, 2)


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        load_config(_write(tmp_path, {"schema": 3, **PLAIN}))


def test_schema_key_beside_base_propagates_to_every_override_variant() -> None:
    raw = {
        "schema": 2,
        "base": dict(PLAIN),
        "override_small": {"resources": {"agg_workers": 1}},
        "zip_override_names": {"name": ["a", "b"], "benchmark": {"command": ["echo a", "echo b"]}},
    }
    variants = generate_override_configs(raw)
    assert len(variants) == 3
    assert all(cfg["schema"] == 2 for _, cfg in variants)
    assert generate_override_configs(raw, selector="base")[0][1]["schema"] == 2


def test_validate_config_file_accepts_schema_on_override_and_sweep_files(tmp_path: Path) -> None:
    override = _write(
        tmp_path,
        {"schema": 2, "base": dict(PLAIN), "override_x": {"benchmark": {"command": "echo x"}}},
        "override.yaml",
    )
    assert validate_config_file(override) == []

    sweep_config = {"schema": 2, **PLAIN, "sweep": {"cmd": ["echo 1", "echo 2"]}}
    sweep_config["benchmark"] = {"type": "custom", "command": "{cmd}"}
    sweep = _write(tmp_path, sweep_config, "sweep.yaml")
    assert validate_config_file(sweep) == []


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
    path = _write(tmp_path, PLAIN)
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "-f", str(path)])
    submit_cli.main()
    out = capsys.readouterr().out
    assert out.startswith("schema: 2\n")
    assert yaml.safe_load(path.read_text()).get("schema") is None, "stdout mode must not touch the file"


def test_cli_migrate_in_place_rewrites_the_file(tmp_path: Path, monkeypatch, capsys) -> None:
    path = _write(tmp_path, PLAIN)
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "-f", str(path), "--in-place"])
    submit_cli.main()
    assert yaml.safe_load(path.read_text())["schema"] == 2
    assert "schema 1 -> 2" in capsys.readouterr().out
    assert load_config(path).schema_version == 2


def test_cli_migrate_output_writes_a_new_file(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, PLAIN)
    output = tmp_path / "out" / "migrated.yaml"
    monkeypatch.setattr(sys, "argv", ["srtctl", "migrate", "-f", str(path), "--output", str(output)])
    submit_cli.main()
    assert yaml.safe_load(output.read_text())["schema"] == 2


def test_every_example_declares_the_current_schema() -> None:
    for path in sorted(EXAMPLES_DIR.rglob("*.yaml")):
        declared = yaml.safe_load(path.read_text()).get("schema")
        assert declared == CURRENT_SCHEMA_VERSION, f"{path} declares schema {declared!r}"
