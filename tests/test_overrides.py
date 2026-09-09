# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.overrides import (
    Override,
    apply_override,
    apply_overrides_to_recipe,
    format_path,
    parse_overrides,
    parse_path,
    parse_set,
    parse_value,
)
from srtctl.core.yaml_utils import dump_yaml_with_comments, load_yaml_text_with_comments

PLAIN = {
    "schema": 2,
    "name": "override-test",
    "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1},
    "backend": {"type": "sglang", "sglang_config": {"aggregated": {"tp-size": 1}}},
    "frontend": {"type": "sglang", "enable_multiple_frontends": False},
    "health_check": {"max_attempts": 180, "interval_seconds": 10},
    "benchmark": {"type": "custom", "command": "echo hi"},
}


def _write(tmp_path: Path, data: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("name", ("name",)),
        ("health_check.max_attempts", ("health_check", "max_attempts")),
        ("backend.sglang_config.prefill.dist-timeout", ("backend", "sglang_config", "prefill", "dist-timeout")),
        ('container_mounts."/a/b.c"', ("container_mounts", "/a/b.c")),
        ("host_setup.commands[1]", ("host_setup", "commands", 1)),
        ("a[0].b[2].c", ("a", 0, "b", 2, "c")),
    ],
)
def test_parse_path(text: str, expected: tuple) -> None:
    assert parse_path(text) == expected
    assert parse_path(format_path(expected)) == expected


@pytest.mark.parametrize("bad", ["", "a..b", ".a", "a.", "[0]", "a[x]", 'a."unterminated'])
def test_parse_path_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_path(bad)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("720", 720),
        ('"720"', "720"),
        ("true", True),
        ("0.85", 0.85),
        ("[4, 8]", [4, 8]),
        ("1x2x4", "1x2x4"),
        ("", ""),
        ('{"rope_type": "yarn"}', '{"rope_type": "yarn"}'),
        ("{concurrency}", "{concurrency}"),
    ],
)
def test_parse_value(raw: str, expected: object) -> None:
    assert parse_value(raw) == expected


def test_parse_set_requires_equals() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_set("health_check.max_attempts")
    assert parse_set("a.b=x=y").value == "x=y"


def test_parse_overrides_orders_sets_before_unsets() -> None:
    overrides = parse_overrides(["a=1"], ["b"])
    assert [o.unset for o in overrides] == [False, True]
    assert overrides[0].render() == "--set a=1"
    assert overrides[1].render() == "--unset b"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def test_set_creates_intermediate_mappings() -> None:
    doc: dict = {}
    assert apply_override(doc, parse_set("backend.sglang_config.prefill.dist-timeout=1800"))
    assert doc == {"backend": {"sglang_config": {"prefill": {"dist-timeout": 1800}}}}


def test_set_replaces_existing_values_and_list_items() -> None:
    doc = {"host_setup": {"commands": ["a", "b"]}}
    apply_override(doc, parse_set("host_setup.commands[1]=c"))
    assert doc["host_setup"]["commands"] == ["a", "c"]
    with pytest.raises(ValueError, match="out of range"):
        apply_override(doc, parse_set("host_setup.commands[5]=z"))


def test_unset_is_lenient_on_missing_paths() -> None:
    doc = {"health_check": {"max_attempts": 1}}
    assert apply_override(doc, Override(path=("health_check",), unset=True))
    assert "health_check" not in doc
    assert not apply_override(doc, Override(path=("health_check", "interval_seconds"), unset=True))
    assert not apply_override(doc, Override(path=("nope", "deeper"), unset=True))


def test_set_on_non_mapping_fails_loudly() -> None:
    with pytest.raises(TypeError, match="non-mapping"):
        apply_override({"name": "x"}, parse_set("name.child=1"))


def test_override_file_semantics_write_to_base_and_every_variant() -> None:
    doc = {
        "base": {"name": "b", "resources": {"agg_workers": 2}, "health_check": {"max_attempts": 10}},
        "override_small": {"resources": {"agg_workers": 1}},
        "zip_override_ctx": {"name": ["x", "y"], "resources": {"agg_workers": [3, 4]}},
    }
    applied = apply_overrides_to_recipe(doc, parse_overrides(["resources.agg_workers=8"], ["health_check"]))
    assert applied == ["--set resources.agg_workers=8", "--unset health_check"]
    assert doc["base"]["resources"]["agg_workers"] == 8
    assert doc["override_small"]["resources"]["agg_workers"] == 8, "a variant must not shadow --set"
    assert doc["zip_override_ctx"]["resources"]["agg_workers"] == [8], "zip groups get a broadcast list"
    assert "health_check" not in doc["base"]


def test_overrides_preserve_comments_on_round_trip_documents() -> None:
    text = "# recipe\nname: x  # job\nhealth_check:\n  max_attempts: 10\n"
    doc = load_yaml_text_with_comments(text)
    apply_overrides_to_recipe(doc, parse_overrides(["health_check.max_attempts=720", "resources.gpu_type=h100"], []))
    out = dump_yaml_with_comments(doc) or ""
    assert out.startswith("# recipe\nname: x  # job\n")
    assert "max_attempts: 720" in out
    assert "gpu_type: h100" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_dry_run_applies_set_and_unset(tmp_path: Path, monkeypatch, capsys) -> None:
    path = _write(tmp_path, PLAIN)
    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "dry-run", "-f", str(path), "--set", "name=renamed-by-set", "--unset", "health_check"],
    )
    submit_cli.main()
    out = capsys.readouterr().out
    assert "renamed-by-set" in out
    assert yaml.safe_load(path.read_text())["name"] == "override-test", "the source file is never modified"


def test_set_value_wins_over_a_cluster_default_block() -> None:
    from srtctl.core.config import resolve_config_with_defaults

    recipe = {k: v for k, v in PLAIN.items() if k != "health_check"}
    apply_overrides_to_recipe(recipe, parse_overrides(["health_check.max_attempts=720"], []))
    resolved = resolve_config_with_defaults(
        recipe, {"default_health_check": {"max_attempts": 5, "interval_seconds": 10}}
    )
    assert resolved["health_check"] == {"max_attempts": 720}, "an explicit --set beats default_health_check"


def test_apply_mock_json_records_applied_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    path = _write(tmp_path, PLAIN)
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srtctl",
            "apply",
            "-f",
            str(path),
            "-o",
            str(outputs),
            "--mock",
            "--mock-tick-s",
            "0.05",
            "--json",
            "--set",
            "benchmark.command=echo overridden",
            "--unset",
            "health_check",
        ],
    )
    submit_cli.main()
    record = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert record["status"] == "submitted"
    assert record["applied_overrides"] == ["--set benchmark.command=echo overridden", "--unset health_check"]
    written = yaml.safe_load((Path(record["output_dir"]) / "config.yaml").read_text())
    assert written["benchmark"]["command"] == "echo overridden"
    assert "health_check" not in written


def test_resolve_override_stdout_applies_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    path = _write(
        tmp_path,
        {"schema": 2, "base": dict(PLAIN), "override_small": {"resources": {"agg_workers": 4}}},
        "override.yaml",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["srtctl", "resolve-override", "-f", f"{path}:override_small", "--stdout", "--set", "resources.agg_workers=2"],
    )
    submit_cli.main()
    out = capsys.readouterr().out
    assert "agg_workers: 2" in out
    assert "agg_workers: 4" not in out
