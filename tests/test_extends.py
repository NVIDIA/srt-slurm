# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`extends`: shared partial recipes merged under a recipe (srtctl.core.compose, docs/extends.md)."""

from __future__ import annotations

import io
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.core.compose import (
    ComposedRecipe,
    ExtendsError,
    compose_recipe,
    compose_recipe_text,
    composition_header,
    layer_files,
    load_recipe,
)
from srtctl.core.config import generate_override_configs, load_config, validate_config_file
from srtctl.core.schema import SrtConfig

MODEL_LAYER = textwrap.dedent(
    """\
    # The model and the worker shape (a partial recipe).
    schema: 2
    name: "shared"
    model:
      path: "hf:fake/mock-model"
      container: "nvcr.io/fake:latest"
      precision: "fp8"
    resources:
      gpu_type: "h100"
      gpus_per_node: 8
    engine: sglang
    frontend:
      type: sglang-router
      enable_multiple_frontends: false
    roles:
      agg:
        nodes: 1
        workers: 1
        env:
          KEEP: "1"
          DROP: "2"
          FLAG: "n"           # a string PyYAML would read as a bool unquoted
        args:
          tp-size: 1
          context-length: 4096
    extra_mount: ["/a:/a", "/b:/b"]
    environment:
      SHARED: "model-layer"
    """
)
BENCH_LAYER = textwrap.dedent(
    """\
    slurm:
      time_limit: "00:30:00"
    benchmark:
      type: custom
      command: "echo layer"
    environment:
      SHARED: "bench-layer"   # a later file wins
    """
)
RECIPE = textwrap.dedent(
    """\
    # A recipe built from the two shared files.
    schema: 2
    extends: [shared/model.yaml, shared/bench.yaml]
    name: "composed"
    roles:
      agg:
        env:
          DROP: null          # null deletes a key a shared file set
        args:
          context-length: 8192
    extra_mount: ["/c:/c"]   # lists replace
    """
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


@pytest.fixture
def recipes(tmp_path: Path) -> Path:
    """tmp/recipes/{shared/model.yaml, shared/bench.yaml, recipe.yaml}; returns the recipe."""
    root = tmp_path / "recipes"
    _write(root / "shared" / "model.yaml", MODEL_LAYER)
    _write(root / "shared" / "bench.yaml", BENCH_LAYER)
    return _write(root / "recipe.yaml", RECIPE)


def _composed_text(path: Path) -> str:
    composed = compose_recipe_text(path.read_text(), path=path)
    assert composed is not None
    return composed.text()


# ---------------------------------------------------------------------------
# Merge rules
# ---------------------------------------------------------------------------


class TestCompose:
    def test_files_merge_left_to_right_and_the_recipe_wins(self, recipes: Path) -> None:
        merged = load_recipe(recipes)
        assert "extends" not in merged
        assert merged["name"] == "composed"  # the recipe over the model layer
        assert merged["environment"] == {"SHARED": "bench-layer"}  # the later layer over the earlier
        assert merged["benchmark"] == {"type": "custom", "command": "echo layer"}
        assert merged["slurm"] == {"time_limit": "00:30:00"}

    def test_merge_rules(self, recipes: Path) -> None:
        agg = load_recipe(recipes)["roles"]["agg"]
        assert agg["env"] == {"KEEP": "1", "FLAG": "n"}  # mapping merged, null deleted DROP
        assert agg["args"] == {"tp-size": 1, "context-length": 8192}  # nested merge, scalar replaced
        assert agg["nodes"] == 1 and agg["workers"] == 1
        assert load_recipe(recipes)["extra_mount"] == ["/c:/c"]  # the list replaced, not appended

    def test_a_null_that_deletes_nothing_is_kept(self, tmp_path: Path) -> None:
        """`telemetry.dcgm_exporter: null` keeps a cluster default exporter out; moving other keys
        to a shared file must not change what it means."""
        _write(
            tmp_path / "layer.yaml",
            """\
            telemetry:
              enabled: true
              node_exporter: null      # the layer's own null
              process_exporter: {port: 9256}
            environment: {A: "1"}
            """,
        )
        recipe = _write(
            tmp_path / "recipe.yaml",
            """\
            schema: 2
            extends: layer.yaml
            telemetry:
              dcgm_exporter: null      # nothing below sets it: kept
              process_exporter: null   # the layer set it: deleted
            environment: {A: null}     # deleted
            output: {nested: {key: null}}  # inside a new block: kept
            """,
        )
        for merged in (load_recipe(recipe), yaml.safe_load(_composed_text(recipe))):
            assert merged["telemetry"] == {"enabled": True, "node_exporter": None, "dcgm_exporter": None}
            assert merged["environment"] == {}
            assert merged["output"] == {"nested": {"key": None}}

        from srtctl.core.config import resolve_config_with_defaults

        cluster = {"default_gpu_exporter": {"port": 9400, "container_image": "dcgm"}}
        resolved = resolve_config_with_defaults(load_recipe(recipe), cluster)
        assert resolved["telemetry"]["dcgm_exporter"] is None  # the cluster default stayed out

    def test_schema_and_name_lead_the_merged_recipe(self, recipes: Path) -> None:
        assert list(load_recipe(recipes))[:2] == ["schema", "name"]
        assert list(yaml.safe_load(_composed_text(recipes)))[:2] == ["schema", "name"]

    def test_comment_keeping_and_plain_composition_agree(self, recipes: Path) -> None:
        """The CLI composes the ruamel document it writes for the job; loaders compose plain dicts."""
        assert yaml.safe_load(_composed_text(recipes)) == load_recipe(recipes)

    def test_written_recipe_keeps_comments_quotes_and_names_its_files(self, recipes: Path) -> None:
        text = _composed_text(recipes)
        root = recipes.parent.resolve()
        assert text.splitlines()[:4] == [
            "# Composed by srtctl from these files (later files win):",
            f"#   {root / 'shared' / 'model.yaml'}",
            f"#   {root / 'shared' / 'bench.yaml'}",
            f"#   {root / 'recipe.yaml'}",
        ]
        assert "# A recipe built from the two shared files." in text  # the recipe's own header
        assert "# a string PyYAML would read as a bool unquoted" in text  # a layer's comment
        assert 'FLAG: "n"' in text and 'time_limit: "00:30:00"' in text  # quotes kept
        assert "extends" not in yaml.safe_load(text)
        assert composition_header(text) == "\n".join(text.splitlines()[:4]) + "\n"

    def test_a_single_path_is_accepted(self, tmp_path: Path) -> None:
        _write(tmp_path / "model.yaml", MODEL_LAYER)
        recipe = _write(tmp_path / "recipe.yaml", "schema: 2\nextends: model.yaml\nname: one\n")
        assert load_recipe(recipe)["model"]["path"] == "hf:fake/mock-model"

    def test_paths_resolve_against_the_recipe_not_the_working_directory(
        self, recipes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert load_recipe(Path("../recipes/recipe.yaml"))["name"] == "composed"
        assert compose_recipe_text(RECIPE, path=Path("../recipes/recipe.yaml")) is not None

    def test_absolute_paths(self, recipes: Path, tmp_path: Path) -> None:
        model = (recipes.parent / "shared" / "model.yaml").resolve()
        recipe = _write(tmp_path / "other" / "recipe.yaml", f"schema: 2\nextends: ['{model}']\nname: abs\n")
        assert load_recipe(recipe)["resources"]["gpu_type"] == "h100"

    def test_symlinks_name_files_as_written_and_count_once(self, recipes: Path, tmp_path: Path) -> None:
        (tmp_path / "link").symlink_to(recipes.parent / "shared", target_is_directory=True)
        recipe = _write(tmp_path / "linked.yaml", "schema: 2\nextends: [link/model.yaml]\nname: linked\n")
        composed = compose_recipe_text(recipe.read_text(), path=recipe)
        assert composed is not None
        assert composed.sources == (str(tmp_path / "link" / "model.yaml"), str(tmp_path / "linked.yaml"))
        assert layer_files([recipe]) == {(recipes.parent / "shared" / "model.yaml").resolve()}
        twice = _write(tmp_path / "twice.yaml", "schema: 2\nextends: [link/model.yaml, recipes/shared/model.yaml]\n")
        with pytest.raises(ExtendsError, match="more than once"):
            load_recipe(twice)

    def test_a_layer_may_repeat_the_schema(self, recipes: Path) -> None:
        assert load_recipe(recipes)["schema"] == 2  # model.yaml declares it too; bench.yaml does not

    def test_recipe_without_extends_is_untouched(self, tmp_path: Path) -> None:
        plain = _write(tmp_path / "plain.yaml", "schema: 2\nname: plain\n")
        assert compose_recipe_text(plain.read_text(), path=plain) is None
        assert load_recipe(plain) == {"schema": 2, "name": "plain"}
        assert compose_recipe({"schema": 2}, base_dir=tmp_path) == ComposedRecipe({"schema": 2}, ("<recipe>",))

    def test_inputs_are_not_mutated(self, recipes: Path) -> None:
        document = yaml.safe_load(recipes.read_text())
        before = yaml.safe_dump(document)
        compose_recipe(document, base_dir=recipes.parent)
        assert yaml.safe_dump(document) == before


# ---------------------------------------------------------------------------
# What stops composition
# ---------------------------------------------------------------------------


def _recipe_extending(tmp_path: Path, layer_text: str | None, extends: Any = "layer.yaml", **extra: Any) -> Path:
    if layer_text is not None:
        (tmp_path / "layer.yaml").write_text(textwrap.dedent(layer_text))
    document = {"schema": 2, "extends": extends, "name": "r", **extra}
    return _write(tmp_path / "recipe.yaml", yaml.safe_dump(document, sort_keys=False))


class TestComposeErrors:
    @pytest.mark.parametrize(
        ("layer_text", "message"),
        [
            (None, "file not found"),
            ("", "file is empty"),
            ("- a\n- b\n", "not a YAML mapping"),
            ("a: [1, 2\n", "layer.yaml"),
            ("extends: other.yaml\n", "cannot use extends itself"),
            ("base: {name: x}\n", "holds base"),
            ("override_x: {name: x}\n", "holds override_x"),
            ("zip_override_y: {name: [a, b]}\n", "holds zip_override_y"),
            ("sweep: {tp: [1, 2]}\n", "holds sweep"),
            ("schema: 1\n", "declares schema 1"),
        ],
    )
    def test_bad_shared_file(self, tmp_path: Path, layer_text: str | None, message: str) -> None:
        recipe = _recipe_extending(tmp_path, layer_text)
        for compose in (load_recipe, lambda path: compose_recipe_text(path.read_text(), path=path)):
            with pytest.raises(ExtendsError, match=message):
                compose(recipe)

    @pytest.mark.parametrize(
        ("extends", "message"),
        [
            (3, "must be a file path or a non-empty list"),
            ("", "must be a file path or a non-empty list"),
            ([], "must be a file path or a non-empty list"),
            (["layer.yaml", 3], "must be a file path or a non-empty list"),
            ({"a": "b"}, "must be a file path or a non-empty list"),
            (["layer.yaml", "./layer.yaml"], "more than once"),
            ("$HOME/layer.yaml", "not expanded"),
            ("~/layer.yaml", "not expanded"),
            ("*.yaml", "not expanded"),
        ],
    )
    def test_bad_extends_value(self, tmp_path: Path, extends: Any, message: str) -> None:
        recipe = _recipe_extending(tmp_path, "environment: {A: '1'}\n", extends)
        with pytest.raises(ExtendsError, match=message):
            load_recipe(recipe)

    def test_a_directory_is_not_a_file(self, tmp_path: Path) -> None:
        (tmp_path / "layer.yaml").mkdir()
        with pytest.raises(ExtendsError, match="not a file"):
            load_recipe(_recipe_extending(tmp_path, None))

    def test_the_recipe_must_declare_its_schema(self, tmp_path: Path) -> None:
        (tmp_path / "layer.yaml").write_text("schema: 2\nname: x\n")
        recipe = _write(tmp_path / "recipe.yaml", "extends: layer.yaml\n")
        with pytest.raises(ExtendsError, match="must declare its schema"):
            load_recipe(recipe)

    @pytest.mark.parametrize("section", ["base", "override_x", "zip_override_y"])
    def test_extends_is_top_level_only(self, tmp_path: Path, section: str) -> None:
        (tmp_path / "layer.yaml").write_text("environment: {A: '1'}\n")
        document = {"schema": 2, "extends": "layer.yaml", section: {"extends": "layer.yaml"}}
        recipe = _write(tmp_path / "recipe.yaml", yaml.safe_dump(document))
        with pytest.raises(ExtendsError, match=f"not inside {section}"):
            load_recipe(recipe)

    def test_errors_name_the_recipe(self, tmp_path: Path) -> None:
        recipe = _recipe_extending(tmp_path, None)
        with pytest.raises(ExtendsError) as excinfo:
            load_recipe(recipe)
        assert str(recipe.resolve()) in str(excinfo.value)
        assert str((tmp_path / "layer.yaml").resolve()) in str(excinfo.value)


# ---------------------------------------------------------------------------
# Override files
# ---------------------------------------------------------------------------


class TestOverrideFiles:
    def test_without_base_the_shared_files_are_the_base(self, recipes: Path) -> None:
        bundle = _write(
            recipes.parent / "variants.yaml",
            """\
            schema: 2
            extends: [shared/model.yaml, shared/bench.yaml]
            override_tp2:
              roles: {agg: {args: {tp-size: 2}}}
            zip_override_ctx:
              roles: {agg: {args: {context-length: [2048, 8192]}}}
            """,
        )
        raw = load_recipe(bundle)
        assert list(raw) == ["schema", "base", "override_tp2", "zip_override_ctx"]
        assert raw["base"]["name"] == "shared" and "schema" not in raw["base"]
        variants = dict(generate_override_configs(raw))
        assert variants["tp2"]["name"] == "shared_tp2"
        assert variants["tp2"]["roles"]["agg"]["args"] == {"tp-size": 2, "context-length": 4096}
        assert variants["tp2"]["environment"] == {"SHARED": "bench-layer"}
        assert [v["roles"]["agg"]["args"]["context-length"] for k, v in variants.items() if k != "tp2"] == [2048, 8192]
        [(_, base)] = generate_override_configs(raw, selector="base")
        assert base["benchmark"]["command"] == "echo layer" and base["schema"] == 2
        assert validate_config_file(bundle) == []

    def test_base_wins_over_the_shared_files_and_variants_over_base(self, recipes: Path) -> None:
        bundle = _write(
            recipes.parent / "with-base.yaml",
            """\
            schema: 2
            extends: shared/model.yaml
            base:
              name: "bundle"
              roles: {agg: {workers: 2}}
              benchmark: {type: custom, command: "echo base"}
            override_big:
              roles: {agg: {workers: 4}}
            """,
        )
        raw = load_recipe(bundle)
        [(_, big)] = generate_override_configs(raw, selector="override_big")
        assert big["name"] == "bundle_big"
        assert big["roles"]["agg"]["workers"] == 4
        assert big["roles"]["agg"]["args"]["context-length"] == 4096  # from the shared file
        assert raw["base"]["roles"]["agg"]["workers"] == 2
        assert yaml.safe_load(_composed_text(bundle)) == raw

    def test_detected_as_an_override_file_before_composition(self, recipes: Path) -> None:
        bundle = _write(recipes.parent / "variants.yaml", "schema: 2\nextends: shared/model.yaml\noverride_a: {}\n")
        assert submit_cli.is_override_config(bundle)
        assert not submit_cli.is_override_config(recipes)  # extends, but no variant


# ---------------------------------------------------------------------------
# Helpers for callers
# ---------------------------------------------------------------------------


class TestLayerFiles:
    def test_finds_the_files_others_extend(self, recipes: Path, tmp_path: Path) -> None:
        root = recipes.parent
        _write(root / "plain.yaml", "schema: 2\nname: plain\n")
        _write(root / "single.yaml", "schema: 2\nextends: shared/model.yaml\n")
        _write(root / "broken.yaml", "a: [1\n")
        files = sorted(root.rglob("*.yaml"))
        assert layer_files(files) == {
            (root / "shared" / "model.yaml").resolve(),
            (root / "shared" / "bench.yaml").resolve(),
        }

    def test_layers_parameter_merges_below_the_recipes_own_extends(self, recipes: Path, tmp_path: Path) -> None:
        """A launcher that picks shared files at run time passes them as `layers`."""
        launcher_file = _write(tmp_path / "launcher.yaml", "environment: {SHARED: launcher, LAUNCHER: '1'}\n")
        document = yaml.safe_load(recipes.read_text())
        composed = compose_recipe(
            document,
            base_dir=recipes.parent,
            source="recipe",
            layers=[
                {"benchmark": {"type": "custom", "command": "echo mapping"}, "slurm": {"account": "a"}},
                launcher_file,
            ],
        )
        merged = composed.document
        assert merged["environment"] == {"SHARED": "bench-layer", "LAUNCHER": "1"}  # extends win over layers
        assert merged["slurm"] == {"account": "a", "time_limit": "00:30:00"}
        assert merged["benchmark"]["command"] == "echo layer"
        assert composed.sources[0] == "<layer 0>" and composed.sources[1] == str(launcher_file.resolve())
        assert composed.sources[-1] == "recipe"
        with pytest.raises(ExtendsError, match="holds base"):
            compose_recipe({"schema": 2, "name": "x"}, base_dir=tmp_path, layers=[{"base": {}}])
        without_extends = compose_recipe(
            {"schema": 2, "name": "x"}, base_dir=tmp_path, layers=[{"slurm": {"account": "b"}}]
        )
        assert without_extends.document == {"schema": 2, "name": "x", "slurm": {"account": "b"}}


class TestLoaders:
    def test_load_config_and_from_yaml_compose(self, recipes: Path) -> None:
        with patch("srtctl.core.config.load_cluster_config", return_value=None):
            config = load_config(recipes)
        assert config.name == "composed"
        assert config.model.path == "hf:fake/mock-model"
        assert SrtConfig.from_yaml(recipes).name == "composed"

    def test_validate_config_file(self, recipes: Path, tmp_path: Path) -> None:
        assert validate_config_file(recipes) == []
        broken = _recipe_extending(tmp_path, None)
        [error] = validate_config_file(broken)
        assert "file not found" in error

    def test_mcp_spec_tools_reject_inline_extends(self) -> None:
        from srtctl.mcp.spec_tools import _load_raw_config

        with pytest.raises(ValueError, match="resolve-override"):
            _load_raw_config(config_yaml="schema: 2\nextends: base.yaml\n")
        with pytest.raises(ValueError, match="resolve-override"):
            _load_raw_config(config={"schema": 2, "extends": "base.yaml"})

    def test_submit_single_refuses_an_unstaged_recipe(self, recipes: Path) -> None:
        """The job copies the submitted file into its output directory, away from the files in extends."""
        with (
            patch("srtctl.core.config.load_cluster_config", return_value=None),
            pytest.raises(ValueError, match="has `extends`"),
        ):
            submit_cli.submit_single(config_path=recipes, dry_run=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["srtctl", *argv])
    submit_cli.main()


class TestMaterialize:
    def test_stages_the_composed_recipe(self, recipes: Path) -> None:
        source = recipes.read_text()
        with submit_cli.materialize_config_path(recipes) as staged:
            assert staged != recipes
            text = staged.read_text()
        assert not staged.exists()  # the staged copy is removed afterwards
        assert recipes.read_text() == source  # the recipe file is never modified
        assert text.startswith("# Composed by srtctl from these files (later files win):\n")
        assert yaml.safe_load(text) == load_recipe(recipes)

    def test_a_recipe_without_extends_is_not_staged(self, tmp_path: Path) -> None:
        plain = _write(tmp_path / "plain.yaml", "schema: 2\nname: plain\n")
        with submit_cli.materialize_config_path(plain) as staged:
            assert staged == plain

    def test_set_and_unset_apply_after_composing(self, recipes: Path) -> None:
        from srtctl.core.overrides import parse_overrides

        overrides = parse_overrides(["environment.SHARED=cli", "roles.agg.args.context-length=1024"], ["slurm"])
        with submit_cli.materialize_config_path(recipes, overrides) as staged:
            merged = yaml.safe_load(staged.read_text())
            header = composition_header(staged.read_text())
        assert merged["environment"]["SHARED"] == "cli"  # over both shared files
        assert merged["roles"]["agg"]["args"]["context-length"] == 1024  # over the recipe itself
        assert "slurm" not in merged  # a key that only a shared file set
        assert header

    def test_set_reaches_every_variant_of_a_composed_override_file(self, recipes: Path) -> None:
        from srtctl.core.overrides import parse_overrides

        bundle = _write(
            recipes.parent / "variants.yaml",
            "schema: 2\nextends: shared/model.yaml\noverride_a:\n  roles: {agg: {workers: 3}}\n",
        )
        with submit_cli.materialize_config_path(bundle, parse_overrides(["roles.agg.workers=5"], [])) as staged:
            raw = yaml.safe_load(staged.read_text())
        assert all(config["roles"]["agg"]["workers"] == 5 for _, config in generate_override_configs(raw))

    def test_stdin_resolves_paths_against_the_working_directory(
        self, recipes: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(recipes.parent)
        monkeypatch.setattr(sys, "stdin", io.StringIO(RECIPE))
        with submit_cli.materialize_config_path(Path("-")) as staged:
            text = staged.read_text()
        assert "#   <stdin>" in text
        assert yaml.safe_load(text) == load_recipe(recipes)

    def test_pins_a_source_block_that_comes_from_a_shared_file(self, tmp_path: Path) -> None:
        _write(tmp_path / "layer.yaml", "dynamo:\n  source:\n    git: https://example.com/dynamo.git\n    rev: main\n")
        recipe = _write(tmp_path / "recipe.yaml", "schema: 2\nextends: layer.yaml\nname: pin\n")
        with (
            patch("srtctl.core.source.resolve_rev", return_value="0" * 40) as resolve,
            submit_cli.materialize_config_path(recipe, pin_sources=True) as staged,
        ):
            merged = yaml.safe_load(staged.read_text())
        resolve.assert_called_once_with("https://example.com/dynamo.git", "main")
        assert merged["dynamo"]["source"]["sha"] == "0" * 40


class TestCli:
    def test_dry_run(self, recipes: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _main(monkeypatch, "dry-run", "-f", str(recipes))

    def test_dry_run_reports_a_composition_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recipe = _recipe_extending(tmp_path, None)
        with pytest.raises(SystemExit) as excinfo:
            _main(monkeypatch, "dry-run", "-f", str(recipe))
        assert excinfo.value.code == 1
        assert "file not found" in " ".join(capsys.readouterr().out.split())  # rich wraps long lines

    def test_resolve_override_prints_a_composed_recipe(
        self, recipes: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _main(monkeypatch, "resolve-override", "-f", str(recipes), "--stdout")
        out = capsys.readouterr().out
        assert out.startswith("# Composed by srtctl from these files (later files win):\n")
        assert yaml.safe_load(out) == load_recipe(recipes)

    def test_resolve_override_writes_next_to_the_users_file(
        self, recipes: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _main(monkeypatch, "resolve-override", "-f", str(recipes))
        written = recipes.parent / "recipe_resolved.yaml"
        assert yaml.safe_load(written.read_text()) == load_recipe(recipes)

        bundle = _write(
            recipes.parent / "variants.yaml",
            "schema: 2\nextends: shared/model.yaml\noverride_a:\n  roles: {agg: {workers: 3}}\n",
        )
        _main(monkeypatch, "resolve-override", "-f", str(bundle), "--set", "roles.agg.args.tp-size=4")
        variant = yaml.safe_load((recipes.parent / "variants_a.yaml").read_text())
        assert variant["roles"]["agg"]["workers"] == 3
        assert variant["roles"]["agg"]["args"]["tp-size"] == 4

    def test_resolve_override_rejects_a_sweep(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write(tmp_path / "model.yaml", MODEL_LAYER)
        sweep = _write(
            tmp_path / "sweep.yaml",
            "schema: 2\nextends: model.yaml\nname: s-{tp}\nsweep: {tp: [1, 2]}\nroles: {agg: {args: {tp-size: '{tp}'}}}\n",
        )
        with pytest.raises(SystemExit) as excinfo:
            _main(monkeypatch, "resolve-override", "-f", str(sweep))
        assert excinfo.value.code == 1

    def test_sweep_jobs_keep_the_composition_header(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write(tmp_path / "model.yaml", MODEL_LAYER)
        _write(
            tmp_path / "sweep.yaml",
            """\
            schema: 2
            extends: model.yaml
            name: "sweep-tp{tp}"
            sweep:
              tp: [1, 2]
            roles:
              agg:
                args:
                  tp-size: "{tp}"
            benchmark: {type: custom, command: "echo sweep"}
            """,
        )
        monkeypatch.chdir(tmp_path)
        _main(monkeypatch, "dry-run", "-f", "sweep.yaml")
        configs = sorted((tmp_path / "dry-runs").glob("*/job_*/config.yaml"))
        assert len(configs) == 2
        for config in configs:
            text = config.read_text()
            assert composition_header(text)
            job = yaml.safe_load(text)
            assert job["model"]["path"] == "hf:fake/mock-model"
            assert job["roles"]["agg"]["env"]["KEEP"] == "1"
        assert sorted(str(yaml.safe_load(c.read_text())["roles"]["agg"]["args"]["tp-size"]) for c in configs) == [
            "1",
            "2",
        ]

    def test_directory_skips_the_shared_files(self, recipes: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write(
            recipes.parent / "variants.yaml",
            "schema: 2\nextends: shared/model.yaml\noverride_a:\n  roles: {agg: {workers: 3}}\n",
        )
        submitted: list[tuple[str, dict]] = []

        def record(kind: str):
            def submit(config_path: Path | None = None, *args: Any, **kwargs: Any) -> None:
                path = config_path or kwargs["config_path"]
                submitted.append((kind, yaml.safe_load(Path(path).read_text())))

            return submit

        with (
            patch.object(submit_cli, "submit_single", side_effect=record("single")),
            patch.object(submit_cli, "submit_override", side_effect=record("override")),
            patch.object(submit_cli, "submit_sweep", side_effect=record("sweep")),
        ):
            submit_cli.submit_directory(recipes.parent, dry_run=True)
        assert sorted(kind for kind, _ in submitted) == ["override", "single"]  # the shared files are not jobs
        assert all("extends" not in config for _, config in submitted)
        assert {kind: config.get("name", config.get("base", {}).get("name")) for kind, config in submitted} == {
            "single": "composed",
            "override": "shared",
        }


# ---------------------------------------------------------------------------
# End to end: `apply --mock` stages the composed recipe and the job runs it
# ---------------------------------------------------------------------------


def _wait_for(path: Path, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.1)
    return False


def _apply_mock(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: str, outputs: Path
) -> dict:
    _main(monkeypatch, "apply", "-f", config, "-o", str(outputs), "--mock", "--mock-tick-s", "0.05", "--json")
    [record] = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert record["status"] == "submitted"
    return record


class TestApplyMock:
    def test_the_job_records_and_runs_the_composed_recipe(
        self, recipes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record = _apply_mock(monkeypatch, capsys, str(recipes), tmp_path / "outputs")
        output_dir = Path(record["output_dir"])
        saved = (output_dir / "config.yaml").read_text()
        assert composition_header(saved)
        assert yaml.safe_load(saved) == load_recipe(recipes)
        assert _wait_for(output_dir / "result.json"), (output_dir / "mock_worker.log").read_text()
        result = json.loads((output_dir / "result.json").read_text())
        assert result["status"] == "completed" and result["exit_code"] == 0

    def test_an_override_variant_of_a_composed_file(
        self, recipes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle = _write(
            recipes.parent / "variants.yaml",
            "schema: 2\nextends: [shared/model.yaml, shared/bench.yaml]\noverride_a:\n  roles: {agg: {workers: 2}}\n",
        )
        record = _apply_mock(monkeypatch, capsys, f"{bundle}:override_a", tmp_path / "outputs")
        output_dir = Path(record["output_dir"])
        assert composition_header((output_dir / "config.yaml").read_text())
        variant = yaml.safe_load((output_dir / "config_a.yaml").read_text())
        assert variant["name"] == "shared_a"
        assert variant["roles"]["agg"]["workers"] == 2
        assert variant["benchmark"]["command"] == "echo layer"
        assert _wait_for(output_dir / "result.json"), (output_dir / "mock_worker.log").read_text()
        assert json.loads((output_dir / "result.json").read_text())["status"] == "completed"


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def test_interactive_mode_lists_recipes_not_shared_files_and_stages_them(recipes: Path, tmp_path: Path) -> None:
    from srtctl.cli import interactive

    examples = tmp_path / "examples"
    _write(examples / "shared" / "model.yaml", MODEL_LAYER)
    _write(examples / "shared" / "bench.yaml", BENCH_LAYER)
    _write(examples / "recipe.yaml", RECIPE)
    assert interactive.find_examples(tmp_path) == [examples / "recipe.yaml"]
    staged = interactive._staged(recipes)
    try:
        assert yaml.safe_load(staged.read_text()) == load_recipe(recipes)
    finally:
        staged.unlink()
    plain = _write(tmp_path / "plain.yaml", "schema: 2\nname: plain\n")
    assert interactive._staged(plain) == plain
