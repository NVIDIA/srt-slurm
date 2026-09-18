# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared ``source:`` shape: validation, ref pinning, and the Dynamo mapping."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.core.schema import DynamoConfig, SrtConfig
from srtctl.core.source import DynamoSourceConfig, SourceConfig, is_commit_sha, pin_source_revs, resolve_rev
from srtctl.core.yaml_utils import dump_yaml_with_comments, load_yaml_text_with_comments

SHA = "2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a"
UPSTREAM = "https://github.com/ai-dynamo/dynamo.git"


# --- SourceConfig -------------------------------------------------------------------


def test_is_commit_sha() -> None:
    assert is_commit_sha(SHA)
    assert is_commit_sha("abc1234")
    assert is_commit_sha("abc123")  # legacy recipes use short hashes
    assert not is_commit_sha("v1.4.2")
    assert not is_commit_sha("refs/pull/14000/head")
    assert not is_commit_sha("main")


def test_source_config_rules() -> None:
    src = SourceConfig(git="https://example.com/r", rev="refs/pull/1/head")
    assert src.checkout == "refs/pull/1/head"
    assert SourceConfig(git="https://example.com/r", rev="v1", sha=SHA).checkout == SHA
    for rev in ("main", "master", "HEAD", " "):
        with pytest.raises(ValidationError, match="rev must be"):
            SourceConfig(git="https://example.com/r", rev=rev)
    with pytest.raises(ValidationError, match="sha must be a commit SHA"):
        SourceConfig(git="https://example.com/r", rev="v1", sha="not-a-sha")
    with pytest.raises(ValidationError, match="git must be"):
        SourceConfig(git="", rev="v1")


# --- resolve_rev ---------------------------------------------------------------------


def _ls_remote(stdout: str, returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = "" if returncode == 0 else "fatal: repository not found"
    return result


def test_resolve_rev_returns_sha_unchanged_without_network() -> None:
    with patch("srtctl.core.source.subprocess.run") as run:
        assert resolve_rev(UPSTREAM, SHA) == SHA
    run.assert_not_called()


def test_resolve_rev_prefers_peeled_tag_and_handles_pr_refs() -> None:
    tag_obj, commit = "1" * 40, "2" * 40
    with patch(
        "srtctl.core.source.subprocess.run",
        return_value=_ls_remote(f"{tag_obj}\trefs/tags/v1.4.2\n{commit}\trefs/tags/v1.4.2^{{}}\n"),
    ) as run:
        assert resolve_rev(UPSTREAM, "v1.4.2") == commit
    cmd = run.call_args.args[0]
    assert cmd[:4] == ["git", "-c", "http.version=HTTP/1.1", "ls-remote"]
    assert "refs/tags/v1.4.2" in cmd
    assert run.call_args.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"

    with patch("srtctl.core.source.subprocess.run", return_value=_ls_remote(f"{commit}\trefs/pull/14000/head\n")):
        assert resolve_rev(UPSTREAM, "refs/pull/14000/head") == commit


def test_resolve_rev_failures_raise() -> None:
    with (
        patch("srtctl.core.source.subprocess.run", return_value=_ls_remote("", returncode=128)),
        pytest.raises(RuntimeError, match="repository not found"),
    ):
        resolve_rev(UPSTREAM, "v1")
    with (
        patch("srtctl.core.source.subprocess.run", return_value=_ls_remote("")),
        pytest.raises(RuntimeError, match="not found in"),
    ):
        resolve_rev(UPSTREAM, "refs/pull/99/head")
    with (
        patch("srtctl.core.source.subprocess.run", side_effect=subprocess.TimeoutExpired("git", 60)),
        pytest.raises(RuntimeError, match="failed"),
    ):
        resolve_rev(UPSTREAM, "v1")


# --- pin_source_revs ------------------------------------------------------------------

RECIPE = """\
name: pin-test
dynamo:
  source:
    rev: refs/pull/14000/head   # the PR under test
services:
  - name: router
    command: [python3, -m, router]
    source:
      git: https://example.com/r
      rev: v2.0.0
  - name: pinned
    command: [/bin/true]
    source:
      git: https://example.com/r
      rev: v1
      sha: 2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a
"""


def test_pin_source_revs_pins_every_unpinned_source_and_keeps_comments() -> None:
    document = load_yaml_text_with_comments(RECIPE)
    document["dynamo"]["source"]["git"] = UPSTREAM  # what DynamoSourceConfig defaults to
    seen: list[tuple[str, str]] = []

    def fake_resolve(git: str, rev: str) -> str:
        seen.append((git, rev))
        return SHA

    notes = pin_source_revs(document, resolve=fake_resolve)

    assert seen == [(UPSTREAM, "refs/pull/14000/head"), ("https://example.com/r", "v2.0.0")]
    assert notes == [
        f"dynamo.source: refs/pull/14000/head -> {SHA}",
        f"services.0.source: v2.0.0 -> {SHA}",
    ]
    assert document["dynamo"]["source"]["sha"] == SHA
    assert document["dynamo"]["source"]["rev"] == "refs/pull/14000/head"
    text = dump_yaml_with_comments(document)
    assert "# the PR under test" in text  # ruamel round-trip keeps the recipe's comments


def test_pin_source_revs_walks_override_format_and_leaves_failures_unpinned(caplog) -> None:
    document = load_yaml_text_with_comments(
        "base:\n  dynamo:\n    source:\n      git: https://example.com/d\n      rev: refs/pull/1/head\n"
        "override_x:\n  dynamo:\n    source:\n      git: https://example.com/d\n      rev: refs/pull/2/head\n"
    )

    def flaky(git: str, rev: str) -> str:
        if rev.endswith("2/head"):
            raise RuntimeError("boom")
        return SHA

    notes = pin_source_revs(document, resolve=flaky)
    assert notes == [f"base.dynamo.source: refs/pull/1/head -> {SHA}"]
    assert document["override_x"]["dynamo"]["source"].get("sha") is None
    assert "could not pin rev 'refs/pull/2/head'" in caplog.text


# --- dynamo.source -----------------------------------------------------------------------


def test_dynamo_source_git_maps_to_hash_and_defaults_upstream_repo() -> None:
    config = DynamoConfig(source=DynamoSourceConfig(rev="refs/pull/14000/head"))
    assert config.source is not None and config.source.git == UPSTREAM
    assert config.hash == "refs/pull/14000/head"
    assert config.version is None
    assert config.needs_source_install
    cmd = config.get_install_commands()
    # An unpinned ref is fetched by name; a plain clone has no PR refs.
    assert "git fetch origin refs/pull/14000/head && git checkout FETCH_HEAD" in cmd
    assert "/configs/dynamo-wheels/refs-pull-14000-head" in cmd
    assert f"clone {UPSTREAM} dynamo" in cmd


def test_dynamo_source_pinned_sha_and_fork_and_patches() -> None:
    patch_line = 'dynamo-tokenizers = { git = "https://github.com/ai-dynamo/frontend-crates", branch = "feat" }'
    config = DynamoConfig(
        source=DynamoSourceConfig(
            git="https://github.com/me/dynamo-fork", rev="refs/pull/3/head", sha=SHA, patches=[patch_line]
        )
    )
    assert config.hash == SHA
    assert config.cargo_patches == [patch_line]
    cmd = config.get_install_commands()
    assert "clone https://github.com/me/dynamo-fork dynamo" in cmd
    assert f"git checkout {SHA}" in cmd
    assert f"/configs/dynamo-wheels/{SHA}-patch-" in cmd
    assert patch_line in cmd


def test_dynamo_source_pypi_and_wheel() -> None:
    assert DynamoConfig(source=DynamoSourceConfig(pypi="1.4.2")).version == "1.4.2"
    assert "ai-dynamo==1.4.2" in DynamoConfig(source=DynamoSourceConfig(pypi="1.4.2")).get_install_commands()
    wheel = DynamoConfig(source=DynamoSourceConfig(wheel="1.5.0.dev20260901"))
    assert wheel.wheel == "1.5.0.dev20260901"
    assert wheel.version is None
    assert not wheel.needs_source_install


def test_dynamo_source_validation() -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        DynamoSourceConfig()
    with pytest.raises(ValidationError, match="exactly one of"):
        DynamoSourceConfig(pypi="1.0", wheel="1.0")
    with pytest.raises(ValidationError, match="requires rev"):
        DynamoSourceConfig(git="https://example.com/d")
    with pytest.raises(ValidationError, match="immutable ref"):
        DynamoSourceConfig(rev="main")
    with pytest.raises(ValidationError, match="only apply to a git source"):
        DynamoSourceConfig(pypi="1.0", patches=["x = 1"])
    with pytest.raises(ValueError, match="cannot be combined with dynamo.hash"):
        DynamoConfig(hash="abc1234", source=DynamoSourceConfig(pypi="1.0"))
    with pytest.raises(ValueError, match="cannot be combined with dynamo.top_of_tree, dynamo.cargo_patches"):
        DynamoConfig(top_of_tree=True, cargo_patches=["x = 1"], source=DynamoSourceConfig(rev="v1"))


def test_dynamo_source_loads_from_recipe_yaml() -> None:
    raw = yaml.safe_load(
        """
name: t
model:
  path: /m
  container: /c.sqsh
  precision: bf16
resources:
  gpu_type: h100
  gpus_per_node: 8
  agg_nodes: 1
  agg_workers: 2
  gpus_per_agg: 1
frontend:
  type: dynamo
dynamo:
  source:
    git: https://github.com/ai-dynamo/dynamo
    rev: v1.4.2
    sha: 2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a
backend:
  type: sglang
benchmark:
  type: manual
"""
    )
    config = SrtConfig.Schema().load(raw)
    assert config.dynamo.hash == SHA
    assert config.dynamo.version is None
    assert config.dynamo.source is not None and config.dynamo.source.rev == "v1.4.2"


# --- submit-time pinning through materialize_config_path -----------------------------------


def test_materialize_pins_sources_only_on_apply(tmp_path) -> None:
    from srtctl.cli.submit import materialize_config_path

    recipe = tmp_path / "r.yaml"
    recipe.write_text("name: t\ndynamo:\n  source:\n    git: https://example.com/d\n    rev: refs/pull/7/head\n")

    with patch("srtctl.core.source.resolve_rev", return_value=SHA) as resolve:
        with materialize_config_path(recipe, pin_sources=False) as path:
            assert path == recipe  # dry-run / preflight never touch the network
        resolve.assert_not_called()

        with materialize_config_path(recipe, pin_sources=True) as path:
            assert path != recipe
            staged = yaml.safe_load(path.read_text())
        resolve.assert_called_once_with("https://example.com/d", "refs/pull/7/head")

    assert staged["dynamo"]["source"]["sha"] == SHA
    assert staged["dynamo"]["source"]["rev"] == "refs/pull/7/head"
    assert recipe.read_text().count("sha") == 0  # the source file is never modified


def test_materialize_without_sources_is_a_passthrough(tmp_path) -> None:
    from srtctl.cli.submit import materialize_config_path

    recipe = tmp_path / "r.yaml"
    recipe.write_text("name: t\ndynamo:\n  version: '1.4.2'\n")
    with patch("srtctl.core.source.resolve_rev") as resolve, materialize_config_path(recipe, pin_sources=True) as path:
        assert path == recipe
    resolve.assert_not_called()
