# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import yaml

from srtctl.core.assets import ensure_assets_for_config_variants
from srtctl.core.config import load_cluster_config, load_config


def _write_cluster_config(tmp_path: Path, *, mode: str = "login", login_probe_command: str | None = None) -> Path:
    cluster_config = {
        "default_account": "acct",
        "default_partition": "gpu",
        "asset_materialization": {
            "mode": mode,
            "models_root": str(tmp_path / "models"),
            "containers_root": str(tmp_path / "containers"),
            "lock_path": str(tmp_path / "asset.lock"),
            "model_pull_template": "mkdir -p {q_target}",
            "container_pull_template": (
                f"{sys.executable} -c "
                '"from pathlib import Path; p=Path({py_target}); '
                "p.parent.mkdir(parents=True, exist_ok=True); p.write_text('container')\""
            ),
        },
    }
    if login_probe_command is not None:
        cluster_config["asset_materialization"]["login_probe_command"] = login_probe_command
    path = tmp_path / "srtslurm.yaml"
    path.write_text(yaml.safe_dump(cluster_config, sort_keys=False))
    return path


def _recipe() -> dict:
    return {
        "name": "asset-test",
        "model": {
            "path": "qwen32b",
            "container": "sglang-dev",
            "precision": "fp8",
        },
        "identity": {
            "model": {"repo": "Qwen/Qwen3-32B", "revision": "abc123"},
            "container": {"image": "nvcr.io/example/sglang:latest"},
        },
        "resources": {
            "gpu_type": "h100",
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 1,
        },
    }


def test_ensure_assets_pulls_and_registers_missing_aliases(tmp_path, monkeypatch):
    cluster_path = _write_cluster_config(tmp_path)
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster_path))

    result = ensure_assets_for_config_variants(_recipe())

    assert result.ok is True
    assert result.changed is True
    assert {action.status for action in result.actions} == {"pulled"}
    assert Path(result.model_paths["qwen32b"]).is_dir()
    assert Path(result.containers["sglang-dev"]).is_file()

    updated = yaml.safe_load(cluster_path.read_text())
    assert updated["model_paths"]["qwen32b"] == str(tmp_path / "models" / "qwen32b")
    assert updated["containers"]["sglang-dev"] == str(tmp_path / "containers" / "sglang-dev.sqsh")

    loaded = load_cluster_config()
    assert loaded is not None
    assert loaded["asset_materialization"]["mode"] == "login"


def test_ensure_assets_preserves_recipe_identity_after_alias_resolution(tmp_path, monkeypatch):
    cluster_path = _write_cluster_config(tmp_path)
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(_recipe(), sort_keys=False))
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster_path))

    ensure_assets_for_config_variants(_recipe())
    resolved = load_config(recipe_path)

    assert resolved.model.path == str(tmp_path / "models" / "qwen32b")
    assert resolved.model.container == str(tmp_path / "containers" / "sglang-dev.sqsh")
    assert resolved.identity.model.repo == "Qwen/Qwen3-32B"
    assert resolved.identity.model.revision == "abc123"
    assert resolved.identity.container.image == "nvcr.io/example/sglang:latest"


def test_ensure_assets_dry_run_does_not_mutate(tmp_path, monkeypatch):
    cluster_path = _write_cluster_config(tmp_path)
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster_path))

    result = ensure_assets_for_config_variants(_recipe(), dry_run=True)

    assert result.ok is True
    assert result.changed is False
    assert {action.status for action in result.actions} == {"would-pull"}
    assert not (tmp_path / "models").exists()
    assert "model_paths" not in yaml.safe_load(cluster_path.read_text())


def test_auto_mode_uses_srun_when_login_probe_fails(tmp_path, monkeypatch):
    cluster_path = _write_cluster_config(tmp_path, mode="auto", login_probe_command="false")
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster_path))

    result = ensure_assets_for_config_variants(_recipe(), dry_run=True)

    assert result.ok is True
    assert {action.mode for action in result.actions} == {"srun"}
    assert all(action.command and action.command.startswith("srun ") for action in result.actions)


def test_missing_target_root_is_not_ok(tmp_path, monkeypatch):
    cluster_path = tmp_path / "srtslurm.yaml"
    cluster_path.write_text(
        yaml.safe_dump(
            {
                "asset_materialization": {
                    "mode": "login",
                    "containers_root": str(tmp_path / "containers"),
                    "model_pull_template": "mkdir -p {q_target}",
                    "container_pull_template": "touch {q_target}",
                }
            },
            sort_keys=False,
        )
    )
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster_path))

    result = ensure_assets_for_config_variants(_recipe(), dry_run=True)

    assert result.ok is False
    assert any(action.status == "missing-target-root" for action in result.actions)


def test_ensure_assets_cli_json(tmp_path, monkeypatch):
    cluster_path = _write_cluster_config(tmp_path)
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(_recipe(), sort_keys=False))
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster_path))

    result = subprocess.run(
        [sys.executable, "-m", "srtctl.cli.submit", "ensure-assets", "-f", str(recipe_path), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["changed"] is True
    assert {action["status"] for action in payload["actions"]} == {"pulled"}
