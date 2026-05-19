# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `srtctl install` flow.

Tests are network-free: model download / enroot import / `make setup` are
patched out. We only exercise the pieces with real logic (registry lookup,
srtslurm.yaml round-trip with comment preservation, bootstrap detection).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srtctl.install.container import container_filename
from srtctl.install.registry import REGISTRY, available_models, get_spec
from srtctl.install.registry import ModelInstallSpec as _MIS
from srtctl.install.setup import is_bootstrapped
from srtctl.install.slurm import build_sbatch_script
from srtctl.install.srtslurm_yaml_writer import register_aliases

# ----------------------------- registry -----------------------------


def test_registry_has_glm5():
    assert "glm5" in REGISTRY
    spec = get_spec("glm5")
    assert spec.hf_repo_id == "nvidia/GLM-5-NVFP4"
    assert spec.model_alias == "nvidia/GLM5-NVFP4"  # matches recipe `model.path`
    assert spec.container_image.startswith("nvcr.io/")


def test_registry_unknown_model_raises_with_available_list():
    with pytest.raises(KeyError, match="Unknown model 'nope'.*Available:.*glm5"):
        get_spec("nope")


def test_available_models_sorted():
    assert available_models() == sorted(REGISTRY)


# ----------------------------- container filename --------------------


def test_container_filename_matches_enroot_convention():
    name = container_filename("nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.3")
    # enroot replaces '/' and ':' with '+', then appends .sqsh
    assert name == "nvcr.io+nvidia+ai-dynamo+tensorrtllm-runtime+1.1.0-dev.3.sqsh"


# ----------------------------- bootstrap detection -------------------


def test_is_bootstrapped_requires_all_three(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    assert is_bootstrapped(tmp_path) is False

    nats = tmp_path / "configs" / "nats-server"
    etcd = tmp_path / "configs" / "etcd"
    nats.write_text("#!/bin/bash\n")
    etcd.write_text("#!/bin/bash\n")
    assert is_bootstrapped(tmp_path) is False  # missing srtslurm.yaml

    (tmp_path / "srtslurm.yaml").touch()
    assert is_bootstrapped(tmp_path) is False  # not executable

    nats.chmod(0o755)
    etcd.chmod(0o755)
    assert is_bootstrapped(tmp_path) is True


# ----------------------------- srtslurm.yaml writer ------------------


SRTSLURM_YAML_FIXTURE = """\
# Cluster config
default_account: "infra_rd_gsw"
default_partition: "gb300"

# Model paths (do not delete this comment)
model_paths:
  "existing/model": "/data/existing"

containers:
  "existing/container:v1": "/data/existing.sqsh"
"""


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "srtslurm.yaml"
    path.write_text(SRTSLURM_YAML_FIXTURE)
    return path


def test_register_aliases_adds_new_entries(tmp_path: Path):
    yml = _write_fixture(tmp_path)
    actions = register_aliases(
        yml,
        model_alias="nvidia/GLM5-NVFP4",
        model_path=Path("/install/models/GLM-5-NVFP4"),
        container_alias="nvcr.io/foo:1.0",
        container_path=Path("/install/containers/foo.sqsh"),
    )
    assert actions == {"model": "added", "container": "added"}

    content = yml.read_text()
    # New aliases present
    assert "nvidia/GLM5-NVFP4" in content
    assert "/install/models/GLM-5-NVFP4" in content
    assert "nvcr.io/foo:1.0" in content
    # Existing entries preserved
    assert '"existing/model"' in content
    assert "/data/existing" in content
    # User comments preserved
    assert "Cluster config" in content
    assert "do not delete this comment" in content


def test_register_aliases_overwrites_existing_with_warning(tmp_path: Path, caplog):
    yml = _write_fixture(tmp_path)
    actions = register_aliases(
        yml,
        model_alias="existing/model",
        model_path=Path("/new/path"),
        container_alias="existing/container:v1",
        container_path=Path("/new/sqsh.sqsh"),
    )
    assert actions == {"model": "updated", "container": "updated"}
    content = yml.read_text()
    assert "/new/path" in content
    assert "/data/existing" not in content


def test_register_aliases_unchanged_when_value_matches(tmp_path: Path):
    yml = _write_fixture(tmp_path)
    actions = register_aliases(
        yml,
        model_alias="existing/model",
        model_path=Path("/data/existing"),
        container_alias="existing/container:v1",
        container_path=Path("/data/existing.sqsh"),
    )
    assert actions == {"model": "unchanged", "container": "unchanged"}


def test_register_aliases_creates_sections_when_missing(tmp_path: Path):
    yml = tmp_path / "srtslurm.yaml"
    yml.write_text('default_account: "a"\ndefault_partition: "b"\n')
    actions = register_aliases(
        yml,
        model_alias="m",
        model_path=Path("/m"),
        container_alias="c",
        container_path=Path("/c.sqsh"),
    )
    assert actions == {"model": "added", "container": "added"}
    content = yml.read_text()
    assert "model_paths:" in content
    assert "containers:" in content


def test_register_aliases_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="srtslurm.yaml not found"):
        register_aliases(
            tmp_path / "nope.yaml",
            model_alias="m",
            model_path=Path("/m"),
            container_alias="c",
            container_path=Path("/c.sqsh"),
        )


# ----------------------------- model download guard ------------------


def test_download_model_errors_without_hf_token(tmp_path: Path, monkeypatch):
    from srtctl.install.model import download_model, model_storage_dirname

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN is not set"):
        download_model("nvidia/GLM-5-NVFP4", tmp_path / "model", hf_token=None)
    assert model_storage_dirname("nvidia/GLM-5-NVFP4") == "nvidia__GLM-5-NVFP4"
    assert model_storage_dirname("orgA/modelX") != model_storage_dirname("orgB/modelX")


# ----------------------------- container guard -----------------------


def test_import_container_skips_existing(tmp_path: Path):
    from srtctl.install.container import import_container

    existing = tmp_path / "already.sqsh"
    existing.write_text("placeholder")
    # Should not invoke enroot — we don't patch it; if it ran, it would fail in CI.
    result = import_container("nvcr.io/x:1", existing)
    assert result == existing


def test_import_container_errors_without_enroot(tmp_path: Path):
    from srtctl.install.container import import_container

    dest = tmp_path / "new.sqsh"
    with (
        patch("srtctl.install.container.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="`enroot` not found"),
    ):
        import_container("nvcr.io/x:1", dest)


def test_import_container_strict_mode_requires_nvcr_credentials(tmp_path: Path):
    from srtctl.install.container import import_container

    dest = tmp_path / "new.sqsh"
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    with (
        patch("srtctl.install.container.Path.home", return_value=fake_home),
        patch("srtctl.install.container.shutil.which", return_value="/usr/bin/enroot"),
        pytest.raises(RuntimeError, match="Missing registry credentials for nvcr.io"),
    ):
        import_container("nvcr.io/x:1", dest, strict_auth_preflight=True)


def test_import_container_warn_mode_allows_missing_nvcr_credentials(tmp_path: Path, caplog):
    from srtctl.install.container import import_container

    dest = tmp_path / "new.sqsh"
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    with (
        patch("srtctl.install.container.Path.home", return_value=fake_home),
        patch("srtctl.install.container.shutil.which", return_value="/usr/bin/enroot"),
        patch("srtctl.install.container.subprocess.run"),
    ):
        import_container("nvcr.io/x:1", dest, strict_auth_preflight=False)
    assert "Proceeding with enroot import attempt" in caplog.text


def test_import_container_allows_non_nvcr_without_credentials(tmp_path: Path):
    from srtctl.install.container import import_container

    dest = tmp_path / "new.sqsh"
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    with (
        patch("srtctl.install.container.Path.home", return_value=fake_home),
        patch("srtctl.install.container.shutil.which", return_value="/usr/bin/enroot"),
        patch("srtctl.install.container.subprocess.run"),
    ):
        out = import_container("docker.io/library/busybox:latest", dest)
    assert out == dest


# ----------------------------- slurm sbatch builder -------------------


def _make_spec(name: str = "glm5") -> _MIS:
    return _MIS(
        name=name,
        hf_repo_id="nvidia/GLM-5-NVFP4",
        model_alias="nvidia/GLM5-NVFP4",
        container_image="nvcr.io/test/img:1.0",
        default_recipe="recipes/test/x.yaml",
        description="test spec",
    )


def test_build_sbatch_script_uses_cluster_config():
    """Account/partition/time MUST come from srtslurm.yaml, never hardcoded."""
    cluster = {
        "default_account": "my_account",
        "default_partition": "gpu_partition",
        "default_time_limit": "06:00:00",
        "use_exclusive_sbatch_directive": True,
    }
    script = build_sbatch_script(
        spec=_make_spec(),
        srtctl_root=Path("/repo"),
        install_base=Path("/repo/install"),
        log_path=Path("/repo/install/install_glm5_%j.log"),
        venv_path=Path("/repo/.venv"),
        cluster_config=cluster,
    )
    assert "#SBATCH --account=my_account" in script
    assert "#SBATCH --partition=gpu_partition" in script
    assert "#SBATCH --time=06:00:00" in script
    assert "#SBATCH --exclusive" in script
    assert '#SBATCH --job-name="srtctl-install-glm5"' in script
    # Inline format: model + container + alias-register all live in this single script.
    assert "snapshot_download(" in script
    assert "repo_id='nvidia/GLM-5-NVFP4'" in script
    assert "/repo/install/models/nvidia__GLM-5-NVFP4" in script
    assert "enroot import --output" in script
    assert 'CONTAINER_IMAGE="nvcr.io/test/img:1.0"' in script
    assert 'docker://${CONTAINER_IMAGE}' in script
    assert "STRICT_AUTH_PREFLIGHT=0" in script
    assert "WARNING: Missing common nvcr.io credential files" in script
    # Alias-write step uses ruamel.yaml; alias keys must match the spec exactly.
    assert "model_paths['nvidia/GLM5-NVFP4']" in script
    assert "containers['nvcr.io/test/img:1.0']" in script
    # Alias updates should execute at script top-level (not nested under helper function).
    assert "\nmodel_paths = ensure(doc, 'model_paths')" in script
    assert "\ncontainers = ensure(doc, 'containers')" in script
    # No shell-level call back into srtctl from inside the sbatch (banner echo is fine).
    # The compute node only needs hf_hub + ruamel.yaml — not srtctl itself.
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("srtctl "):
            raise AssertionError(f"sbatch script invokes srtctl as a command: {line!r}")
    # Activates the user-supplied venv path.
    assert 'source "/repo/.venv/bin/activate"' in script


def test_build_sbatch_script_omits_exclusive_when_false():
    cluster = {
        "default_account": "a",
        "default_partition": "p",
        "default_time_limit": "01:00:00",
        "use_exclusive_sbatch_directive": False,
    }
    script = build_sbatch_script(
        spec=_make_spec("m"),
        srtctl_root=Path("/r"),
        install_base=Path("/r/install"),
        log_path=Path("/r/install/m_%j.log"),
        venv_path=Path("/r/.venv"),
        cluster_config=cluster,
    )
    assert "#SBATCH --exclusive" not in script


def test_build_sbatch_script_raises_on_missing_account():
    cluster = {"default_partition": "p", "default_time_limit": "01:00:00"}
    with pytest.raises(RuntimeError, match="missing 'default_account'"):
        build_sbatch_script(
            spec=_make_spec("m"),
            srtctl_root=Path("/r"),
            install_base=Path("/r/install"),
            log_path=Path("/r/install/m_%j.log"),
            venv_path=Path("/r/.venv"),
            cluster_config=cluster,
        )


def test_build_sbatch_script_raises_when_no_srtslurm_yaml():
    with pytest.raises(RuntimeError, match="srtslurm.yaml not found"):
        build_sbatch_script(
            spec=_make_spec("m"),
            srtctl_root=Path("/r"),
            install_base=Path("/r/install"),
            log_path=Path("/r/install/m_%j.log"),
            venv_path=Path("/r/.venv"),
            cluster_config=None,
        )


def test_build_sbatch_script_uses_custom_venv_path():
    """User can override the venv path via --venv. The path appears verbatim."""
    cluster = {"default_account": "a", "default_partition": "p", "default_time_limit": "01:00:00"}
    custom = Path("/my/custom/.venv-aarch64")
    script = build_sbatch_script(
        spec=_make_spec("m"),
        srtctl_root=Path("/r"),
        install_base=Path("/r/install"),
        log_path=Path("/r/install/m_%j.log"),
        venv_path=custom,
        cluster_config=cluster,
    )
    assert f'source "{custom}/bin/activate"' in script


def test_build_sbatch_script_strict_auth_preflight_enabled():
    cluster = {"default_account": "a", "default_partition": "p", "default_time_limit": "01:00:00"}
    script = build_sbatch_script(
        spec=_make_spec("m"),
        srtctl_root=Path("/r"),
        install_base=Path("/r/install"),
        log_path=Path("/r/install/m_%j.log"),
        venv_path=Path("/r/.venv"),
        cluster_config=cluster,
        strict_auth_preflight=True,
    )
    assert "STRICT_AUTH_PREFLIGHT=1" in script


# ----------------------------- install cli pathing --------------------


def test_install_loads_repo_local_srtslurm_yaml_even_outside_cwd(tmp_path: Path, monkeypatch):
    """`srtctl install --slurm` should not depend on current working directory."""
    from srtctl.cli.install import _load_cluster_config_for_install

    root = tmp_path / "repo"
    root.mkdir()
    (root / "srtslurm.yaml").write_text(
        "\n".join(
            [
                'default_account: "acct"',
                'default_partition: "gpu"',
                'default_time_limit: "01:00:00"',
            ]
        )
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.delenv("SRTSLURM_CONFIG", raising=False)

    loaded = _load_cluster_config_for_install(root)
    assert loaded is not None
    assert loaded["default_account"] == "acct"
    assert loaded["default_partition"] == "gpu"
    assert loaded["default_time_limit"] == "01:00:00"
    # Helper should not leak env overrides back into the caller shell.
    assert "SRTSLURM_CONFIG" not in os.environ


def test_resolve_spec_requires_explicit_fields():
    from srtctl.cli.install import _resolve_spec

    args = SimpleNamespace(
        model="glm5",
        hf_repo_id=None,
        model_alias=None,
        container_image=None,
        strict_auth_preflight=False,
    )
    with pytest.raises(ValueError, match="are required"):
        _resolve_spec(args)


def test_resolve_spec_supports_generic_model():
    from srtctl.cli.install import _resolve_spec

    args = SimpleNamespace(
        model="my-model",
        hf_repo_id="org/repo",
        model_alias="org/repo-alias",
        container_image="docker.io/org/image:1.0",
        strict_auth_preflight=False,
    )
    spec = _resolve_spec(args)
    assert spec.name == "my-model"
    assert spec.hf_repo_id == "org/repo"
    assert spec.model_alias == "org/repo-alias"
    assert spec.container_image == "docker.io/org/image:1.0"


def test_resolve_spec_rejects_unsafe_values():
    from srtctl.cli.install import _resolve_spec

    args = SimpleNamespace(
        model="my-model",
        hf_repo_id="org/repo",
        model_alias="bad\"alias",
        container_image="docker.io/org/image:1.0",
        strict_auth_preflight=False,
    )
    with pytest.raises(ValueError, match="Invalid model alias"):
        _resolve_spec(args)
