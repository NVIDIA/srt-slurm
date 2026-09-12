# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `srtctl bake-image`."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.benchmarks.base import SCRIPTS_DIR
from srtctl.core.image_bake import (
    BAKE_PATCH_DIR,
    BakePlan,
    bake_image,
    build_srun_options,
    build_vllm_patches,
    default_configs_dir,
    default_output_image,
    infer_vllm_patch_args,
    read_sa_bench_deps,
    resolve_setup_script,
    resolve_vllm_patch,
)


def _plan(tmp_path: Path, **overrides) -> BakePlan:
    source = tmp_path / "base.sqsh"
    source.write_bytes(b"squashfs")
    kwargs = {
        "source_image": source,
        "output_image": tmp_path / "baked.sqsh",
        "dynamo_version": "1.2.0.dev20260526",
    }
    kwargs.update(overrides)
    return BakePlan(**kwargs)


def _script_tree(tmp_path: Path, *, name: str = "overlay.sh", under_patches: bool = False) -> tuple[Path, Path]:
    configs = tmp_path / "configs"
    (configs / "patches").mkdir(parents=True, exist_ok=True)
    host = (configs / "patches" / name) if under_patches else (configs / name)
    host.write_text("#!/bin/bash\necho overlay\n")
    return configs, host


def _script_plan(tmp_path: Path, **overrides) -> BakePlan:
    configs, script = _script_tree(tmp_path)
    kwargs = {
        "source_image": tmp_path / "base.sqsh",
        "output_image": tmp_path / "baked.sqsh",
        "dynamo_version": None,
        "setup_script": script,
        "configs_dir": configs,
    }
    kwargs.update(overrides)
    source = kwargs["source_image"]
    if not source.exists():
        source.write_bytes(b"squashfs")
    return BakePlan(**kwargs)


_GIT_VLLM_DIFF = """\
--- a/vllm/foo.py
+++ b/vllm/foo.py
@@ -1 +1 @@
-old
+new
"""


def _vllm_diff(tmp_path: Path, *, name: str = "mine.diff", body: str = _GIT_VLLM_DIFF) -> Path:
    configs = tmp_path / "configs"
    (configs / "patches").mkdir(parents=True, exist_ok=True)
    path = configs / "patches" / name
    path.write_text(body)
    return path


def _patch_plan(tmp_path: Path, **overrides) -> BakePlan:
    diff = overrides.pop("diff_path", None) or _vllm_diff(tmp_path)
    kwargs = {
        "source_image": tmp_path / "base.sqsh",
        "output_image": tmp_path / "baked.sqsh",
        "dynamo_version": None,
        "vllm_patches": build_vllm_patches([diff]),
    }
    kwargs.update(overrides)
    source = kwargs["source_image"]
    if not source.exists():
        source.write_bytes(b"squashfs")
    return BakePlan(**kwargs)


class TestSaBenchDeps:
    def test_deps_are_read_from_the_shared_script(self):
        packages, imports = read_sa_bench_deps()

        assert "aiohttp" in packages
        assert "Pillow" in packages
        # Pillow imports under a different name, which is why deps.sh carries both.
        assert "PIL" in imports
        assert "Pillow" not in imports

    def test_bench_sh_sources_the_same_list(self):
        """A baked image is only useful if bench.sh checks for exactly these packages."""
        bench_sh = (SCRIPTS_DIR / "sa-bench" / "bench.sh").read_text()

        assert "deps.sh" in bench_sh
        assert "import ${SA_BENCH_IMPORTS}" in bench_sh
        # The list must live only in deps.sh, otherwise the two can drift apart.
        assert not re.search(r"^SA_BENCH_DEPS=", bench_sh, re.MULTILINE)


class TestOutputNaming:
    def test_name_records_what_was_installed(self, tmp_path):
        out = default_output_image(tmp_path / "for-model-nvfp4.sqsh", dynamo_version="1.2.0", sa_bench=True)

        assert out.name == "for-model-nvfp4+dynamo1.2.0+sa-bench.sqsh"
        assert out.parent == tmp_path

    def test_single_tag(self, tmp_path):
        out = default_output_image(tmp_path / "base.sqsh", dynamo_version=None, sa_bench=True)

        assert out.name == "base+sa-bench.sqsh"

    def test_setup_script_tag_uses_stem(self, tmp_path):
        out = default_output_image(
            tmp_path / "base.sqsh",
            dynamo_version=None,
            sa_bench=False,
            setup_script="vllm-pr51924-one-sided.sh",
        )

        assert out.name == "base+vllm-pr51924-one-sided.sqsh"

    def test_patch_tag_uses_stem(self, tmp_path):
        out = default_output_image(
            tmp_path / "base.sqsh",
            dynamo_version=None,
            sa_bench=False,
            vllm_patches=[tmp_path / "mine.diff"],
        )

        assert out.name == "base+mine.sqsh"


class TestPlanValidation:
    def test_rejects_empty_plan(self, tmp_path):
        with pytest.raises(ValueError, match="nothing to install"):
            BakePlan(source_image=tmp_path / "a.sqsh", output_image=tmp_path / "b.sqsh")

    def test_rejects_overwriting_the_source(self, tmp_path):
        same = tmp_path / "a.sqsh"
        with pytest.raises(ValueError, match="must differ"):
            BakePlan(source_image=same, output_image=same, dynamo_version="1.0.0")

    def test_rejects_sa_bench_without_packages(self, tmp_path):
        with pytest.raises(ValueError, match="without a package list"):
            BakePlan(source_image=tmp_path / "a.sqsh", output_image=tmp_path / "b.sqsh", sa_bench=True)

    def test_script_only_plan_is_valid(self, tmp_path):
        plan = _script_plan(tmp_path)

        assert plan.dynamo_version is None
        assert plan.setup_script.name == "overlay.sh"

    def test_patch_only_plan_is_valid(self, tmp_path):
        plan = _patch_plan(tmp_path)

        assert plan.setup_script is None
        assert plan.vllm_patches[0].host_path.name == "mine.diff"
        assert plan.vllm_patches[0].strip == 1
        assert plan.vllm_patches[0].root == "site-packages"

    def test_rejects_script_without_configs_dir(self, tmp_path):
        script = tmp_path / "overlay.sh"
        script.write_text("echo hi\n")
        with pytest.raises(ValueError, match="configs directory"):
            BakePlan(
                source_image=tmp_path / "a.sqsh",
                output_image=tmp_path / "b.sqsh",
                setup_script=script,
            )


class TestInstallScript:
    def test_dynamo_install_is_not_serialized(self, tmp_path):
        """The flock sentinel would be captured in the image and make later jobs skip installing."""
        script = _plan(tmp_path).install_script()

        assert "pip install" in script
        assert "ai-dynamo==1.2.0.dev20260526" in script
        assert "flock" not in script

    def test_script_verifies_and_cleans_up(self, tmp_path):
        script = _plan(tmp_path).install_script()

        assert script.startswith("set -euo pipefail")
        assert "pip show ai-dynamo" in script
        assert "pip cache purge" in script

    def test_sa_bench_installs_and_imports_every_package(self, tmp_path):
        packages, imports = read_sa_bench_deps()
        script = _plan(
            tmp_path, dynamo_version=None, sa_bench=True, sa_bench_deps=packages, sa_bench_imports=imports
        ).install_script()

        for package in packages:
            assert package in script
        assert f'python3 -c "import {imports}"' in script

    def test_setup_script_runs_from_configs_mount(self, tmp_path):
        script = _script_plan(tmp_path).install_script()

        assert "bash /configs/overlay.sh" in script
        assert "test -f /configs/overlay.sh" in script
        # Missing script must fail the bake, not warn-and-continue like job startup.
        assert "WARNING" not in script

    def test_setup_script_runs_before_pip_installs(self, tmp_path):
        configs, overlay = _script_tree(tmp_path)
        script = _plan(
            tmp_path,
            setup_script=overlay,
            configs_dir=configs,
        ).install_script()

        assert script.index("bash /configs/overlay.sh") < script.index("pip install")


class TestVllmPatch:
    def test_git_diff_against_vllm_repo_is_p1_site_packages(self):
        strip, root = infer_vllm_patch_args(_GIT_VLLM_DIFF)

        assert strip == 1
        assert root == "site-packages"

    def test_repo_pr51924_diff_is_p1_site_packages(self):
        diff = default_configs_dir() / "patches" / "vllm-pr51924.diff"
        if not diff.is_file():
            pytest.skip(f"{diff.name} is not in this checkout")

        strip, root = infer_vllm_patch_args(diff.read_text())

        assert strip == 1
        assert root == "site-packages"

    def test_paths_inside_the_package_use_vllm_root(self):
        strip, root = infer_vllm_patch_args(
            "--- a/distributed/device_communicators/all2all.py\n+++ b/distributed/device_communicators/all2all.py\n"
        )

        assert strip == 1
        assert root == "vllm"

    def test_install_script_dry_runs_before_apply(self, tmp_path):
        script = _patch_plan(tmp_path).install_script()

        assert "import vllm" in script
        assert "--dry-run" in script
        assert "refusing to bake" in script
        apply_at = script.rfind("patch --batch --forward -p")
        dry_at = script.find("patch --batch --forward --dry-run")
        assert dry_at != -1
        assert apply_at != -1
        assert dry_at < apply_at
        assert '_bake_apply_patch /bake-patches/mine.diff 1 "${SITE_PACKAGES}"' in script

    def test_setup_script_runs_before_patch(self, tmp_path):
        configs, overlay = _script_tree(tmp_path)
        diff = _vllm_diff(tmp_path)
        script = _plan(
            tmp_path,
            dynamo_version=None,
            setup_script=overlay,
            configs_dir=configs,
            vllm_patches=build_vllm_patches([diff]),
        ).install_script()

        assert script.index("bash /configs/overlay.sh") < script.index("_bake_apply_patch")

    def test_patch_runs_before_pip_installs(self, tmp_path):
        diff = _vllm_diff(tmp_path)
        script = _plan(
            tmp_path,
            vllm_patches=build_vllm_patches([diff]),
        ).install_script()

        assert script.index("_bake_apply_patch") < script.index("pip install")

    def test_mounts_the_diff_and_records_manifest(self, tmp_path):
        plan = _patch_plan(tmp_path)
        proc = MagicMock()
        proc.wait.return_value = 0

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value="12345"),
            patch("srtctl.core.image_bake.start_srun_process", return_value=proc) as srun,
        ):
            assert bake_image(plan) == 0

        patch_file = plan.vllm_patches[0].host_path
        assert srun.call_args.kwargs["container_mounts"][patch_file] == BAKE_PATCH_DIR / "mine.diff"
        command = srun.call_args.kwargs["command"]
        assert "refusing to bake" in command[2]
        manifest = json.loads(plan.manifest_path.read_text())
        assert manifest["installed"]["vllm_patches"] == ["mine.diff"]

    def test_conflict_does_not_keep_a_new_image(self, tmp_path):
        plan = _patch_plan(tmp_path)
        proc = MagicMock()
        proc.wait.return_value = 1

        def write_then_fail(**_kwargs):
            plan.output_image.write_bytes(b"partial-sqsh")
            return proc

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value="12345"),
            patch("srtctl.core.image_bake.start_srun_process", side_effect=write_then_fail),
        ):
            assert bake_image(plan) == 1

        assert not plan.output_image.exists()
        assert not plan.manifest_path.exists()

    def test_failed_bake_does_not_delete_a_preexisting_image(self, tmp_path):
        plan = _patch_plan(tmp_path)
        plan.output_image.write_bytes(b"previous")
        proc = MagicMock()
        proc.wait.return_value = 1

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value="12345"),
            patch("srtctl.core.image_bake.start_srun_process", return_value=proc),
        ):
            assert bake_image(plan) == 1

        assert plan.output_image.read_bytes() == b"previous"


class TestResolveVllmPatch:
    def test_name_under_patches(self, tmp_path):
        path = _vllm_diff(tmp_path)
        configs = tmp_path / "configs"

        assert resolve_vllm_patch("mine.diff", configs_dir=configs) == path.resolve()

    def test_explicit_path(self, tmp_path):
        path = _vllm_diff(tmp_path)

        assert resolve_vllm_patch(str(path), configs_dir=tmp_path / "configs") == path.resolve()

    def test_missing_name_raises(self, tmp_path):
        configs = tmp_path / "configs"
        configs.mkdir()

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_vllm_patch("missing.diff", configs_dir=configs)

    def test_empty_patch_is_rejected(self, tmp_path):
        empty = tmp_path / "empty.diff"
        empty.write_text("\n")

        with pytest.raises(ValueError, match="empty"):
            build_vllm_patches([empty])


class TestBakeImage:
    def test_dry_run_prints_without_launching_srun(self, tmp_path, capsys):
        with patch("srtctl.core.image_bake.start_srun_process") as srun:
            assert bake_image(_plan(tmp_path), dry_run=True) == 0

        srun.assert_not_called()
        out = capsys.readouterr().out
        assert "--container-save=" in out
        assert "--container-writable" in out
        assert "ai-dynamo==1.2.0.dev20260526" in out

    def test_allocates_a_node_when_run_outside_a_job(self, tmp_path):
        """Without an allocation srun requests one, using the cluster defaults."""
        plan = _plan(tmp_path, time_limit="0:20:00")
        proc = MagicMock()
        proc.wait.return_value = 0

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value=None),
            patch("srtctl.core.image_bake.get_srtslurm_setting", side_effect=["acct", "part"]),
            patch("srtctl.core.image_bake.start_srun_process", return_value=proc) as srun,
        ):
            assert bake_image(plan) == 0

        options = srun.call_args.kwargs["srun_options"]
        assert options["time"] == "0:20:00"
        assert options["account"] == "acct"
        assert options["partition"] == "part"
        # --overlap only makes sense for a step inside an existing allocation.
        assert srun.call_args.kwargs["overlap"] is False

    def test_joins_an_existing_allocation_without_slurm_flags(self, tmp_path):
        plan = _plan(tmp_path)
        proc = MagicMock()
        proc.wait.return_value = 0

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value="12345"),
            patch("srtctl.core.image_bake.start_srun_process", return_value=proc) as srun,
        ):
            assert bake_image(plan) == 0

        options = srun.call_args.kwargs["srun_options"]
        assert "time" not in options
        assert "account" not in options
        assert srun.call_args.kwargs["overlap"] is True

    def test_cli_override_beats_cluster_default(self, tmp_path):
        plan = _plan(tmp_path, slurm_overrides={"account": "mine", "partition": None})

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value=None),
            patch("srtctl.core.image_bake.get_srtslurm_setting", return_value="from-yaml"),
        ):
            options = build_srun_options(plan)

        assert options["account"] == "mine"
        assert options["partition"] == "from-yaml"

    def test_saves_image_and_writes_manifest(self, tmp_path):
        plan = _plan(tmp_path)
        proc = MagicMock()
        proc.wait.return_value = 0

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value="12345"),
            patch("srtctl.core.image_bake.start_srun_process", return_value=proc) as srun,
        ):
            assert bake_image(plan) == 0

        options = srun.call_args.kwargs["srun_options"]
        assert options["container-save"] == str(plan.output_image)
        assert options["container-writable"] == ""
        assert srun.call_args.kwargs["container_image"] == str(plan.source_image)
        # enroot needs to remap us to root before pip can write into /usr/local.
        assert srun.call_args.kwargs["srun_export_env"] == {"ENROOT_REMAP_ROOT": "yes"}

        manifest = json.loads(plan.manifest_path.read_text())
        assert manifest["installed"]["ai-dynamo"] == "1.2.0.dev20260526"
        assert manifest["source_image"] == str(plan.source_image)
        assert re.match(r"\d{4}-\d{2}-\d{2}T", manifest["created_utc"])

    def test_failed_install_leaves_no_manifest(self, tmp_path):
        plan = _plan(tmp_path)
        proc = MagicMock()
        proc.wait.return_value = 1

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value="12345"),
            patch("srtctl.core.image_bake.start_srun_process", return_value=proc),
        ):
            assert bake_image(plan) == 1

        assert not plan.manifest_path.exists()

    def test_setup_script_mounts_configs_and_records_manifest(self, tmp_path):
        plan = _script_plan(tmp_path)
        proc = MagicMock()
        proc.wait.return_value = 0

        with (
            patch("srtctl.core.image_bake.get_slurm_job_id", return_value="12345"),
            patch("srtctl.core.image_bake.start_srun_process", return_value=proc) as srun,
        ):
            assert bake_image(plan) == 0

        assert srun.call_args.kwargs["container_mounts"] == {plan.configs_dir.resolve(): Path("/configs")}
        command = srun.call_args.kwargs["command"]
        assert command[0] == "bash"
        assert "bash /configs/overlay.sh" in command[2]

        manifest = json.loads(plan.manifest_path.read_text())
        assert manifest["installed"]["setup_script"] == "overlay.sh"
        assert manifest["installed"]["ai-dynamo"] is None

    def test_script_outside_configs_gets_an_extra_mount(self, tmp_path):
        configs = tmp_path / "configs"
        configs.mkdir()
        outside = tmp_path / "elsewhere" / "custom.sh"
        outside.parent.mkdir()
        outside.write_text("echo hi\n")
        plan = _script_plan(tmp_path, setup_script=outside, configs_dir=configs)

        assert plan.container_script_path() == Path("/bake-script.sh")
        assert plan.container_mounts()[outside.resolve()] == Path("/bake-script.sh")
        assert "bash /bake-script.sh" in plan.install_script()

    def test_dry_run_prints_configs_mount(self, tmp_path, capsys):
        with patch("srtctl.core.image_bake.start_srun_process") as srun:
            assert bake_image(_script_plan(tmp_path), dry_run=True) == 0

        srun.assert_not_called()
        out = capsys.readouterr().out
        assert "--container-mounts" in out
        assert "/configs" in out
        assert "bash /configs/overlay.sh" in out


class TestResolveSetupScript:
    def test_name_under_configs(self, tmp_path):
        configs, script = _script_tree(tmp_path)

        assert resolve_setup_script("overlay.sh", configs_dir=configs) == script.resolve()

    def test_name_under_patches(self, tmp_path):
        configs, script = _script_tree(tmp_path, name="inside.sh", under_patches=True)

        assert resolve_setup_script("inside.sh", configs_dir=configs) == script.resolve()

    def test_explicit_path(self, tmp_path):
        configs, script = _script_tree(tmp_path)

        assert resolve_setup_script(str(script), configs_dir=configs) == script.resolve()

    def test_missing_name_raises(self, tmp_path):
        configs, _ = _script_tree(tmp_path)

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_setup_script("missing.sh", configs_dir=configs)

    def test_repo_setup_script_is_resolvable(self):
        configs = default_configs_dir()
        candidates = sorted(path for path in configs.glob("*.sh") if path.is_file())
        assert candidates, f"no setup scripts under {configs}"
        path = resolve_setup_script(candidates[0].name, configs_dir=configs)

        assert path == candidates[0].resolve()


class TestCliWiring:
    def test_script_flag_builds_a_script_only_plan(self, tmp_path):
        from argparse import Namespace

        from srtctl.cli.submit import _run_bake_image

        source = tmp_path / "base.sqsh"
        source.write_bytes(b"squashfs")
        configs, script = _script_tree(tmp_path)
        captured: dict = {}

        def fake_bake(plan, *, dry_run=False):
            captured["plan"] = plan
            captured["dry_run"] = dry_run
            return 0

        args = Namespace(
            source_image=str(source),
            output_image=None,
            dynamo_version=None,
            sa_bench=False,
            setup_script="overlay.sh",
            force=True,
            bake_time_limit="0:30:00",
            bake_account=None,
            bake_partition=None,
            bake_dry_run=True,
            patches=None,
        )
        with (
            patch("srtctl.cli.submit.bake_image", side_effect=fake_bake),
            patch("srtctl.cli.submit.default_configs_dir", return_value=configs),
            patch("srtctl.cli.submit._resolve_container_alias", return_value=source),
        ):
            assert _run_bake_image(args) == 0

        plan = captured["plan"]
        assert captured["dry_run"] is True
        assert plan.setup_script == script.resolve()
        assert plan.configs_dir == configs
        assert plan.dynamo_version is None
        assert plan.output_image.name == "base+overlay.sqsh"

    def test_patch_flag_builds_a_patch_only_plan(self, tmp_path):
        from argparse import Namespace

        from srtctl.cli.submit import _run_bake_image

        source = tmp_path / "base.sqsh"
        source.write_bytes(b"squashfs")
        diff = _vllm_diff(tmp_path)
        configs = tmp_path / "configs"
        captured: dict = {}

        def fake_bake(plan, *, dry_run=False):
            captured["plan"] = plan
            return 0

        args = Namespace(
            source_image=str(source),
            output_image=None,
            dynamo_version=None,
            sa_bench=False,
            setup_script=None,
            patches=["mine.diff"],
            force=True,
            bake_time_limit="0:30:00",
            bake_account=None,
            bake_partition=None,
            bake_dry_run=True,
        )
        with (
            patch("srtctl.cli.submit.bake_image", side_effect=fake_bake),
            patch("srtctl.cli.submit.default_configs_dir", return_value=configs),
            patch("srtctl.cli.submit._resolve_container_alias", return_value=source),
        ):
            assert _run_bake_image(args) == 0

        plan = captured["plan"]
        assert plan.vllm_patches[0].host_path == diff.resolve()
        assert plan.setup_script is None
        assert plan.output_image.name == "base+mine.sqsh"
