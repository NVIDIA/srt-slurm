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
    BakePlan,
    bake_image,
    build_srun_options,
    default_output_image,
    read_sa_bench_deps,
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
        script = _plan(tmp_path, dynamo_version=None, sa_bench=True, sa_bench_deps=packages, sa_bench_imports=imports).install_script()

        for package in packages:
            assert package in script
        assert f'python3 -c "import {imports}"' in script


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
