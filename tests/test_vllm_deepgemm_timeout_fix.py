# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the midcurve DeepGEMM startup hotfix."""

import subprocess
import sys
from pathlib import Path

from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.core.schema import SrtConfig

REPO_ROOT = Path(__file__).parents[1]
PATCHER = REPO_ROOT / "configs/patches/vllm_deepgemm_timeout_fix.py"
WRAPPER = REPO_ROOT / "configs/patches/vllm-container-deps-deepgemm-timeout.sh"

UNPATCHED_SOURCE = """\
        // Update status and wait arrival (with 30s timeout, at 2 GHz)
        constexpr int64_t kNumTimeoutCycles = 30ll * 2000000000ll;
                    printf("DeepGEMM NVLink barrier timeout (30s): rank=%d\\n", rank);
"""


def run_patcher(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PATCHER), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_patches_timeout_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "barrier.cuh"
    target.write_text(UNPATCHED_SOURCE)

    first = run_patcher(target)

    assert first.returncode == 0
    assert "Increased NVLink barrier timeout to 300s" in first.stderr
    patched = target.read_text()
    assert "300ll * 2000000000ll" in patched
    assert "timeout (300s)" in patched
    assert "timeout (30s)" not in patched

    second = run_patcher(target)

    assert second.returncode == 0
    assert "Already patched, skipping" in second.stderr
    assert target.read_text() == patched


def test_rejects_drifted_source_without_modifying_it(tmp_path: Path) -> None:
    target = tmp_path / "barrier.cuh"
    drifted = UNPATCHED_SOURCE.replace("30ll * 2000000000ll", "45ll * 2000000000ll")
    target.write_text(drifted)

    result = run_patcher(target)

    assert result.returncode == 1
    assert "vendored DeepGEMM source may have drifted" in result.stderr
    assert target.read_text() == drifted


def test_rejects_ambiguous_anchors_without_modifying_source(tmp_path: Path) -> None:
    target = tmp_path / "barrier.cuh"
    ambiguous = UNPATCHED_SOURCE + UNPATCHED_SOURCE
    target.write_text(ambiguous)

    result = run_patcher(target)

    assert result.returncode == 1
    assert "found old=(2, 2)" in result.stderr
    assert target.read_text() == ambiguous


def test_midcurve_uses_wrapper_after_base_setup() -> None:
    config = SrtConfig.from_yaml(REPO_ROOT / "midcurve.yaml")
    wrapper = WRAPPER.read_text()
    mixin = WorkerStageMixin()
    mixin.config = config
    preamble = mixin._build_worker_preamble()

    assert config.setup_script == WRAPPER.name
    assert wrapper.index("vllm-container-deps.sh") < wrapper.index("vllm_deepgemm_timeout_fix.py")
    assert preamble is not None
    assert WRAPPER.name in preamble

