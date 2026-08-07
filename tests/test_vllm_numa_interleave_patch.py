# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the B300 vLLM NUMA interleave experiment patch."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
PATCHER = REPO_ROOT / "configs/patches/vllm_numa_interleave.py"
WRAPPER = REPO_ROOT / "configs/patches/vllm-numa-interleave.sh"

UNPATCHED_SOURCE = """\
        return f"--physcpubind={cpu_binding} --membind={numa_node}"
    return f"--cpunodebind={numa_node} --membind={numa_node}"
        return f"--physcpubind={cpu_binding} --membind={membind_arg}"
    return f"--cpunodebind={membind_arg} --membind={membind_arg}"
    cpu_only = " ".join(
        t for t in numactl_args.split() if not t.startswith("--membind=")
    )
"""


def run_patcher(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PATCHER), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_patches_all_numactl_paths_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "numa_utils.py"
    target.write_text(UNPATCHED_SOURCE)

    first = run_patcher(target)

    assert first.returncode == 0
    patched = target.read_text()
    assert patched.count("--interleave=0,1") == 4
    assert patched.count("srt-slurm-sa: interleave NUMA memory") == 4
    assert 'startswith(("--membind=", "--interleave="))' in patched

    second = run_patcher(target)

    assert second.returncode == 0
    assert "Already patched, skipping" in second.stderr
    assert target.read_text() == patched


def test_rejects_source_drift_without_modifying_it(tmp_path: Path) -> None:
    target = tmp_path / "numa_utils.py"
    drifted = UNPATCHED_SOURCE.replace("--membind={numa_node}", "--preferred={numa_node}", 1)
    target.write_text(drifted)

    result = run_patcher(target)

    assert result.returncode == 1
    assert "image source may have drifted" in result.stderr
    assert target.read_text() == drifted


def test_wrapper_checks_numactl_before_patching() -> None:
    wrapper = WRAPPER.read_text()

    assert wrapper.index("command -v numactl") < wrapper.index("vllm_numa_interleave.py")
