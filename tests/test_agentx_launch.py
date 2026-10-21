# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``benchmarks/agentx.sh`` (repo root): the harness it fetches follows the InferenceX ref.

InferenceX vendored the AgentX harness as the ``inferencex-e2e/utils/aiperf`` submodule until
SemiAnalysisAI/InferenceX#3940 replaced it with a pinned ``agentx-harness`` PyPI release. The
fixture repository has one commit on each side of that change.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

AGENTX = Path(__file__).resolve().parents[1] / "benchmarks" / "agentx.sh"
REQUIRED_ENV = {
    "MODEL": "m",
    "MODEL_PREFIX": "p",
    "FRAMEWORK": "f",
    "PRECISION": "fp8",
    "CONC": "1",
    "RESULT_FILENAME": "r",
    "DURATION": "60",
}
# The fixture repository's URL, its submodule-era commit, and the git environment to run under.
InferenceX = tuple[str, str, dict[str, str]]


def _git_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        # Local file:// submodules are blocked by default since git 2.38.1.
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "protocol.file.allow",
        "GIT_CONFIG_VALUE_0": "always",
    }


def _git(cwd: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def _commit_all(repo: Path, env: dict[str, str], message: str) -> str:
    _git(repo, env, "add", "-A")
    _git(repo, env, "commit", "-qm", message)
    return _git(repo, env, "rev-parse", "HEAD")


@pytest.fixture
def inferencex(tmp_path: Path) -> InferenceX:
    """An InferenceX repo whose ``main`` pins the PyPI harness; returns its URL and the submodule-era commit."""
    env = _git_env(tmp_path)
    aiperf = tmp_path / "aiperf"
    aiperf.mkdir()
    (aiperf / "pyproject.toml").write_text("[project]\nname = 'aiperf'\n")
    _git(aiperf, env, "init", "-q", "-b", "main")
    _commit_all(aiperf, env, "aiperf")

    repo = tmp_path / "InferenceX"
    entrypoint = repo / "inferencex-e2e" / "benchmarks" / "srt_agentic.sh"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text('echo "entrypoint ran in $INFMAX_CONTAINER_WORKSPACE"\n')
    _git(repo, env, "init", "-q", "-b", "main")
    _git(repo, env, "config", "uploadpack.allowAnySHA1InWant", "true")
    _git(repo, env, "submodule", "add", "-q", aiperf.as_uri(), "inferencex-e2e/utils/aiperf")
    submodule_era = _commit_all(repo, env, "harness as the utils/aiperf submodule")

    _git(repo, env, "rm", "-q", "inferencex-e2e/utils/aiperf")
    requirements = repo / "inferencex-e2e" / "infx" / "bench" / "agentic" / "requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("agentx-harness==1.0.12\ndatasets>=4.7.0\n")
    _commit_all(repo, env, "harness from PyPI")
    return repo.as_uri(), submodule_era, env


def _run(tmp_path: Path, url: str, env: dict[str, str], ref: str = "") -> subprocess.CompletedProcess[str]:
    run_env = {
        **env,
        **REQUIRED_ENV,
        "TMPDIR": str(tmp_path),
        "RESULT_DIR": str(tmp_path / "logs"),
        "AGENTX_INFERENCEX_REPO_URL": url,
        "AGENTX_INFERENCEX_REF": ref,
    }
    return subprocess.run(["bash", str(AGENTX)], env=run_env, capture_output=True, text=True, timeout=60, check=False)


def test_pypi_harness_ref_runs_without_the_aiperf_submodule(tmp_path: Path, inferencex: InferenceX) -> None:
    url, _, env = inferencex
    result = _run(tmp_path, url, env)
    assert result.returncode == 0, result.stderr
    assert "Using agentx-harness==1.0.12" in result.stdout
    assert "entrypoint ran in" in result.stdout


def test_submodule_era_ref_still_fetches_aiperf(tmp_path: Path, inferencex: InferenceX) -> None:
    url, submodule_era, env = inferencex
    result = _run(tmp_path, url, env, ref=submodule_era)
    assert result.returncode == 0, result.stderr
    assert f"Running InferenceX AgentX at {submodule_era}" in result.stdout
    assert re.search(r"Using AIPerf at [0-9a-f]{40}", result.stdout)
    assert "entrypoint ran in" in result.stdout
