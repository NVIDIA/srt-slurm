# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the mounted AgentX launcher with a local InferenceX fixture."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "agentx.sh"


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_ALLOW_PROTOCOL": "file"},
    )
    return result.stdout.strip()


def test_agentx_launcher_checks_required_metadata_before_cloning(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "AGENTX_INFERENCEX_REPO_URL": str(tmp_path / "missing")},
        check=False,
    )

    assert result.returncode == 1
    assert "MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC RESULT_FILENAME DURATION" in result.stderr
    assert "Cloning into" not in result.stderr


def test_agentx_launcher_fetches_pinned_harness_and_preserves_results(tmp_path: Path) -> None:
    aiperf = tmp_path / "aiperf"
    aiperf.mkdir()
    _git("init", "-q", cwd=aiperf)
    (aiperf / "pyproject.toml").write_text("[project]\nname = 'aiperf-fixture'\nversion = '0.0.1'\n")
    _git("add", ".", cwd=aiperf)
    _git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture", cwd=aiperf)

    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    _git("init", "-q", cwd=inferencex)
    benchmarks = inferencex / "inferencex-e2e" / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "srt_agentic.sh").write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$PORT" "$AIPERF_PYTHON_VERSION" "$INFMAX_CONTAINER_WORKSPACE" '
        '"$MODEL" > "$AGENTIC_OUTPUT_DIR/capture.txt"\n'
    )
    _git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(aiperf),
        "inferencex-e2e/utils/aiperf",
        cwd=inferencex,
    )
    _git("add", ".", cwd=inferencex)
    _git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture", cwd=inferencex)
    pinned_ref = _git("rev-parse", "HEAD", cwd=inferencex)

    (benchmarks / "srt_agentic.sh").write_text("exit 99\n")
    _git("add", ".", cwd=inferencex)
    _git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "newer", cwd=inferencex)

    result_dir = tmp_path / "results"
    env = {
        **os.environ,
        "GIT_ALLOW_PROTOCOL": "file",
        "AGENTX_INFERENCEX_REPO_URL": str(inferencex),
        "AGENTX_INFERENCEX_REF": pinned_ref,
        "RESULT_DIR": str(result_dir),
        "SRT_FRONTEND_PORT": "9123",
        "AIPERF_PYTHON_VERSION": "3.12",
        "MODEL": "test/model",
        "MODEL_PREFIX": "test",
        "FRAMEWORK": "sglang",
        "PRECISION": "fp8",
        "CONC": "8",
        "RESULT_FILENAME": "agentx_c8",
        "DURATION": "900",
    }
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert f"Running InferenceX AgentX at {pinned_ref}" in result.stdout
    port, python_version, checkout, model = (result_dir / "capture.txt").read_text().splitlines()
    assert (port, python_version, model) == ("9123", "3.12", "test/model")
    assert not Path(checkout).exists()
