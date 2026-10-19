# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise container setup with fake apt/tools; never change host packages."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SETUP = ROOT / "tools/setup_mooncake_rails.sh"


@pytest.fixture
def setup_environment(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    binaries = tmp_path / "fake-bin"
    binaries.mkdir()
    run_dir = tmp_path / "shared run"
    run_dir.mkdir()
    (run_dir / "mooncake_rail_worker.py").write_text("# Worker execution is intercepted by the fake python3.\n")
    # PATH contains only these harmless utilities and our explicit fakes. An
    # installed host perftest or package manager cannot accidentally be used.
    for name in ("bash", "mkdir", "tee", "date", "cat", "sleep", "env", "chmod", "rm", "touch"):
        resolved = shutil.which(name)
        assert resolved, name
        (binaries / name).symlink_to(resolved)

    def executable(name: str, body: str) -> None:
        path = binaries / name
        path.write_text("#!/bin/bash\nset -eu\n" + body)
        path.chmod(0o755)

    executable("id", "printf '0\\n'\n")
    executable(
        "timeout",
        'while [[ $1 == --* ]]; do shift; done\nif [[ $1 == -k ]]; then shift 2; fi\nshift\nexec "$@"\n',
    )
    executable(
        "apt-get",
        r"""
printf '%s\n' "$*" >> "$MC_TEST_APT_LOG"
if [[ ${MC_TEST_APT_FAIL:-0} == 1 ]]; then
    printf 'synthetic apt repository failure\n' >&2
    exit 42
fi
while [[ $1 == -o ]]; do shift 2; done
if [[ $1 == install ]]; then
    for name in ib_read_bw ib_write_bw numactl; do
        printf '#!/bin/bash\nexit 0\n' > "$MC_TEST_BIN/$name"
        chmod +x "$MC_TEST_BIN/$name"
    done
fi
""",
    )
    executable(
        "python3",
        r"""
if [[ $1 == -c || $1 == - ]]; then
    exec "$MC_TEST_PYTHON" "$@"
fi
printf '%s\n' "$@" > "$MC_TEST_WORKER_ARGS"
""",
    )
    environment = {
        **os.environ,
        "PATH": str(binaries),
        "MC_TEST_BIN": str(binaries),
        "MC_TEST_APT_LOG": str(tmp_path / "apt.log"),
        "MC_TEST_WORKER_ARGS": str(tmp_path / "worker.args"),
        "MC_TEST_PYTHON": sys.executable,
    }
    return run_dir, binaries, environment


@pytest.mark.parametrize("preinstalled", [True, False])
def test_setup_skips_installed_tools_or_installs_then_executes_worker(
    setup_environment: tuple[Path, Path, dict[str, str]], preinstalled: bool
) -> None:
    run_dir, binaries, environment = setup_environment
    if preinstalled:
        for name in ("ib_read_bw", "ib_write_bw", "numactl"):
            path = binaries / name
            path.write_text("#!/bin/bash\nexit 0\n")
            path.chmod(0o755)

    result = subprocess.run(
        ["/bin/bash", str(SETUP), str(run_dir), "synthetic-node"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    args = Path(environment["MC_TEST_WORKER_ARGS"]).read_text().splitlines()
    assert args == [str(run_dir / "mooncake_rail_worker.py"), "--run-dir", str(run_dir), "--node", "synthetic-node"]
    apt_log = Path(environment["MC_TEST_APT_LOG"])
    if preinstalled:
        assert not apt_log.exists()
    else:
        calls = [line.split() for line in apt_log.read_text().splitlines()]
        assert len(calls) == 2 and calls[0][-1] == "update"
        assert calls[1][calls[1].index("install") :] == [
            "install",
            "-y",
            "--no-install-recommends",
            "perftest",
            "numactl",
        ]
        assert all((binaries / name).is_file() for name in ("ib_read_bw", "ib_write_bw", "numactl"))


def test_setup_apt_failure_publishes_failed_inventory_and_preserves_the_error(
    setup_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    run_dir, _, environment = setup_environment
    environment["MC_TEST_APT_FAIL"] = "1"

    result = subprocess.run(
        ["/bin/bash", str(SETUP), str(run_dir), "synthetic-node"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert not Path(environment["MC_TEST_WORKER_ARGS"]).exists()
    inventory = json.loads((run_dir / "logs/agents/synthetic-node.inventory.json").read_text())
    assert inventory["ready"] is False and inventory["errors"]
    assert inventory["setup_log"]
    setup_log = Path(inventory["setup_log"])
    if not setup_log.is_absolute():
        setup_log = run_dir / setup_log
    assert "synthetic apt repository failure" in setup_log.read_text()
    calls = Path(environment["MC_TEST_APT_LOG"]).read_text().splitlines()
    assert len(calls) == 1 and calls[0].split()[-1] == "update"


def test_batch_runs_package_setup_only_inside_the_writable_root_remapped_container() -> None:
    source = (ROOT / "tools/check_mooncake_rails.sbatch").read_text()
    agent_launch = source[source.index("srun --overlap") : source.index("AGENTS_PID=$!")]

    assert "--container-remap-root" in agent_launch
    assert "--container-writable" in agent_launch
    assert "--container-image=" in agent_launch
    assert "setup_mooncake_rails.sh" in agent_launch
    commands = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"\bapt-get\s+(?:update|install)\b", commands)
