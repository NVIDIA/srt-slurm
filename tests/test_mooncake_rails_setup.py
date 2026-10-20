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
    for name in ("bash", "mkdir", "tee", "date", "cat", "sleep", "env", "chmod", "rm", "touch", "grep"):
        resolved = shutil.which(name)
        assert resolved, name
        (binaries / name).symlink_to(resolved)

    def executable(name: str, body: str) -> None:
        path = binaries / name
        path.write_text("#!/bin/bash\nset -eu\n" + body)
        path.chmod(0o755)

    executable("id", "printf '0\\n'\n")
    for name in ("ibstat", "ibv_devinfo", "iblinkinfo"):
        executable(name, "exit 0\n")
    executable(
        "fake-perftest",
        r"""
if [[ ${MC_TEST_LOADER_FAIL:-0} == 1 ]]; then
    printf 'error while loading shared libraries: libibverbs.so.1: cannot open shared object file\n' >&2
    exit 127
fi
printf '%s\n' "${MC_TEST_VERSION_TEXT:-Version: 6.20}"
exit "${MC_TEST_VERSION_RC:-1}"
""",
    )
    executable(
        "timeout",
        r"""
printf '%s\n' "$*" >> "$MC_TEST_TIMEOUT_LOG"
if [[ $1 == --signal=TERM && $2 == --kill-after=10s && ${MC_TEST_SETUP_TIMEOUT:-0} == 1 ]]; then
    exit 124
fi
while [[ $1 == --* ]]; do shift; done
if [[ $1 == -k ]]; then shift 2; fi
shift
exec "$@"
""",
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
    if [[ $* == *ibverbs-utils* ]]; then
        if [[ ${MC_TEST_OPTIONAL_APT_FAIL:-0} == 1 ]]; then
            printf 'synthetic optional topology package failure\n' >&2
            exit 42
        fi
        for name in ibstat ibv_devinfo iblinkinfo; do
            printf '#!/bin/bash\nexit 0\n' > "$MC_TEST_BIN/$name"
            chmod +x "$MC_TEST_BIN/$name"
        done
    else
        for name in ib_read_bw ib_write_bw; do
            cat "$MC_TEST_PERFTEST_SOURCE" > "$MC_TEST_BIN/$name"
            chmod +x "$MC_TEST_BIN/$name"
        done
        printf '#!/bin/bash\nexit 0\n' > "$MC_TEST_BIN/numactl"
        chmod +x "$MC_TEST_BIN/numactl"
    fi
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
        "MC_TEST_TIMEOUT_LOG": str(tmp_path / "timeout.log"),
        "MC_TEST_WORKER_ARGS": str(tmp_path / "worker.args"),
        "MC_TEST_PYTHON": sys.executable,
        "MC_TEST_PERFTEST_SOURCE": str(binaries / "fake-perftest"),
    }
    return run_dir, binaries, environment


@pytest.mark.parametrize("preinstalled", [True, False])
@pytest.mark.parametrize("version_rc", [0, 1])
def test_setup_skips_installed_tools_or_installs_then_executes_worker(
    setup_environment: tuple[Path, Path, dict[str, str]], preinstalled: bool, version_rc: int
) -> None:
    run_dir, binaries, environment = setup_environment
    environment["MC_TEST_VERSION_RC"] = str(version_rc)
    if preinstalled:
        for name in ("ib_read_bw", "ib_write_bw", "numactl"):
            path = binaries / name
            path.write_text(
                "#!/bin/bash\nexit 0\n"
                if name == "numactl"
                else Path(environment["MC_TEST_PERFTEST_SOURCE"]).read_text()
            )
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
    assert "Version: 6.20" in (run_dir / "logs/agents/synthetic-node.install.log").read_text()
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
    assert inventory["errors"] == ["apt installation inside the container failed (exit 42)"]
    assert "timed out" not in result.stderr
    setup_log = Path(inventory["setup_log"])
    if not setup_log.is_absolute():
        setup_log = run_dir / setup_log
    assert "synthetic apt repository failure" in setup_log.read_text()
    calls = Path(environment["MC_TEST_APT_LOG"]).read_text().splitlines()
    assert len(calls) == 1 and calls[0].split()[-1] == "update"


@pytest.mark.parametrize("setup_timeout", [None, "420"])
def test_setup_timeout_uses_configured_budget_and_publishes_timeout_reason(
    setup_environment: tuple[Path, Path, dict[str, str]], setup_timeout: str | None
) -> None:
    run_dir, _, environment = setup_environment
    environment.pop("MC_RAIL_SETUP_TIMEOUT", None)
    if setup_timeout is not None:
        environment["MC_RAIL_SETUP_TIMEOUT"] = setup_timeout
    environment["MC_TEST_SETUP_TIMEOUT"] = "1"

    result = subprocess.run(
        ["/bin/bash", str(SETUP), str(run_dir), "synthetic-node"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    expected_timeout = setup_timeout or "240"
    assert result.returncode != 0
    inventory = json.loads((run_dir / "logs/agents/synthetic-node.inventory.json").read_text())
    assert inventory["ready"] is False
    assert inventory["errors"] == [f"apt installation inside the container timed out after {expected_timeout} seconds"]
    timeout_calls = Path(environment["MC_TEST_TIMEOUT_LOG"]).read_text().splitlines()
    assert timeout_calls[0].startswith(f"--signal=TERM --kill-after=10s {expected_timeout}s bash -c")
    assert not Path(environment["MC_TEST_APT_LOG"]).exists()
    assert not Path(environment["MC_TEST_WORKER_ARGS"]).exists()


@pytest.mark.parametrize("setup_timeout", ["0", "-1", "bad", "1.5"])
def test_setup_rejects_invalid_budget_before_installing_packages(
    setup_environment: tuple[Path, Path, dict[str, str]], setup_timeout: str
) -> None:
    run_dir, _, environment = setup_environment
    environment["MC_RAIL_SETUP_TIMEOUT"] = setup_timeout

    result = subprocess.run(
        ["/bin/bash", str(SETUP), str(run_dir), "synthetic-node"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    inventory = json.loads((run_dir / "logs/agents/synthetic-node.inventory.json").read_text())
    assert inventory["ready"] is False
    assert inventory["errors"] == [f"MC_RAIL_SETUP_TIMEOUT must be a positive integer, got: {setup_timeout}"]
    assert not Path(environment["MC_TEST_APT_LOG"]).exists()
    assert not Path(environment["MC_TEST_WORKER_ARGS"]).exists()


@pytest.mark.parametrize("failure", ["loader", "malformed_version"])
def test_setup_rejects_a_loader_failure_or_unanchored_version_line(
    setup_environment: tuple[Path, Path, dict[str, str]], failure: str
) -> None:
    run_dir, _, environment = setup_environment
    if failure == "loader":
        environment["MC_TEST_LOADER_FAIL"] = "1"
        expected = "error while loading shared libraries: libibverbs.so.1"
    else:
        expected = "diagnostic mentions Version: 6.20 but no version result"
        environment["MC_TEST_VERSION_TEXT"] = expected
        environment["MC_TEST_VERSION_RC"] = "1"

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
    log_path = Path(inventory["setup_log"])
    if not log_path.is_absolute():
        log_path = run_dir / log_path
    assert expected in log_path.read_text()


@pytest.mark.parametrize("required_preinstalled", [True, False])
@pytest.mark.parametrize("optional_failure", [True, False])
def test_optional_topology_install_is_best_effort_and_reuses_required_apt_update(
    setup_environment: tuple[Path, Path, dict[str, str]], required_preinstalled: bool, optional_failure: bool
) -> None:
    run_dir, binaries, environment = setup_environment
    for name in ("ibstat", "ibv_devinfo", "iblinkinfo"):
        (binaries / name).unlink()
    if required_preinstalled:
        for name in ("ib_read_bw", "ib_write_bw", "numactl"):
            path = binaries / name
            path.write_text(
                "#!/bin/bash\nexit 0\n"
                if name == "numactl"
                else Path(environment["MC_TEST_PERFTEST_SOURCE"]).read_text()
            )
            path.chmod(0o755)
    environment["MC_TEST_OPTIONAL_APT_FAIL"] = str(int(optional_failure))

    result = subprocess.run(
        ["/bin/bash", str(SETUP), str(run_dir), "synthetic-node"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(environment["MC_TEST_WORKER_ARGS"]).exists()
    assert not (run_dir / "logs/agents/synthetic-node.inventory.json").exists()
    calls = Path(environment["MC_TEST_APT_LOG"]).read_text().splitlines()
    assert sum(line.endswith(" update") for line in calls) == 1
    assert len(calls) == (2 if required_preinstalled else 3)
    assert calls[-1].endswith("install -y --no-install-recommends ibverbs-utils infiniband-diags")
    setup_log = (run_dir / "logs/agents/synthetic-node.install.log").read_text()
    assert f"Optional topology package setup exit code: {42 if optional_failure else 0}" in setup_log
    assert (binaries / "iblinkinfo").exists() == (not optional_failure)


def test_batch_runs_package_setup_only_inside_the_writable_root_remapped_container() -> None:
    source = (ROOT / "tools/check_mooncake_rails.sbatch").read_text()
    agent_launch = source[source.index("srun --overlap") : source.index("AGENTS_PID=$!")]

    assert "--container-remap-root" in agent_launch
    assert "--container-writable" in agent_launch
    assert "--container-image=" in agent_launch
    assert "setup_mooncake_rails.sh" in agent_launch
    commands = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"\bapt-get\s+(?:update|install)\b", commands)
