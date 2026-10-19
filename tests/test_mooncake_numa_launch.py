# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the standalone launch protocol without Slurm, containers or RDMA."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/benchmark_mooncake_numa.sbatch"
BASH = shutil.which("bash") or "/bin/bash"


def executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def launch(
    tmp_path: Path,
    cpu_spec: str = "128(x2)",
    srun_rc: int = 0,
    overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    source = tmp_path / "tools"
    source.mkdir(exist_ok=True)
    for name in (
        "benchmark_mooncake_numa.py",
        "mooncake_numa_workload.py",
        "diagnose_mooncake.py",
        "check_mooncake_rails.py",
    ):
        (source / name).write_text("# Fixture: the fake srun never executes this module.\n")
    capture = tmp_path / "srun.jsonl"
    executable(
        fake_bin / "srun",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['SRUN_CAPTURE'], 'a') as output:\n"
        "    output.write(json.dumps({'argv': sys.argv[1:], 'env': "
        "{k: v for k, v in os.environ.items() if k.startswith('MC_NUMA_') or k == 'ENROOT_MAX_PROCESSORS'}}) + '\\n')\n"
        "raise SystemExit(int(os.environ['FAKE_SRUN_RC']))\n",
    )
    executable(fake_bin / "scontrol", "#!/bin/bash\nprintf 'JobState=RUNNING\\n'\n")
    executable(
        fake_bin / "timeout",
        '#!/bin/bash\nwhile [[ $1 == -* ]]; do shift; done\nshift\nexec "$@"\n',
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MC_NUMA_") and key != "ENROOT_MAX_PROCESSORS"
    }
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        SLURM_SUBMIT_DIR=str(tmp_path),
        SLURM_JOB_ID="123",
        SLURM_JOB_NUM_NODES="2",
        SLURM_NNODES="2",
        SLURM_JOB_CPUS_PER_NODE=cpu_spec,
        SLURM_JOB_NODELIST="node[0-1]",
        MC_NUMA_IMAGE="/images/test.sqsh",
        SRUN_CAPTURE=str(capture),
        FAKE_SRUN_RC=str(srun_rc),
    )
    env.update(overrides or {})
    result = subprocess.run(
        [BASH, str(LAUNCHER)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=15, check=False
    )
    return result, tmp_path / "outputs/mooncake-numa-123", capture


def test_shell_syntax_and_allocation_contract() -> None:
    subprocess.run([BASH, "-n", str(LAUNCHER)], check=True)
    directives = [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --account=root" in directives
    assert "#SBATCH --nodes=2" in directives
    assert "#SBATCH --exclusive" in directives
    assert "#SBATCH --mem=0" in directives
    assert "#SBATCH --time=00:25:00" in directives
    assert "#SBATCH --signal=B:USR1@180" in directives
    assert not any("--partition" in line or "--nodelist" in line or "--gres" in line for line in directives)


@pytest.mark.parametrize(("cpu_spec", "cpus"), [("128(x2)", 128), ("96,96", 96)])
def test_one_persistent_step_uses_actual_full_node_cpus(tmp_path: Path, cpu_spec: str, cpus: int) -> None:
    result, run_dir, capture = launch(
        tmp_path, cpu_spec, overrides={"MC_NUMA_CLUSTER": "hecate", "MC_NUMA_ALLOW_CROSS_RAIL": "1"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1
    argv, env = calls[0]["argv"], calls[0]["env"]
    for arg in (
        "--nodes=2",
        "--ntasks=2",
        "--ntasks-per-node=1",
        f"--cpus-per-task={cpus}",
        "--cpu-bind=none",
        "--mpi=none",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        "--container-remap-root",
        "--container-writable",
        "--container-workdir=/bench-run",
        "--container-image=/images/test.sqsh",
        f"--container-mounts={run_dir}:/bench-run",
    ):
        assert arg in argv
    assert argv[-3:] == ["bash", "/bench-run/launch-node.sh", "/bench-run"]
    assert env["MC_NUMA_CPUS_PER_TASK"] == str(cpus)
    assert env["MC_NUMA_CLUSTER"] == "hecate"
    assert env["MC_NUMA_ALLOW_CROSS_RAIL"] == "1"
    assert env["MC_NUMA_BUDGET"] == "1320"
    assert env["MC_NUMA_SECONDS"] == "10"
    assert env["MC_NUMA_BATCH_SIZE"] == "512"
    assert len(list((run_dir / "tools").glob("*.py"))) == 4
    assert (run_dir / "logs/perf-analysis.md").exists()


@pytest.mark.parametrize("configured,expected", [(None, "8"), ("4", "4")])
def test_container_extraction_concurrency_is_exported_to_host_srun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured: str | None, expected: str
) -> None:
    # An unrelated invoking shell must not accidentally supply this test's default.
    monkeypatch.setenv("ENROOT_MAX_PROCESSORS", "999")
    overrides = {} if configured is None else {"ENROOT_MAX_PROCESSORS": configured}

    result, _, capture = launch(tmp_path, overrides=overrides)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["env"]["ENROOT_MAX_PROCESSORS"] == expected
    assert any("ENROOT_MAX_PROCESSORS" in line and expected in line for line in result.stdout.splitlines())


def test_heterogeneous_nodes_fail_before_launch(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, "128,96")
    assert result.returncode != 0
    assert not capture.exists()
    assert "homogeneous" in result.stdout + result.stderr
    report = (run_dir / "logs/perf-analysis.md").read_text()
    assert "pending or incomplete" in report
    assert "not launched" in report


def test_step_failure_and_stale_artifacts_are_preserved(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, srun_rc=19)
    assert result.returncode == 19
    report = run_dir / "logs/perf-analysis.md"
    original = report.read_bytes()
    assert "launcher exit code: 19" in original.decode()
    assert "Batch exit code: 19" in original.decode()
    result, _, _ = launch(tmp_path)
    assert result.returncode == 2
    assert "Refusing stale" in result.stderr
    assert report.read_bytes() == original
    assert len(capture.read_text().splitlines()) == 1


@pytest.mark.parametrize(("rank", "role"), [("0", "controller"), ("1", "agent")])
def test_container_wrapper_dispatches_ranks_without_changing_existing_install(
    tmp_path: Path, rank: str, role: str
) -> None:
    result, run_dir, _ = launch(tmp_path)
    assert result.returncode == 0
    fake_bin = tmp_path / "wrapper-bin"
    fake_bin.mkdir()
    args_file, apt_file = tmp_path / "python-args", tmp_path / "apt-called"
    executable(fake_bin / "numactl", "#!/bin/bash\nexit 0\n")
    executable(fake_bin / "apt-get", f"#!/bin/bash\ntouch '{apt_file}'\nexit 99\n")
    executable(fake_bin / "python3", '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$WRAPPER_ARGS"\n')
    env = os.environ | {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SLURMD_NODENAME": "test-node",
        "SLURM_PROCID": rank,
        "WRAPPER_ARGS": str(args_file),
    }
    result = subprocess.run(
        [BASH, str(run_dir / "launch-node.sh"), str(run_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert args_file.read_text().splitlines() == [
        "-m",
        "tools.benchmark_mooncake_numa",
        role,
        "--run-dir",
        str(run_dir),
    ]
    assert not apt_file.exists()
    assert "preserving the Mooncake installation" in (run_dir / "logs/agents/test-node.install.log").read_text()


def test_container_wrapper_installs_only_missing_numactl(tmp_path: Path) -> None:
    result, run_dir, _ = launch(tmp_path)
    assert result.returncode == 0
    fake_bin = tmp_path / "minimal-bin"
    fake_bin.mkdir()
    for name, target in (("bash", BASH), ("mkdir", shutil.which("mkdir"))):
        assert target
        (fake_bin / name).symlink_to(target)
    apt_log, args_file = tmp_path / "apt-args", tmp_path / "python-args"
    executable(fake_bin / "id", "#!/bin/bash\nprintf '0\\n'\n")
    executable(fake_bin / "timeout", '#!/bin/bash\nwhile [[ $1 == -* ]]; do shift; done\nshift\nexec "$@"\n')
    executable(fake_bin / "python3", '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$WRAPPER_ARGS"\n')
    executable(
        fake_bin / "apt-get",
        "#!/bin/bash\n"
        'printf \'%s\\n\' "$*" >> "$APT_ARGS"\n'
        "if [[ \" $* \" == *' install '* ]]; then\n"
        "    printf '#!/bin/bash\\nexit 0\\n' > \"$FAKE_BIN/numactl\"\n"
        '    "$REAL_CHMOD" +x "$FAKE_BIN/numactl"\n'
        "fi\n",
    )
    env = os.environ | {
        "PATH": str(fake_bin),
        "SLURMD_NODENAME": "test-node",
        "SLURM_PROCID": "1",
        "WRAPPER_ARGS": str(args_file),
        "APT_ARGS": str(apt_log),
        "FAKE_BIN": str(fake_bin),
        "REAL_CHMOD": shutil.which("chmod") or "/bin/chmod",
    }
    result = subprocess.run(
        [BASH, str(run_dir / "launch-node.sh"), str(run_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = apt_log.read_text().splitlines()
    assert len(calls) == 2
    assert calls[0].endswith(" update")
    assert calls[1].endswith(" install -y --no-install-recommends numactl")
    assert not any("perftest" in call or "mooncake" in call for call in calls)
    assert args_file.read_text().splitlines()[2] == "agent"
