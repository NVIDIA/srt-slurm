# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Hecate allocation and the shared launch protocol without Slurm or RDMA."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/check_hecate_rails.sbatch"
COMMON = ROOT / "tools/check_mooncake_rails.sbatch"
BASH = shutil.which("bash") or "/bin/bash"


def executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def launch(
    tmp_path: Path,
    cpu_spec: str = "352(x2)",
    *,
    overrides: dict[str, str] | None = None,
    srun_rc: int = 0,
    controller_rc: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    binaries, source = tmp_path / "bin", tmp_path / "tools"
    binaries.mkdir(exist_ok=True)
    source.mkdir(exist_ok=True)
    shutil.copyfile(COMMON, source / COMMON.name)
    for name in ("check_mooncake_rails.py", "mooncake_rail_worker.py", "setup_mooncake_rails.sh"):
        (source / name).write_text("# Fixture: the launcher never executes this module.\n")
    srun_capture, controller_capture = tmp_path / "srun.json", tmp_path / "controller.json"
    executable(
        binaries / "srun",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SRUN_CAPTURE']).write_text(json.dumps({'argv': sys.argv[1:], 'env': "
        "{k: v for k, v in os.environ.items() if k.startswith('MC_RAIL_') or k == 'ENROOT_MAX_PROCESSORS'}}))\n"
        "raise SystemExit(int(os.environ['FAKE_SRUN_RC']))\n",
    )
    executable(
        binaries / "python3",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-':\n"
        "    os.execv(os.environ['REAL_PYTHON'], [os.environ['REAL_PYTHON'], *sys.argv[1:]])\n"
        "Path(os.environ['CONTROLLER_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "raise SystemExit(int(os.environ['FAKE_CONTROLLER_RC']))\n",
    )
    executable(
        binaries / "scontrol",
        "#!/bin/bash\nif [[ $1 == show && $2 == hostnames ]]; then\n"
        "    printf 'fixture-node-a\\nfixture-node-b\\n'\n"
        "else\n    printf 'JobState=RUNNING Segment=fixture-rack\\n'\nfi\n",
    )
    executable(binaries / "timeout", '#!/bin/bash\nshift\nexec "$@"\n')
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MC_RAIL_") and key != "ENROOT_MAX_PROCESSORS"
    }
    environment.update(
        PATH=f"{binaries}{os.pathsep}{os.environ['PATH']}",
        SLURM_SUBMIT_DIR=str(tmp_path),
        SLURM_JOB_ID="456",
        SLURM_JOB_CPUS_PER_NODE=cpu_spec,
        SLURM_JOB_NODELIST="fixture-node-[a-b]",
        SLURM_JOB_SEGMENT_SIZE="2",
        SRUN_CAPTURE=str(srun_capture),
        CONTROLLER_CAPTURE=str(controller_capture),
        REAL_PYTHON=sys.executable,
        FAKE_SRUN_RC=str(srun_rc),
        FAKE_CONTROLLER_RC=str(controller_rc),
    )
    environment.update(overrides or {})
    result = subprocess.run(
        [BASH, str(LAUNCHER)], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=10, check=False
    )
    return result, tmp_path / "outputs/mooncake-rails-456", srun_capture, controller_capture


def test_hecate_allocation_and_shell_syntax() -> None:
    for path in (LAUNCHER, COMMON):
        subprocess.run([BASH, "-n", str(path)], check=True)
    directives = [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH")]
    for option in (
        "--account=root",
        "--partition=batch-xdr",
        "--nodes=2",
        "--segment=2",
        "--ntasks-per-node=1",
        "--exclusive",
        "--mem=0",
        "--no-requeue",
        "--time=00:30:00",
        "--signal=B:USR1@180",
    ):
        assert f"#SBATCH {option}" in directives
    assert not any("--nodelist" in line or "--gres" in line or "--gpus" in line for line in directives)
    assert "#SBATCH --mem=0" in COMMON.read_text()
    assert "#SBATCH --cpus-per-task=" not in COMMON.read_text()


@pytest.mark.parametrize(("cpu_spec", "cpus"), [("352(x2)", 352), ("128,128", 128)])
def test_hecate_wrapper_reuses_one_persistent_step_with_full_node_resources(
    tmp_path: Path, cpu_spec: str, cpus: int
) -> None:
    result, run_dir, srun_capture, controller_capture = launch(tmp_path, cpu_spec)
    assert result.returncode == 0, result.stdout + result.stderr
    step = json.loads(srun_capture.read_text())
    argv, env = step["argv"], step["env"]
    for argument in (
        "--nodes=2",
        "--ntasks=2",
        "--ntasks-per-node=1",
        f"--cpus-per-task={cpus}",
        "--cpu-bind=none",
        "--export=ALL",
        "--container-remap-root",
        "--container-writable",
        "--container-workdir=/rail-run",
        f"--container-mounts={run_dir}:/rail-run",
    ):
        assert argument in argv
    assert argv[-3:] == [
        "bash",
        "-c",
        'exec bash /rail-run/setup_mooncake_rails.sh /rail-run "$SLURMD_NODENAME"',
    ]
    assert env["MC_RAIL_IMAGE"] == (
        "/lustre/fsw/coreai_comparch_inferencex/rihuo/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh"
    )
    assert env["MC_RAIL_CLUSTER"] == "hecate"
    assert env["MC_RAIL_CPUS_PER_TASK"] == str(cpus)
    assert env["ENROOT_MAX_PROCESSORS"] == "8"
    assert env["MC_RAIL_SETUP_TIMEOUT"] == "420"
    assert env["MC_RAIL_INVENTORY_TIMEOUT"] == "600"
    assert env["MC_RAIL_NICS"].split(",") == [
        "mlx5_0",
        "mlx5_1",
        "mlx5_2",
        "mlx5_5",
        "mlx5_6",
        "mlx5_7",
        "mlx5_8",
        "mlx5_11",
    ]
    controller = json.loads(controller_capture.read_text())
    for flag, expected in (
        ("--reader", "fixture-node-a"),
        ("--donor", "fixture-node-b"),
        ("--duration", "3"),
        ("--case-timeout", "8"),
        ("--budget", "1620"),
        ("--inventory-timeout", "600"),
    ):
        assert controller[controller.index(flag) + 1] == expected
    assert (run_dir / "submitted.sbatch").read_text() == LAUNCHER.read_text()
    assert (run_dir / "check_mooncake_rails.sbatch").read_text() == COMMON.read_text()
    assert (run_dir / "logs/perf-analysis.md").exists()
    assert "segment-size=2" in (run_dir / "logs/controller.log").read_text()


def test_hecate_runtime_overrides_reach_the_shared_launcher(tmp_path: Path) -> None:
    result, _, srun_capture, controller_capture = launch(
        tmp_path,
        overrides={
            "MC_RAIL_IMAGE": "/images/custom.sqsh",
            "MC_RAIL_NICS": "mlx5_0,mlx5_6",
            "MC_RAIL_DURATION": "5",
            "MC_RAIL_CASE_TIMEOUT": "12",
            "MC_RAIL_BUDGET": "1500",
            "MC_RAIL_SETUP_TIMEOUT": "300",
            "MC_RAIL_INVENTORY_TIMEOUT": "400",
            "ENROOT_MAX_PROCESSORS": "4",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    step = json.loads(srun_capture.read_text())
    assert "--container-image=/images/custom.sqsh" in step["argv"]
    assert step["env"]["MC_RAIL_NICS"] == "mlx5_0,mlx5_6"
    assert step["env"]["ENROOT_MAX_PROCESSORS"] == "4"
    assert step["env"]["MC_RAIL_SETUP_TIMEOUT"] == "300"
    controller = json.loads(controller_capture.read_text())
    for flag, expected in (
        ("--nics", "mlx5_0,mlx5_6"),
        ("--duration", "5"),
        ("--case-timeout", "12"),
        ("--budget", "1500"),
        ("--inventory-timeout", "400"),
    ):
        assert controller[controller.index(flag) + 1] == expected


@pytest.mark.parametrize("cpu_spec", ["352,128", "bad-allocation", "352(x3)"])
def test_invalid_cpu_allocation_stops_before_any_remote_step(tmp_path: Path, cpu_spec: str) -> None:
    result, run_dir, srun_capture, controller_capture = launch(tmp_path, cpu_spec)
    assert result.returncode != 0
    assert not srun_capture.exists() and not controller_capture.exists()
    assert (run_dir / "logs/perf-analysis.md").exists()


@pytest.mark.parametrize(("srun_rc", "controller_rc"), [(19, 0), (0, 7)])
def test_launcher_preserves_failure_and_rejects_stale_results(tmp_path: Path, srun_rc: int, controller_rc: int) -> None:
    result, run_dir, _, _ = launch(tmp_path, srun_rc=srun_rc, controller_rc=controller_rc)
    assert result.returncode != 0
    report = run_dir / "logs/perf-analysis.md"
    original = report.read_bytes()
    assert f"launcher exit code: {srun_rc}" in original.decode()
    assert "Batch exit code:" in original.decode()
    repeated, _, _, _ = launch(tmp_path)
    assert repeated.returncode == 2
    assert "Refusing to reuse" in repeated.stderr
    assert report.read_bytes() == original
