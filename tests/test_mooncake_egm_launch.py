# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the two-node EGM launch contract without Slurm or GPU libraries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/check_mooncake_egm.sbatch"
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
    # Run the real report command: this must work on the batch host even when
    # container setup, Mooncake imports or srun fail before producing results.
    shutil.copyfile(ROOT / "tools/check_mooncake_egm.py", source / "check_mooncake_egm.py")
    shutil.copyfile(ROOT / "tools/mooncake_egm_alloc_diagnostics.py", source / "mooncake_egm_alloc_diagnostics.py")
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    for name in ("mooncake_egm_donor.py", "setup_mooncake_egm.sh"):
        shutil.copyfile(ROOT / "configs" / name, configs / name)
    wheels = tmp_path / "wheels"
    wheels.mkdir(exist_ok=True)
    (wheels / "mooncake_transfer_engine_cuda13-0.0.0-cp312-cp312-linux_aarch64.whl").write_bytes(b"fixture")
    image = tmp_path / "image.sqsh"
    image.touch()
    capture = tmp_path / "srun.jsonl"
    executable(
        fake_bin / "srun",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['SRUN_CAPTURE'], 'a') as output:\n"
        "    output.write(json.dumps({'argv': sys.argv[1:], 'env': "
        "{k: v for k, v in os.environ.items() if k.startswith('MC_EGM_') or k == 'ENROOT_MAX_PROCESSORS'}}) + '\\n')\n"
        "raise SystemExit(int(os.environ['FAKE_SRUN_RC']))\n",
    )
    executable(fake_bin / "scontrol", "#!/bin/bash\nprintf 'JobState=RUNNING\\n'\n")
    executable(fake_bin / "timeout", '#!/bin/bash\nwhile [[ $1 == -* ]]; do shift; done\nshift\nexec "$@"\n')
    if not (fake_bin / "python3").exists():
        (fake_bin / "python3").symlink_to(sys.executable)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MC_EGM_") and key not in ("ENROOT_MAX_PROCESSORS", "MC_USE_NVLINK_IPC")
    }
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        SLURM_SUBMIT_DIR=str(tmp_path),
        SLURM_JOB_ID="123",
        SLURM_JOB_NUM_NODES="2",
        SLURM_NNODES="2",
        SLURM_JOB_CPUS_PER_NODE=cpu_spec,
        SLURM_JOB_NODELIST="node[0-1]",
        MC_EGM_IMAGE=str(image),
        MC_EGM_WHEEL_DIR=str(wheels),
        SRUN_CAPTURE=str(capture),
        FAKE_SRUN_RC=str(srun_rc),
    )
    env.update(overrides or {})
    result = subprocess.run(
        [BASH, str(LAUNCHER)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=15, check=False
    )
    return result, tmp_path / "outputs/mooncake-egm-123", capture


def test_shell_syntax_and_two_node_nvlink_allocation() -> None:
    subprocess.run([BASH, "-n", str(LAUNCHER)], check=True)
    wrapper = LAUNCHER.read_text().split("<<'WRAPPER'\n", 1)[1].split("\nWRAPPER\n", 1)[0]
    subprocess.run([BASH, "-n"], input=wrapper, text=True, check=True)
    directives = [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH")]
    for directive in (
        "#SBATCH --nodes=2",
        "#SBATCH --segment=2",
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --exclusive",
        "#SBATCH --mem=0",
        "#SBATCH --time=00:15:00",
    ):
        assert directive in directives
    assert not any("--partition" in line or "--nodelist" in line for line in directives)


@pytest.mark.parametrize(("cpu_spec", "cpus"), [("128(x2)", 128), ("96,96", 96)])
def test_one_step_uses_allocated_cpus_and_read_only_wheel_mount(tmp_path: Path, cpu_spec: str, cpus: int) -> None:
    result, run_dir, capture = launch(tmp_path, cpu_spec)
    # A successful srun without the client's verified-completion marker is incomplete.
    assert result.returncode == 1, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1
    argv, env = calls[0]["argv"], calls[0]["env"]
    for arg in (
        "--nodes=2",
        "--ntasks=2",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        f"--cpus-per-task={cpus}",
        "--cpu-bind=none",
        "--mpi=none",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        "--container-remap-root",
        "--container-writable",
        "--no-container-mount-home",
        "--container-workdir=/egm-run",
        f"--container-image={tmp_path / 'image.sqsh'}",
        f"--container-mounts={run_dir}:/egm-run,{(tmp_path / 'wheels').resolve()}:/egm-wheels:ro",
    ):
        assert arg in argv
    assert argv[-3:] == ["bash", "/egm-run/launch-node.sh", "/egm-run"]
    assert env["MC_EGM_CPUS_PER_TASK"] == str(cpus)
    assert env["MC_EGM_TIMEOUT"] == "720"
    assert env["MC_EGM_POOL_SIZE"] == "4GiB"
    assert env["MC_EGM_ALLOC_DIAGNOSTICS"] == "1"
    assert env["MC_EGM_ALLOC_SIZES"] == "4GiB,260GiB"
    assert env["MC_EGM_ALLOC_GPU"] == "0"
    assert env["MC_EGM_ALLOC_NUMA_NODES"] == "0,auto"
    assert env["MC_EGM_ALLOC_TIMEOUT"] == "30"
    assert env["MC_EGM_ALLOC_BUDGET"] == "150"
    assert env["ENROOT_MAX_PROCESSORS"] == "8"
    for source, copied in (
        (ROOT / "tools/check_mooncake_egm.py", run_dir / "tools/check_mooncake_egm.py"),
        (ROOT / "tools/mooncake_egm_alloc_diagnostics.py", run_dir / "tools/mooncake_egm_alloc_diagnostics.py"),
        (ROOT / "configs/mooncake_egm_donor.py", run_dir / "configs/mooncake_egm_donor.py"),
        (ROOT / "configs/setup_mooncake_egm.sh", run_dir / "configs/setup_mooncake_egm.sh"),
        (LAUNCHER, run_dir / "submitted.sbatch"),
    ):
        assert copied.read_bytes() == source.read_bytes()
    report = (run_dir / "logs/perf-analysis.md").read_text()
    # The fake step produces no CUDA evidence, even though its exit code is zero.
    assert "FAIL / INCOMPLETE" in report
    assert "TTFT p50/p95/p99" in report
    assert "no measured successful sample" in report


@pytest.mark.parametrize("cpu_spec", ["128,96", "128(x3)", "128garbage"])
def test_invalid_cpu_allocation_never_starts_step(tmp_path: Path, cpu_spec: str) -> None:
    result, run_dir, capture = launch(tmp_path, cpu_spec)
    assert result.returncode != 0
    assert not capture.exists()
    assert "SLURM_JOB_CPUS_PER_NODE" in result.stdout or "homogeneous" in result.stdout
    report = (run_dir / "logs/perf-analysis.md").read_text()
    assert "pending or incomplete" in report
    assert "Batch exit code:" in report


def test_wrong_node_count_never_starts_step(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, overrides={"SLURM_JOB_NUM_NODES": "1"})
    assert result.returncode == 2
    assert not capture.exists()
    assert "exactly two allocated nodes" in result.stdout
    assert "Batch exit code: 2" in (run_dir / "logs/perf-analysis.md").read_text()


@pytest.mark.parametrize("wheel_count", [0, 2])
def test_ambiguous_or_missing_wheels_fail_with_report_before_launch(tmp_path: Path, wheel_count: int) -> None:
    wheels = tmp_path / "selected-wheels"
    wheels.mkdir()
    for index in range(wheel_count):
        (wheels / f"mooncake_transfer_engine_cuda13-{index}.whl").touch()
    result, run_dir, capture = launch(tmp_path, overrides={"MC_EGM_WHEEL_DIR": str(wheels)})
    assert result.returncode == 2
    assert not capture.exists()
    assert f"found {wheel_count}" in result.stdout
    report = (run_dir / "logs/perf-analysis.md").read_text()
    assert "FAIL / INCOMPLETE" in report
    assert "launcher exit code: `2`" in report
    assert "Batch exit code: 2" in report


def test_step_failure_preserves_exit_code_report_and_stale_run(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, srun_rc=19)
    assert result.returncode == 19, result.stdout + result.stderr
    report = run_dir / "logs/perf-analysis.md"
    original = report.read_bytes()
    assert "FAIL / INCOMPLETE" in original.decode()
    assert "launcher exit code: `19`" in original.decode()
    assert "Batch exit code: 19" in original.decode()

    repeated, _, _ = launch(tmp_path)
    assert repeated.returncode == 2
    assert "Refusing stale" in repeated.stderr
    assert report.read_bytes() == original
    assert len(capture.read_text().splitlines()) == 1


def test_wrapper_clears_conflicting_transport_and_mpi_settings_before_setup(tmp_path: Path) -> None:
    result, run_dir, _ = launch(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    env_file, args_file = tmp_path / "setup-env", tmp_path / "setup-args"
    (run_dir / "configs/setup_mooncake_egm.sh").write_text(
        'env > "$EGM_WRAPPER_ENV"\nprintf \'%s\\n\' "$@" > "$EGM_WRAPPER_ARGS"\nexit 27\n'
    )
    env = os.environ | {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "SLURM_PROCID": "1",
        "MC_EGM_SETUP_TIMEOUT": "240",
        "EGM_WRAPPER_ENV": str(env_file),
        "EGM_WRAPPER_ARGS": str(args_file),
        "PMIX_RANK": "9",
        "OMPI_COMM_WORLD_SIZE": "8",
        "MC_USE_NVLINK_IPC": "0",
        "MC_TE_FILTERS": "rdma",
        "MC_ENABLE_DEST_DEVICE_AFFINITY": "1",
        "MC_FORCE_TCP": "1",
        "MC_USE_TENT": "1",
        "MOONCAKE_STORE_BACKEND": "rs",
    }
    result = subprocess.run(
        [BASH, str(run_dir / "launch-node.sh"), str(run_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 27, result.stdout + result.stderr
    installed_env = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    for name in (
        "PMIX_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "MC_USE_NVLINK_IPC",
        "MC_TE_FILTERS",
        "MC_ENABLE_DEST_DEVICE_AFFINITY",
        "MC_FORCE_TCP",
    ):
        assert name not in installed_env
    assert installed_env["MC_USE_TENT"] == "0"
    assert installed_env["MOONCAKE_STORE_BACKEND"] == "cpp"
    assert installed_env["SRT_MOONCAKE_EGM_WHEEL_DIR"] == "/egm-wheels"
    assert installed_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert args_file.read_text().splitlines() == ["--mooncake-only"]
    assert "exit_code=27 cleanup_failed=0" in (run_dir / "logs/shutdown-1.txt").read_text()


@pytest.mark.parametrize(
    ("enabled", "diagnostic_rc", "interrupt"), [(True, 0, False), (True, 7, False), (False, 0, False), (True, 0, True)]
)
def test_provider_wrapper_runs_allocation_diagnostics_before_smoke_and_stops_owned_children(
    tmp_path: Path, enabled: bool, diagnostic_rc: int, interrupt: bool
) -> None:
    result, run_dir, _ = launch(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    (run_dir / "configs/setup_mooncake_egm.sh").write_text("exit 0\n")
    capture = tmp_path / "wrapper-events.jsonl"
    python = tmp_path / "bin/python3"
    python.unlink()
    executable(
        python,
        f"#!{sys.executable}\n"
        "import json, os, signal, sys, time\n"
        "from pathlib import Path\n"
        "def event(name):\n"
        "    with open(os.environ['WRAPPER_CAPTURE'], 'a') as stream:\n"
        "        stream.write(json.dumps({'name': name, 'args': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
        "def hold(name):\n"
        "    def stop(signum, frame):\n"
        "        event(name + '-stop')\n"
        "        raise SystemExit(0)\n"
        "    signal.signal(signal.SIGTERM, stop)\n"
        "    event(name + '-waiting')\n"
        "    while True: time.sleep(0.02)\n"
        "if sys.argv[1] == '-':\n"
        "    source = sys.stdin.read()\n"
        "    if 'with ExitStack()' in source:\n"
        "        run_dir = Path(sys.argv[2])\n"
        "        (run_dir / 'master-config.json').write_text(json.dumps({'master_server_address': '127.0.0.1:12345', "
        "'provider_hostname': 'provider-node', 'metrics_port': 12346}))\n"
        "        (run_dir / 'master-ports').write_text('12345 12346\\n')\n"
        "    else: event('master-ready')\n"
        "elif sys.argv[1:3] == ['-m', 'mooncake.cli']:\n"
        "    event('master-start')\n"
        "    hold('master')\n"
        "elif sys.argv[1].endswith('mooncake_egm_alloc_diagnostics.py'):\n"
        "    event('allocation-diagnostics')\n"
        "    output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "    output.mkdir(parents=True, exist_ok=True)\n"
        "    (output / 'summary.json').write_text(json.dumps({'cases': [{'status': 'allocation_failed', "
        "'cuda_result': 2}], 'cleanup_confirmed': True}))\n"
        "    if os.environ['WRAPPER_INTERRUPT'] == '1': hold('allocation-diagnostics')\n"
        "    raise SystemExit(int(os.environ['WRAPPER_DIAGNOSTIC_RC']))\n"
        "elif sys.argv[1].endswith('mooncake_egm_donor.py'):\n"
        "    event('provider-start')\n"
        "    ready = Path(sys.argv[sys.argv.index('--ready-file') + 1])\n"
        "    ready.write_text('{}')\n"
        "    (ready.parents[1] / 'client.done').touch()\n"
        "    hold('provider')\n"
        "else: raise SystemExit('Unexpected Python invocation: ' + repr(sys.argv))\n",
    )
    executable(tmp_path / "bin/nvidia-smi", "#!/bin/bash\nexit 0\n")
    executable(tmp_path / "bin/cat", '#!/bin/bash\ncase "$1" in /proc/*) exit 0 ;; esac\nexec /bin/cat "$@"\n')
    env = os.environ | {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "SLURM_PROCID": "0",
        "MC_EGM_SETUP_TIMEOUT": "1",
        "MC_EGM_READY_TIMEOUT": "1",
        "MC_EGM_CLIENT_TIMEOUT": "1",
        "MC_EGM_POOL_SIZE": "4GiB",
        "MC_EGM_ALLOC_DIAGNOSTICS": "1" if enabled else "0",
        "MC_EGM_ALLOC_SIZES": "4GiB,260GiB",
        "MC_EGM_ALLOC_GPU": "0",
        "MC_EGM_ALLOC_NUMA_NODES": "0,auto",
        "MC_EGM_ALLOC_TIMEOUT": "30",
        "MC_EGM_ALLOC_BUDGET": "150",
        "WRAPPER_CAPTURE": str(capture),
        "WRAPPER_DIAGNOSTIC_RC": str(diagnostic_rc),
        "WRAPPER_INTERRUPT": "1" if interrupt else "0",
        "MC_USE_NVLINK_IPC": "0",
        "MC_USE_TENT": "1",
        "MC_TE_FILTERS": "rdma",
        "MOONCAKE_STORE_BACKEND": "rs",
    }
    with subprocess.Popen(
        [BASH, str(run_dir / "launch-node.sh"), str(run_dir)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        if interrupt:
            deadline = time.monotonic() + 4
            while not capture.exists() or '"name": "allocation-diagnostics-waiting"' not in capture.read_text():
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "Allocation diagnostics never started"
                time.sleep(0.02)
            process.terminate()
        stdout, stderr = process.communicate(timeout=10)
    expected_rc = 143 if interrupt else diagnostic_rc if enabled else 0
    assert process.returncode == expected_rc, stdout + stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    names = [call["name"] for call in calls]
    assert "master-ready" in names
    if enabled:
        diagnostics = next(call for call in calls if call["name"] == "allocation-diagnostics")
        assert diagnostics["args"] == [
            str(run_dir / "tools/mooncake_egm_alloc_diagnostics.py"),
            "--config",
            str(run_dir / "master-config.json"),
            "--output-dir",
            str(run_dir / "logs/allocation-diagnostics"),
            "--sizes",
            "4GiB,260GiB",
            "--gpu",
            "0",
            "--numa-nodes",
            "0,auto",
            "--timeout-seconds",
            "30",
            "--budget-seconds",
            "150",
        ]
        assert names.index("master-ready") < names.index("allocation-diagnostics")
        # The helper narrows its allocation children to physical GPU 0. The
        # surrounding container keeps all GPUs visible for the original smoke.
        assert diagnostics["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert diagnostics["env"]["MC_USE_TENT"] == "0"
        assert diagnostics["env"]["MOONCAKE_STORE_BACKEND"] == "cpp"
        assert "MC_USE_NVLINK_IPC" not in diagnostics["env"]
        assert "MC_TE_FILTERS" not in diagnostics["env"]
    else:
        assert "allocation-diagnostics" not in names
    if expected_rc:
        assert "provider-start" not in names
        assert not (run_dir / "client.done").exists()
    else:
        provider = next(call for call in calls if call["name"] == "provider-start")
        assert provider["args"][provider["args"].index("--segment-size") + 1] == "4GiB"
        assert provider["args"][provider["args"].index("--numa-nodes") + 1] == "auto"
        assert provider["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        if enabled:
            assert names.index("allocation-diagnostics") < names.index("provider-start")
        assert names.index("provider-stop") < names.index("master-stop")
    assert "master-stop" in names
    if interrupt:
        assert names.index("allocation-diagnostics-stop") < names.index("master-stop")
    assert f"exit_code={expected_rc} cleanup_failed=0" in (run_dir / "logs/shutdown-0.txt").read_text()
