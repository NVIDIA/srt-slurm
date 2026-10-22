# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the allocation-reclamation batch launcher without Slurm or CUDA."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/check_mooncake_egm_reclaim.sbatch"
BASH = shutil.which("bash") or "/bin/bash"


def executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def launch(
    tmp_path: Path,
    *,
    cpu_spec: str = "352",
    srun_rc: int = 0,
    overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    source = tmp_path / "tools"
    source.mkdir(exist_ok=True)
    for name in ("mooncake_egm_reclaim.py", "mooncake_egm_alloc_diagnostics.py"):
        shutil.copyfile(ROOT / "tools" / name, source / name)
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
    env = {key: value for key, value in os.environ.items() if not key.startswith("MC_EGM_")}
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        SLURM_SUBMIT_DIR=str(tmp_path),
        SLURM_JOB_ID="123",
        SLURM_JOB_NUM_NODES="1",
        SLURM_NNODES="1",
        SLURM_JOB_CPUS_PER_NODE=cpu_spec,
        SLURM_JOB_NODELIST="node0",
        MC_EGM_IMAGE=str(image),
        MC_EGM_WHEEL_DIR=str(wheels),
        SRUN_CAPTURE=str(capture),
        FAKE_SRUN_RC=str(srun_rc),
    )
    env.update(overrides or {})
    result = subprocess.run(
        [BASH, str(LAUNCHER)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=20, check=False
    )
    return result, tmp_path / "outputs/mooncake-egm-reclaim-123", capture


def test_shell_syntax_and_exclusive_host_memory_allocation() -> None:
    subprocess.run([BASH, "-n", str(LAUNCHER)], check=True)
    wrapper = LAUNCHER.read_text().split("<<'WRAPPER'\n", 1)[1].split("\nWRAPPER\n", 1)[0]
    subprocess.run([BASH, "-n"], input=wrapper, text=True, check=True)
    directives = [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH")]
    for directive in (
        "#SBATCH --nodes=1",
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --exclusive",
        "#SBATCH --mem=0",
        "#SBATCH --time=00:20:00",
    ):
        assert directive in directives
    assert not any(any(option in line for option in ("--account", "--partition", "--nodelist")) for line in directives)


@pytest.mark.parametrize("cpu_spec", ["352", "352(x1)"])
def test_one_container_gets_all_cpus_and_read_only_wheel_mount(tmp_path: Path, cpu_spec: str) -> None:
    result, run_dir, capture = launch(tmp_path, cpu_spec=cpu_spec)
    # A zero-exit fake srun has produced no allocation evidence, so cannot pass.
    assert result.returncode == 1, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1
    argv, env = calls[0]["argv"], calls[0]["env"]
    for arg in (
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        "--cpus-per-task=352",
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
    for name, expected in {
        "MC_EGM_RECLAIM_POOL_SIZE": "250GiB",
        "MC_EGM_RECLAIM_CASES": "same-numa,cross-numa,reverse-gpu,chunked",
        "MC_EGM_RECLAIM_TIMEOUT": "60",
        "MC_EGM_RECLAIM_BUDGET": "600",
        "MC_EGM_RECLAIM_SAMPLE_INTERVAL": "1",
        "MC_EGM_RECLAIM_HOLD_SECONDS": "2",
        "MC_EGM_RECLAIM_CHUNK_SIZE": "16GiB",
        "MC_EGM_SETUP_TIMEOUT": "240",
        "MC_EGM_READY_TIMEOUT": "60",
        "MC_EGM_STEP_TIMEOUT": "960",
    }.items():
        assert env[name] == expected
    for source, copied in (
        (ROOT / "tools/mooncake_egm_reclaim.py", run_dir / "tools/mooncake_egm_reclaim.py"),
        (ROOT / "tools/mooncake_egm_alloc_diagnostics.py", run_dir / "tools/mooncake_egm_alloc_diagnostics.py"),
        (ROOT / "configs/mooncake_egm_donor.py", run_dir / "configs/mooncake_egm_donor.py"),
        (ROOT / "configs/setup_mooncake_egm.sh", run_dir / "configs/setup_mooncake_egm.sh"),
        (LAUNCHER, run_dir / "submitted.sbatch"),
    ):
        assert copied.read_bytes() == source.read_bytes()
    metadata = json.loads((run_dir / "logs/launch.json").read_text())
    assert metadata["wheel_sha256"] == hashlib.sha256(b"fixture").hexdigest()
    for phase in ("before", "after"):
        assert isinstance(json.loads((run_dir / f"logs/host-{phase}.json").read_text()), dict)
    report = (run_dir / "logs/perf-analysis.md").read_text()
    assert "INCOMPLETE" in report
    assert "TTFT p50/p95/p99" in report
    assert "Batch exit code: 1" in report


@pytest.mark.parametrize("cpu_spec", ["352,352", "352(x2)", "352garbage"])
def test_bad_cpu_allocation_keeps_report_and_never_starts_step(tmp_path: Path, cpu_spec: str) -> None:
    result, run_dir, capture = launch(tmp_path, cpu_spec=cpu_spec)
    assert result.returncode != 0
    assert not capture.exists()
    assert "SLURM_JOB_CPUS_PER_NODE" in result.stdout or "one" in result.stdout.lower()
    report = (run_dir / "logs/perf-analysis.md").read_text()
    assert "Batch exit code:" in report
    assert "INCOMPLETE" in report.upper()


def test_wrong_node_count_is_rejected_before_step(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, overrides={"SLURM_JOB_NUM_NODES": "2"})
    assert result.returncode == 2
    assert not capture.exists()
    assert "one allocated node" in result.stdout.lower()
    assert "Batch exit code: 2" in (run_dir / "logs/perf-analysis.md").read_text()


@pytest.mark.parametrize("wheel_count", [0, 2])
def test_missing_or_ambiguous_wheel_keeps_failure_report(tmp_path: Path, wheel_count: int) -> None:
    wheels = tmp_path / "selected-wheels"
    wheels.mkdir()
    for index in range(wheel_count):
        (wheels / f"mooncake_transfer_engine_cuda13-{index}.whl").touch()
    result, run_dir, capture = launch(tmp_path, overrides={"MC_EGM_WHEEL_DIR": str(wheels)})
    assert result.returncode == 2
    assert not capture.exists()
    assert f"found {wheel_count}" in result.stdout
    report = (run_dir / "logs/perf-analysis.md").read_text()
    assert "INCOMPLETE" in report
    assert "Batch exit code: 2" in report


def test_step_failure_keeps_its_code_and_refuses_to_overwrite_stale_results(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, srun_rc=19)
    assert result.returncode == 19, result.stdout + result.stderr
    report = run_dir / "logs/perf-analysis.md"
    original = report.read_bytes()
    assert "INCOMPLETE" in original.decode()
    assert "Batch exit code: 19" in original.decode()

    repeated, _, _ = launch(tmp_path)
    assert repeated.returncode == 2
    assert "Refusing stale" in repeated.stderr
    assert report.read_bytes() == original
    assert len(capture.read_text().splitlines()) == 1


@pytest.mark.parametrize(("diagnostic_rc", "interrupt"), [(0, False), (23, False), (0, True)])
def test_wrapper_scrubs_environment_and_stops_owned_diagnostic_before_master(
    tmp_path: Path, diagnostic_rc: int, interrupt: bool
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
        "    hold('master')\n"
        "elif sys.argv[1:4] == ['-m', 'tools.mooncake_egm_reclaim', 'inventory']:\n"
        "    Path(sys.argv[sys.argv.index('--output') + 1]).write_text('{}')\n"
        "    event('inventory')\n"
        "elif sys.argv[1:4] == ['-m', 'tools.mooncake_egm_reclaim', 'run']:\n"
        "    event('diagnostic')\n"
        "    if os.environ['WRAPPER_INTERRUPT'] == '1': hold('diagnostic')\n"
        "    raise SystemExit(int(os.environ['WRAPPER_DIAGNOSTIC_RC']))\n"
        "else: raise SystemExit('Unexpected Python invocation: ' + repr(sys.argv))\n",
    )
    executable(tmp_path / "bin/nvidia-smi", "#!/bin/bash\nexit 0\n")
    env = os.environ | {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "MC_EGM_SETUP_TIMEOUT": "1",
        "MC_EGM_READY_TIMEOUT": "1",
        "MC_EGM_RECLAIM_POOL_SIZE": "250GiB",
        "MC_EGM_RECLAIM_CASES": "same-numa,cross-numa,reverse-gpu,chunked",
        "MC_EGM_RECLAIM_TIMEOUT": "60",
        "MC_EGM_RECLAIM_BUDGET": "600",
        "MC_EGM_RECLAIM_SAMPLE_INTERVAL": "1",
        "MC_EGM_RECLAIM_HOLD_SECONDS": "2",
        "MC_EGM_RECLAIM_CHUNK_SIZE": "16GiB",
        "WRAPPER_CAPTURE": str(capture),
        "WRAPPER_DIAGNOSTIC_RC": str(diagnostic_rc),
        "WRAPPER_INTERRUPT": "1" if interrupt else "0",
        "PMIX_RANK": "9",
        "OMPI_COMM_WORLD_SIZE": "8",
        "MC_USE_NVLINK_IPC": "0",
        "MC_USE_TENT": "1",
        "MC_TE_FILTERS": "rdma",
        "MC_FORCE_TCP": "1",
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
            while not capture.exists() or '"name": "diagnostic-waiting"' not in capture.read_text():
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "Diagnostic never started"
                time.sleep(0.02)
            process.terminate()
        stdout, stderr = process.communicate(timeout=10)
    expected_rc = 143 if interrupt else diagnostic_rc
    assert process.returncode == expected_rc, stdout + stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    names = [call["name"] for call in calls]
    assert names.index("inventory") < names.index("master-ready") < names.index("diagnostic")
    diagnostic = next(call for call in calls if call["name"] == "diagnostic")
    assert diagnostic["args"] == [
        "-m",
        "tools.mooncake_egm_reclaim",
        "run",
        "--config",
        str(run_dir / "master-config.json"),
        "--output-dir",
        str(run_dir / "logs/reclaim"),
        "--pool-size",
        "250GiB",
        "--cases",
        "same-numa,cross-numa,reverse-gpu,chunked",
        "--timeout-seconds",
        "60",
        "--budget-seconds",
        "600",
        "--sample-interval",
        "1",
        "--hold-seconds",
        "2",
        "--chunk-size",
        "16GiB",
    ]
    for name in ("PMIX_RANK", "OMPI_COMM_WORLD_SIZE", "MC_USE_NVLINK_IPC", "MC_TE_FILTERS", "MC_FORCE_TCP"):
        assert name not in diagnostic["env"]
    assert diagnostic["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert diagnostic["env"]["MC_USE_TENT"] == "0"
    assert diagnostic["env"]["MOONCAKE_STORE_BACKEND"] == "cpp"
    assert diagnostic["env"]["SRT_MOONCAKE_EGM_WHEEL_DIR"] == "/egm-wheels"
    assert "master-stop" in names
    if interrupt:
        assert names.index("diagnostic-stop") < names.index("master-stop")
    assert f"exit_code={expected_rc} cleanup_failed=0" in (run_dir / "logs/shutdown.txt").read_text()
