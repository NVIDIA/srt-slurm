# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the standalone CUDA diagnostic launch without Slurm or a GPU."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/check_cuda_host_numa.sbatch"
BASH = shutil.which("bash") or "/bin/bash"


def executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def helper_source() -> str:
    """A stdlib-only controller fake; real controller behavior has its own tests."""
    return "\n".join(
        [
            "import json, os, signal, sys, time",
            "from pathlib import Path",
            "def argument(name): return sys.argv[sys.argv.index(name) + 1]",
            "def event(name):",
            "    if os.environ.get('WRAPPER_CAPTURE'):",
            "        with open(os.environ['WRAPPER_CAPTURE'], 'a') as stream:",
            "            stream.write(json.dumps({'name': name, 'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')",
            "if sys.argv[1] == 'inventory':",
            "    Path(argument('--output')).write_text(json.dumps({'inventory': 'fake'}))",
            "elif sys.argv[1] == 'report':",
            "    run_dir = Path(argument('--run-dir'))",
            "    summary = run_dir / 'logs/probe/summary.json'",
            "    complete = summary.exists() and json.loads(summary.read_text()).get('complete')",
            "    (run_dir / 'logs/perf-analysis.md').write_text(('COMPLETE' if complete else 'INCOMPLETE') + '\\n')",
            "    raise SystemExit(0 if complete else 1)",
            "elif sys.argv[1] == 'run':",
            "    event('run')",
            "    if os.environ.get('WRAPPER_INTERRUPT') == '1':",
            "        def stop(signum, frame):",
            "            event('run-stopped')",
            "            raise SystemExit(0)",
            "        signal.signal(signal.SIGTERM, stop)",
            "        event('run-waiting')",
            "        while True: time.sleep(0.02)",
            "    raise SystemExit(int(os.environ.get('WRAPPER_DIAGNOSTIC_RC', '0')))",
            "else: raise SystemExit('Unexpected invocation: ' + repr(sys.argv))",
            "",
        ]
    )


def launch(
    tmp_path: Path,
    *,
    cpu_spec: str = "352",
    srun_rc: int = 0,
    complete: bool = False,
    overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    source = tmp_path / "tools"
    source.mkdir(exist_ok=True)
    (source / "cuda_host_numa_probe.py").write_text(helper_source())
    (source / "cuda_host_numa_repro.cpp").write_text("// fake native source; no GPU used\n")
    (source / "check_cuda_host_numa.md").write_text("Standalone diagnostic README\n")
    capture = tmp_path / "srun.jsonl"
    executable(
        fake_bin / "srun",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['SRUN_CAPTURE'], 'a') as output:\n"
        "    output.write(json.dumps({'argv': sys.argv[1:], 'env': "
        "{k: v for k, v in os.environ.items() if k.startswith('CUDA_EGM_') or k == 'ENROOT_MAX_PROCESSORS'}}) + '\\n')\n"
        "if os.environ['FAKE_COMPLETE'] == '1':\n"
        "    summary = Path(os.environ['SLURM_SUBMIT_DIR']) / 'outputs/cuda-host-numa-123/logs/probe/summary.json'\n"
        "    summary.parent.mkdir(parents=True)\n"
        "    summary.write_text(json.dumps({'complete': True}))\n"
        "raise SystemExit(int(os.environ['FAKE_SRUN_RC']))\n",
    )
    executable(fake_bin / "scontrol", "#!/bin/bash\nprintf 'JobState=RUNNING\\n'\n")
    executable(fake_bin / "timeout", '#!/bin/bash\nwhile [[ $1 == -* ]]; do shift; done\nshift\nexec "$@"\n')
    if not (fake_bin / "python3").exists():
        (fake_bin / "python3").symlink_to(sys.executable)
    env = {key: value for key, value in os.environ.items() if not key.startswith("CUDA_EGM_")}
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        SLURM_SUBMIT_DIR=str(tmp_path),
        SLURM_JOB_ID="123",
        SLURM_JOB_NUM_NODES="1",
        SLURM_NNODES="1",
        SLURM_JOB_CPUS_PER_NODE=cpu_spec,
        SLURM_JOB_NODELIST="node0",
        CUDA_EGM_IMAGE=str(tmp_path / "cuda-devel.sqsh"),
        SRUN_CAPTURE=str(capture),
        FAKE_SRUN_RC=str(srun_rc),
        FAKE_COMPLETE="1" if complete else "0",
    )
    env.update(overrides or {})
    result = subprocess.run(
        [BASH, str(LAUNCHER)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=20, check=False
    )
    return result, tmp_path / "outputs/cuda-host-numa-123", capture


def test_shell_syntax_and_allocation_resources() -> None:
    subprocess.run([BASH, "-n", str(LAUNCHER)], check=True)
    script = LAUNCHER.read_text()
    wrapper = script.split("<<'WRAPPER'\n", 1)[1].split("\nWRAPPER\n", 1)[0]
    subprocess.run([BASH, "-n"], input=wrapper, text=True, check=True)
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH")]
    for directive in (
        "#SBATCH --nodes=1",
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --exclusive",
        "#SBATCH --mem=0",
        "#SBATCH --no-requeue",
        "#SBATCH --time=00:20:00",
        "#SBATCH --signal=B:USR1@180",
    ):
        assert directive in directives
    assert not any(any(option in line for option in ("--account", "--partition", "--nodelist")) for line in directives)
    for unexpected in ("pip install", "mooncake", "drop_caches", "sudo ", "--gpu-reset", "pkill", "killall"):
        assert unexpected not in script


@pytest.mark.parametrize("cpu_spec", ["352", "352(x1)"])
def test_one_step_gets_all_cpus_and_only_standalone_sources(tmp_path: Path, cpu_spec: str) -> None:
    result, run_dir, capture = launch(tmp_path, cpu_spec=cpu_spec)
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
        "--container-workdir=/cuda-egm-run",
        f"--container-image={tmp_path / 'cuda-devel.sqsh'}",
        f"--container-mounts={run_dir}:/cuda-egm-run",
    ):
        assert arg in argv
    assert argv[-3:] == ["bash", "/cuda-egm-run/launch-node.sh", "/cuda-egm-run"]
    for name, expected in {
        "CUDA_EGM_SIZE": "250GiB",
        "CUDA_EGM_CASES": "same-numa,cross-numa,reverse-gpu,chunked",
        "CUDA_EGM_ALLOCATION_TIMEOUT": "60",
        "CUDA_EGM_BUDGET": "600",
        "CUDA_EGM_SAMPLE_INTERVAL": "1",
        "CUDA_EGM_HOLD_SECONDS": "5",
        "CUDA_EGM_CHUNK_SIZE": "16GiB",
        "CUDA_EGM_BUILD_TIMEOUT": "120",
        "CUDA_EGM_STEP_TIMEOUT": "960",
    }.items():
        assert env[name] == expected
    assert sorted(path.name for path in (run_dir / "tools").iterdir()) == [
        "cuda_host_numa_probe.py",
        "cuda_host_numa_repro.cpp",
    ]
    for name in ("cuda_host_numa_probe.py", "cuda_host_numa_repro.cpp"):
        assert (run_dir / "tools" / name).read_bytes() == (tmp_path / "tools" / name).read_bytes()
    assert (run_dir / "submitted.sbatch").read_bytes() == LAUNCHER.read_bytes()
    assert (run_dir / "README.md").read_text() == "Standalone diagnostic README\n"
    metadata = json.loads((run_dir / "logs/launch.json").read_text())
    assert (
        metadata["source_sha256"]["cuda_host_numa_repro.cpp"]
        == hashlib.sha256((tmp_path / "tools/cuda_host_numa_repro.cpp").read_bytes()).hexdigest()
    )
    for phase in ("before", "after"):
        assert json.loads((run_dir / f"logs/host-{phase}.json").read_text()) == {"inventory": "fake"}
    assert "INCOMPLETE" in (run_dir / "logs/perf-analysis.md").read_text()


def test_host_mode_has_no_container_dependency(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, complete=True, overrides={"CUDA_EGM_IMAGE": "host"})
    assert result.returncode == 0, result.stdout + result.stderr
    argv = json.loads(capture.read_text())["argv"]
    assert not any("container" in item for item in argv)
    assert argv[-3:] == ["bash", str(run_dir / "launch-node.sh"), str(run_dir)]
    assert "Batch exit code: 0" in (run_dir / "logs/perf-analysis.md").read_text()


@pytest.mark.parametrize("cpu_spec", ["352,352", "352(x2)", "352garbage"])
def test_bad_cpu_allocation_preserves_report_without_starting_step(tmp_path: Path, cpu_spec: str) -> None:
    result, run_dir, capture = launch(tmp_path, cpu_spec=cpu_spec)
    assert result.returncode != 0
    assert not capture.exists()
    report = (run_dir / "logs/perf-analysis.md").read_text()
    assert "incomplete" in report.lower()
    assert "TTFT p50/p95/p99" in report
    assert "Batch exit code:" in report


def test_wrong_node_count_never_starts_step(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, overrides={"SLURM_JOB_NUM_NODES": "2"})
    assert result.returncode == 2
    assert not capture.exists()
    assert "one allocated node" in result.stdout.lower()
    assert "Batch exit code: 2" in (run_dir / "logs/perf-analysis.md").read_text()


def test_step_failure_keeps_its_code_and_stale_results_are_not_overwritten(tmp_path: Path) -> None:
    result, run_dir, capture = launch(tmp_path, srun_rc=19)
    assert result.returncode == 19, result.stdout + result.stderr
    report = run_dir / "logs/perf-analysis.md"
    original = report.read_bytes()
    assert "Batch exit code: 19" in original.decode()
    repeated, _, _ = launch(tmp_path)
    assert repeated.returncode == 2
    assert "Refusing stale" in repeated.stderr
    assert report.read_bytes() == original
    assert len(capture.read_text().splitlines()) == 1


@pytest.mark.parametrize(("diagnostic_rc", "interrupt"), [(0, False), (23, False), (0, True)])
def test_wrapper_builds_driver_only_source_and_cleans_up_owned_process(
    tmp_path: Path, diagnostic_rc: int, interrupt: bool
) -> None:
    result, run_dir, _ = launch(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    fake_bin = tmp_path / "bin"
    include = tmp_path / "include"
    include.mkdir()
    (include / "cuda.h").write_text("// fixture\n")
    library = tmp_path / "libcuda.so.1"
    library.touch()
    executable(
        fake_bin / "g++",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if '--version' in sys.argv:\n"
        "    print('fake compiler')\n"
        "else:\n"
        "    Path(os.environ['CXX_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "    output = Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "    output.write_text('#!/bin/bash\\nexit 0\\n')\n"
        "    output.chmod(0o755)\n",
    )
    executable(fake_bin / "ldd", "#!/bin/bash\nprintf 'libcuda.so.1 => driver library\\n'\n")
    executable(fake_bin / "nvidia-smi", "#!/bin/bash\nexit 0\n")
    executable(fake_bin / "dmesg", "#!/bin/bash\nprintf 'dmesg: permission denied\\n' >&2\nexit 1\n")
    capture = tmp_path / "wrapper-events.jsonl"
    env = os.environ | {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CUDA_EGM_CXX": str(fake_bin / "g++"),
        "CUDA_EGM_INCLUDE": str(include),
        "CUDA_EGM_LIBCUDA": str(library),
        "CUDA_EGM_BUILD_TIMEOUT": "120",
        "CUDA_EGM_SIZE": "250GiB",
        "CUDA_EGM_CASES": "same-numa,cross-numa,reverse-gpu,chunked",
        "CUDA_EGM_ALLOCATION_TIMEOUT": "60",
        "CUDA_EGM_BUDGET": "600",
        "CUDA_EGM_SAMPLE_INTERVAL": "1",
        "CUDA_EGM_HOLD_SECONDS": "5",
        "CUDA_EGM_CHUNK_SIZE": "16GiB",
        "PMIX_RANK": "17",
        "OMPI_COMM_WORLD_RANK": "17",
        "SLURM_JOB_ID": "123",
        "WRAPPER_CAPTURE": str(capture),
        "WRAPPER_DIAGNOSTIC_RC": str(diagnostic_rc),
        "WRAPPER_INTERRUPT": "1" if interrupt else "0",
        "CXX_CAPTURE": str(tmp_path / "compiler.json"),
    }
    process = subprocess.Popen(
        [BASH, str(run_dir / "launch-node.sh"), str(run_dir)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if interrupt:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if capture.exists() and '"run-waiting"' in capture.read_text():
                    break
                time.sleep(0.02)
            else:
                pytest.fail("Fake diagnostic did not reach the retained-allocation stage")
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    assert process.returncode == (143 if interrupt else diagnostic_rc), stdout + stderr
    events = [json.loads(line) for line in capture.read_text().splitlines()]
    run = next(event for event in events if event["name"] == "run")
    assert run["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert run["env"]["SLURM_JOB_ID"] == "123"
    assert "PMIX_RANK" not in run["env"]
    assert "OMPI_COMM_WORLD_RANK" not in run["env"]
    if interrupt:
        assert "run-stopped" in [event["name"] for event in events]
    assert "cleanup_failed=0" in (run_dir / "logs/shutdown.txt").read_text()
    assert "permission denied" in (run_dir / "logs/kernel-after.txt").read_text()
    assert (run_dir / "logs/container-after.json").exists()
    command = json.loads((tmp_path / "compiler.json").read_text())
    assert "-std=c++17" in command
    assert str(library) == command[-1]
    assert str(run_dir / "tools/cuda_host_numa_repro.cpp") in command
    assert "nvcc" not in command
    assert not any("stubs" in item for item in run["env"].get("LD_LIBRARY_PATH", "").split(os.pathsep))
