# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSH transport contracts with a local fake SSH executable and real detached processes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from srtctl.cli import remote, submit
from srtctl.core.config import load_cluster_config
from srtctl.core.schema import ClusterConfig, SSHConfig
from srtctl.runtime_scripts import ssh_job


def executable(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o755)
    return path


def wait_for_state(client: remote.SSHClient, job_id: str, expected: tuple[str, ...]) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.run(client.request("status", job_id))
        assert isinstance(result, dict)
        if result["state"] in expected:
            return result
        time.sleep(0.05)
    pytest.fail(f"job did not reach {expected}: {result}")


FAKE_UV = """\
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
recipe = Path(args[args.index("--file") + 1].split(":")[0])
spec = json.loads(recipe.read_text())
cluster = json.loads(Path(os.environ["SRTSLURM_CONFIG"]).read_text())
print(json.dumps({"args": args, "cwd": os.getcwd(), "cluster": cluster,
    "venv": os.environ.get("VIRTUAL_ENV"), "path": os.environ["PATH"]}), flush=True)
if spec.get("schema") == 2:
    from srtctl.mock import MockOptions, run_mock_sweep
    def capture(call):
        print("mock-step=" + str(call.get("step_name")), flush=True)
    sys.exit(run_mock_sweep(config_path=recipe, output_dir=recipe.parent / "mock-run",
        job_id="4242", options=MockOptions(on_srun=capture)))
if spec.get("sleep"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    print("child=" + str(child.pid), flush=True)
    try:
        child.wait()
    except KeyboardInterrupt:
        child.wait(timeout=5)
        print("graceful shutdown", flush=True)
sys.exit(spec.get("exit_code", 0))
"""


@pytest.fixture
def target(tmp_path, monkeypatch):
    """Execute the exact remote helper locally, with no SSH server, GPU or Slurm."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    executable(
        bindir / "ssh",
        "import os, shlex, sys\n"
        "assert '-T' in sys.argv and 'BatchMode=yes' in sys.argv\n"
        "assert 'StrictHostKeyChecking=yes' in sys.argv\n"
        "command = shlex.split(sys.argv[-1])\n"
        "assert command.pop(0) == 'exec'\n"
        "os.execv(command[0], command)\n",
    )
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    workdir = tmp_path / "remote checkout's files"
    host = workdir / "src" / "srtctl" / "core" / "host.py"
    host.parent.mkdir(parents=True)
    host.touch()
    uv = executable(bindir / "fake uv", FAKE_UV)
    venv = tmp_path / "engine env"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python3").symlink_to(sys.executable)
    config = SSHConfig(host="gpu-box", workdir=str(workdir), python=sys.executable, uv=str(uv), venv=str(venv))
    client = remote.SSHClient(config)
    jobs: list[str] = []

    def launch(**recipe):
        request = client.request("submit", "ssh-" + uuid.uuid4().hex)
        request["recipe"] = json.dumps(recipe)
        request["cluster_config"] = {"launcher": "host"}
        result = client.run(request)
        assert isinstance(result, dict)
        jobs.append(result["job_id"])
        return result

    yield client, launch
    for job_id in jobs:
        state = client.run(client.request("status", job_id))
        if state["state"] not in ssh_job.TERMINAL_STATES:
            client.run(client.request("stop", job_id))
            wait_for_state(client, job_id, ssh_job.TERMINAL_STATES)


@pytest.mark.parametrize("launcher", ["slurm", "docker"])
def test_ssh_requires_capable_launcher(launcher):
    with pytest.raises(ValueError, match="ssh requires launcher: host"):
        ClusterConfig.Schema().load({"launcher": launcher, "ssh": {"host": "gpu", "workdir": "/repo"}})


@pytest.mark.parametrize(
    "settings",
    [
        {"host": "-oProxyCommand=bad", "workdir": "/repo"},
        {"host": "gpu box", "workdir": "/repo"},
        {"host": "gpu", "workdir": "relative/repo"},
        {"host": "gpu", "workdir": "/repo", "venv": "relative/env"},
    ],
)
def test_invalid_ssh_config_never_falls_back_to_local(settings, tmp_path, monkeypatch):
    config = tmp_path / "srtslurm.yaml"
    config.write_text(yaml.safe_dump({"launcher": "host", "ssh": settings}))
    monkeypatch.setenv("SRTSLURM_CONFIG", str(config))
    with pytest.raises(ValueError, match="Invalid SSH cluster config"):
        load_cluster_config()


def test_ssh_uses_passwordless_auth_and_existing_known_hosts():
    command = remote.SSHClient(SSHConfig(host="user@gpu-box", workdir="~/srt-slurm")).command()
    assert command[command.index("--") + 1] == "user@gpu-box"
    assert "BatchMode=yes" in command
    assert "StrictHostKeyChecking=yes" in command


def test_detached_submit_stages_recipe_and_remote_config(target):
    client, launch = target
    state = launch()
    state = wait_for_state(client, state["job_id"], ("succeeded",))
    root = Path(state["output_dir"])
    assert root.stat().st_mode & 0o777 == 0o700
    assert json.loads((root / "recipe.yaml").read_text()) == {}
    record = json.loads((root / "apply.log").read_text().splitlines()[0])
    assert record["cwd"] == client.config.workdir
    assert "ssh" not in record["cluster"]
    assert record["cluster"]["srtctl_root"] == client.config.workdir
    assert record["venv"] == client.config.venv
    assert record["path"].split(os.pathsep)[0] == str(Path(client.config.venv) / "bin")
    assert "--yes" in record["args"] and "--json" in record["args"]
    assert state["exit_code"] == 0


def test_transferred_recipe_runs_a_host_sweep_through_the_mock_orchestrator(target):
    client, launch = target
    recipe = Path(__file__).parents[1] / "examples" / "host" / "vllm-agg-1gpu.yaml"
    state = launch(**yaml.safe_load(recipe.read_text()))
    state = wait_for_state(client, state["job_id"], ("succeeded", "failed"))
    log = Path(state["log_file"]).read_text()
    assert state["state"] == "succeeded", log
    assert "mock-step=agg_0_" in log
    assert "mock-step=benchmark" in log


def test_job_survives_submit_connection_and_can_be_stopped_later(target):
    client, launch = target
    state = launch(sleep=True)
    # The submit SSH process has exited. A second SSH connection can still see and stop the job.
    state = wait_for_state(client, state["job_id"], ("running",))
    root = Path(state["output_dir"])
    deadline = time.monotonic() + 5
    while "child=" not in (root / "apply.log").read_text() and time.monotonic() < deadline:
        time.sleep(0.05)
    listing = client.run(client.request("list"))
    assert any(item["job_id"] == state["job_id"] for item in listing)
    child_pid = int((root / "apply.log").read_text().split("child=")[1].splitlines()[0])
    os.kill(child_pid, 0)
    response = client.run(client.request("stop", state["job_id"]))
    assert response["stop_requested"]
    state = wait_for_state(client, state["job_id"], ("stopped",))
    assert "graceful shutdown" in (root / "apply.log").read_text()
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_failure_preserves_exit_code_and_logs(target, capfd):
    client, launch = target
    state = launch(exit_code=7)
    state = wait_for_state(client, state["job_id"], ("failed",))
    assert state["exit_code"] == 7
    client.run(client.request("logs", state["job_id"]))
    assert '"cwd":' in capfd.readouterr().out


def test_interrupting_log_follow_leaves_the_job_running(target):
    client, launch = target
    state = launch(sleep=True)
    follower = subprocess.Popen(
        client.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            follower.communicate(json.dumps(client.request("logs", state["job_id"], follow=True)), timeout=0.5)
        follower.send_signal(signal.SIGINT)
        follower.communicate(timeout=5)
        current = client.run(client.request("status", state["job_id"]))
        assert current["state"] == "running" and not current["stop_requested"]
    finally:
        if follower.poll() is None:
            follower.kill()
        follower.communicate(timeout=5)


def test_stale_state_is_unknown_and_never_signals_a_saved_pid(tmp_path):
    directory = tmp_path / "outputs" / "ssh" / ("ssh-" + "a" * 32)
    directory.mkdir(parents=True)
    state = {"state": "running", "updated_at": time.time() - 60, "pid": os.getpid()}
    (directory / "status.json").write_text(json.dumps(state))
    request = remote.SSHClient(SSHConfig(host="gpu", workdir=str(tmp_path))).request("stop", directory.name)
    with patch.object(ssh_job.os, "killpg") as killpg:
        assert ssh_job.read_state(directory)["state"] == "unknown"
        with pytest.raises(RuntimeError, match="heartbeat is stale"):
            ssh_job.handle(request)
        killpg.assert_not_called()


@pytest.mark.parametrize("job_id", ["../other-job", "123", "ssh-../../x", "ssh-" + "g" * 32])
def test_control_rejects_invalid_job_ids(job_id):
    with pytest.raises(ValueError, match="job ID"):
        ssh_job.job_directory("/repo", job_id)


def test_missing_remote_checkout_fails_before_staging(tmp_path):
    client = remote.SSHClient(SSHConfig(host="gpu", workdir=str(tmp_path)))
    request = client.request("submit", "ssh-" + "a" * 32)
    with pytest.raises(ValueError, match="checkout"):
        ssh_job.submit(request)
    assert not (tmp_path / "outputs").exists()


def test_cli_dispatches_before_local_path_checks_and_preserves_arguments(target, tmp_path, monkeypatch, capsys):
    client, _launch = target
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("{}")
    config = tmp_path / "cluster.yaml"
    settings = {"launcher": "host", "ssh": SSHConfig.Schema().dump(client.config)}
    config.write_text(yaml.safe_dump(settings))
    monkeypatch.setenv("SRTSLURM_CONFIG", str(config))
    value = "name=do not expand $(touch /no-such-marker)"
    monkeypatch.setattr(
        sys, "argv", ["srtctl", "apply", "-f", str(recipe) + ":override_variant", "--json", "--set", value]
    )
    with (
        patch.object(submit, "materialize_config_path", side_effect=AssertionError("must resolve remotely")),
        pytest.raises(SystemExit) as result,
    ):
        submit.main()
    assert result.value.code == 0
    state = json.loads(capsys.readouterr().out)
    state = wait_for_state(client, state["job_id"], ("succeeded",))
    record = json.loads(Path(state["log_file"]).read_text().splitlines()[0])
    assert record["args"][record["args"].index("--set") + 1] == value
    assert record["args"][record["args"].index("--file") + 1].endswith("recipe.yaml:override_variant")
