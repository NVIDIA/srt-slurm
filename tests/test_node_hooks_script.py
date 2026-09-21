import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "configs" / "node-hooks.sh"


def run(phase: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    clean = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "HOOK_SNAPSHOT": "0"}
    clean.update(env or {})
    return subprocess.run(["bash", str(SCRIPT), phase, *args], env=clean, text=True, capture_output=True, check=False)


def test_rejects_unknown_phase():
    result = run("during")
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_runs_numbered_commands_in_numeric_order():
    result = run("pre", env={"HOOK_PRE_10": "echo tenth", "HOOK_PRE_2": "echo second", "HOOK_PRE_1": "echo first"})
    assert result.returncode == 0
    lines = [line for line in result.stdout.splitlines() if line in {"first", "second", "tenth"}]
    assert lines == ["first", "second", "tenth"]


def test_command_line_values_override_environment():
    result = run("pre", "HOOK_PRE_1=echo from-arg", env={"HOOK_PRE_1": "echo from-env"})
    assert result.returncode == 0
    assert "from-arg" in result.stdout
    assert "from-env" not in result.stdout.splitlines()


def test_pre_failure_stops_and_exits_nonzero_with_real_code():
    result = run("pre", env={"HOOK_PRE_1": "exit 3", "HOOK_PRE_2": "echo should-not-run"})
    assert result.returncode == 1
    assert "HOOK_PRE_1 exited 3" in result.stdout
    assert "should-not-run" not in result.stdout


def test_pre_lenient_mode_continues():
    result = run("pre", "HOOK_PRE_STRICT=0", env={"HOOK_PRE_1": "false", "HOOK_PRE_2": "echo continues"})
    assert result.returncode == 0
    assert "continues" in result.stdout


def test_post_failure_never_changes_exit_code():
    result = run("post", env={"HOOK_POST_1": "false", "HOOK_POST_2": "echo still-runs"})
    assert result.returncode == 0
    assert "HOOK_POST_1 exited 1" in result.stdout
    assert "still-runs" in result.stdout


def test_no_commands_is_a_noop():
    result = run("post")
    assert result.returncode == 0
    assert "commands=0" in result.stdout


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_ignores_non_hook_arguments(phase):
    result = run(phase, "FOO=bar")
    assert result.returncode == 0
    assert "ignoring argument without HOOK_ prefix: FOO=bar" in result.stderr
