"""keepalive_command: the srun task must outlive a time-windowed nsys session."""

import os
import shlex
import stat
import subprocess
import time

from srtctl.core.nsys_keepalive import keepalive_command


def test_wrapper_shape():
    cmd = keepalive_command(["nsys", "profile", "--delay", "5", "-o", "/logs/x y", "python3", "-m", "dynamo.frontend"])
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    assert script.startswith(shlex.join(["nsys", "profile", "--delay", "5", "-o", "/logs/x y", "python3", "-m", "dynamo.frontend"]) + " &")
    assert "pgrep -P" in script and 'wait "$NSYS"' in script and 'kill -0 "$APP"' in script
    assert script.endswith('exit "$rc"')


def _fake_nsys(tmp_path, *, exit_code: int, child_secs: float):
    """A stand-in for nsys: fork a child that lives child_secs, exit after 0.5 s with exit_code."""
    path = tmp_path / "nsys"
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"sleep {child_secs} &\n"
        "sleep 0.5\n"
        f"exit {exit_code}\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_task_outlives_nsys_and_keeps_its_exit_code(tmp_path):
    nsys = _fake_nsys(tmp_path, exit_code=0, child_secs=3)
    t0 = time.monotonic()
    proc = subprocess.run(keepalive_command([nsys]), capture_output=True, text=True, timeout=60)
    elapsed = time.monotonic() - t0
    assert proc.returncode == 0
    # nsys itself exits after 0.5 s; the wrapper must stay until the child (3 s) is gone
    assert elapsed >= 2.5, (elapsed, proc.stderr)
    assert "keeping task alive while pid" in proc.stderr


def test_nsys_failure_is_propagated(tmp_path):
    nsys = _fake_nsys(tmp_path, exit_code=3, child_secs=1)
    proc = subprocess.run(keepalive_command([nsys]), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 3


def test_no_child_degrades_to_plain_wait(tmp_path):
    path = tmp_path / "nsys"
    path.write_text("#!/usr/bin/env bash\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    t0 = time.monotonic()
    proc = subprocess.run(keepalive_command([str(path)]), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and time.monotonic() - t0 < 5
    assert os.environ is not None  # keep os imported for readers extending the fake
