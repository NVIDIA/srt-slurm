"""keepalive_command: the srun task must outlive a time-windowed nsys session."""

import os
import shlex
import signal
import stat
import subprocess
import time

from srtctl.core.nsys_keepalive import keepalive_command


def test_wrapper_shape():
    cmd = keepalive_command(["nsys", "profile", "--delay", "5", "-o", "/logs/x y", "python3", "-m", "dynamo.frontend"])
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    launch = shlex.join(["nsys", "profile", "--delay", "5", "-o", "/logs/x y", "python3", "-m", "dynamo.frontend"])
    assert f"setsid {launch} &" in script and f"else {launch} &" in script
    assert "pgrep -P" in script and 'wait "$NSYS"' in script and 'kill -0 "$APP"' in script
    assert "trap fwd TERM INT" in script and 'kill -TERM "$APP"' in script
    assert script.endswith('exit "$rc"')


def _fake_nsys(tmp_path, *, exit_code: int, child_secs: float):
    """A stand-in for nsys: fork a child that lives child_secs, exit after 0.5 s with exit_code."""
    path = tmp_path / "nsys"
    path.write_text("#!/usr/bin/env bash\n" f"sleep {child_secs} &\n" "sleep 0.5\n" f"exit {exit_code}\n")
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


def test_sigterm_stops_app_and_lets_nsys_finish(tmp_path):
    """Teardown path: SIGTERM to the wrapper must stop the app only; nsys (waiting on it) then exits normally."""
    path = tmp_path / "nsys"
    marker = tmp_path / "report_written"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "sleep 300 &\n"  # the profiled app
        "wait $!\n"  # nsys --wait all: block until the app is gone ...
        f"touch {marker}\n"  # ... then write the report
        "exit 0\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    proc = subprocess.Popen(keepalive_command([str(path)]), stderr=subprocess.PIPE, text=True)
    time.sleep(2.0)  # let the wrapper find the app pid
    t0 = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    assert time.monotonic() - t0 < 30, "wrapper should return as soon as the app dies and nsys finishes"
    assert marker.exists(), "nsys must have survived the teardown long enough to write its report"
    assert "SIGTERM: stopping profiled app" in err


def test_stubborn_app_tree_is_killed_after_app_grace(tmp_path):
    """The app ignores SIGTERM (like the TRT-LLM MPI ranks); after app_exit_grace_secs the wrapper kills the app tree,
    nsys (waiting on it) then finalises and exits 0."""
    path = tmp_path / "nsys"
    marker = tmp_path / "report_written"
    app = tmp_path / "app.sh"
    app.write_text("#!/usr/bin/env bash\ntrap '' TERM\nsleep 300\n")  # ignores SIGTERM
    app.chmod(app.stat().st_mode | stat.S_IXUSR)
    path.write_text("#!/usr/bin/env bash\n" f"{app} &\n" "wait $!\n" f"touch {marker}\n" "exit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    proc = subprocess.Popen(keepalive_command([str(path)], app_exit_grace_secs=2), stderr=subprocess.PIPE, text=True)
    time.sleep(2.0)
    t0 = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=120)
    elapsed = time.monotonic() - t0
    assert proc.returncode == 0, err
    assert marker.exists(), err
    assert 2 <= elapsed < 60, (elapsed, err)  # grace 2 s + TERM (ignored) + 20 s + KILL
    assert "app tree still alive 2s after SIGTERM" in err
