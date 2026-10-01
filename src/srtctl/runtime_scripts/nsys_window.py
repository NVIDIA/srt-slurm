# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acknowledged workload windows for nsys; standalone inside serving/client images.

Workers launch the app under an inactive named nsys session. The benchmark sends
start/stop requests through the shared log mount and waits for every profiler's
acknowledgment. No benchmark or GPU-framework imports are needed here.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

POLL_SECS = 0.1


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def expected_participants(root: Path) -> set[str]:
    return {
        f"{path.stem}-{rank}" for path in (root / "steps").glob("*.json") for rank in range(read_json(path)["ranks"])
    }


def _wait_for_acks(root: Path, directory: Path, expected: set[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        errors = list((root / "failed").glob("*.json"))
        if errors:
            raise RuntimeError(f"nsys participant failed: {read_json(errors[0])}")
        missing = []
        for participant in expected:
            path = directory / f"{participant}.json"
            if not path.exists():
                missing.append(participant)
            elif not read_json(path)["ok"]:
                raise RuntimeError(f"nsys request failed: {read_json(path)}")
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"nsys acknowledgment timed out; missing {missing}")
        time.sleep(POLL_SECS)


def request(root: Path, action: str, timeout: float = 1800) -> None:
    """Start after warmup, or stop after measured traffic; returns after all acks.

    A new report is produced for each start/stop pair. Start is never implicit:
    an opaque custom benchmark must call this at its actual phase boundary.
    """
    if action not in {"start", "stop"}:
        raise ValueError(f"Unknown nsys window action: {action}")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "client.lock").open("a") as lock:
        # Do not wait forever on a second client driving the same capture.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "client.json"
        state = read_json(state_path) if state_path.exists() else {"sequence": 0, "window": 0, "active": False}
        if action == "stop" and not state["active"]:
            return
        if action == "start" and state["active"]:
            raise RuntimeError("nsys capture is already active; stop it before starting another window")
        expected = expected_participants(root)
        if not expected:
            raise RuntimeError("No nsys participants were configured")
        _wait_for_acks(root, root / "ready", expected, timeout)
        if action == "start":
            state["window"] += 1
            state["active"] = True
        state["sequence"] += 1
        # Persist active before publishing: interrupted clients can be detected.
        write_json(state_path, state)
        request_id = f"{state['sequence']:06d}"
        write_json(root / "requests" / f"{request_id}.json", {"action": action, "window": state["window"]})
        _wait_for_acks(root, root / "acks" / request_id, expected, timeout)
        if action == "stop":
            state["active"] = False
            state["completed"] = state.get("completed", 0) + 1
            write_json(state_path, state)


def finish(root: Path, timeout: float) -> None:
    """Require a completed capture; flush an unfinished one, then report failure."""
    state_path = root / "client.json"
    if not state_path.exists():
        raise RuntimeError("No nsys workload window was recorded. Custom clients must call the start/stop hooks.")
    state = read_json(state_path)
    if state["active"]:
        request(root, "stop", timeout)
        raise RuntimeError("Benchmark exited with an active nsys window; the stop hook was missing or interrupted")
    if not state.get("completed"):
        raise RuntimeError("No nsys workload window completed")
    errors = list((root / "failed").glob("*.json"))
    if errors:
        raise RuntimeError(f"nsys participant failed: {read_json(errors[0])}")


def _descendants(pid: int) -> set[int]:
    """Snapshot only this profiler's descendants for bounded application cleanup."""
    children: set[int] = set()
    # Nsight's launcher is forked by a helper thread, not the main thread.
    # Linux exposes children separately for each thread in /proc.
    for path in Path(f"/proc/{pid}/task").glob("*/children"):
        with suppress(FileNotFoundError, ProcessLookupError):
            children.update(int(child) for child in path.read_text().split())
    result = set(children)
    for child in children:
        result.update(_descendants(child))
    return result


def _signal(pids: set[int], sig: int) -> None:
    for pid in pids:
        with suppress(ProcessLookupError):
            os.kill(pid, sig)


def worker(spec: dict[str, Any], command: list[str]) -> int:
    """Own one rank's named interactive session for the application's lifetime."""
    root = Path(spec["control_dir"])
    rank = os.environ.get("SLURM_PROCID", "0")
    participant = f"{spec['step']}-{rank}"
    session = f"srtctl-{participant}-{os.getpid()}"
    nsys = spec["nsys"]
    output = spec["output"].replace("%q{SLURM_PROCID}", rank)
    timeout = spec["timeout"]
    stopping = False
    active = False
    failed = False
    sequence = 0
    report: Path | None = None

    def stop_requested(_sig: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    def control(action: str, *options: str) -> None:
        subprocess.run(
            [nsys, action, f"--session={session}", *options],
            check=True,
            timeout=min(timeout, 10) if action == "shutdown" else timeout,
        )

    def stop_capture() -> None:
        control("stop")
        if report is None or not report.is_file() or not report.stat().st_size:
            raise RuntimeError(f"nsys stop did not produce a nonempty report: {report}")

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    try:
        proc = subprocess.Popen(
            [nsys, "launch", f"--session-new={session}", "--trace=nvtx", "--show-output=true", *command],
            start_new_session=True,
        )
    except OSError as exc:
        write_json(root / "failed" / f"{participant}.json", {"error": str(exc)})
        print(f"[srtctl] nsys launch failed: {exc}", file=sys.stderr, flush=True)
        return 1
    try:
        # The application child proves launch has initialized the named session.
        deadline = time.monotonic() + timeout
        while not _descendants(proc.pid) and proc.poll() is None and not stopping:
            if time.monotonic() >= deadline:
                raise TimeoutError("nsys did not launch the application")
            time.sleep(POLL_SECS)
        if stopping:
            raise RuntimeError("nsys was stopped before application readiness")
        if proc.poll() is not None:
            raise RuntimeError(f"nsys launch exited before readiness ({proc.returncode})")
        write_json(root / "ready" / f"{participant}.json", {"ok": True, "session": session})
        while proc.poll() is None and not stopping:
            pending = sorted((root / "requests").glob("*.json"))
            for path in pending:
                if stopping:
                    break
                if int(path.stem) <= sequence:
                    continue
                event = read_json(path)
                try:
                    if event["action"] == "start":
                        if active:
                            raise RuntimeError("nsys received start while already capturing")
                        report_base = f"{output}_window{event['window']:03d}"
                        report = Path(f"{report_base}.nsys-rep")
                        control("start", *spec["start_args"], f"--output={report_base}", "--force-overwrite=true")
                        active = True
                    else:
                        if not active:
                            raise RuntimeError("nsys received stop without an active capture")
                        stop_capture()
                        active = False
                    write_json(root / "acks" / path.stem / f"{participant}.json", {"ok": True})
                    sequence = int(path.stem)
                except Exception as exc:
                    write_json(root / "acks" / path.stem / f"{participant}.json", {"ok": False, "error": str(exc)})
                    raise
            time.sleep(POLL_SECS)
        if not stopping:
            raise RuntimeError(f"Profiled application exited before teardown ({proc.returncode})")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        failed = True
        print(f"[srtctl] nsys: {exc}", file=sys.stderr, flush=True)
        write_json(root / "failed" / f"{participant}.json", {"error": str(exc)})
    finally:
        # On cancellation, finish active reports before any rank exits. Normal
        # workload stop has already exported them while applications stayed up.
        deadline = time.monotonic() + timeout
        if active:
            try:
                stop_capture()
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                failed = True
                write_json(root / "failed" / f"{participant}.json", {"error": str(exc)})
                print(f"[srtctl] nsys stop failed: {exc}", file=sys.stderr, flush=True)
        stopped = root / "teardown" / spec["step"]
        write_json(stopped / f"{participant}.json", {"ok": not failed})
        expected = {f"{spec['step']}-{index}" for index in range(spec["ranks"])}
        try:
            _wait_for_acks(root, stopped, expected, max(0, deadline - time.monotonic()))
        except (RuntimeError, TimeoutError) as exc:
            failed = True
            print(f"[srtctl] nsys teardown barrier: {exc}", file=sys.stderr, flush=True)
        tree = _descendants(proc.pid)
        try:
            control("shutdown", "--kill=sigterm")
        except (OSError, subprocess.SubprocessError):
            _signal(tree, signal.SIGTERM)
        try:
            proc.wait(timeout=spec.get("app_grace_secs", 120))
        except subprocess.TimeoutExpired:
            _signal(tree | _descendants(proc.pid), signal.SIGKILL)
            proc.kill()
            proc.wait(timeout=5)
        # nsys shutdown may detach an application that ignores TERM.
        _signal(tree, signal.SIGKILL)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "worker"))
    parser.add_argument("--spec")
    args, command = parser.parse_known_args()
    if args.action == "worker":
        if command[:1] == ["--"]:
            command = command[1:]
        if not args.spec or not command:
            parser.error("worker requires --spec and an application command")
        return worker(json.loads(args.spec), command)
    root = os.environ.get("SRT_NSYS_CONTROL_DIR")
    if root:
        request(Path(root), args.action, float(os.environ.get("SRT_NSYS_CONTROL_TIMEOUT", "1800")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
