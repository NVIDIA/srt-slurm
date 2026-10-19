# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded cleanup failures retain ownership and the attempted matrix cell."""

from __future__ import annotations

import io
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tools import check_mooncake_rails as controller
from tools import mooncake_rail_worker as worker


class StubbornProcess:
    pid = 12345

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.waits: list[int] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: int) -> int:
        self.waits.append(timeout)
        raise subprocess.TimeoutExpired(["synthetic-perftest"], timeout)


def active_worker(tmp_path: Path, *, pending_ready: bool = False) -> tuple[worker.Worker, StubbornProcess, io.StringIO]:
    agent = worker.Worker(tmp_path, "donor", {})
    process, stream = StubbornProcess(), io.StringIO()
    agent.child = {
        "process": process,
        "stream": stream,
        "command_id": "000001",
        "case_id": "read-mlx5_0-mlx5_0",
        "pid": process.pid,
        "port": 18515,
        "argv": ["synthetic-perftest"],
        "log_path": "logs/cases/read-mlx5_0-mlx5_0.server.log",
        "started": time.monotonic() - 1,
        "deadline": time.monotonic() - 0.5,
        "pending_ready": pending_ready,
    }
    return agent, process, stream


@pytest.mark.parametrize("pending_ready", [False, True])
def test_unreaped_child_returns_teardown_error_and_can_be_reaped_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pending_ready: bool
) -> None:
    agent, process, stream = active_worker(tmp_path, pending_ready=pending_ready)
    signals = []
    monkeypatch.setattr(worker.os, "killpg", lambda pid, signum: signals.append((pid, signum)))

    failure = agent.stop_child("deadline")

    assert failure is not None and failure["status"] == "teardown_error"
    assert process.waits == [2, 2]
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert agent.child is not None and not stream.closed
    assert agent.stop_child("stop") == failure
    assert agent.handle("000002", {"action": "shutdown"})["status"] == "teardown_error"
    assert not agent.stopping and process.waits == [2, 2]
    with pytest.raises(ValueError, match="busy"):
        agent.handle("000003", {"action": "start"})
    if pending_ready:
        assert json.loads((agent.results / "000001.json").read_text())["status"] == "teardown_error"

    process.returncode = -9
    completed = agent.stop_child("stop")
    assert completed is not None and completed["returncode"] == -9
    assert agent.child is None and stream.closed
    assert json.loads((agent.results / "000001.done.json").read_text())["returncode"] == -9
    assert agent.handle("000004", {"action": "shutdown"})["status"] == "stopped"
    assert agent.stopping


def test_deadline_cleanup_failure_keeps_mailbox_responsive_without_repeated_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, process, stream = active_worker(tmp_path)
    monkeypatch.setattr(worker.os, "killpg", lambda *args: None)
    for command_id, action in (("000002", "stop"), ("000003", "shutdown"), ("000004", "ping")):
        worker.atomic_json(agent.commands / f"{command_id}.json", {"action": action})
    sleeps = []

    def pause(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            agent.stopping = True

    monkeypatch.setattr(worker.time, "sleep", pause)
    agent.run()

    assert process.waits == [2, 2]
    assert agent.child is not None and not stream.closed
    for command_id in ("000002", "000003"):
        assert json.loads((agent.results / f"{command_id}.json").read_text())["status"] == "teardown_error"
    assert json.loads((agent.results / "000004.json").read_text())["status"] == "pong"
    stream.close()


@pytest.mark.parametrize("failed_phase", ["server_startup", "client_startup", "transfer", "completed"])
def test_controller_persists_inflight_case_before_aborting_on_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_phase: str
) -> None:
    test = controller.RailController(tmp_path, "reader", "donor", ["mlx5_0"], budget=30)
    inventories = {
        node: {"ready": True, "ip": ip, "nics": {"mlx5_0": {"numa_node": 0}}}
        for node, ip in (("reader", "192.0.2.1"), ("donor", "192.0.2.2"))
    }
    calls = []

    def wait_json(path: Path, deadline: float) -> dict[str, Any]:
        if path.name.endswith(".inventory.json"):
            return inventories[path.name.split(".")[0]]
        calls.append(path)
        if len(calls) == 1:
            if failed_phase == "server_startup":
                raise TimeoutError("synthetic server ACK timeout")
            return {"status": "ready", "pid": 10, "port": 18515}
        if len(calls) == 2:
            if failed_phase == "client_startup":
                raise TimeoutError("synthetic client ACK timeout")
            return {"status": "started", "pid": 20}
        if failed_phase == "transfer":
            raise TimeoutError("synthetic transfer timeout")
        return {"status": "completed", "returncode": 0}

    def stop(nodes: tuple[str, ...]) -> None:
        raise RuntimeError("synthetic teardown_error: donor child exit unconfirmed")

    monkeypatch.setattr(test, "wait_json", wait_json)
    monkeypatch.setattr(test, "check_mailboxes", lambda: None)
    monkeypatch.setattr(test, "stop", stop)

    assert test.run() == 1

    rows = [json.loads(line) for line in (tmp_path / "logs/results.jsonl").read_text().splitlines()]
    assert len(rows) == len(test.results) == 1
    row = rows[0]
    assert row["status"] == "INCOMPLETE" and "child exit unconfirmed" in row["teardown_error"]
    assert row["phase"] == failed_phase
    assert row["command_ids"]["server"] == "000001"
    assert bool(row["command_ids"]["client"]) == (failed_phase != "server_startup")
    if failed_phase in ("client_startup", "transfer"):
        assert row["server"]["pid"] == 10
    if failed_phase == "transfer":
        assert row["client"]["pid"] == 20
    commands = [json.loads(path.read_text()) for path in (tmp_path / "logs/agents").glob("*.commands/*.json")]
    assert not any(command.get("op") == "write" for command in commands)
    assert sum(command["action"] == "shutdown" for command in commands) == 2
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "INCOMPLETE=1" in report and "NOT RUN=1" in report
