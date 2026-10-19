# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rail diagnostic checks with synthetic perftest output and local mailboxes."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tools import check_mooncake_rails as controller
from tools import mooncake_rail_worker as worker


def bandwidth_log(average: str = "312.50", *, size: int = 2_006_016, iterations: int = 1200) -> str:
    return (
        "---------------------------------------------------------------\n"
        "                    RDMA_Read BW Test\n"
        "#bytes #iterations BW peak[Gb/sec] BW average[Gb/sec] MsgRate[Mpps]\n"
        f"{size} {iterations} 0.00 {average} 0.019473\n"
        "---------------------------------------------------------------\n"
    )


def test_bandwidth_uses_average_gbits_for_the_requested_transfer_size() -> None:
    text = "Conflicting CPU frequency values detected.\n" + bandwidth_log()

    assert controller.parse_bandwidth(text) == 312.5
    assert controller.parse_bandwidth(text, size=4096) is None


def test_success_requires_complete_endpoints_and_positive_finite_bandwidth() -> None:
    assert controller.classify_case(0, 0, "Server completed", bandwidth_log()) == "PASS"
    assert controller.classify_case(0, 0, "", "Waiting for connection") == "FAIL"
    assert controller.classify_case(0, None, "", bandwidth_log()) != "PASS"
    assert controller.classify_case(None, 0, "", bandwidth_log()) != "PASS"
    for invalid in ("0.00", "-1.0", "nan", "inf"):
        assert controller.classify_case(0, 0, "", bandwidth_log(invalid)) != "PASS"
    assert controller.classify_case(0, 0, "", bandwidth_log(iterations=0)) != "PASS"


@pytest.mark.parametrize("failed_side", ["server", "client"])
def test_endpoint_errors_cannot_be_hidden_by_a_valid_bandwidth_row(failed_side: str) -> None:
    server_rc, client_rc = (1, 0) if failed_side == "server" else (0, 1)
    assert controller.classify_case(server_rc, client_rc, "", bandwidth_log()) == "FAIL"
    completion_error = "Completion with error at client\nFailed status 12: wr_id 0\n"
    server_text = completion_error if failed_side == "server" else ""
    client_text = bandwidth_log() + (completion_error if failed_side == "client" else "")
    assert controller.classify_case(0, 0, server_text, client_text) == "FAIL"


def test_matrix_covers_every_directed_pair_for_both_operations_and_preserves_numa() -> None:
    nics = [f"mlx5_{number}" for number in (0, 1, 2, 5, 6, 7, 8, 11)]
    inventory = {
        "reader": {"nics": {nic: {"numa_node": index % 2} for index, nic in enumerate(nics)}},
        "donor": {"nics": {nic: {"numa_node": (index + 1) % 2} for index, nic in enumerate(nics)}},
    }
    cases = controller.make_cases(nics, inventory)

    assert len(cases) == 128
    assert len({case["case_id"] for case in cases}) == 128
    assert {(case["op"], case["reader_nic"], case["donor_nic"]) for case in cases} == {
        (operation, source, destination) for operation in ("read", "write") for source in nics for destination in nics
    }
    for case in cases:
        assert case["reader_numa"] == inventory["reader"]["nics"][case["reader_nic"]]["numa_node"]
        assert case["donor_numa"] == inventory["donor"]["nics"][case["donor_nic"]]["numa_node"]


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("role", ["server", "client"])
def test_perftest_command_binds_the_selected_nic_numa_and_operation(operation: str, role: str) -> None:
    info = {
        "tools": {name: {"path": f"/usr/bin/{name}"} for name in ("numactl", "ib_read_bw", "ib_write_bw")},
        "nics": {"mlx5_2": {"numa_node": 1}},
    }
    command = {"op": operation, "role": role, "nic": "mlx5_2", "numa": 1, "peer_ip": "192.0.2.2"}

    argv = worker.build_command(command, info, 18515)

    assert argv[:4] == ["/usr/bin/numactl", "--cpunodebind=1", "--membind=1", f"/usr/bin/ib_{operation}_bw"]
    for flag, value in (("-d", "mlx5_2"), ("-i", "1"), ("-x", "0"), ("-p", "18515"), ("-s", "2006016")):
        assert argv[argv.index(flag) + 1] == value
    assert "--report_gbits" in argv
    if operation == "read":
        assert argv[argv.index("-o") + 1] == "16"
    else:
        assert "-o" not in argv
    assert (argv[-1] == "192.0.2.2") == (role == "client")
    assert not any("cuda" in argument.lower() for argument in argv)
    with pytest.raises(ValueError, match="NUMA"):
        worker.build_command({**command, "numa": 0}, info, 18515)


@pytest.mark.parametrize(
    "inode,state,family,expected",
    [(111, "0A", "tcp", True), (111, "0A", "tcp6", True), (222, "0A", "tcp", False), (111, "01", "tcp", False)],
)
def test_listener_readiness_requires_a_socket_owned_by_the_actual_process(
    monkeypatch: pytest.MonkeyPatch, inode: int, state: str, family: str, expected: bool
) -> None:
    descriptor = Path("/proc/123/fd/9")
    monkeypatch.setattr(
        Path, "glob", lambda self, pattern: iter([descriptor]) if str(self) == "/proc/123/fd" else iter([])
    )
    monkeypatch.setattr(worker.os, "readlink", lambda path: "socket:[111]")
    table = f"header\n0: 00000000:4853 00000000:0000 {state} 0:0 00:0 0 0 0 {inode} 1\n"
    monkeypatch.setattr(worker, "read_text", lambda path: table if str(path).endswith(f"/net/{family}") else "header\n")

    def forbidden_probe(*args: object, **kwargs: object) -> None:
        pytest.fail("Readiness must not consume perftest's one client connection")

    monkeypatch.setattr(worker.socket, "create_connection", forbidden_probe)

    assert worker.listener_owned(123, 18515) is expected
    assert not worker.listener_owned(123, 18516)


@pytest.mark.parametrize("failure", ["", "zero_gid", "numa_permission"])
def test_worker_preflight_checks_real_binding_and_gid_instead_of_current_cpu_mask(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setenv("MC_RAIL_NICS", "mlx5_0")
    monkeypatch.setattr(worker.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(worker.socket, "getaddrinfo", lambda *args: [(2, 1, 6, "", ("192.0.2.2", 0))])
    monkeypatch.setattr(worker, "cgroup_evidence", lambda: {})

    def read(path: Path) -> str | None:
        value = str(path)
        if value == "/proc/self/status":
            return "Cpus_allowed_list: 0-3\nMems_allowed_list: 0\n"
        for suffix, content in (
            ("/device/numa_node", "1"),
            ("/ports/1/state", "4: ACTIVE"),
            ("/ports/1/link_layer", "InfiniBand"),
            ("/ports/1/gids/0", "::" if failure == "zero_gid" else "fe80::1"),
        ):
            if value.endswith(suffix):
                return content
        return None

    def capture(argv: list[str]) -> dict[str, Any]:
        failed = failure == "numa_permission" and "--cpunodebind=1" in argv
        return {"argv": argv, "returncode": int(failed), "stdout": "", "stderr": "EPERM" if failed else ""}

    monkeypatch.setattr(worker, "read_text", read)
    monkeypatch.setattr(worker, "capture", capture)

    info = worker.inventory("donor")

    assert info["ready"] == (not failure)
    assert info["ip"] == "192.0.2.2"
    assert "--membind=1" in info["numa_bind_checks"]["1"]["argv"]
    if failure:
        assert any(("GID" if failure == "zero_gid" else "bind failed") in error for error in info["errors"])


class FakeRailAgents:
    """Honor the real filesystem protocol without starting perftest or Slurm."""

    def __init__(
        self,
        run_dir: Path,
        nics: list[str],
        *,
        fail_case: str = "",
        stalled: str = "",
        startup: str = "",
        ping_failure: str = "",
        drop_ping: bool = False,
    ):
        self.run_dir = run_dir
        self.fail_case, self.stalled, self.startup = fail_case, stalled, startup
        self.ping_failure = ping_failure
        self.drop_ping = drop_ping
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.active: dict[str, tuple[str, dict[str, Any]]] = {}
        self.errors: list[Exception] = []
        self.threads = []
        for node, ip in (("reader", "192.0.2.1"), ("donor", "192.0.2.2")):
            worker.atomic_json(
                run_dir / "logs/agents" / f"{node}.inventory.json",
                {"ready": True, "ip": ip, "nics": {nic: {"numa_node": index % 2} for index, nic in enumerate(nics)}},
            )
            for suffix in ("commands", "results"):
                (run_dir / "logs/agents" / f"{node}.{suffix}").mkdir(parents=True)
            self.threads.append(threading.Thread(target=self.serve, args=(node,), daemon=True))

    def __enter__(self) -> FakeRailAgents:
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=1)
            assert not thread.is_alive()
        assert not self.errors, self.errors

    def result(self, node: str, command_id: str, value: dict[str, Any], *, completed: bool = False) -> None:
        suffix = ".done.json" if completed else ".json"
        worker.atomic_json(self.run_dir / "logs/agents" / f"{node}.results" / f"{command_id}{suffix}", value)

    def complete(self, node: str, command_id: str, command: dict[str, Any], rc: int, text: str) -> None:
        relative = Path("logs/cases") / f"{command['case_id']}.{command['role']}.log"
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.result(
            node,
            command_id,
            {
                "status": "completed",
                "returncode": rc,
                "runtime_s": 0.001,
                "argv": ["synthetic-perftest", "-d", command["nic"]],
                "log_path": str(relative),
            },
            completed=True,
        )

    def handle(self, node: str, command_id: str, command: dict[str, Any]) -> None:
        self.commands.append((node, command))
        if command["action"] == "ping":
            if node == self.ping_failure and self.drop_ping:
                return
            self.result(
                node,
                command_id,
                {"status": "error", "error": "synthetic mailbox failure"}
                if node == self.ping_failure
                else {"status": "pong", "published_utc": "2026-10-19T00:00:00Z"},
            )
            return
        if command["action"] in ("stop", "shutdown"):
            pending = self.active.pop(node, None)
            if pending:
                pending_id, pending_command = pending
                done = self.run_dir / "logs/agents" / f"{node}.results" / f"{pending_id}.done.json"
                if not done.exists():
                    self.complete(node, pending_id, pending_command, -15, "Stopped by controller\n")
            self.result(node, command_id, {"status": "stopped"})
            return
        self.active[node] = (command_id, command)
        if command["role"] == "server":
            if command["case_id"] == self.startup:
                self.result(node, command_id, {"status": "error", "error": "synthetic server startup failure"})
            else:
                self.result(node, command_id, {"status": "ready", "port": 18515})
            return
        self.result(node, command_id, {"status": "started"})
        if command["case_id"] == self.stalled:
            return
        failed = command["case_id"] == self.fail_case
        text = bandwidth_log() + ("Completion with error at client\n" if failed else "")
        self.complete(node, command_id, command, int(failed), text)
        server_id, server_command = self.active["donor"]
        self.complete("donor", server_id, server_command, 0, "Server completed\n")

    def serve(self, node: str) -> None:
        seen = set()
        directory = self.run_dir / "logs/agents" / f"{node}.commands"
        while not self.stop_event.is_set():
            for path in sorted(directory.glob("*.json")):
                if path.stem in seen:
                    continue
                seen.add(path.stem)
                try:
                    with self.lock:
                        self.handle(node, path.stem, json.loads(path.read_text()))
                except Exception as exc:
                    self.errors.append(exc)
            self.stop_event.wait(0.002)


@pytest.mark.parametrize("fail_case", ["", "read-mlx5_0-mlx5_1"])
def test_controller_collects_real_mailbox_results_and_reports_failed_pairs(tmp_path: Path, fail_case: str) -> None:
    nics = ["mlx5_0", "mlx5_1"]
    with FakeRailAgents(tmp_path, nics, fail_case=fail_case) as agents:
        test = controller.RailController(tmp_path, "reader", "donor", nics, case_timeout=0.3, budget=20)
        rc = test.run()

    rows = [json.loads(line) for line in (tmp_path / "logs/results.jsonl").read_text().splitlines()]
    assert rc == bool(fail_case)
    assert len(rows) == 8
    assert sum(row["status"] == "PASS" for row in rows) == (7 if fail_case else 8)
    if fail_case:
        failed = next(row for row in rows if row["case_id"] == fail_case)
        assert failed["status"] == "FAIL" and failed["client_rc"] == 1
        assert "Completion with error" in (tmp_path / failed["client"]["log_path"]).read_text()
    starts = [(node, command) for node, command in agents.commands if command["action"] == "start"]
    assert len(starts) == 16
    assert all(command["peer_ip"] == "192.0.2.2" for _, command in starts if command["role"] == "client")
    assert sum(command["action"] == "stop" for _, command in agents.commands) == 16
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "NOT RUN=0" in report and "## READ matrix" in report and "## WRITE matrix" in report
    assert (tmp_path / "logs/summary.csv").exists()
    checks = json.loads((tmp_path / "logs/mailbox-checks.json").read_text())
    assert {check["node"] for check in checks} == {"reader", "donor"}
    assert all(check["ack"]["status"] == "pong" and check["round_trip_s"] >= 0 for check in checks)
    assert len({check["command_id"] for check in checks}) == 2


def test_controller_reads_an_ack_despite_a_cached_negative_exists_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ack = tmp_path / "logs/agents/donor.results/000001.json"
    payload = {"status": "ready", "port": 18515}
    worker.atomic_json(ack, payload)
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path == ack else exists(path))
    test = controller.RailController(tmp_path, "reader", "donor", ["mlx5_0"], budget=20)

    assert not ack.exists()
    assert test.wait_json(ack, time.monotonic() + 0.2) == payload


@pytest.mark.parametrize("missing_ack", [False, True])
def test_controller_mailbox_ping_failure_prevents_perftest_and_preserves_report(
    tmp_path: Path, missing_ack: bool
) -> None:
    with FakeRailAgents(tmp_path, ["mlx5_0", "mlx5_1"], ping_failure="donor", drop_ping=missing_ack) as agents:
        test = controller.RailController(tmp_path, "reader", "donor", ["mlx5_0", "mlx5_1"], case_timeout=0.3, budget=20)
        assert test.run() == 1

    assert not test.results
    assert {node for node, command in agents.commands if command["action"] == "ping"} == {"reader", "donor"}
    assert not any(command["action"] == "start" for _, command in agents.commands)
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "PASS=0" in report and "NOT RUN=8" in report
    assert ("waiting for" if missing_ack else "synthetic mailbox failure") in report
    checks = json.loads((tmp_path / "logs/mailbox-checks.json").read_text())
    assert checks[-1]["node"] == "donor" and checks[-1]["error"]


@pytest.mark.parametrize("failure", ["startup", "stalled"])
def test_controller_stops_a_failed_case_before_reusing_agents(tmp_path: Path, failure: str) -> None:
    case_id = "read-mlx5_0-mlx5_0"
    with FakeRailAgents(tmp_path, ["mlx5_0"], **{failure: case_id}) as agents:
        test = controller.RailController(tmp_path, "reader", "donor", ["mlx5_0"], case_timeout=0.15, budget=20)
        assert test.run() == 1

    assert [row["status"] for row in test.results] == ["FAIL", "PASS"]
    assert ("server startup" if failure == "startup" else "waiting for") in test.results[0]["reason"]
    writes = [index for index, (_, command) in enumerate(agents.commands) if command.get("op") == "write"]
    stopped = [index for index, (_, command) in enumerate(agents.commands) if command["action"] == "stop"]
    assert len(writes) == 2 and max(stopped[:2]) < min(writes)


def test_controller_preserves_budget_excluded_matrix_cells_as_not_run(tmp_path: Path) -> None:
    nics = ["mlx5_0", "mlx5_1"]
    with FakeRailAgents(tmp_path, nics) as agents:
        test = controller.RailController(tmp_path, "reader", "donor", nics, case_timeout=0.15, budget=1)
        assert test.run() == 1

    assert len(test.planned) == 8 and not test.results
    assert not any(command["action"] == "start" for _, command in agents.commands)
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "PASS=0" in report and "NOT RUN=8" in report
    assert "remaining cells NOT RUN" in report


def test_controller_preflight_failure_does_not_claim_connectivity(tmp_path: Path) -> None:
    worker.atomic_json(
        tmp_path / "logs/agents/reader.inventory.json",
        {"ready": False, "errors": ["Required tool missing: ib_read_bw"]},
    )
    test = controller.RailController(tmp_path, "reader", "donor", ["mlx5_0", "mlx5_1"], budget=20)

    assert test.run() == 1

    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "ib_read_bw" in report and "PASS=0" in report and "NOT RUN=8" in report
    assert not test.results


def test_controller_rejects_a_stale_run_before_reusing_results(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"job_id": "previous-job"}\n')
    test = controller.RailController(tmp_path, "reader", "donor", ["mlx5_0"], budget=20)

    with pytest.raises(RuntimeError, match="fresh run directory"):
        test.run()

    assert metadata.read_text() == '{"job_id": "previous-job"}\n'
    assert not list(tmp_path.rglob("*.commands/*.json"))
    assert not (tmp_path / "logs/perf-analysis.md").exists()


@pytest.mark.parametrize("listener_ready", [False, True])
def test_worker_publishes_readiness_and_stops_its_child_without_tcp_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listener_ready: bool
) -> None:
    # The real supervisor owns a harmless local child; no TCP listener or RDMA
    # process is started. A pending readiness check must not block stop commands.
    monkeypatch.setattr(worker, "build_command", lambda *args: ["/bin/sleep", "60"])
    monkeypatch.setattr(worker, "listener_owned", lambda *args: listener_ready)

    def forbidden_probe(*args: object, **kwargs: object) -> None:
        pytest.fail("Worker readiness must not connect to the single-client perftest listener")

    monkeypatch.setattr(worker.socket, "create_connection", forbidden_probe)
    agent = worker.Worker(tmp_path, "synthetic-node", {})
    errors = []

    def run_agent() -> None:
        try:
            agent.run()
        except Exception as exc:
            errors.append(exc)

    def wait_for(path: Path) -> None:
        deadline = time.monotonic() + 2
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert path.exists(), f"Agent did not respond to {path.name}"

    worker.atomic_json(agent.commands / "000000.json", {"action": "ping"})
    worker.atomic_json(
        agent.commands / "000001.json",
        {"action": "start", "case_id": "pending-listener", "op": "read", "role": "server", "port": 18515},
    )
    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()
    try:
        wait_for(agent.results / "000000.json")
        pong = json.loads((agent.results / "000000.json").read_text())
        assert pong["status"] == "pong" and pong["published_utc"]
        deadline = time.monotonic() + 2
        while agent.child is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert agent.child is not None
        process = agent.child["process"]
        assert process.poll() is None
        if listener_ready:
            wait_for(agent.results / "000001.json")
            readiness = json.loads((agent.results / "000001.json").read_text())
            assert readiness["status"] == "ready"
            assert readiness["pid"] == process.pid and readiness["port"] == 18515
        worker.atomic_json(agent.commands / "000002.json", {"action": "stop"})
        wait_for(agent.results / "000002.json")

        ack = json.loads((agent.results / "000002.json").read_text())
        completed = json.loads((agent.results / "000001.done.json").read_text())
        assert ack["status"] == "stopped"
        assert completed["status"] == "completed" and completed["returncode"] != 0
        assert completed["reason"] == "stop"
        assert process.poll() is not None and agent.child is None
        assert json.loads((agent.results / "000001.json").read_text())["status"] == (
            "ready" if listener_ready else "error"
        )
        worker.atomic_json(agent.commands / "000003.json", {"action": "shutdown"})
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert not errors
    finally:
        agent.stopping = True
        thread.join(timeout=2)
        if thread.is_alive():
            agent.stop_child("test cleanup")
            thread.join(timeout=2)
