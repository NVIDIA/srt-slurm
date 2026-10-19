#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate two persistent node agents to measure explicit RDMA NIC pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

PAGE_BYTES = 2_006_016
DEFAULT_NICS = "mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_mailbox_json(path: Path) -> dict[str, Any] | None:
    """Discover new replies by listing, as agents do for incoming commands.

    Repeated stat calls on an absent name can retain a negative lookup on a
    shared filesystem. Do not prime that lookup before the writer publishes it.
    """
    try:
        with os.scandir(path.parent) as entries:
            if not any(entry.name == path.name for entry in entries):
                return None
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def parse_bandwidth(text: str, size: int = PAGE_BYTES) -> float | None:
    """Read the average Gbit/s column, including legitimate zero peak values."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.search(r"BW\s+average\[Gb/sec\]", line, re.IGNORECASE):
            continue
        peak = bool(re.search(r"BW\s+peak", line, re.IGNORECASE))
        for row in lines[index + 1 :]:
            columns = row.split()
            try:
                if len(columns) < (5 if peak else 4) or int(columns[0]) != size:
                    continue
                iterations = int(columns[1])
                average = float(columns[3 if peak else 2])
            except (ValueError, IndexError):
                continue
            if iterations > 0 and math.isfinite(average) and average > 0:
                return average
    return None


def classify_case(server_rc: int | None, client_rc: int | None, server_text: str, client_text: str) -> str:
    if server_rc is None or client_rc is None:
        return "INCOMPLETE"
    if server_rc != 0 or client_rc != 0:
        return "FAIL"
    errors = r"(?:completion.*(?:error|syndrome)|failed\s+status|work completion.*error)"
    if re.search(errors, server_text + "\n" + client_text, re.IGNORECASE):
        return "FAIL"
    return "PASS" if parse_bandwidth(client_text) is not None else "FAIL"


def make_cases(nics: list[str], inventory: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for op in ("read", "write"):
        for reader_nic in nics:
            for donor_nic in nics:
                cases.append(
                    {
                        "case_id": f"{op}-{reader_nic}-{donor_nic}",
                        "op": op,
                        "reader_nic": reader_nic,
                        "donor_nic": donor_nic,
                        "reader_numa": int(inventory["reader"]["nics"][reader_nic]["numa_node"]),
                        "donor_numa": int(inventory["donor"]["nics"][donor_nic]["numa_node"]),
                    }
                )
    return cases


def write_report(
    run_dir: Path,
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
    planned: list[dict[str, Any]],
    reason: str,
) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "op",
        "reader_nic",
        "reader_numa",
        "donor_nic",
        "donor_numa",
        "status",
        "average_gbit_s",
        "server_rc",
        "client_rc",
        "elapsed_s",
        "reason",
    ]
    with (logs / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    counts = {status: sum(row["status"] == status for row in results) for status in ("PASS", "FAIL", "INCOMPLETE")}
    not_run = max(0, len(planned) - len(results))
    lines = [
        "# Mooncake cross-rail RDMA diagnostic",
        "",
        "Goal: establish which explicit reader-NIC / donor-NIC pairs complete host-memory RDMA reads and writes.",
        "Each endpoint binds CPU and memory to its own NIC's NUMA node. No model or GPU is loaded.",
        "",
        f"Reader: `{metadata.get('reader', 'N/A')}`; donor: `{metadata.get('donor', 'N/A')}`; "
        f"job: `{metadata.get('job_id', 'N/A')}`.",
        f"Image: `{metadata.get('image', 'N/A')}`.",
        f"Bytes per operation: {PAGE_BYTES}; {metadata.get('duration', 'N/A')} s per test, "
        "one-second measurement margins, two QPs, read depth 16/QP, port 1, GID index 0.",
        "",
        f"Status: {reason}. PASS={counts['PASS']}, FAIL={counts['FAIL']}, "
        f"INCOMPLETE={counts['INCOMPLETE']}, NOT RUN={not_run}.",
        "A PASS requires both processes to exit zero and a positive, finite client average-bandwidth result.",
        "A failure is not automatically a fabric defect: inspect logs for NUMA permissions, TCP bootstrap, "
        "tool availability, registration, or completion failures.",
        "",
    ]
    nics = metadata.get("nics", [])
    by_case = {row["case_id"]: row for row in results}
    for op in ("read", "write"):
        lines.extend(
            [
                f"## {op.upper()} matrix",
                "",
                "Rows are reader NICs; columns are donor NICs. Cells show average Gbit/s for PASS, "
                "or an explicit failure/missing status.",
                "",
                "| Reader / donor | " + " | ".join(nics) + " |",
                "| --- | " + " | ".join("---:" for _ in nics) + " |",
            ]
        )
        for reader_nic in nics:
            cells = []
            for donor_nic in nics:
                row = by_case.get(f"{op}-{reader_nic}-{donor_nic}")
                cells.append(
                    "NOT RUN"
                    if row is None
                    else f"{row['average_gbit_s']:.2f}"
                    if row["status"] == "PASS"
                    else row["status"]
                )
            lines.append("| " + " | ".join([reader_nic, *cells]) + " |")
        lines.append("")
    lines.extend(
        [
            "## Sources and interpretation",
            "",
            "`metadata.json` records configuration and time budget; `logs/results.jsonl` records case outcomes, "
            "commands, exit codes and paths to raw output; `logs/summary.csv` is the flat comparison table.",
            "`logs/agents/*.inventory.json` records installed tools/versions, NIC NUMA/GID/link information, "
            "CPU/memory allowances and collection errors. Per-case output is under `logs/cases/`.",
            "`logs/agents/*.install.log` records container-only apt setup or preinstalled-tool detection. "
            "Setup failure is a preflight failure, not an RDMA connectivity result.",
            "`logs/mailbox-checks.json` records controller/agent round-trip checks before perftest starts. "
            "A missing acknowledgment is a coordination failure, not evidence of failed RDMA connectivity.",
            "Compare same-name and cross-name pairs. A complete passing matrix supports cross-rail reachability "
            "for these two nodes, NICs, port/GID settings and this test window only. Other nodes need their own test.",
            "Different-socket numbers across nodes are valid; these tests keep NIC-to-memory access local "
            "within each endpoint. They do not measure an intentional remote-socket memory penalty.",
            "Missing/unrun cases do not establish connectivity. These short single-pair tests do not measure "
            "aggregate multi-rail bandwidth or establish a Mooncake/TTFT speedup. The perftest operation is a "
            "whole cache page; Mooncake currently splits pages into smaller RDMA work requests.",
            "",
            "TTFT p50/p95/p99 (ms), ITL p50/p99 (ms/token), input/output/total token throughput (tokens/s), "
            "and agentic Pareto/SLO metrics: N/A because this diagnostic has no inference requests.",
            "",
            "The `slurm-job-sizing` skill (`src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md`) limits the "
            "allocation to 25 minutes, with a 22-minute controller budget and time reserved for cleanup/reporting. "
            "The controller budget includes up to four minutes of container apt setup. "
            "The default 128 five-second tests require about 11 minutes of configured traffic plus setup; "
            "timeouts can leave explicit unrun cases. No DSight/dashboard is needed: raw perftest completion "
            "results and the two matrices directly answer the connectivity hypothesis.",
        ]
    )
    (logs / "perf-analysis.md").write_text("\n".join(lines) + "\n")


class RailController:
    def __init__(
        self,
        run_dir: Path,
        reader: str,
        donor: str,
        nics: list[str],
        duration: int = 5,
        case_timeout: float = 15,
        budget: float = 1320,
        inventory_timeout: float = 60,
    ) -> None:
        self.run_dir = run_dir
        self.reader = reader
        self.donor = donor
        self.nics = nics
        self.duration = duration
        self.case_timeout = case_timeout
        self.inventory_timeout = inventory_timeout
        self.deadline = time.monotonic() + budget
        self.sequence = 0
        self.interrupted = False
        self.results: list[dict[str, Any]] = []
        self.planned: list[dict[str, Any]] = []
        self.metadata = {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "reader": reader,
            "donor": donor,
            "nics": nics,
            "duration": duration,
            "case_timeout": case_timeout,
            "budget": budget,
            "inventory_timeout": inventory_timeout,
            "image": os.environ.get("MC_RAIL_IMAGE"),
            "start_utc": time.strftime("%FT%TZ", time.gmtime()),
            "planned_cases": len(nics) ** 2 * 2,
        }

    def wait_json(self, path: Path, deadline: float) -> dict[str, Any]:
        while time.monotonic() < min(deadline, self.deadline):
            if self.interrupted:
                raise InterruptedError("controller interrupted")
            value = read_mailbox_json(path)
            if value is not None:
                return value
            time.sleep(0.05)
        raise TimeoutError(f"waiting for {path.relative_to(self.run_dir)}")

    def send(self, node: str, command: dict[str, Any]) -> tuple[str, Path]:
        self.sequence += 1
        command_id = f"{self.sequence:06d}"
        base = self.run_dir / "logs" / "agents"
        atomic_json(base / f"{node}.commands" / f"{command_id}.json", command)
        return command_id, base / f"{node}.results"

    def check_mailboxes(self) -> None:
        """Require a round trip before launching any perftest processes."""
        checks = []
        try:
            for node in (self.reader, self.donor):
                started = time.monotonic()
                command_id, directory = self.send(node, {"action": "ping"})
                check: dict[str, Any] = {"node": node, "command_id": command_id}
                checks.append(check)
                try:
                    ack = self.wait_json(directory / f"{command_id}.json", started + self.case_timeout)
                    check["ack"] = ack
                    if ack.get("status") != "pong":
                        raise RuntimeError(f"mailbox round trip failed for {node}: {ack}")
                except (TimeoutError, RuntimeError, InterruptedError) as exc:
                    check["error"] = str(exc)
                    raise RuntimeError(f"mailbox preflight for {node}: {exc}") from exc
                finally:
                    check["round_trip_s"] = round(time.monotonic() - started, 6)
                print(f"{node}: mailbox round trip {check['round_trip_s']:.3f} s", flush=True)
        finally:
            atomic_json(self.run_dir / "logs/mailbox-checks.json", checks)

    def stop(self, nodes: tuple[str, ...]) -> None:
        pending = [self.send(node, {"action": "stop"}) for node in nodes]
        # Agents also enforce their own child timeouts. Stop acknowledgments
        # must arrive before a new case can reuse an agent or its resources.
        deadline = time.monotonic() + 8
        for command_id, directory in pending:
            path = directory / f"{command_id}.json"
            ack = read_mailbox_json(path)
            while ack is None and time.monotonic() < deadline:
                time.sleep(0.05)
                ack = read_mailbox_json(path)
            if ack is None or ack.get("status") != "stopped":
                raise RuntimeError(f"node-agent teardown unconfirmed; refusing another case: {path.name}: {ack}")

    def read_output(self, result: dict[str, Any] | None) -> str:
        if not result or not result.get("log_path"):
            return ""
        path = Path(result["log_path"])
        # Worker paths are relative to the shared run directory, never host paths.
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid worker log path")
        try:
            return (self.run_dir / path).read_text(errors="replace")
        except FileNotFoundError:
            return ""

    def run_case(self, case: dict[str, Any], donor_ip: str) -> dict[str, Any]:
        started = time.monotonic()
        deadline = min(self.deadline, started + self.case_timeout)
        common = {
            "action": "start",
            "case_id": case["case_id"],
            "op": case["op"],
            "size": PAGE_BYTES,
            "duration": self.duration,
            "timeout": self.case_timeout,
        }
        server: dict[str, Any] | None = None
        client: dict[str, Any] | None = None
        reason = ""
        teardown_error = ""
        phase = "server_startup"
        server_id = client_id = ""
        server_dir = client_dir = self.run_dir
        try:
            server_id, server_dir = self.send(
                self.donor,
                {
                    **common,
                    "role": "server",
                    "nic": case["donor_nic"],
                    "numa": case["donor_numa"],
                },
            )
            ready = self.wait_json(server_dir / f"{server_id}.json", deadline)
            if ready.get("status") != "ready":
                raise RuntimeError(f"server startup: {ready}")
            server = ready
            phase = "client_startup"
            client_id, client_dir = self.send(
                self.reader,
                {
                    **common,
                    "role": "client",
                    "nic": case["reader_nic"],
                    "numa": case["reader_numa"],
                    "port": ready["port"],
                    "peer_ip": donor_ip,
                },
            )
            ack = self.wait_json(client_dir / f"{client_id}.json", deadline)
            if ack.get("status") != "started":
                raise RuntimeError(f"client startup: {ack}")
            client = ack
            phase = "transfer"
            client = self.wait_json(client_dir / f"{client_id}.done.json", deadline)
            server = self.wait_json(server_dir / f"{server_id}.done.json", deadline)
            phase = "completed"
        except (TimeoutError, RuntimeError, InterruptedError) as exc:
            reason = str(exc)
        finally:
            try:
                self.stop((self.reader, self.donor))
            except (TimeoutError, RuntimeError, InterruptedError) as exc:
                teardown_error = str(exc)
        # Stopping a timed-out process produces final exit status and keeps its
        # output available instead of treating absent output as a successful cell.
        for command_id, directory, role in ((server_id, server_dir, "server"), (client_id, client_dir, "client")):
            path = directory / f"{command_id}.done.json"
            completed = read_mailbox_json(path) if command_id else None
            if completed is not None:
                if role == "server":
                    server = completed
                else:
                    client = completed
        server_text, client_text = self.read_output(server), self.read_output(client)
        server_rc = server.get("returncode") if server else None
        client_rc = client.get("returncode") if client else None
        status = classify_case(server_rc, client_rc, server_text, client_text)
        if reason:
            status = "INCOMPLETE" if self.interrupted or time.monotonic() >= self.deadline else "FAIL"
        if teardown_error:
            status = "INCOMPLETE"
        return {
            **case,
            "status": status,
            "average_gbit_s": parse_bandwidth(client_text),
            "server_rc": server_rc,
            "client_rc": client_rc,
            "elapsed_s": round(time.monotonic() - started, 3),
            "reason": reason,
            "teardown_error": teardown_error,
            "phase": phase,
            "command_ids": {"server": server_id, "client": client_id},
            "server": server,
            "client": client,
        }

    def run(self) -> int:
        logs = self.run_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "metadata.json").exists() or (logs / "results.jsonl").exists():
            raise RuntimeError("Refusing stale commands/results: use a fresh run directory")
        atomic_json(self.run_dir / "metadata.json", self.metadata)
        reason, rc = "preflight incomplete", 1
        try:
            inventories = {}
            inventory_deadline = time.monotonic() + self.inventory_timeout
            for role, node in (("reader", self.reader), ("donor", self.donor)):
                inventory = self.wait_json(logs / "agents" / f"{node}.inventory.json", inventory_deadline)
                if not inventory.get("ready"):
                    raise RuntimeError(f"{node} preflight failed: {inventory.get('errors', inventory)}")
                for nic in self.nics:
                    if int(inventory["nics"][nic]["numa_node"]) < 0:
                        raise RuntimeError(f"{node}/{nic} has unknown NUMA locality")
                inventories[role] = inventory
            self.check_mailboxes()
            self.planned = make_cases(self.nics, inventories)
            atomic_json(self.run_dir / "plan.json", self.planned)
            with (logs / "results.jsonl").open("a", buffering=1) as output:
                for case in self.planned:
                    if self.interrupted or time.monotonic() + self.case_timeout + 8 > self.deadline:
                        reason = "interrupted or time budget exhausted; remaining cells NOT RUN"
                        break
                    print(f"{case['case_id']}: starting", flush=True)
                    row = self.run_case(case, inventories["donor"]["ip"])
                    self.results.append(row)
                    output.write(json.dumps(row) + "\n")
                    print(f"{case['case_id']}: {row['status']} {row['average_gbit_s']} Gbit/s", flush=True)
                    if row.get("teardown_error"):
                        reason = row["teardown_error"]
                        break
                else:
                    reason = "matrix completed"
            rc = int(len(self.results) != len(self.planned) or any(row["status"] != "PASS" for row in self.results))
        except (TimeoutError, RuntimeError, InterruptedError, KeyError, ValueError) as exc:
            reason = str(exc)
            print(reason, flush=True)
        finally:
            self.metadata["finish_utc"] = time.strftime("%FT%TZ", time.gmtime())
            atomic_json(self.run_dir / "metadata.json", self.metadata)
            # Retain the planned count even when preflight prevented plan construction.
            planned = self.planned or [{}] * self.metadata["planned_cases"]
            write_report(self.run_dir, self.metadata, self.results, planned, reason)
            for node in (self.reader, self.donor):
                self.send(node, {"action": "shutdown"})
        return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--nics", default=DEFAULT_NICS)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--case-timeout", type=float, default=15)
    parser.add_argument("--budget", type=float, default=1320)
    parser.add_argument("--inventory-timeout", type=float, default=60)
    args = parser.parse_args()
    nics = args.nics.split(",")
    if len(set(nics)) != len(nics) or not all(re.fullmatch(r"[A-Za-z0-9_]+", nic) for nic in nics):
        parser.error("NIC names must be unique, comma-separated device names")
    if args.reader == args.donor or args.duration < 3 or args.case_timeout <= args.duration + 2:
        parser.error("use different nodes, duration >=3 and case timeout > duration+2")
    controller = RailController(
        args.run_dir.resolve(),
        args.reader,
        args.donor,
        nics,
        args.duration,
        args.case_timeout,
        args.budget,
        args.inventory_timeout,
    )
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: setattr(controller, "interrupted", True))
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
