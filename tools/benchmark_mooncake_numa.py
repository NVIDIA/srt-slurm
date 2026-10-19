#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate two persistent Slurm tasks for real Mooncake NUMA cache GETs.

The donor task owns the controller, master, and NUMA-pinned donor subprocess.
The other task executes bounded reader subprocesses using the proven directory-
scanned mailbox protocol. No SSH or model/GPU workload is involved.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from tools.check_mooncake_rails import atomic_json, read_mailbox_json
from tools.diagnose_mooncake import PAGE_BYTES, inventory, versions


@dataclass(frozen=True)
class Settings:
    cluster: str = "raplab"
    allow_cross_rail: bool = False
    rounds: int = 3
    seconds: float = 10
    warmup_seconds: float = 2
    batch_size: int = 512
    page_bytes: int = PAGE_BYTES
    pool_bytes: int = 4 * 1024**3
    working_set_batches: int = 2
    budget: float = 1320

    @classmethod
    def from_env(cls) -> Settings:
        result = cls(
            cluster=os.environ.get("MC_NUMA_CLUSTER", "raplab"),
            allow_cross_rail=os.environ.get("MC_NUMA_ALLOW_CROSS_RAIL", "0") == "1",
            rounds=int(os.environ.get("MC_NUMA_ROUNDS", "3")),
            seconds=float(os.environ.get("MC_NUMA_SECONDS", "10")),
            warmup_seconds=float(os.environ.get("MC_NUMA_WARMUP", "2")),
            batch_size=int(os.environ.get("MC_NUMA_BATCH_SIZE", "512")),
            page_bytes=int(os.environ.get("MC_NUMA_PAGE_BYTES", str(PAGE_BYTES))),
            pool_bytes=int(float(os.environ.get("MC_NUMA_POOL_GIB", "4")) * 1024**3),
            working_set_batches=int(os.environ.get("MC_NUMA_WORKING_BATCHES", "2")),
            budget=float(os.environ.get("MC_NUMA_BUDGET", "1320")),
        )
        numeric = (result.seconds, result.warmup_seconds, result.budget)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Durations must be finite")
        if result.budget <= 0 or result.budget > 1320:
            raise ValueError("Body budget must be positive and at most 1320 seconds, leaving cleanup reserve")
        if not 1 <= result.rounds <= 10 or not 1 <= result.seconds <= 120 or not 0 <= result.warmup_seconds <= 30:
            raise ValueError("Use 1..10 rounds, 1..120 measurement seconds, and 0..30 warmup seconds")
        if min(result.batch_size, result.page_bytes, result.pool_bytes, result.working_set_batches) <= 0:
            raise ValueError("Batch, page, pool and working-set sizes must be positive")
        working_bytes = result.batch_size * result.page_bytes * result.working_set_batches
        if working_bytes > result.pool_bytes * 0.75:
            raise ValueError("Working set must fit within 75% of the donor pool; leave allocation/metadata headroom")
        return result


def make_plan(settings: Settings) -> list[dict[str, Any]]:
    plan = []
    for repeat in range(settings.rounds):
        donors = (0, 1) if repeat % 2 == 0 else (1, 0)
        for donor in donors:
            modes = [(reader, affinity) for reader in (0, 1) for affinity in (True, False)]
            random.Random(104729 + repeat * 2 + donor).shuffle(modes)
            for reader, affinity in modes:
                plan.append(
                    {
                        "case_id": f"round{repeat}-d{donor}-r{reader}-affinity-{'on' if affinity else 'off'}",
                        "round": repeat,
                        "donor_numa": donor,
                        "reader_numa": reader,
                        "affinity": affinity,
                        "enabled": affinity or settings.allow_cross_rail,
                    }
                )
    return plan


def discover_groups(reader: dict[str, Any], donor: dict[str, Any]) -> dict[int, list[str]]:
    def active(info: dict[str, Any], numa: int) -> set[str]:
        return {
            name
            for name, nic in info["infiniband"].items()
            if str(nic.get("numa_node")) == str(numa)
            and str(nic.get("ports", {}).get("1", {}).get("state", "")).startswith("4:")
            and nic.get("ports", {}).get("1", {}).get("link_layer") == "InfiniBand"
        }

    groups = {}
    for numa in (0, 1):
        available = active(reader, numa) & active(donor, numa)
        override = os.environ.get(f"MC_NUMA_GROUP{numa}")
        selected = override.split(",") if override else sorted(available)
        if not selected or len(set(selected)) != len(selected) or any(name not in available for name in selected):
            raise ValueError(
                f"NUMA {numa}: need common ACTIVE port-1 IB NIC names with matching NUMA locality: {selected}"
            )
        groups[numa] = selected
    if set(groups[0]) & set(groups[1]):
        raise ValueError("NIC groups must be disjoint")
    return groups


def nic_evidence(
    before: dict[str, Any], after: dict[str, Any], groups: dict[int, list[str]], expected: int
) -> dict[str, Any]:
    """Responder TX counters verify the whole reader process's donor NIC group."""
    first, last = before.get("ib_counters", {}), after.get("ib_counters", {})
    deltas = {}
    for nic in groups[0] + groups[1]:
        key = f"{nic}/ports/1/counters/port_xmit_data"
        if key not in first or key not in last:
            return {"verified": False, "reason": f"counter unavailable: {key}"}
        change = int(last[key]) - int(first[key])
        if change < 0:
            return {"verified": False, "reason": f"counter decreased/reset: {key}"}
        deltas[nic] = change * 4
    total = sum(deltas.values())
    selected = sum(deltas[nic] for nic in groups[expected])
    share = selected / total if total else 0.0
    return {
        "verified": total > 1024**2 and share >= 0.95,
        "expected_peer_numa": expected,
        "expected_group_share": share,
        "tx_bytes_by_nic": deltas,
        "scope": "whole reader process including setup, verification and warmup; not the measured API interval",
    }


def transport_errors(native_log: str) -> list[str]:
    """A recovered GET is unsuitable for a clean NUMA-cost comparison."""
    pattern = (
        r"Completion with error|Failed status|retry_cnt\s*[:=]\s*[1-9]|CQ completion timeout|"
        r"Worker: Process failed for slice|Worker: Cannot make connection for endpoint|"
        r"transport retry counter exceeded|pausing peer rail and retrying|"
        r"Failed to complete run_iter"
    )
    return re.findall(rf"(?im)^.*(?:{pattern}).*$", native_log)[:20]


def comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Matched per-round ratios, with no zero-fill for failed or missing cases."""
    by_key = {(row["round"], row["donor_numa"], row["reader_numa"], row["affinity"]): row for row in rows}
    results = []
    for repeat in sorted({row["round"] for row in rows}):
        for reader in (0, 1):
            pairs = [
                (
                    "remote-memory/local-memory, affinity ON",
                    (repeat, 1 - reader, reader, True),
                    (repeat, reader, reader, True),
                ),
                (
                    "affinity OFF/ON, remote memory",
                    (repeat, 1 - reader, reader, False),
                    (repeat, 1 - reader, reader, True),
                ),
            ]
            for label, numerator, denominator in pairs:
                a, b = by_key.get(numerator), by_key.get(denominator)
                valid = bool(
                    a
                    and b
                    and all(
                        row.get("status") == "PASS"
                        and row.get("placement_verified") is True
                        and row.get("nic_evidence", {}).get("verified")
                        for row in (a, b)
                    )
                    and all(
                        math.isfinite(row.get("goodput_gb_s", 0)) and row.get("goodput_gb_s", 0) > 0 for row in (a, b)
                    )
                )
                results.append(
                    {
                        "comparison": label,
                        "round": repeat,
                        "reader_numa": reader,
                        "status": "VALID" if valid else "INCONCLUSIVE",
                        "numerator": a.get("case_id") if a else None,
                        "denominator": b.get("case_id") if b else None,
                        "bandwidth_ratio": a["goodput_gb_s"] / b["goodput_gb_s"] if valid else None,
                    }
                )
    return results


def write_report(
    run_dir: Path, metadata: dict[str, Any], plan: list[dict[str, Any]], rows: list[dict[str, Any]], reason: str
) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # Include the final pool placement-verification outcome in durable raw rows.
    (logs / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    fields = ["case_id", "round", "donor_numa", "reader_numa", "affinity", "status", "goodput_gb_s", "reason"]
    with (logs / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    ratios = comparisons(rows)
    atomic_json(logs / "comparisons.json", ratios)
    status = {value: sum(row.get("status") == value for row in rows) for value in ("PASS", "FAIL", "INCOMPLETE")}
    skipped = sum(not case["enabled"] for case in plan)
    unrun = max(0, sum(case["enabled"] for case in plan) - len(rows))
    lines = [
        "# Mooncake NUMA cache-read benchmark",
        "",
        f"Status: {reason}. PASS={status['PASS']}, FAIL={status['FAIL']}, INCOMPLETE={status['INCOMPLETE']}, "
        f"NOT RUN={unrun}, SKIPPED BY POLICY={skipped}.",
        f"Cluster label: `{metadata.get('settings', {}).get('cluster', 'unknown')}`; job `{metadata.get('job_id')}`; "
        f"donor `{metadata.get('donor_node')}`; reader `{metadata.get('reader_node')}`.",
        f"Image: `{metadata.get('image')}`. Settings: `{json.dumps(metadata.get('settings', {}), sort_keys=True)}`.",
        "",
        "Goal: measure real registered-host-buffer Mooncake cache GET goodput and batch latency when the donor's "
        "cache memory is local or remote to the NIC chosen by destination affinity. The requested policy may produce "
        "no measurable slowdown; the benchmark does not assume an effect.",
        "Only the explicit donor RAM-backed segment supplies capacity. It bypasses automatic NUMA segmentation. "
        "Seed, payload verification, process startup and warmup are outside the measured API interval. "
        "Reader NIC groups and context counts are identical between ON/OFF cases for each reader NUMA node.",
        "",
        "| Case | Donor RAM | Reader RAM | Affinity | Status | GET GB/s | Batch p50/p95/p99 ms | NIC group verified |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | --- |",
    ]
    indexed = {row["case_id"]: row for row in rows}
    for case in plan:
        row = indexed.get(case["case_id"], {})
        latency = row.get("latency_ms", {})
        values = "/".join(str(latency.get(p, "N/A")) for p in ("p50", "p95", "p99"))
        lines.append(
            f"| {case['case_id']} | {case['donor_numa']} | {case['reader_numa']} | {case['affinity']} | "
            f"{row.get('status', 'SKIPPED BY POLICY' if not case['enabled'] else 'NOT RUN')} | "
            f"{row.get('goodput_gb_s', 'N/A')} | {values} | {row.get('nic_evidence', {}).get('verified', False)} |"
        )
    lines.extend(["", "## Matched comparisons", ""])
    for label in sorted({item["comparison"] for item in ratios}):
        for reader in (0, 1):
            selected = [
                item["bandwidth_ratio"]
                for item in ratios
                if item["comparison"] == label and item["reader_numa"] == reader and item["status"] == "VALID"
            ]
            lines.append(
                f"- {label}; reader NUMA{reader}: median ratio={median(selected) if selected else 'N/A'}, matched rounds={len(selected)}."
            )
    lines.extend(
        [
            "",
            "Ratios require both cases to pass payload/NUMA checks and donor NIC-group counter verification. "
            "Failures, missing/reset counters and unrun cases are never converted to zero bandwidth. A ratio below "
            "1 for remote/local means the remote-memory case was slower. OFF/ON above 1 means OFF was faster in "
            "that matched remote-memory comparison. Compare ratios within each cluster; hardware and link speed "
            "differences make absolute RapLab/Hecate comparisons insufficient to attribute causality.",
            "",
            "## Sources and limits",
            "",
            "`metadata.json` and `plan.json` preserve settings and shuffled order; `logs/results.jsonl`, "
            "`logs/summary.csv`, and `logs/comparisons.json` preserve measured results and matched pairs. "
            "`logs/cases/*/` contains process commands/exit status, small separate affinity traces, reader payload "
            "checks, physical NUMA residency samples, per-batch samples and donor/reader counter snapshots. "
            "Counter-based NIC proof covers each entire reader process, not only the API timing window. "
            "NUMA residency is sampled and must not be represented as an exhaustive page audit.",
            "Mooncake may bind internal workers per HCA. Startup CPU binding does not guarantee every native "
            "thread stays on that socket; physical memory residency and NIC counters provide placement evidence.",
            "Node inventories are under `logs/inventory-*.json`; `logs/master.log` and workload native logs include "
            "the installed transport's device/GID messages. Affinity traces use separate short processes, so their "
            "logging overhead is excluded from measurement. Timed-process transport errors invalidate a clean comparison.",
            "",
            "The working set, batching, NIC selection, memory placement, seed and validation must match between "
            "compared cases. No GPUs/model, serving load or inference requests are used. This does not measure "
            "TTFT improvements or guarantee behavior at the serving pool's full capacity/concurrency.",
            "TTFT p50/p95/p99: N/A ms; ITL p50/p99: N/A ms/token; output/input/total throughput: N/A tokens/s; "
            "agentic Pareto/SLO metrics: N/A because this is a host-only cache GET diagnostic.",
            "",
            "The slurm-job-sizing skill (`src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md`) sizes the allocation "
            "to 25 minutes with a 22-minute body budget and cleanup reserve. Default configured measurement "
            "traffic is 120 s for affinity ON only, or 240 s with explicit OFF-case opt-in, plus warmup, "
            "seeding, verification and registration. No DSight/dashboard is needed: API latency, verified bytes, "
            "NUMA samples and selected NIC-group evidence directly test the hypothesis.",
        ]
    )
    (logs / "perf-analysis.md").write_text("\n".join(lines) + "\n")


def collect_inventory(run_dir: Path, label: str) -> dict[str, Any]:
    path = run_dir / "logs" / f"inventory-{label}.json"
    inventory(argparse.Namespace(output=path, scope="container", commands_timeout=3))
    info = json.loads(path.read_text())
    info["numa_bind_checks"] = {}
    for numa in (0, 1):
        completed = subprocess.run(
            ["numactl", f"--cpunodebind={numa}", f"--membind={numa}", "/bin/true"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        info["numa_bind_checks"][str(numa)] = {"returncode": completed.returncode, "stderr": completed.stderr}
        if completed.returncode:
            raise RuntimeError(f"NUMA {numa} binding failed on {label}: {completed.stderr}")
    installed = versions()
    if not installed or any(value.split("+", 1)[0].split(".post", 1)[0] != "0.3.13" for value in installed.values()):
        raise RuntimeError(f"Mooncake 0.3.13 required: {installed}")
    atomic_json(path, info)
    return info


def stop_process(process: subprocess.Popen[Any]) -> bool:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return False
    return True


def workload_argv(config_path: Path, config: dict[str, Any], role: str) -> list[str]:
    return [
        "numactl",
        f"--cpunodebind={config['numa']}",
        f"--membind={config['numa']}",
        sys.executable,
        "-m",
        "tools.mooncake_numa_workload",
        "--config",
        str(config_path),
        "--role",
        role,
    ]


def agent(run_dir: Path) -> int:
    commands, replies = run_dir / "commands", run_dir / "replies"
    commands.mkdir(exist_ok=True)
    replies.mkdir(exist_ok=True)
    try:
        info = collect_inventory(run_dir, "reader")
        atomic_json(run_dir / "reader-ready.json", {"status": "ready", "inventory": info})
    except Exception as exc:
        atomic_json(run_dir / "reader-ready.json", {"status": "error", "error": str(exc)})
        return 1
    seen = set()
    stopping = False
    active: dict[str, Any] | None = None
    shutdown_path: Path | None = None

    def signal_stop(*_: Any) -> None:
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, signal_stop)
    while True:
        # Poll the mailbox while a reader runs, so shutdown can cancel an in-flight
        # GET. A completed/stopped ACK always follows confirmed process reaping.
        for path in sorted(commands.glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            command = json.loads(path.read_text())
            reply = replies / path.name
            if command["action"] == "shutdown":
                stopping, shutdown_path = True, reply
                break
            if command["action"] == "ping":
                atomic_json(reply, {"status": "pong"})
                continue
            if active is not None or stopping:
                atomic_json(reply, {"status": "busy"})
                continue
            try:
                if command["action"] != "run":
                    raise ValueError(f"Unknown action: {command['action']}")
                config_path = Path(command["config"])
                config = json.loads(config_path.read_text())
                argv = workload_argv(config_path, config, command["role"])
                with Path(command["native_log"]).open("w") as log:
                    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active = {
                    "child": child,
                    "reply": reply,
                    "argv": argv,
                    "deadline": time.monotonic() + command.get("timeout", 120),
                }
            except (OSError, ValueError, KeyError) as exc:
                atomic_json(reply, {"status": "completed", "returncode": 1, "error": str(exc)})
        if active is not None:
            child = active["child"]
            timed_out = time.monotonic() >= active["deadline"]
            if child.poll() is not None or stopping or timed_out:
                confirmed = stop_process(child)
                atomic_json(
                    active["reply"],
                    {
                        "status": "completed" if confirmed else "teardown_error",
                        "returncode": child.returncode,
                        "timed_out": timed_out,
                        "cancelled": stopping,
                        "argv": active["argv"],
                    },
                )
                active = None
                if not confirmed:
                    if shutdown_path is not None:
                        atomic_json(shutdown_path, {"status": "teardown_error"})
                    return 1
        if stopping:
            if shutdown_path is not None:
                atomic_json(shutdown_path, {"status": "stopped"})
            return 0
        time.sleep(0.05)


class Controller:
    def __init__(self, run_dir: Path, settings: Settings) -> None:
        self.run_dir, self.settings = run_dir, settings
        self.deadline = time.monotonic() + settings.budget
        self.interrupted = False
        self.sequence = 0
        self.rows: list[dict[str, Any]] = []
        self.plan = make_plan(settings)
        self.children: list[subprocess.Popen[Any]] = []
        self.groups: dict[int, list[str]] = {}
        self.donor_reader_safe = True
        self.metadata = {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "image": os.environ.get("MC_NUMA_IMAGE"),
            "settings": asdict(settings),
            "start_utc": time.strftime("%FT%TZ", time.gmtime()),
        }

    def wait(self, path: Path, seconds: float, *, identifier: str | None = None) -> dict[str, Any]:
        deadline = min(self.deadline, time.monotonic() + seconds)
        while time.monotonic() < deadline:
            if self.interrupted:
                raise InterruptedError("controller interrupted")
            value = read_mailbox_json(path)
            if value is not None and (identifier is None or value.get("id") == identifier):
                return value
            time.sleep(0.05)
        raise TimeoutError(f"waiting for {path.relative_to(self.run_dir)}")

    def request(self, action: str, **fields: Any) -> tuple[str, Path]:
        self.sequence += 1
        identifier = f"{self.sequence:06d}"
        atomic_json(self.run_dir / "commands" / f"{identifier}.json", {"action": action, **fields})
        return identifier, self.run_dir / "replies" / f"{identifier}.json"

    def launch(self, argv: list[str], log_path: Path) -> subprocess.Popen[Any]:
        with log_path.open("w") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(process)
        atomic_json(log_path.with_suffix(".command.json"), {"argv": argv, "pid": process.pid})
        return process

    def execute_reader(self, config: dict[str, Any], role: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
        path = Path(config["output"]).with_suffix(".config.json")
        atomic_json(path, config)
        timeout = max(120, self.settings.seconds + self.settings.warmup_seconds + 90)
        if time.monotonic() + timeout + 20 >= self.deadline:
            raise InterruptedError("insufficient body budget for reader and confirmed teardown")
        identifier, reply_path = self.request(
            "run", config=str(path), role=role, native_log=str(path.parent / f"{label}.native.log"), timeout=timeout
        )
        self.donor_reader_safe = False
        reply = self.wait(reply_path, timeout + 20)
        if reply.get("status") != "completed":
            raise RuntimeError(f"reader teardown unconfirmed: {reply}")
        self.donor_reader_safe = True
        result = read_mailbox_json(Path(config["output"])) or {"status": "FAIL", "error": "reader produced no result"}
        if reply.get("returncode") != 0 or reply.get("timed_out") or reply.get("cancelled"):
            result.update(status="FAIL", error=f"reader process: {reply}; workload: {result.get('error', '')}")
        return result, reply

    def donor_request(self, config: dict[str, Any], action: str, keys: list[str] | None = None) -> dict[str, Any]:
        self.sequence += 1
        identifier = f"donor-{self.sequence}"
        atomic_json(Path(config["donor_verify_request"]), {"id": identifier, "action": action, "keys": keys or []})
        result = self.wait(Path(config["donor_verify_result"]), 30, identifier=identifier)
        if result.get("status") != "PASS":
            raise RuntimeError(f"donor {action} failed: {result}")
        return result

    def start_master(self, ip: str) -> str:
        sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
        try:
            for sock in sockets:
                sock.bind(("", 0))
            rpc_port, metrics_port = (sock.getsockname()[1] for sock in sockets)
        finally:
            for sock in sockets:
                sock.close()
        executable = shutil.which("mooncake_master")
        if not executable:
            raise RuntimeError("mooncake_master is missing from the container")
        process = self.launch(
            [
                executable,
                f"--rpc_port={rpc_port}",
                f"--metrics_port={metrics_port}",
                "--default_kv_lease_ttl=60000",
                "--eviction_ratio=0.02",
                "--allocation_strategy=random",
                "--memory_allocator=offset",
            ],
            self.run_dir / "logs/master.log",
        )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and time.monotonic() < self.deadline:
            if process.poll() is not None:
                raise RuntimeError("Mooncake master exited during startup; see logs/master.log")
            try:
                with socket.create_connection((ip, rpc_port), timeout=0.2):
                    return f"{ip}:{rpc_port}"
            except OSError:
                time.sleep(0.2)
        raise TimeoutError("Mooncake master startup")

    def run_pool(self, repeat: int, numa: int, master: str) -> None:
        folder = self.run_dir / "logs/cases" / f"round{repeat}-donor{numa}"
        folder.mkdir(parents=True)
        config = {
            **asdict(self.settings),
            "run_dir": str(self.run_dir),
            "case_id": folder.name,
            "master": master,
            "numa": numa,
            "nics": self.groups[0] + self.groups[1],
            "affinity": True,
            "trace": False,
            "key_prefix": f"numa/{self.metadata['job_id']}/{folder.name}",
            "donor_ready": str(folder / "donor-ready.json"),
            "donor_verify_request": str(folder / "donor-request.json"),
            "donor_verify_result": str(folder / "donor-response.json"),
            "stop_file": str(folder / "stop.json"),
            "output": str(folder / "donor.json"),
            "native_log": str(folder / "donor.native.log"),
        }
        path = folder / "donor.config.json"
        atomic_json(path, config)
        donor = self.launch(workload_argv(path, config, "donor"), folder / "donor.native.log")
        try:
            ready = self.wait(Path(config["donor_ready"]), 120)
            if ready.get("status") not in ("ready", "PASS"):
                raise RuntimeError(f"donor startup failed: {ready}")
            seed_config = {**config, "numa": 0, "nics": self.groups[0], "output": str(folder / "seed.json")}
            seeded, _ = self.execute_reader(seed_config, "seed", "seed")
            if seeded.get("status") != "PASS":
                raise RuntimeError(f"cache seed failed: {seeded}")
            keys = seeded["keys"]
            # Owner checks cover every key during seeding. Sample physical pages
            # across at most 64 cached objects to bound remote-control overhead.
            sampled_keys = [keys[index * len(keys) // min(64, len(keys))] for index in range(min(64, len(keys)))]
            verification = self.donor_request(config, "verify", sampled_keys)
            atomic_json(folder / "donor-residency-before.json", verification)
            for case in self.plan:
                if case["round"] != repeat or case["donor_numa"] != numa or not case["enabled"]:
                    continue
                if self.interrupted or time.monotonic() + 160 > self.deadline:
                    raise InterruptedError("time budget exhausted; remaining cases NOT RUN")
                case_dir = folder / case["case_id"]
                case_dir.mkdir()
                reader_config = {
                    **config,
                    **case,
                    "numa": case["reader_numa"],
                    "nics": self.groups[case["reader_numa"]],
                    "output": str(case_dir / "reader.json"),
                }
                print(f"{case['case_id']}: starting", flush=True)
                row = dict(case)
                try:
                    if repeat == 0:
                        proof, proof_process = self.execute_reader(
                            {**reader_config, "trace": True, "output": str(case_dir / "trace.json")}, "read", "trace"
                        )
                        row["trace"] = {"result": proof, "process": proof_process}
                        if proof.get("status") != "PASS":
                            raise RuntimeError(f"short affinity proof failed: {proof}")
                        trace_errors = transport_errors((case_dir / "trace.native.log").read_text(errors="replace"))
                        if trace_errors:
                            raise RuntimeError(f"short affinity proof encountered transport errors: {trace_errors}")
                    before = self.donor_request(config, "snapshot")
                    result, process = self.execute_reader(reader_config, "read", "reader")
                    after = self.donor_request(config, "snapshot")
                    atomic_json(case_dir / "donor-counters.json", {"before": before, "after": after})
                    row.update(result, process=process)
                    expected = case["reader_numa"] if case["affinity"] else numa
                    row["nic_evidence"] = nic_evidence(
                        before.get("health", {}), after.get("health", {}), self.groups, expected
                    )
                    native = (case_dir / "reader.native.log").read_text(errors="replace")
                    errors = transport_errors(native)
                    if errors:
                        row.update(
                            status="FAIL",
                            reason="transport error/retry in timed reader process",
                            transport_errors=errors[:20],
                        )
                    if row.get("status") != "PASS":
                        row["reason"] = row.get("reason") or row.get("error") or "reader failed"
                except (RuntimeError, TimeoutError, InterruptedError) as exc:
                    row.update(status="INCOMPLETE" if not self.donor_reader_safe else "FAIL", reason=str(exc))
                self.rows.append(row)
                with (self.run_dir / "logs/results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    f"{case['case_id']}: {row['status']} {row.get('goodput_gb_s')} GB/s; NIC evidence={row.get('nic_evidence')}",
                    flush=True,
                )
                if not self.donor_reader_safe:
                    raise RuntimeError("reader teardown unconfirmed; refusing another case or explicit donor unmount")
            atomic_json(folder / "donor-residency-after.json", self.donor_request(config, "verify", sampled_keys))
            for row in self.rows:
                if row["round"] == repeat and row["donor_numa"] == numa:
                    row["placement_verified"] = True
        finally:
            if self.donor_reader_safe:
                atomic_json(Path(config["stop_file"]), {"stop": True})
                with contextlib.suppress(subprocess.TimeoutExpired):
                    donor.wait(timeout=30)
                if not stop_process(donor):
                    raise RuntimeError("donor teardown unconfirmed; refusing another pool")
                final = read_mailbox_json(Path(config["output"]))
                if donor.returncode != 0 or not final or final.get("status") != "PASS":
                    raise RuntimeError(f"donor cleanup/verification failed: exit={donor.returncode}, result={final}")

    def run(self) -> int:
        logs = self.run_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "metadata.json").exists():
            raise RuntimeError("Refusing stale run directory")
        reason, rc = "preflight incomplete", 1
        try:
            donor_inventory = collect_inventory(self.run_dir, "donor")
            ready = self.wait(self.run_dir / "reader-ready.json", 360)
            if ready.get("status") != "ready":
                raise RuntimeError(f"reader preflight failed: {ready}")
            reader_inventory = ready["inventory"]
            self.groups = discover_groups(reader_inventory, donor_inventory)
            self.metadata.update(
                donor_node=donor_inventory.get("slurm_node") or donor_inventory["host"],
                reader_node=reader_inventory.get("slurm_node") or reader_inventory["host"],
                groups=self.groups,
            )
            atomic_json(self.run_dir / "metadata.json", self.metadata)
            atomic_json(self.run_dir / "plan.json", self.plan)
            _, reply = self.request("ping")
            started = time.monotonic()
            if self.wait(reply, 15).get("status") != "pong":
                raise RuntimeError("reader mailbox round trip failed")
            atomic_json(logs / "mailbox-check.json", {"round_trip_s": time.monotonic() - started})
            master = self.start_master(donor_inventory["ip"])
            for repeat in range(self.settings.rounds):
                for numa in (0, 1) if repeat % 2 == 0 else (1, 0):
                    if self.interrupted or time.monotonic() + 200 > self.deadline:
                        raise InterruptedError("time budget exhausted; remaining cases NOT RUN")
                    self.run_pool(repeat, numa, master)
            reason = "all enabled cases completed"
            rc = int(any(row.get("status") != "PASS" for row in self.rows))
        except (RuntimeError, TimeoutError, InterruptedError, ValueError, KeyError, OSError) as exc:
            reason = str(exc)
            print(reason, flush=True)
        finally:
            shutdown_confirmed = False
            try:
                _, path = self.request("shutdown")
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if (read_mailbox_json(path) or {}).get("status") == "stopped":
                        self.donor_reader_safe = True
                        shutdown_confirmed = True
                        break
                    time.sleep(0.05)
            finally:
                if not shutdown_confirmed:
                    reason += "; reader shutdown not confirmed"
                    rc = 1
                for child in reversed(self.children):
                    if self.donor_reader_safe:
                        confirmed = stop_process(child)
                    else:
                        # No graceful donor unmount while a remote GET may be
                        # active. Terminate the failed job's processes directly.
                        if child.poll() is None:
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(child.pid, signal.SIGKILL)
                        try:
                            child.wait(timeout=5)
                            confirmed = True
                        except subprocess.TimeoutExpired:
                            confirmed = False
                    if not confirmed:
                        reason += "; child teardown unconfirmed"
                        rc = 1
                self.metadata["finish_utc"] = time.strftime("%FT%TZ", time.gmtime())
                atomic_json(self.run_dir / "metadata.json", self.metadata)
                write_report(self.run_dir, self.metadata, self.plan, self.rows, reason)
        return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("controller", "agent"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.role == "agent":
        return agent(args.run_dir.resolve())
    controller = Controller(args.run_dir.resolve(), Settings.from_env())
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: setattr(controller, "interrupted", True))
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
