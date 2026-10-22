#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare overlapping HOST_NUMA EGM pools while observing Linux reclamation.

The native Store configuration and MC_MAX_MR_SIZE semantics follow Mooncake
042ab8a18b0e6ccfe87782261a5556a3dd4446e4. Inventory and reporting use only the
standard library; CUDA and the native Store are loaded only in owned children.
No cache flush, shrinker scan, driver reconfiguration or GPU payload is used.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from tools import mooncake_egm_alloc_diagnostics as allocation

_EXTRA_FILES = (
    "/proc/vmstat",
    "/proc/pagetypeinfo",
    "/proc/zoneinfo",
    "/proc/driver/nvidia/version",
    "/proc/driver/nvidia/params",
    "/proc/sys/vm/min_free_kbytes",
)
_CGROUP_FILES = ("memory.numa_stat", "memory.swap.current", "memory.swap.max")
_CUDA_ERRORS = {2: "CUDA_ERROR_OUT_OF_MEMORY", 100: "CUDA_ERROR_NO_DEVICE"}
_GIB = 1024**3
_CLEANUP_GRACE_SECONDS = 15.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def shrinker_counts(root: Path = Path("/")) -> dict[str, Any]:
    """Read counts only, distinguishing missing interfaces from denied access."""
    directory = root / "sys/kernel/debug/shrinker"
    result: dict[str, Any] = {"directory": "/sys/kernel/debug/shrinker", "counts": {}}
    try:
        entries = sorted(directory.iterdir())
    except OSError as exc:
        result.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        return result
    for entry in entries:
        if re.fullmatch(r"nv-sysmem-alloc-node-\d+-order-.*", entry.name):
            result["counts"][entry.name] = allocation.read_text(entry / "count")
    result["status"] = "available" if result["counts"] else "no_matching_interface"
    return result


def inventory(root: Path = Path("/"), *, lightweight: bool = False) -> dict[str, Any]:
    result = allocation.inventory(root)
    files = result["files"]
    for name in _EXTRA_FILES:
        if not lightweight or name in ("/proc/vmstat",):
            files[name] = allocation.read_text(root / name.lstrip("/"))
    for node in sorted((root / "sys/devices/system/node").glob("node[0-9]*")):
        name = "/" + str((node / "vmstat").relative_to(root))
        files[name] = allocation.read_text(node / "vmstat")
    for group in result["cgroups"]:
        for ancestor in group.get("ancestors", []):
            for name in _CGROUP_FILES:
                path = str(Path(ancestor) / name)
                files[path] = allocation.read_text(root / path.lstrip("/"))
    result.update(time_utc=utc_now(), page_size_bytes=os.sysconf("SC_PAGE_SIZE"), shrinker=shrinker_counts(root))
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result
    except (OSError, ValueError) as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def cuda_errors(log_text: str, child: dict[str, Any]) -> list[dict[str, Any]]:
    # Python exceptions (including cuInit failure) may precede native logging.
    evidence = log_text + "\n" + json.dumps({key: child.get(key) for key in ("error", "close_error", "cuda")})
    codes = sorted({int(value) for value in re.findall(r"(?:CUDA (?:result|error)|cuInit returned)\s+(\d+)", evidence)})
    return [{"code": code, "name": _CUDA_ERRORS.get(code, "unknown CUDA error name")} for code in codes]


@dataclass(frozen=True)
class Placement:
    gpu: str
    numa: str


@dataclass(frozen=True)
class Trial:
    name: str
    placements: tuple[Placement, Placement]
    chunked: bool = False


TRIALS = {
    row.name: row
    for row in (
        Trial("same-numa", (Placement("0", "0"), Placement("1", "0"))),
        Trial("cross-numa", (Placement("0", "0"), Placement("2", "1"))),
        Trial("reverse-gpu", (Placement("1", "0"), Placement("0", "0"))),
        Trial("chunked", (Placement("0", "0"), Placement("1", "0")), chunked=True),
    )
}


@dataclass(frozen=True)
class Settings:
    config: Path
    output_dir: Path
    pool_size: int
    cases: tuple[str, ...]
    timeout_seconds: float
    budget_seconds: float
    sample_interval: float
    hold_seconds: float
    chunk_size: int
    donor_helper: Path


@dataclass(frozen=True)
class RunningDonor:
    process: subprocess.Popen[Any]
    log_stream: TextIO
    result_path: Path
    log_path: Path
    metadata: dict[str, Any]


def child(args: argparse.Namespace) -> int:
    """Keep one native HOST_NUMA pool alive until its parent requests cleanup."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    donor = allocation.load_donor(args.donor_helper)
    stop = threading.Event()
    handlers = {number: signal.signal(number, lambda *_: stop.set()) for number in (signal.SIGTERM, signal.SIGINT)}
    result: dict[str, Any] = {
        "status": "initializing",
        "pid": os.getpid(),
        "time_utc": utc_now(),
        "cleanup_confirmed": False,
        "setup_started": False,
        "ready_seen": False,
        "gpu_selector": args.gpu,
        "numa_nodes": args.numa,
        "max_mr_size": os.environ.get("MC_MAX_MR_SIZE"),
        "before": inventory(),
    }
    store = None
    allocation.write_json(args.result, result)
    try:
        donor.configure_environment()
        result["config"] = donor.provider_config(args.config, args.pool_size, args.numa)
        result["cuda"] = allocation.cuda_inventory()
        donor.require_cuda_devices()
        store = donor.load_store_type()()
        result.update(status="allocating", setup_started=True)
        allocation.write_json(args.result, result)
        started = time.monotonic()
        result["setup_result"] = store.setup(result["config"])
        result["setup_seconds"] = time.monotonic() - started
        if result["setup_result"] == 0:
            result.update(status="ready", ready_seen=True, ready_time_utc=utc_now())
            allocation.write_json(args.result, result)
            stop.wait()
        else:
            result["status"] = "setup_failed"
    except Exception as exc:
        result.update(status="child_failed", error=f"{type(exc).__name__}: {exc}")
        print(result["error"], flush=True)
    finally:
        if store is not None:
            setup_status = result["status"]
            result["status"] = "closing"
            allocation.write_json(args.result, result)
            try:
                result["close_result"] = store.close()
                result["cleanup_confirmed"] = result["close_result"] == 0
                result["status"] = (
                    ("closed" if setup_status == "ready" else setup_status)
                    if result["cleanup_confirmed"]
                    else "cleanup_failed"
                )
            except Exception as exc:
                result.update(status="cleanup_failed", close_error=f"{type(exc).__name__}: {exc}")
                print(result["close_error"], flush=True)
        result.update(after=inventory(), end_time_utc=utc_now())
        allocation.write_json(args.result, result)
        for number, handler in handlers.items():
            signal.signal(number, handler)
    return 0 if result["status"] == "closed" and result["cleanup_confirmed"] else 1


def start_donor(settings: Settings, trial: Trial, index: int, directory: Path) -> RunningDonor:
    placement = trial.placements[index]
    result_path, log_path = directory / f"donor-{index}.json", directory / f"donor-{index}.log"
    result_path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = placement.gpu
    # All unchunked controls use the native default, independently of shell state.
    environment.pop("MC_MAX_MR_SIZE", None)
    if trial.chunked:
        environment["MC_MAX_MR_SIZE"] = str(settings.chunk_size)
    command = [
        sys.executable,
        "-m",
        "tools.mooncake_egm_reclaim",
        "child",
        "--config",
        str(settings.config),
        "--result",
        str(result_path),
        "--gpu",
        placement.gpu,
        "--numa",
        placement.numa,
        "--pool-size",
        str(settings.pool_size),
        "--donor-helper",
        str(settings.donor_helper),
    ]
    metadata: dict[str, Any] = {
        "index": index,
        "gpu_selector": placement.gpu,
        "numa_nodes": placement.numa,
        "segment_size_bytes": settings.pool_size,
        "max_mr_size": environment.get("MC_MAX_MR_SIZE"),
        "result": result_path.name,
        "log": log_path.name,
        "start_time_utc": utc_now(),
    }
    log = log_path.open("w")
    try:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
    except BaseException:
        log.close()
        raise
    metadata["pid"] = process.pid
    return RunningDonor(process, log, result_path, log_path, metadata)


def stop_donor(running: RunningDonor, *, grace: float = _CLEANUP_GRACE_SECONDS) -> None:
    """Signal only this launcher's process group; preserve uncertain cleanup."""
    process, metadata = running.process, running.metadata
    metadata["stop_time_utc"] = utc_now()
    if process.poll() is None:
        metadata["stop_method"] = "SIGTERM"
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            metadata["stop_method"] = "SIGKILL_after_grace"
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                metadata["reap_error"] = "Child did not exit after SIGKILL"
    else:
        metadata["stop_method"] = "already_exited"
    running.log_stream.close()
    result = read_json(running.result_path)
    metadata["child"] = {key: value for key, value in result.items() if key not in ("before", "after")}
    metadata["returncode"] = process.poll()
    metadata["cuda_errors"] = cuda_errors(allocation.read_text(running.log_path).get("text", ""), result)
    expected = {"closed": 0, "setup_failed": 1}
    metadata["cleanup_confirmed"] = (
        result.get("cleanup_confirmed") is True
        and result.get("status") in expected
        and metadata["returncode"] == expected[result["status"]]
        and metadata["stop_method"] != "SIGKILL_after_grace"
    )


class Sampler:
    def __init__(self, directory: Path, interval: float) -> None:
        self.directory = directory
        self.interval = interval
        self.next_sample = 0.0
        self.phase = "baseline"

    def sample(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now >= self.next_sample:
            with (self.directory / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps({"phase": self.phase, **inventory(lightweight=True)}, sort_keys=True) + "\n")
            self.next_sample = now + self.interval

    def snapshot(self, stage: str, record: dict[str, Any]) -> None:
        self.phase = stage
        name = f"{stage}.json"
        allocation.write_json(self.directory / name, inventory())
        record.setdefault("snapshots", {})[stage] = name
        self.sample(force=True)


def wait_for_setup(
    running: RunningDonor,
    first: RunningDonor | None,
    deadline: float,
    budget_deadline: float,
    stop: threading.Event,
    sampler: Sampler,
) -> str:
    while True:
        sampler.sample()
        if stop.is_set():
            return "interrupted"
        if time.monotonic() >= budget_deadline:
            return "budget_exhausted"
        if first is not None and first.process.poll() is not None:
            return "first_donor_died"
        if first is not None and read_json(first.result_path).get("status") != "ready":
            return "first_donor_not_ready"
        result = read_json(running.result_path)
        returncode = running.process.poll()
        if returncode is not None:
            if result.get("status") == "setup_failed" and result.get("cleanup_confirmed") is True and returncode == 1:
                return "setup_failed"
            return "child_died"
        if result.get("status") == "ready" and result.get("setup_result") == 0:
            running.metadata["ready_observed_utc"] = utc_now()
            return "ready"
        if time.monotonic() >= deadline:
            return "timeout"
        stop.wait(min(0.1, max(0.001, deadline - time.monotonic())))


def run_trial(
    settings: Settings, trial: Trial, stop: threading.Event, budget_deadline: float, record: dict[str, Any]
) -> None:
    directory = settings.output_dir / trial.name
    directory.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(directory, settings.sample_interval)
    running: list[RunningDonor] = []
    record.update(name=trial.name, status="running", donors=[], snapshots={}, samples=f"{trial.name}/samples.jsonl")
    failure: str | None = None
    try:
        sampler.snapshot("baseline", record)
        for index in range(2):
            if stop.is_set() or time.monotonic() >= budget_deadline:
                failure = "interrupted" if stop.is_set() else "budget_exhausted"
                break
            if index and running[0].process.poll() is not None:
                failure = "first_donor_died"
                break
            sampler.snapshot("before-first" if index == 0 else "before-second", record)
            sampler.phase = "allocating-first" if index == 0 else "allocating-second"
            donor = start_donor(settings, trial, index, directory)
            running.append(donor)
            record["donors"].append(donor.metadata)
            outcome = wait_for_setup(
                donor,
                running[0] if index else None,
                min(time.monotonic() + settings.timeout_seconds, budget_deadline),
                budget_deadline,
                stop,
                sampler,
            )
            donor.metadata["setup_observation"] = outcome
            sampler.snapshot(
                ("first-ready" if outcome == "ready" else "first-result") if index == 0 else "second-result", record
            )
            if outcome == "setup_failed":
                record["allocation_outcome"] = "first_allocation_failed" if index == 0 else "second_allocation_failed"
                break
            if outcome != "ready":
                failure = outcome
                break
        if len(running) == 2 and all(donor.metadata.get("setup_observation") == "ready" for donor in running):
            record["allocation_outcome"] = "two_pools_allocated"
        # Keep the first allocation resident through the second result/snapshots,
        # including a controlled second failure, before cleaning either child.
        if failure is None and running and running[0].metadata.get("setup_observation") == "ready":
            until = min(time.monotonic() + settings.hold_seconds, budget_deadline)
            while time.monotonic() < until:
                sampler.sample()
                if stop.is_set():
                    failure = "interrupted"
                    break
                if any(
                    donor.process.poll() is not None
                    for donor in running
                    if donor.metadata.get("setup_observation") == "ready"
                ):
                    failure = "child_died_during_hold"
                    break
                stop.wait(min(0.1, max(0.001, until - time.monotonic())))
            if time.monotonic() >= budget_deadline and failure is None:
                failure = "budget_exhausted"
    except Exception as exc:
        failure = f"controller_error: {type(exc).__name__}: {exc}"
    finally:
        for donor in reversed(running):
            stop_donor(donor)
        sampler.snapshot("after-close", record)
        record["cleanup_confirmed"] = bool(running) and all(donor.metadata["cleanup_confirmed"] for donor in running)
        if not record["cleanup_confirmed"] and failure is None:
            failure = "uncertain_cleanup"
        record.update(status="complete" if failure is None else "incomplete", end_time_utc=utc_now())
        if failure:
            record["failure"] = failure
        allocation.write_json(directory / "trial.json", record)


def run(settings: Settings) -> int:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = settings.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Refusing to overwrite existing diagnostic results: {summary_path}")
    summary: dict[str, Any] = {
        "status": "running",
        "time_utc": utc_now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "parameters": {
            "pool_size_bytes": settings.pool_size,
            "cases": list(settings.cases),
            "timeout_seconds": settings.timeout_seconds,
            "budget_seconds": settings.budget_seconds,
            "sample_interval": settings.sample_interval,
            "hold_seconds": settings.hold_seconds,
            "chunk_size_bytes": settings.chunk_size,
            "inherited_max_mr_size": os.environ.get("MC_MAX_MR_SIZE"),
            "baseline_max_mr_size": "unset (native default)",
        },
        "order_caveat": "Serial trials can change Linux/driver cache state; no cache flush or shrinker scan is performed.",
        "native_reference": "042ab8a18b0e6ccfe87782261a5556a3dd4446e4",
        "cases": [],
    }
    stop = threading.Event()
    handlers = {number: signal.signal(number, lambda *_: stop.set()) for number in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    allocation.write_json(summary_path, summary)
    try:
        allocation.write_json(settings.output_dir / "inventory.json", inventory())
        for name in settings.cases:
            if stop.is_set() or time.monotonic() >= started + settings.budget_seconds:
                summary["status"] = "interrupted" if stop.is_set() else "budget_exhausted"
                break
            record: dict[str, Any] = {}
            summary["cases"].append(record)
            run_trial(settings, TRIALS[name], stop, started + settings.budget_seconds, record)
            allocation.write_json(summary_path, summary)
            if record["status"] != "complete":
                summary["status"] = "interrupted" if stop.is_set() else "stopped_incomplete"
                break
        if summary["status"] == "running":
            summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="controller_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        summary.update(elapsed_seconds=time.monotonic() - started, end_time_utc=utc_now())
        allocation.write_json(settings.output_dir / "inventory-after.json", inventory())
        allocation.write_json(summary_path, summary)
        for number, handler in handlers.items():
            signal.signal(number, handler)
    print(json.dumps({"reclaim_diagnostics": summary["status"], "summary": str(summary_path)}), flush=True)
    return 0 if summary["status"] == "complete" else 1


def numeric_fields(text: str, *, meminfo: bool = False) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        parts = re.sub(r"^Node\s+\d+\s+", "", line).split()
        if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
            values[parts[0].rstrip(":")] = int(parts[1]) * (1024 if meminfo and parts[2:] == ["kB"] else 1)
    return values


def snapshot_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
    files = snapshot.get("files", {})
    nodes = {}
    for path, payload in files.items():
        match = re.fullmatch(r"/sys/devices/system/node/node(\d+)/meminfo", path)
        if match:
            nodes[match[1]] = numeric_fields(payload.get("text", ""), meminfo=True)
    cgroups = {}
    for group in snapshot.get("cgroups", []):
        for ancestor in group.get("ancestors", []):
            row: dict[str, Any] = {}
            for name in ("memory.current", "memory.max", "memory.usage_in_bytes", "memory.limit_in_bytes"):
                value = files.get(f"{ancestor}/{name}", {}).get("text", "").strip()
                if value:
                    row[name] = int(value) if value.isdecimal() else value
            row["events"] = numeric_fields(files.get(f"{ancestor}/memory.events", {}).get("text", ""))
            row["numa_stat"] = files.get(f"{ancestor}/memory.numa_stat", {})
            cgroups[ancestor] = row
    return {
        "time_utc": snapshot.get("time_utc"),
        "nodes": nodes,
        "vmstat": numeric_fields(files.get("/proc/vmstat", {}).get("text", "")),
        "cgroups": cgroups,
        "shrinker": snapshot.get("shrinker", {"status": "not_recorded"}),
        "missing_files": {name: value["error"] for name, value in files.items() if "error" in value},
    }


def deltas(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: value - before[name] for name, value in after.items() if name in before}


def gib(value: Any) -> str:
    return f"{value / _GIB:.2f}" if isinstance(value, (int, float)) else "N/A"


def evidence_gaps(directory: Path, case: dict[str, Any]) -> list[str]:
    """Do not call an interrupted or artifact-incomplete observation complete."""
    name = case.get("name", "unknown")
    folder = directory / name
    gaps: list[str] = []
    trial = read_json(folder / "trial.json")
    if trial.get("read_error"):
        gaps.append(f"{name}/trial.json: {trial['read_error']}")
    snapshots = case.get("snapshots", {})
    required = {"baseline", "before-first", "after-close"}
    required |= (
        {"first-result"}
        if case.get("allocation_outcome") == "first_allocation_failed"
        else {"first-ready", "before-second", "second-result"}
    )
    for stage in required:
        snapshot = read_json(folder / snapshots.get(stage, f"missing-{stage}.json"))
        if snapshot.get("read_error") or not isinstance(snapshot.get("files"), dict):
            gaps.append(f"{name}/{stage}: missing or unreadable inventory snapshot")
    expected_donors = 1 if case.get("allocation_outcome") == "first_allocation_failed" else 2
    donors = case.get("donors", [])
    if len(donors) != expected_donors:
        gaps.append(f"{name}: expected {expected_donors} donor records, found {len(donors)}")
    for donor in donors:
        child = read_json(folder / donor.get("result", "missing-child.json"))
        if child.get("read_error") or child.get("cleanup_confirmed") is not True:
            gaps.append(f"{name}/donor-{donor.get('index', '?')}: missing final result or confirmed cleanup")
        if allocation.read_text(folder / donor.get("log", "missing-child.log")).get("error"):
            gaps.append(f"{name}/donor-{donor.get('index', '?')}: missing or unreadable native log")
    try:
        with (directory / case.get("samples", f"{name}/samples.jsonl")).open() as stream:
            first = json.loads(stream.readline())
            if not isinstance(first, dict) or "phase" not in first:
                raise ValueError("No phase-labelled sample")
    except (OSError, ValueError) as exc:
        gaps.append(f"{name}/samples.jsonl: {type(exc).__name__}: {exc}")
    return gaps


def report(run_dir: Path, exit_code: int) -> int:
    """Write the actual run's required report even after early startup failure."""
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    directory = logs / "reclaim"
    summary = read_json(directory / "summary.json")
    raw_cases = summary.get("cases", [])
    cases = [case for case in raw_cases if isinstance(case, dict)] if isinstance(raw_cases, list) else []
    parameters = summary.get("parameters", {})
    parameters = parameters if isinstance(parameters, dict) else {}
    gaps = [gap for case in cases for gap in evidence_gaps(directory, case)]
    complete = (
        exit_code == 0
        and summary.get("status") == "complete"
        and [case.get("name") for case in cases] == parameters.get("cases")
        and bool(cases)
        and not gaps
        and all(case.get("status") == "complete" and case.get("cleanup_confirmed") is True for case in cases)
    )
    lines = [
        "# Mooncake EGM reclamation investigation",
        "",
        f"Diagnostic status: **{'COMPLETE' if complete else 'INCOMPLETE'}**. Launcher exit code: `{exit_code}`. Generated: `{utc_now()}`.",
        "",
        "Completion means the planned observations and cleanup completed; allocation failures are diagnostic outcomes, not successful allocations.",
        "",
        "Goal: determine whether a second resident HOST_NUMA EGM pool fails because of NUMA-local capacity, allocation size, GPU choice or reclaim behavior. "
        "The baseline keeps GPU 0 / NUMA 0 resident while GPU 1 / NUMA 0 allocates; controls change NUMA, reverse GPU order, or cap each allocation chunk.",
        "",
        f"Host: `{summary.get('hostname', 'not recorded')}`. Window: `{summary.get('time_utc', 'not recorded')}` to `{summary.get('end_time_utc', 'not recorded')}`. "
        f"Pool size: {gib(parameters.get('pool_size_bytes'))} GiB per donor. Two donor processes overlap; there are no inference requests or payload transfers.",
        "",
        "All native pools request `HOST_NUMA` via EGM, `local_buffer_size=0`, `protocol=nvlink`, and `rdma_devices=''`; "
        "no RDMA or device-memory fallback is configured. CUDA context/driver bookkeeping can still consume resources.",
        "",
        "| Trial | Observation | Completion | Donor setup / close return codes | CUDA errors |",
        "| --- | --- | --- | --- | --- |",
    ]
    analyses: list[dict[str, Any]] = []
    for case in cases:
        donors = case.get("donors", [])
        outcomes = (
            "; ".join(
                f"GPU {donor.get('gpu_selector')} / NUMA {donor.get('numa_nodes')}: {donor.get('child', {}).get('setup_result', 'missing')} / {donor.get('child', {}).get('close_result', 'missing')}"
                for donor in donors
            )
            or "none attempted"
        )
        errors = (
            "; ".join(
                f"GPU {donor.get('gpu_selector')}: {error['name']} ({error['code']})"
                for donor in donors
                for error in donor.get("cuda_errors", [])
            )
            or "none recorded (see child logs/JSON)"
        )
        lines.append(
            f"| [{case.get('name', '?')}](reclaim/{case.get('name', '')}/trial.json) | {case.get('allocation_outcome', 'not recorded')}{'; failure: ' + case['failure'] if case.get('failure') else ''} | {case.get('status', 'missing')} | {outcomes} | {errors} |"
        )
        stages: dict[str, Any] = {}
        for stage, name in case.get("snapshots", {}).items():
            snapshot = read_json(directory / case["name"] / name)
            stages[stage] = snapshot_metrics(snapshot)
        before = stages.get("before-second", {})
        after = stages.get("second-result", {})
        analyses.append(
            {
                "trial": case.get("name"),
                "stages": stages,
                "second_attempt_deltas": {
                    "nodes_bytes": {
                        node: deltas(values, after.get("nodes", {}).get(node, {}))
                        for node, values in before.get("nodes", {}).items()
                    },
                    "vmstat": deltas(before.get("vmstat", {}), after.get("vmstat", {})),
                    "cgroups": {
                        path: {
                            "events": deltas(
                                row.get("events", {}), after.get("cgroups", {}).get(path, {}).get("events", {})
                            ),
                            "memory_current_delta_bytes": (
                                after["cgroups"][path]["memory.current"] - row["memory.current"]
                                if isinstance(row.get("memory.current"), int)
                                and isinstance(after.get("cgroups", {}).get(path, {}).get("memory.current"), int)
                                else None
                            ),
                        }
                        for path, row in before.get("cgroups", {}).items()
                    },
                },
            }
        )
    lines += [
        "",
        "## NUMA memory snapshots",
        "",
        "Values are GiB. `Inactive(file)` and `KReclaimable` are accounting categories, not guarantees that all bytes are immediately reusable. "
        "The second-result snapshot is taken while the first donor is still resident, unless its recorded liveness check failed.",
        "",
        "| Trial | Stage | NUMA | Free GiB | Inactive file GiB | KReclaimable GiB |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for analysis in analyses:
        for stage, metrics in analysis["stages"].items():
            for node, values in metrics["nodes"].items():
                lines.append(
                    f"| {analysis['trial']} | [{stage}](reclaim/{analysis['trial']}/{stage}.json) | {node} | {gib(values.get('MemFree'))} | {gib(values.get('Inactive(file)'))} | {gib(values.get('KReclaimable'))} |"
                )
    lines += [
        "",
        "## Reclamation and cgroup evidence",
        "",
        "Deltas below cover before-second → second-result; global counters can include unrelated host activity. "
        "Node memory deltas, all captured reclaim counters, cgroup limits/events/NUMA statistics and missing-file details are retained in [reclaim-analysis.json](reclaim-analysis.json).",
        "",
        "| Trial | VM reclaim/compaction counter deltas | Cgroup usage/limit and event deltas | Shrinker availability |",
        "| --- | --- | --- | --- |",
    ]
    for analysis in analyses:
        changes = analysis["second_attempt_deltas"]
        vm = ", ".join(
            f"{name}={value:+d}"
            for name, value in changes["vmstat"].items()
            if value and name.startswith(("pgscan", "pgsteal", "allocstall", "compact_"))
        ) or ("no changed counters recorded" if changes["vmstat"] else "unavailable: no paired VM counters")
        before = analysis["stages"].get("before-second", {})
        after = analysis["stages"].get("second-result", {})
        groups = []
        for path, values in after.get("cgroups", {}).items():
            current = values.get("memory.current", values.get("memory.usage_in_bytes"))
            limit = values.get("memory.max", values.get("memory.limit_in_bytes"))
            events = changes["cgroups"].get(path, {}).get("events", {})
            groups.append(
                f"`{path}`: {gib(current)} GiB / {gib(limit) if limit != 'max' else 'unlimited'}; events {json.dumps(events, sort_keys=True)}"
            )
        shrinker = after.get("shrinker", before.get("shrinker", {"status": "not_recorded"}))
        shrinker_info = str(shrinker.get("status", "not_recorded"))
        if shrinker.get("error"):
            shrinker_info += ": " + shrinker["error"]
        if any("error" in value for value in shrinker.get("counts", {}).values()):
            shrinker_info += "; some count files unreadable"
        lines.append(f"| {analysis['trial']} | {vm} | {'; '.join(groups) or 'not recorded'} | {shrinker_info} |")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "A same-NUMA failure with a cross-NUMA success supports NUMA-local pressure or an allocation constraint. "
        "A failure that follows the second position after reversing GPU order argues against a single bad GPU. "
        "A chunked success with an unchunked failure supports allocation-size sensitivity; it does not by itself prove physical fragmentation. "
        "Correlate `pgscan*`, `pgsteal*`, `allocstall*`, `compact_*`, node cache changes, buddy/pagetype information, and actual donor cgroups before concluding why reclaim failed.",
        "",
        "Shrinker counts are retained verbatim: an unreadable or absent interface does not mean empty pools, and counts are not converted to bytes without a verified unit. "
        "Full stage snapshots include driver version/parameters, page size, buddy/pagetype/zone information and per-node VM counters. "
        "Each child JSON also records its own cgroup membership and CUDA identity; compare those with the controller snapshots.",
        "",
        "Trials run serially and can change driver/Linux cache state. No cache drop, shrinker scan, driver reset, privileged mount, or driver-parameter change is performed. "
        "Reorder cases in another fresh job to assess order effects. Controlled allocation failure with confirmed close is a complete diagnostic outcome. "
        "Timeout, early child death, forced kill, missing artifacts or uncertain cleanup make results incomplete and stop later trials.",
        "",
        "## Required serving metrics",
        "",
        "| Metric | Value | Reason/source |",
        "| --- | --- | --- |",
        "| TTFT p50/p95/p99 (ms) | N/A | No inference requests; allocation-only workload. |",
        "| ITL p50/p99 (ms/token) | N/A | No generated tokens. |",
        "| Output/input/total throughput (tokens/s) | N/A | No inference traffic. |",
        "| Agentic Pareto/SLO, goodput and concurrency metrics | N/A | No agentic benchmark. |",
        "| Transfer latency/bandwidth | N/A | No payload transfers; native Store allocation only. |",
        "",
        "## Sources and collection",
        "",
        "- [Diagnostic summary](reclaim/summary.json), per-trial `trial.json`, `donor-*.json`/`.log`, full stage snapshots, and timestamped `samples.jsonl` connect each GPU/NUMA attempt with host/cgroup changes.",
        "- [Launch metadata](launch.json), Slurm job snapshots, controller/rank/setup logs and archived tools/configs identify the allocation, image and dependency setup when present. Missing files remain missing evidence; they are not interpreted as zero counters.",
        "- [Job-sizing skill](../../../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md) sizes this as a bounded allocation investigation: no warmup or measured serving traffic; explicit setup, probe and cleanup budgets.",
        "- [Per-run analysis requirements](../../../docs/perf-analysis.md) define the N/A serving metrics and evidence handling. DSight/dashboard skills were not used: no requests or GPU execution trace exists for this allocation-only test.",
        "- Native EGM HOST_NUMA/Fabric allocation and chunk configuration are grounded in [Mooncake commit 042ab8a](https://github.com/kvcache-ai/Mooncake/tree/042ab8a18b0e6ccfe87782261a5556a3dd4446e4).",
    ]
    if "read_error" in summary:
        lines += [
            "",
            f"Missing diagnostic summary: `{summary['read_error']}`. Check setup/master/controller logs; no allocation conclusion is available.",
        ]
    if summary.get("error"):
        lines += ["", f"Controller error: `{summary['error']}`."]
    if gaps:
        lines += ["", "Missing or incomplete evidence:", "", *(f"- {gap}" for gap in gaps)]
    allocation.write_json(
        logs / "reclaim-analysis.json",
        {"complete": complete, "summary_status": summary.get("status"), "evidence_gaps": gaps, "trials": analyses},
    )
    (logs / "perf-analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {logs / 'perf-analysis.md'}", flush=True)
    return 0 if complete else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="Collect host/cgroup/driver counters without initializing CUDA")
    inv.add_argument("--output", type=Path, required=True)
    rep = commands.add_parser("report", help="Write logs/perf-analysis.md without CUDA/native dependencies")
    rep.add_argument("--run-dir", type=Path, required=True)
    rep.add_argument("--exit-code", type=int, required=True)
    runner = commands.add_parser("run", help="Compare two overlapping host-only EGM pools in serial controlled trials")
    runner.add_argument("--config", type=Path, required=True)
    runner.add_argument("--output-dir", type=Path, required=True)
    runner.add_argument("--pool-size", default="250GiB")
    runner.add_argument("--cases", default=",".join(TRIALS))
    runner.add_argument("--timeout-seconds", type=float, default=60)
    runner.add_argument("--budget-seconds", type=float, default=600)
    runner.add_argument("--sample-interval", type=float, default=1)
    runner.add_argument("--hold-seconds", type=float, default=2)
    runner.add_argument("--chunk-size", default="16GiB")
    default_helper = Path(__file__).resolve().parent.parent / "configs/mooncake_egm_donor.py"
    runner.add_argument("--donor-helper", type=Path, default=default_helper)
    native = commands.add_parser("child", help=argparse.SUPPRESS)
    native.add_argument("--config", type=Path, required=True)
    native.add_argument("--result", type=Path, required=True)
    native.add_argument("--gpu", required=True)
    native.add_argument("--numa", required=True)
    native.add_argument("--pool-size", type=int, required=True)
    native.add_argument("--donor-helper", type=Path, default=default_helper)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if args.command == "inventory":
        allocation.write_json(args.output, inventory())
        return 0
    if args.command == "report":
        return report(args.run_dir, args.exit_code)
    if args.command == "child":
        return child(args)
    try:
        donor = allocation.load_donor(args.donor_helper)
        size, chunk = donor.parse_size(args.pool_size), donor.parse_size(args.chunk_size)
        cases = tuple(item.strip() for item in args.cases.split(","))
        if not cases or len(set(cases)) != len(cases) or any(item not in TRIALS for item in cases):
            raise ValueError(f"Choose unique cases from {', '.join(TRIALS)}")
        timing = (args.timeout_seconds, args.budget_seconds, args.sample_interval, args.hold_seconds)
        if any(not math.isfinite(value) or value <= 0 for value in timing):
            raise ValueError("Timeout, budget, sample interval and hold duration must be finite and positive")
        donor.provider_config(args.config, size, "0")
        return run(
            Settings(
                args.config.resolve(),
                args.output_dir.resolve(),
                size,
                cases,
                args.timeout_seconds,
                args.budget_seconds,
                args.sample_interval,
                args.hold_seconds,
                chunk,
                args.donor_helper.resolve(),
            )
        )
    except (OSError, ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
