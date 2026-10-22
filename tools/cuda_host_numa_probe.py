#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe overlapping CUDA HOST_NUMA allocations using only the standard library.

CUDA calls live in cuda_host_numa_repro.cpp. This supervisor does not import or
install any GPU/framework packages. Copy the two files and the batch launcher
to reproduce the allocation investigation independently of the serving stack.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

GIB = 1024**3
CASES = {
    "same-numa": (("0", "0"), ("1", "0")),
    "cross-numa": (("0", "0"), ("2", "1")),
    "reverse-gpu": (("1", "0"), ("0", "0")),
    "chunked": (("0", "0"), ("1", "0")),
}
BASE_FILES = (
    "/proc/self/status",
    "/proc/self/limits",
    "/proc/self/cgroup",
    "/proc/self/mountinfo",
    "/proc/meminfo",
    "/proc/buddyinfo",
    "/proc/vmstat",
    "/sys/devices/system/node/online",
)
FULL_FILES = (
    "/proc/pagetypeinfo",
    "/proc/zoneinfo",
    "/proc/driver/nvidia/version",
    "/proc/driver/nvidia/params",
    "/proc/sys/vm/min_free_kbytes",
)
CGROUP_FILES = (
    "memory.current",
    "memory.max",
    "memory.peak",
    "memory.events",
    "memory.events.local",
    "memory.stat",
    "memory.numa_stat",
    "memory.swap.current",
    "memory.swap.max",
    "cpuset.cpus.effective",
    "cpuset.mems.effective",
    "memory.usage_in_bytes",
    "memory.limit_in_bytes",
    "memory.failcnt",
    "memory.oom_control",
    "cpuset.mems",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> dict[str, str]:
    try:
        return {"text": path.read_text()}
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_size(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)(B|KiB|MiB|GiB|TiB)?", value)
    if not match:
        raise argparse.ArgumentTypeError("Use positive bytes or a binary size such as 250GiB")
    return int(match[1]) * {None: 1, "B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": GIB, "TiB": 1024**4}[match[2]]


def cgroups(membership: str, mountinfo: str) -> list[dict[str, Any]]:
    """Resolve visible membership and ancestors, including cgroup namespaces."""
    members = [line.split(":", 2) for line in membership.splitlines() if len(line.split(":", 2)) == 3]
    result = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        first, last = left.split(), right.split()
        if not separator or len(first) < 5 or len(last) < 3 or last[0] not in ("cgroup", "cgroup2"):
            continue
        root, mount = (Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), p)) for p in first[3:5])
        for _, controllers, member in members:
            unified = last[0] == "cgroup2" and not controllers
            if not unified and not (set(controllers.split(",")) & set(last[2].split(",")) & {"memory", "cpuset"}):
                continue
            row: dict[str, Any] = {"membership": member, "mount_root": str(root), "mountpoint": str(mount)}
            try:
                relative = Path(".") if member == "/" else Path(member).relative_to(root)
                leaf = mount / relative
                row["ancestors"] = [str(leaf), *(str(p) for p in leaf.parents if p.is_relative_to(mount))]
            except ValueError:
                row["error"] = "Membership is outside the visible mount root"
            result.append(row)
    return result


def inventory(root: Path = Path("/"), *, full: bool = True, pids: tuple[int, ...] = ()) -> dict[str, Any]:
    files = {name: read_text(root / name.lstrip("/")) for name in BASE_FILES + (FULL_FILES if full else ())}
    for node in sorted((root / "sys/devices/system/node").glob("node[0-9]*")):
        for name in ("meminfo", "vmstat", "numastat", "cpulist"):
            files["/" + str((node / name).relative_to(root))] = read_text(node / name)
    groups = cgroups(files["/proc/self/cgroup"].get("text", ""), files["/proc/self/mountinfo"].get("text", ""))
    for group in groups:
        for ancestor in group.get("ancestors", []):
            for name in CGROUP_FILES:
                path = str(Path(ancestor) / name)
                files[path] = read_text(root / path.lstrip("/"))
    shrinker: dict[str, Any] = {"counts": {}}
    try:
        entries = sorted((root / "sys/kernel/debug/shrinker").iterdir())
        for entry in entries:
            if re.fullmatch(r"nv-sysmem-alloc-node-\d+-order-.*", entry.name):
                shrinker["counts"][entry.name] = read_text(entry / "count")
        shrinker["status"] = "available" if shrinker["counts"] else "no_matching_interface"
    except OSError as exc:
        shrinker.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
    return {
        "time_utc": now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
        "files": files,
        "cgroups": groups,
        "shrinker": shrinker,
        "processes": {
            str(pid): {name: read_text(root / f"proc/{pid}/{name}") for name in ("status", "cgroup", "limits")}
            for pid in pids
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("SLURM_", "CUDA_EGM_")) or key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        },
    }


@dataclass(frozen=True)
class Settings:
    binary: Path
    output_dir: Path
    size: int
    cases: tuple[str, ...]
    allocation_timeout: float
    budget: float
    sample_interval: float
    hold_seconds: float
    chunk_size: int


@dataclass(frozen=True)
class Child:
    process: subprocess.Popen[Any]
    stream: TextIO
    result: Path
    ready: Path
    record: dict[str, Any]


def start_child(settings: Settings, case: str, index: int, directory: Path) -> Child:
    gpu, numa = CASES[case][index]
    result, ready, log = (directory / f"rank-{index}.{suffix}" for suffix in ("json", "ready", "log"))
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": gpu}
    command = [
        str(settings.binary),
        "--device",
        "0",
        "--numa",
        numa,
        "--size-bytes",
        str(settings.size),
        "--chunk-bytes",
        str(settings.chunk_size if case == "chunked" else 0),
        "--ready-file",
        str(ready),
        "--result-file",
        str(result),
    ]
    stream = log.open("w")
    try:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
    except BaseException:
        stream.close()
        raise
    record: dict[str, Any] = {
        "index": index,
        "gpu": gpu,
        "numa": numa,
        "pid": process.pid,
        "result": result.name,
        "log": log.name,
        "ready": ready.name,
        "started_utc": now(),
        "command": command,
    }
    return Child(process, stream, result, ready, record)


def stop_child(child: Child, *, grace: float = 20) -> None:
    process = child.process
    forced = False
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            forced = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
    child.stream.close()
    final = read_json(child.result)
    expected_code = 0 if final.get("allocation_success") is True else 2
    child.record.update(
        final=final,
        returncode=process.poll(),
        forced_kill=forced,
        stopped_utc=now(),
        cleanup_confirmed=(not forced and final.get("cleanup_confirmed") is True and process.poll() == expected_code),
    )


class Sampler:
    def __init__(self, directory: Path, interval: float, children: list[Child]) -> None:
        self.directory, self.interval, self.children = directory, interval, children
        self.next_sample = 0.0
        self.phase = "baseline"

    def capture(self, *, full: bool = False) -> dict[str, Any]:
        return inventory(full=full, pids=tuple(child.process.pid for child in self.children))

    def sample(self) -> None:
        if time.monotonic() >= self.next_sample:
            with (self.directory / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps({"phase": self.phase, **self.capture()}) + "\n")
            self.next_sample = time.monotonic() + self.interval

    def snapshot(self, phase: str, record: dict[str, Any]) -> None:
        self.phase = phase
        write_json(self.directory / f"{phase}.json", self.capture(full=True))
        record.setdefault("snapshots", {})[phase] = f"{phase}.json"
        self.next_sample = 0
        self.sample()


def wait_allocation(child: Child, first: Child | None, sampler: Sampler, stop: threading.Event, deadline: float) -> str:
    while True:
        sampler.sample()
        if stop.is_set():
            return "interrupted"
        if first is not None and first.process.poll() is not None:
            return "first_process_died"
        if time.monotonic() >= deadline:
            return "allocation_timeout"
        result = read_json(child.result)
        if child.process.poll() is not None:
            return "child_died"
        if child.ready.exists() and result.get("setup_complete") is True:
            child.record["observed"] = result
            if result.get("allocation_success") is True:
                return "ready"
            # Non-OOM errors are capability/permission/API evidence, not this
            # capacity reproduction. Stop further trials rather than claiming PASS.
            if result.get("failed_api") == "cuMemCreate" and result.get("failed_result") == 2:
                return "allocation_oom"
            return "cuda_or_setup_error"
        stop.wait(0.05)


def trial(settings: Settings, case: str, stop: threading.Event, budget_deadline: float, record: dict[str, Any]) -> None:
    directory = settings.output_dir / case
    directory.mkdir()
    children: list[Child] = []
    sampler = Sampler(directory, settings.sample_interval, children)
    record.update(name=case, status="running", ranks=[], started_utc=now())
    failure = None
    try:
        sampler.snapshot("baseline", record)
        for index in range(2):
            if stop.is_set() or time.monotonic() >= budget_deadline:
                failure = "interrupted" if stop.is_set() else "budget_exhausted"
                break
            sampler.snapshot("before-first" if index == 0 else "before-second", record)
            sampler.phase = "allocating-first" if index == 0 else "allocating-second"
            child = start_child(settings, case, index, directory)
            children.append(child)
            record["ranks"].append(child.record)
            outcome = wait_allocation(
                child,
                children[0] if index else None,
                sampler,
                stop,
                min(budget_deadline, time.monotonic() + settings.allocation_timeout),
            )
            child.record["observation"] = outcome
            sampler.snapshot("first-result" if index == 0 else "second-result", record)
            if outcome == "allocation_oom":
                record["outcome"] = "first_allocation_oom" if index == 0 else "second_allocation_oom"
                break
            if outcome != "ready":
                failure = outcome
                break
        if len(children) == 2 and all(child.record.get("observation") == "ready" for child in children):
            record["outcome"] = "two_allocations_succeeded"
        if failure is None:
            sampler.phase = "holding-results"
            until = min(time.monotonic() + settings.hold_seconds, budget_deadline)
            while time.monotonic() < until:
                sampler.sample()
                if stop.is_set() or any(child.process.poll() is not None for child in children):
                    failure = "interrupted" if stop.is_set() else "child_died_during_hold"
                    break
                stop.wait(0.05)
            if time.monotonic() >= budget_deadline and failure is None:
                failure = "budget_exhausted"
    except Exception as exc:
        failure = f"controller_error: {type(exc).__name__}: {exc}"
    finally:
        # Both allocation results (including partial failure) are captured before
        # releasing second, then first. Never kill other Slurm jobs or processes.
        for child in reversed(children):
            stop_child(child)
        sampler.snapshot("after-release", record)
        record["cleanup_confirmed"] = bool(children) and all(c.record["cleanup_confirmed"] for c in children)
        if not record["cleanup_confirmed"] and failure is None:
            failure = "uncertain_cleanup"
        record.update(status="complete" if failure is None else "incomplete", failure=failure, ended_utc=now())
        write_json(directory / "trial.json", record)


def run(settings: Settings) -> int:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if any(settings.output_dir.iterdir()):
        raise ValueError(f"Refusing existing diagnostic data: {settings.output_dir}")
    stop = threading.Event()
    handlers = {number: signal.signal(number, lambda *_: stop.set()) for number in (signal.SIGTERM, signal.SIGINT)}
    summary: dict[str, Any] = {
        "status": "running",
        "hostname": socket.gethostname(),
        "started_utc": now(),
        "size_bytes": settings.size,
        "chunk_cap_bytes": settings.chunk_size,
        "planned_cases": list(settings.cases),
        "cases": [],
    }
    started = time.monotonic()
    path = settings.output_dir / "summary.json"
    write_json(path, summary)
    try:
        for case in settings.cases:
            if stop.is_set() or time.monotonic() >= started + settings.budget:
                summary["status"] = "interrupted" if stop.is_set() else "budget_exhausted"
                break
            record: dict[str, Any] = {}
            summary["cases"].append(record)
            trial(settings, case, stop, started + settings.budget, record)
            write_json(path, summary)
            if record["status"] != "complete":
                summary["status"] = "incomplete"
                break
        if summary["status"] == "running":
            summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="controller_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        summary.update(ended_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(path, summary)
        for number, handler in handlers.items():
            signal.signal(number, handler)
    print(json.dumps({"status": summary["status"], "summary": str(path)}), flush=True)
    return 0 if summary["status"] == "complete" else 1


def fields(payload: dict[str, str], *, meminfo: bool = False) -> dict[str, int]:
    result = {}
    for line in payload.get("text", "").splitlines():
        parts = re.sub(r"^Node\s+\d+\s+", "", line).split()
        if len(parts) > 1 and parts[1].lstrip("-").isdigit():
            result[parts[0].rstrip(":")] = int(parts[1]) * (1024 if meminfo and parts[2:] == ["kB"] else 1)
    return result


def second_attempt_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Retain exact counters and missing evidence alongside human-readable deltas."""
    previous, current = before.get("files", {}), after.get("files", {})
    vm_before, vm_after = fields(previous.get("/proc/vmstat", {})), fields(current.get("/proc/vmstat", {}))
    counters = {
        key: vm_after[key] - value
        for key, value in vm_before.items()
        if key in vm_after and key.startswith(("pgscan", "pgsteal", "allocstall", "compact_"))
    }
    groups = {}
    for group in after.get("cgroups", []):
        for ancestor in group.get("ancestors", []):
            events_before = fields(previous.get(f"{ancestor}/memory.events", {}))
            events_after = fields(current.get(f"{ancestor}/memory.events", {}))
            groups[ancestor] = {
                "current": current.get(
                    f"{ancestor}/memory.current", current.get(f"{ancestor}/memory.usage_in_bytes", {})
                ),
                "limit": current.get(f"{ancestor}/memory.max", current.get(f"{ancestor}/memory.limit_in_bytes", {})),
                "events_delta": {
                    key: events_after[key] - value for key, value in events_before.items() if key in events_after
                },
            }
    return {
        "vmstat_delta": counters,
        "cgroups": groups,
        "shrinker": after.get("shrinker", {"status": "not_recorded"}),
        "missing_files": {name: value["error"] for name, value in current.items() if "error" in value},
    }


def report(run_dir: Path, exit_code: int) -> int:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    folder = logs / "probe"
    summary = read_json(folder / "summary.json")
    cases = summary.get("cases", [])
    gaps = []
    for case in cases:
        directory = folder / case["name"]
        expected_ranks = 1 if case.get("outcome") == "first_allocation_oom" else 2
        if len(case.get("ranks", [])) != expected_ranks:
            gaps.append(f"{case['name']} has incomplete rank records")
        for filename in ("trial.json", "baseline.json", "first-result.json", "after-release.json"):
            if read_json(directory / filename).get("read_error"):
                gaps.append(f"{case['name']}/{filename} is missing or unreadable")
        if len(case.get("ranks", [])) == 2 and read_json(directory / "second-result.json").get("read_error"):
            gaps.append(f"{case['name']}/second-result.json is missing or unreadable")
        if not read_text(directory / "samples.jsonl").get("text", "").strip():
            gaps.append(f"{case['name']}/samples.jsonl has no readable samples")
        for rank in case.get("ranks", []):
            final = read_json(directory / rank["result"])
            if final.get("cleanup_confirmed") is not True or read_text(directory / rank["log"]).get("error"):
                gaps.append(f"{case['name']}/rank-{rank['index']} has incomplete result/log/cleanup evidence")
    complete = (
        exit_code == 0
        and summary.get("status") == "complete"
        and bool(cases)
        and not gaps
        and [c.get("name") for c in cases] == summary.get("planned_cases")
        and all(c.get("status") == "complete" and c.get("cleanup_confirmed") is True for c in cases)
    )
    lines = [
        "# CUDA HOST_NUMA allocation investigation",
        "",
        f"Diagnostic: **{'COMPLETE' if complete else 'INCOMPLETE'}**; launcher exit code `{exit_code}`.",
        "",
        "Completion means observations and cleanup finished; controlled allocation OOM is an outcome, not a successful allocation.",
        "",
        f"Node: `{summary.get('hostname', 'unavailable')}`. Window: `{summary.get('started_utc', 'unavailable')}` to "
        f"`{summary.get('ended_utc', 'unavailable')}`. Requested bytes per rank: `{summary.get('size_bytes', 'unavailable')}`.",
        "",
        "Goal: isolate the native CUDA backing allocation with the first handle still resident when the second process calls cuMemCreate. "
        "The C++ executable uses PINNED, explicit HOST_NUMA and FABRIC properties, queries minimum granularity, "
        "and retains successful handles until directed to call cuMemRelease. It uses no serving framework or GPU payload allocations.",
        "",
        "| Case | Observation | Held GiB at result, by rank | Completion / failure |",
        "| --- | --- | --- | --- |",
    ]
    for case in cases:
        rank_values = []
        for rank in case.get("ranks", []):
            observed = rank.get("observed", {})
            held = observed.get("held_bytes", observed.get("allocated_bytes"))
            amount = f"{held / GIB:.3f}" if isinstance(held, (int, float)) else "N/A"
            rank_values.append(f"GPU {rank['gpu']} / NUMA {rank['numa']}: {amount}")
        ranks = "; ".join(rank_values)
        lines.append(
            f"| [{case['name']}](probe/{case['name']}/trial.json) | {case.get('outcome', 'unavailable')} | "
            f"{ranks} | {case.get('status')}; {case.get('failure') or 'none'} |"
        )
    lines += [
        "",
        "## NUMA memory around allocations",
        "",
        "GiB below come from node meminfo. The second-result snapshot holds partial allocations after a failed create, "
        "before releasing either process's handles. Serial cases can warm driver caches; compare their baselines.",
        "",
        "| Case | Stage | NUMA | Free GiB | FilePages GiB | KReclaimable GiB |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in cases:
        for stage, filename in case.get("snapshots", {}).items():
            snapshot = read_json(folder / case["name"] / filename)
            for path, payload in snapshot.get("files", {}).items():
                match = re.fullmatch(r"/sys/devices/system/node/node(\d+)/meminfo", path)
                values = fields(payload, meminfo=True) if match else {}
                if match and values.get("MemTotal", 0) > 0:
                    amounts = " | ".join(
                        f"{values[key] / GIB:.3f}" if key in values else "N/A"
                        for key in ("MemFree", "FilePages", "KReclaimable")
                    )
                    lines.append(
                        f"| {case['name']} | [{stage}](probe/{case['name']}/{filename}) | {match[1]} | {amounts} |"
                    )
    lines += [
        "",
        "## Second-allocation reclaim and cgroup evidence",
        "",
        "Deltas cover before-second to second-result, while both processes retain their handles. Global VM counters "
        "can include background activity. Detailed counters, cgroup usage/limits/events and missing paths are saved in "
        "[counter-analysis.json](counter-analysis.json). Unavailable observations are not treated as zero activity.",
        "",
        "| Case | Reclaim/compaction counter deltas | Cgroup event deltas | NVIDIA shrinker view |",
        "| --- | --- | --- | --- |",
    ]
    counter_analysis = {}
    for case in cases:
        directory = folder / case["name"]
        metrics = second_attempt_metrics(
            read_json(directory / "before-second.json"), read_json(directory / "second-result.json")
        )
        counter_analysis[case["name"]] = metrics
        counters = metrics["vmstat_delta"]
        changed = ", ".join(f"{key}={value:+d}" for key, value in counters.items() if value)
        vm_text = changed or ("no changed paired counters" if counters else "unavailable: no paired counters")
        groups = (
            "; ".join(
                f"`{name}`: {json.dumps(row['events_delta'], sort_keys=True)}"
                for name, row in metrics["cgroups"].items()
            )
            or "unavailable"
        )
        shrinker = metrics["shrinker"]
        lines.append(
            f"| {case['name']} | {vm_text} | {groups} | {shrinker.get('status')}; {shrinker.get('error', '')} |"
        )
    lines += [
        "",
        "## Interpretation and evidence",
        "",
        "A failure following the second allocation on NUMA0, with successful allocations across nodes and GPU order reversed, "
        "supports NUMA-local allocation constraints. Smaller chunks can show partial resident capacity, but API chunk size "
        "does not establish the driver's page order or a physically contiguous full-pool requirement. "
        "Inspect each native rank log/JSON for failed_api, failed_result, exact error names, granularity and GPU identity.",
        "",
        "The snapshots/samples retain /proc/vmstat reclaim/compaction counters, buddyinfo, node meminfo/vmstat, actual child "
        "cgroup membership, visible ancestor memory limits/events, driver parameters, page size and read-only NVIDIA shrinker counts. "
        "Unavailable paths and permission errors are evidence gaps, not zero counters or empty pools. "
        "Check kernel-before/after logs when accessible; no privileged operation or cache eviction is attempted.",
        "",
        "This creation-only experiment stops before VA mapping, Fabric handle export/import and NVLink copies. "
        "If the direct API reproduces OOM, the failure is independent of the serving stack; if it succeeds, "
        "the original failure is not reproduced and context/mapping/lifetime or changed node state may matter. "
        "A cached allocation released successfully need not immediately appear as Linux free memory.",
        "",
        "## Required serving metrics",
        "",
        "| Metric | Value | Reason |",
        "| --- | --- | --- |",
        "| TTFT p50/p95/p99 (ms) | N/A | No inference requests |",
        "| ITL p50/p99 (ms/token) | N/A | No generated tokens |",
        "| Output/input/total throughput (tokens/s) | N/A | Allocation-only workload |",
        "| Agentic Pareto/SLO, goodput, concurrency | N/A | No agentic workload |",
        "| Transfer bandwidth/latency | N/A | No payload transfers |",
        "",
        "## Sources and collection",
        "",
        "- [Summary](probe/summary.json), per-case trial/rank JSON, native rank logs, phase snapshots and samples.jsonl: "
        "connect actual CUDA calls and resident handles to memory state and cleanup.",
        "- [Launch metadata](launch.json), Slurm/hardware/compiler/kernel logs and archived C++/Python/batch sources: "
        "identify the environment and make the experiment reproducible without the serving repository.",
        "- The slurm-job-sizing skill sets a bounded 20-minute allocation investigation; no serving warmup or traffic is needed. "
        "No DSight/dashboard skill is used because native allocation calls and memory evidence answer this goal.",
        "- [CUDA Driver API VMM documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-driver-api/group__CUDA__VA.html): "
        "defines HOST_NUMA, FABRIC, allocation granularity and cuMemCreate/cuMemRelease.",
    ]
    if summary.get("read_error"):
        gaps.append(f"Diagnostic summary: {summary['read_error']}")
    if gaps:
        lines += ["", "Evidence gaps:", "", *(f"- {gap}" for gap in gaps)]
    write_json(logs / "counter-analysis.json", counter_analysis)
    (logs / "perf-analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {logs / 'perf-analysis.md'}", flush=True)
    return 0 if complete else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--output", type=Path, required=True)
    rep = commands.add_parser("report")
    rep.add_argument("--run-dir", type=Path, required=True)
    rep.add_argument("--exit-code", type=int, required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--binary", type=Path, required=True)
    runner.add_argument("--output-dir", type=Path, required=True)
    runner.add_argument("--size", type=parse_size, default="250GiB")
    runner.add_argument("--cases", default=",".join(CASES))
    runner.add_argument("--allocation-timeout", type=float, default=60)
    runner.add_argument("--budget", type=float, default=600)
    runner.add_argument("--sample-interval", type=float, default=1)
    runner.add_argument("--hold-seconds", type=float, default=5)
    runner.add_argument("--chunk-size", type=parse_size, default="16GiB")
    args = parser.parse_args(argv)
    if args.command == "inventory":
        write_json(args.output, inventory())
        return 0
    if args.command == "report":
        return report(args.run_dir, args.exit_code)
    cases = tuple(item.strip() for item in args.cases.split(","))
    if not cases or len(set(cases)) != len(cases) or any(item not in CASES for item in cases):
        parser.error(f"Choose unique cases from {', '.join(CASES)}")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.allocation_timeout, args.budget, args.sample_interval, args.hold_seconds)
    ):
        parser.error("Timeout, budget, sample interval and hold time must be finite and positive")
    try:
        return run(
            Settings(
                args.binary.resolve(),
                args.output_dir.resolve(),
                args.size,
                cases,
                args.allocation_timeout,
                args.budget,
                args.sample_interval,
                args.hold_seconds,
                args.chunk_size,
            )
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
