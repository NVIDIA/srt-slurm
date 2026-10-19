#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only Mooncake 0.3.13 diagnostic; no model, GPU load, or DSight required.

APIs verified against Mooncake v0.3.13, commit e5598b0992cc258b06d22f24d875480e8931a28e.
Run a fresh master and exactly one donor per case. Drain clients before stopping the donor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import ctypes
import hashlib
import importlib.metadata
import json
import mmap
import os
import platform
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAGE_BYTES = 2_006_016
PACKAGES = ("mooncake-transfer-engine", "mooncake-transfer-engine-cuda13")


def versions() -> dict[str, str]:
    result = {}
    for package in PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            result[package] = importlib.metadata.version(package)
    return result


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def numeric_fields(path: Path) -> dict[str, int]:
    result = {}
    for line in (read_text(path) or "").splitlines():
        fields = line.replace(":", " ").split()
        if len(fields) >= 2 and fields[1].isdigit():
            result[fields[0]] = int(fields[1])
    return result


def host_ip(master: str | None = None) -> str:
    override = os.environ.get("MOONCAKE_LOCAL_HOST")
    if override:
        return override
    if master:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((master.rsplit(":", 1)[0], 9))
            return str(sock.getsockname()[0])
    return socket.gethostbyname(socket.gethostname())


def health() -> dict[str, Any]:
    counters = {}
    for category in ("counters", "hw_counters"):
        for path in Path("/sys/class/infiniband").glob(f"*/ports/*/{category}/*"):
            value = read_text(path)
            if value is not None and value.isdigit():
                counters[str(path.relative_to("/sys/class/infiniband"))] = int(value)
    cgroup_text = read_text(Path("/proc/self/cgroup")) or ""
    cgroup: dict[str, Any] = {"membership": cgroup_text}
    for line in cgroup_text.splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            base = Path("/sys/fs/cgroup") / relative.lstrip("/")
            for name in (
                "memory.current",
                "memory.peak",
                "memory.max",
                "memory.events",
                "memory.events.local",
                "memory.pressure",
            ):
                cgroup[name] = read_text(base / name)
        elif "memory" in controllers.split(","):
            base = Path("/sys/fs/cgroup/memory") / relative.lstrip("/")
            for name in ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.failcnt", "memory.oom_control"):
                cgroup[name] = read_text(base / name)
    return {
        "pid": os.getpid(),
        "proc_status_kib": {
            key: value
            for key, value in numeric_fields(Path("/proc/self/status")).items()
            if key.startswith(("Vm", "Rss"))
        },
        "meminfo_kib": {
            key: value
            for key, value in numeric_fields(Path("/proc/meminfo")).items()
            if key
            in (
                "MemTotal",
                "MemFree",
                "MemAvailable",
                "Buffers",
                "Cached",
                "SwapFree",
                "Mlocked",
                "Unevictable",
                "Committed_AS",
                "CommitLimit",
            )
        },
        "cgroup": cgroup,
        "uptime": read_text(Path("/proc/uptime")),
        "memory_pressure": read_text(Path("/proc/pressure/memory")),
        "ib_counters": counters,
        "ib_data_counter_unit": "4 octets for port_xmit_data and port_rcv_data; other counters are counts",
    }


class Jsonl:
    def __init__(self, path: Path, **context: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", buffering=1)
        self.lock = threading.Lock()
        self.context = context

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch_s": time.time(),
            **self.context,
            "event": event,
            **fields,
        }
        with self.lock:
            self.file.write(json.dumps(record, default=str) + "\n")
            self.file.flush()


def probe(endpoint: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        host, port = endpoint.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=0.5):
            pass
        return {"endpoint": endpoint, "ok": True, "duration_ms": (time.monotonic() - started) * 1000}
    except (OSError, ValueError) as exc:
        return {"endpoint": endpoint, "ok": False, "error": str(exc)}


def monitor(log: Jsonl, stop: threading.Event, endpoints: set[str], lock: threading.Lock) -> None:
    while True:
        with lock:
            peers = sorted(endpoints)
        log.emit("health", health=health(), probes=[probe(peer) for peer in peers])
        if stop.wait(2):
            break


def setup_store(master: str, gib: int = 0, local_bytes: int = 0) -> tuple[Any, dict[str, str], float]:
    installed = versions()
    if not installed or any(value.split("+", 1)[0].split(".post", 1)[0] != "0.3.13" for value in installed.values()):
        raise RuntimeError(f"Mooncake 0.3.13 is required; installed distributions: {installed}")
    from mooncake.store import MooncakeDistributedStore  # type: ignore[import-not-found]

    store = MooncakeDistributedStore()
    started = time.monotonic()
    rc = store.setup(
        local_hostname=host_ip(master),
        metadata_server="P2PHANDSHAKE",
        global_segment_size=gib * 1024**3,
        local_buffer_size=local_bytes,
        protocol="rdma",
        rdma_devices=os.environ.get("MOONCAKE_RDMA_DEVICES", os.environ.get("MC_TE_FILTERS", "")),
        master_server_addr=master,
    )
    if rc != 0:
        raise RuntimeError(f"Mooncake setup failed: {rc}")
    return store, installed, time.monotonic() - started


def inventory_command(argv: list[str], timeout: float) -> dict[str, Any]:
    """Run one read-only probe with bounded retained output and no shell."""
    started = time.monotonic()
    record: dict[str, Any] = {"argv": argv, "returncode": None, "status": "error"}
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            record.update(returncode=result.returncode, status="ok" if result.returncode == 0 else "error")
        except subprocess.TimeoutExpired as exc:
            record.update(status="timeout", error=str(exc))
        except OSError as exc:
            record.update(status="missing" if isinstance(exc, FileNotFoundError) else "error", error=str(exc))
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            record[name] = stream.read(65536).decode(errors="replace")
            record[f"{name}_truncated"] = size > 65536
    record["duration_s"] = time.monotonic() - started
    return record


def inventory(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}

    def read(path: Path) -> str | None:
        try:
            return path.read_text()[:65536].strip()
        except OSError as exc:
            errors[str(path)] = str(exc)
            return None

    def fields(base: Path, names: tuple[str, ...]) -> dict[str, str | None]:
        return {name: read(base / name) for name in names}

    def device(base: Path) -> dict[str, Any]:
        target = base / "device"
        return {
            "pci_address": target.resolve().name if target.exists() else None,
            "driver": (target / "driver").resolve().name if (target / "driver").exists() else None,
            **(fields(target, ("numa_node", "vendor", "device")) if target.exists() else {}),
        }

    commands = {
        "uname": ["uname", "-a"],
        "lscpu": ["lscpu"],
        "numactl": ["numactl", "--hardware"],
        "free": ["free", "-b"],
        "ip_address": ["ip", "-br", "address"],
        "ip_route": ["ip", "route"],
        "rdma_link": ["rdma", "link"],
        "ibstat": ["ibstat"],
        "ibv_devinfo": ["ibv_devinfo"],
        "lspci": ["lspci", "-nn"],
        "dmesg": ["dmesg"],
        "nvidia_gpu": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,pci.bus_id,driver_version",
            "--format=csv,nounits",
        ],
        "nvidia_ecc": ["nvidia-smi", "-q", "-d", "ECC"],
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = dict(
            zip(
                commands,
                executor.map(
                    lambda argv: inventory_command(argv, getattr(args, "commands_timeout", 3.0)), commands.values()
                ),
                strict=True,
            )
        )
    kernel = {key: value for key, value in results["dmesg"].items() if key != "stdout"}
    kernel["messages"] = [
        line
        for line in results["dmesg"]["stdout"].splitlines()
        if re.search(
            r"oom|killed process|out of memory|mlx5|infiniband|rdma|NETDEV|link|hardware error|EDAC|AER",
            line,
            re.IGNORECASE,
        )
    ][-300:]
    cpuinfo = (read(Path("/proc/cpuinfo")) or "") + "\n" + results["lscpu"]["stdout"]
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        affinity = None
        errors["sched_getaffinity"] = str(exc)
    adapters = {}
    for base in sorted(Path("/sys/class/infiniband").glob("*")):
        ports = {}
        for port in sorted((base / "ports").glob("*")):
            ports[port.name] = {
                **fields(port, ("state", "phys_state", "rate", "link_layer", "active_mtu")),
                "gids": {path.name: read(path) for path in sorted((port / "gids").glob("*"))},
            }
        adapters[base.name] = {
            **device(base),
            **fields(base, ("fw_ver", "node_guid", "sys_image_guid")),
            "ports": ports,
        }
    try:
        ip = host_ip()
    except OSError as exc:
        ip = None
        errors["host_ip"] = str(exc)
    args.output.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scope": getattr(args, "scope", "container"),
                "host": socket.gethostname(),
                "slurm_node": os.environ.get("SLURMD_NODENAME"),
                "ip": ip,
                "versions": versions(),
                "health": health(),
                "kernel": kernel,
                "platform": {
                    "kernel": platform.release(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "os_release": read(Path("/etc/os-release")),
                },
                "cpu": {
                    "logical_count": os.cpu_count(),
                    "affinity": affinity,
                    "limits": read(Path("/proc/self/limits")),
                    "model_names": sorted(
                        {
                            line.split(":", 1)[1].strip()
                            for line in cpuinfo.splitlines()
                            if ":" in line
                            and line.split(":", 1)[0].strip() in ("model name", "Model name", "Hardware", "Processor")
                        }
                    ),
                },
                "numa": {
                    base.name: fields(base, ("cpulist", "distance", "meminfo"))
                    for base in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
                },
                "interfaces": {
                    base.name: {
                        **device(base),
                        **fields(base, ("operstate", "carrier", "mtu", "address", "speed")),
                        "statistics": {path.name: read(path) for path in sorted((base / "statistics").glob("*"))},
                    }
                    for base in sorted(Path("/sys/class/net").glob("*"))
                },
                "infiniband": adapters,
                "commands": results,
                "collection_errors": errors,
            },
            indent=2,
        )
        + "\n"
    )


def inventory_summary(run_dir: Path) -> list[str]:
    """Summarize native-host snapshots separately from container OS evidence."""

    def network_counters(row: dict[str, Any]) -> dict[str, int]:
        values = dict(row.get("health", {}).get("ib_counters", {}))
        for interface, data in row.get("interfaces", {}).items():
            for name, value in data.get("statistics", {}).items():
                if isinstance(value, str) and value.isdigit():
                    values[f"net/{interface}/{name}"] = int(value)
        return values

    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    container_sources = []
    malformed = []
    for path in sorted(run_dir.rglob("*inventory-*.json")):
        try:
            row = json.loads(path.read_text())
            if row.get("scope") == "host":
                grouped.setdefault(row.get("slurm_node") or row["host"], []).append((path, row))
            else:
                container_sources.append(str(path.relative_to(run_dir)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            malformed.append(f"{path.relative_to(run_dir)}: {exc}")
    lines = [
        "",
        "## Node inventory",
        "",
        "Host OS evidence comes from native host-scope snapshots; container OS evidence is listed separately. "
        "Historical nonzero counters are not attributed to this run. Unavailable probes do not establish a clean node. "
        "Negative counter deltas indicate a counter reset/decrease and are not evidence of improvement.",
        "",
    ]
    if not grouped:
        lines.append("Host inventory unavailable; host OS/kernel and node counter changes cannot be established.")
    for host, snapshots in sorted(grouped.items()):
        snapshots.sort(key=lambda item: (item[0].parent != run_dir / "logs", item[1].get("timestamp", "")))
        first_path, first = snapshots[0]
        last_path, last = snapshots[-1]
        counters = network_counters(first)
        final = network_counters(last)

        def relevant(name: str) -> bool:
            return any(
                word in name.lower() for word in ("err", "discard", "link_downed", "constraint", "timeout", "recovery")
            )

        comparable = {name for name in counters.keys() & final.keys() if relevant(name)}
        delta = {name: final[name] - counters[name] for name in sorted(comparable) if final[name] != counters[name]}
        historical = sum(value != 0 for name, value in counters.items() if relevant(name))
        platform_info = last.get("platform", {})
        os_name = next(
            (
                line.partition("=")[2].strip('"')
                for line in (platform_info.get("os_release") or "").splitlines()
                if line.startswith("PRETTY_NAME=")
            ),
            "N/A",
        )
        ports = {
            f"{hca}/{port}": values.get("state")
            for hca, adapter in last.get("infiniband", {}).items()
            for port, values in adapter.get("ports", {}).items()
        }
        net_states = {name: data.get("operstate") for name, data in last.get("interfaces", {}).items()}
        commands = last.get("commands", {})
        unavailable = sum(row.get("status") != "ok" for row in commands.values())
        kernel_known = first.get("kernel", {}).get("status") == last.get("kernel", {}).get("status") == "ok"
        previous_messages = set(first.get("kernel", {}).get("messages", []))
        oom = (
            [
                message
                for message in last.get("kernel", {}).get("messages", [])
                if message not in previous_messages
                and re.search(r"oom|killed process|out of memory", message, re.IGNORECASE)
            ]
            if kernel_known
            else None
        )
        baseline = first_path.parent == run_dir / "logs"
        lines.extend(
            [
                f"Node `{host}`: OS={os_name}; kernel={platform_info.get('kernel', 'N/A')}; "
                f"CPU={last.get('cpu', {}).get('model_names') or 'N/A'}; "
                f"MemTotal={last.get('health', {}).get('meminfo_kib', {}).get('MemTotal', 'N/A')} KiB.",
                f"IB port states: {ports or 'unavailable'}. Command errors/missing/timeouts={unavailable}/{len(commands)}; "
                f"sysfs/proc collection errors={len(last.get('collection_errors', {}))}.",
                f"Network interface states: {net_states or 'unavailable'}.",
                f"Preflight baseline={'present' if baseline else 'missing'}; snapshots={len(snapshots)}; "
                f"nonzero link/error counter changes={delta if baseline and len(snapshots) > 1 and comparable else 'N/A'}; "
                f"comparable counters={len(comparable)}; historical nonzero counters={historical}. "
                f"New OOM kernel messages={oom if baseline and len(snapshots) > 1 and oom is not None else 'N/A'} "
                "(only retained kernel log windows are compared).",
                "Sources: " + ", ".join(f"`{path.relative_to(run_dir)}`" for path, _ in snapshots) + ".",
                "",
            ]
        )
        if baseline:
            for path, row in snapshots[1:]:
                current = network_counters(row)
                changes = {
                    name: current[name] - value
                    for name, value in counters.items()
                    if relevant(name) and name in current and current[name] != value
                }
                if changes:
                    lines.append(
                        f"Post snapshot `{path.relative_to(run_dir)}` counter changes from preflight: {changes}."
                    )
    lines.append(
        "Container inventory sources: "
        + (", ".join(f"`{source}`" for source in container_sources) or "unavailable")
        + "."
    )
    if malformed:
        lines.append("Unreadable inventory: " + "; ".join(malformed))
    return lines


def donor(args: argparse.Namespace) -> None:
    log = Jsonl(args.output, role="donor", host=socket.gethostname(), gib=args.gib)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    endpoints: set[str] = set()
    lock = threading.Lock()
    watcher = threading.Thread(target=monitor, args=(log, stop, endpoints, lock), daemon=True)
    watcher.start()
    store = None
    try:
        available = numeric_fields(Path("/proc/meminfo")).get("MemAvailable")
        if available is not None and available * 1024 < (args.gib + 64) * 1024**3:
            raise RuntimeError(
                f"Insufficient host memory: MemAvailable={available} KiB; need donor {args.gib} GiB + 64 GiB margin"
            )
        cgroup = health()["cgroup"]
        limit = cgroup.get("memory.max") or cgroup.get("memory.limit_in_bytes")
        current = cgroup.get("memory.current") or cgroup.get("memory.usage_in_bytes")
        if (
            limit
            and current
            and limit.isdigit()
            and current.isdigit()
            and int(limit) - int(current) < (args.gib + 64) * 1024**3
        ):
            raise RuntimeError(
                f"Insufficient cgroup memory: limit={limit}, current={current}; "
                f"need donor {args.gib} GiB + 64 GiB margin"
            )
        store, installed, elapsed = setup_store(args.master, args.gib, 64 * 1024**2)
        native_log = read_text(args.native_log) if args.native_log else None
        listeners = re.findall(r"P2P handshake, listening on (\S+)", native_log or "")
        ready = {
            "host": socket.gethostname(),
            "ip": host_ip(args.master),
            "pid": os.getpid(),
            "store_endpoint": store.get_hostname(),
            "handshake_endpoint": listeners[-1] if listeners else None,
            "gib": args.gib,
            "setup_sec": elapsed,
            "versions": installed,
        }
        log.emit("ready", **ready)
        args.ready.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.ready.with_suffix(args.ready.suffix + ".tmp")
        temporary.write_text(json.dumps(ready) + "\n")
        temporary.replace(args.ready)
        print(json.dumps(ready), flush=True)
        stop.wait()
    finally:
        log.emit("cleanup_start", reason="stop requested; caller must already have drained clients")
        stop.set()
        watcher.join()
        if store is not None:
            log.emit("cleanup_complete", returncode=store.close())


def replica_samples(store: Any, keys: list[str]) -> list[dict[str, Any]]:
    result = []
    for key, replicas in store.batch_get_replica_desc(keys).items():
        for replica in replicas:
            if replica.is_memory_replica():
                descriptor = replica.get_memory_descriptor().buffer_descriptor
                result.append(
                    {
                        "key": key,
                        "address": descriptor.buffer_address,
                        "size": descriptor.size,
                        "endpoint": descriptor.transport_endpoint,
                        "status": str(replica.status),
                    }
                )
    return result


def client(args: argparse.Namespace) -> None:
    log = Jsonl(args.output, role="client", host=socket.gethostname(), case=args.case)
    stop = threading.Event()
    monitor_stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    endpoints: set[str] = set()
    if args.donor_ready:
        ready = json.loads(args.donor_ready.read_text())
        endpoint = ready.get("handshake_endpoint") or ready.get("p2p_endpoint")
        if endpoint:
            endpoints.add(endpoint)
    lock = threading.Lock()
    watcher = threading.Thread(target=monitor, args=(log, monitor_stop, endpoints, lock), daemon=True)
    watcher.start()
    store = None
    try:
        store, installed, elapsed = setup_store(args.master)
        log.emit(
            "ready",
            versions=installed,
            setup_sec=elapsed,
            batch_size=args.batch_size,
            workers=args.workers,
            page_bytes=PAGE_BYTES,
            seconds=args.seconds,
            store_endpoint=store.get_hostname(),
        )
        barrier = threading.Barrier(args.workers)
        deadline = [0.0]

        def worker(worker_id: int) -> bool:
            mapping = None
            pointer = 0
            registered = False
            passed = True
            try:
                keys = [
                    f"diag/{args.case}/{socket.gethostname()}/{os.getpid()}/{worker_id}/{i}"
                    for i in range(args.batch_size)
                ]
                mapping = mmap.mmap(-1, args.batch_size * PAGE_BYTES)
                pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                pointers = [[pointer + i * PAGE_BYTES] for i in range(args.batch_size)]
                sizes = [[PAGE_BYTES] for _ in keys]
                markers = [hashlib.blake2b(key.encode(), digest_size=8).digest() for key in keys]
                rc = store.register_buffer(pointer, len(mapping))
                if rc != 0:
                    raise RuntimeError(f"register_buffer failed: {rc}")
                registered = True
                for i, marker in enumerate(markers):
                    mapping[i * PAGE_BYTES : i * PAGE_BYTES + 8] = marker
                    mapping[(i + 1) * PAGE_BYTES - 8 : (i + 1) * PAGE_BYTES] = marker
                started = time.monotonic()
                result = store.batch_put_from_multi_buffers(keys, pointers, sizes)
                errors = Counter(str(value) for value in result if value != 0)
                if len(result) != len(keys):
                    errors["result_count_mismatch"] += 1
                seed_ok = len(result) == len(keys) and not errors
                log.emit(
                    "put_batch",
                    worker=worker_id,
                    duration_ms=(time.monotonic() - started) * 1000,
                    keys=len(keys),
                    returned=len(result),
                    errors=dict(errors),
                    ok=seed_ok,
                )
                if not seed_ok:
                    barrier.abort()
                    return False
                replicas = replica_samples(store, keys)
                log.emit("replicas", worker=worker_id, replicas=replicas)
                if barrier.wait(timeout=30) == 0:
                    deadline[0] = time.monotonic() + args.seconds
                barrier.wait(timeout=30)
                batches = 0
                while time.monotonic() < deadline[0] and not stop.is_set():
                    for i in range(len(keys)):
                        mapping[i * PAGE_BYTES : i * PAGE_BYTES + 8] = b"\xff" * 8
                        mapping[(i + 1) * PAGE_BYTES - 8 : (i + 1) * PAGE_BYTES] = b"\xff" * 8
                    started = time.monotonic()
                    result = store.batch_get_into_multi_buffers(keys, pointers, sizes)
                    duration_ms = (time.monotonic() - started) * 1000
                    errors = Counter(str(value) for value in result if value != PAGE_BYTES)
                    if len(result) != len(keys):
                        errors["result_count_mismatch"] += 1
                    failures = sum(
                        mapping[i * PAGE_BYTES : i * PAGE_BYTES + 8] != markers[i]
                        or mapping[(i + 1) * PAGE_BYTES - 8 : (i + 1) * PAGE_BYTES] != markers[i]
                        for i, value in enumerate(result[: len(keys)])
                        if value == PAGE_BYTES
                    )
                    successes = sum(value == PAGE_BYTES for value in result[: len(keys)])
                    valid = len(result) == len(keys) and not errors and failures == 0
                    passed = passed and valid
                    batches += 1
                    log.emit(
                        "get_batch",
                        worker=worker_id,
                        duration_ms=duration_ms,
                        keys=len(keys),
                        returned=len(result),
                        errors=dict(errors),
                        verification_failures=failures,
                        verified_bytes=(successes - failures) * PAGE_BYTES,
                        ok=valid,
                    )
                log.emit("worker_complete", worker=worker_id, batches=batches, ok=passed and batches > 0)
                return passed and batches > 0
            except Exception:
                barrier.abort()
                log.emit("worker_error", worker=worker_id, error=traceback.format_exc())
                return False
            finally:
                if registered:
                    log.emit("unregister", worker=worker_id, returncode=store.unregister_buffer(pointer))
                if mapping is not None:
                    mapping.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(worker, range(args.workers)))
        log.emit("client_complete", ok=all(results), workers_ok=sum(results), workers=args.workers)
        if not all(results):
            raise RuntimeError("One or more diagnostic workers failed; inspect JSONL")
    finally:
        stop.set()
        monitor_stop.set()
        watcher.join()
        if store is not None:
            log.emit("cleanup_complete", returncode=store.close())


def percentile(values: list[float], percent: float) -> str:
    if not values:
        return "N/A"
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return f"{values[lower] + (values[upper] - values[lower]) * (position - lower):.3f}"


def report(args: argparse.Namespace) -> None:
    logs = args.run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    sources = []
    for path in sorted(args.run_dir.rglob("*.jsonl")):
        sources.append(str(path.relative_to(args.run_dir)))
        for line in path.read_text().splitlines():
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("JSONL record is not an object")
                records.append(
                    {
                        "event": "external_health",
                        "case": path.parent.name,
                        **payload,
                        "source": str(path.relative_to(args.run_dir)),
                    }
                )
            except (json.JSONDecodeError, ValueError):
                records.append({"event": "malformed_record", "source": str(path.relative_to(args.run_dir))})
    cases = sorted(
        {record["case"] for record in records if "case" in record}
        | {path.parent.name for path in logs.glob("*/incomplete.txt")}
    )
    metadata = read_text(args.run_dir / "metadata.json")
    settings = json.loads(metadata) if metadata else {}
    expected_clients = len(settings.get("readers", "").split(",")) if settings.get("readers") else None
    text = [
        "# Mooncake host-memory/RDMA diagnostic",
        "",
        "Goal: compare isolated donor placement/capacity under concurrent registered-host-memory reads.",
        "No model or GPU memory pressure is present. This short diagnostic does not reproduce the original roughly 40-minute serving workload.",
        "TTFT p50/p95/p99, ITL p50/p99, token throughput, agentic Pareto/SLO metrics: N/A (host-only byte transfer).",
        "",
        "Batch latencies below are milliseconds; percentiles pool completed calls across threads/nodes. "
        "Verified throughput is bytes/s of pages whose first/last 8 bytes matched after successful reads; "
        "it is not a full-payload checksum. Transfer payload is 2,006,016 bytes/page.",
        "",
        "| Case | Put p50/p95/p99 ms | Get p50/p95/p99 ms | Verified bytes/s | Errors | Verification failures |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    details = []
    for case in cases:
        selected = [record for record in records if record.get("case") == case]
        gets = [record for record in selected if record["event"] == "get_batch"]
        puts = [record for record in selected if record["event"] == "put_batch"]
        put_percentiles = "/".join(percentile([row["duration_ms"] for row in puts], p) for p in (50, 95, 99))
        get_percentiles = "/".join(percentile([row["duration_ms"] for row in gets], p) for p in (50, 95, 99))
        errors: Counter[str] = Counter()
        for row in gets + puts:
            errors.update(row["errors"])
        window = max((row["epoch_s"] for row in gets), default=0) - min(
            (row["epoch_s"] - row["duration_ms"] / 1000 for row in gets), default=0
        )
        rate = f"{sum(row['verified_bytes'] for row in gets) / window:.0f}" if window > 0 else "N/A"
        failures = sum(row["verification_failures"] for row in gets)
        text.append(f"| {case} | {put_percentiles} | {get_percentiles} | {rate} | {dict(errors)} | {failures} |")
        complete = [row for row in selected if row["event"] == "client_complete"]
        details.extend(
            [
                "",
                f"Case `{case}`: {len(gets)} read batches, {len(puts)} seed batches; "
                f"{sum(bool(row['ok']) for row in complete)}/{expected_clients or len(complete)} expected/recorded "
                f"clients completed successfully; measured interval {window:.3f} s.",
            ]
        )
        snapshots = [row for row in selected if row["event"] == "health" and isinstance(row.get("health"), dict)]
        available = [
            row["health"]["meminfo_kib"]["MemAvailable"]
            for row in snapshots
            if "MemAvailable" in row["health"].get("meminfo_kib", {})
        ]
        rss = [
            row["health"]["proc_status_kib"]["VmRSS"]
            for row in snapshots
            if "VmRSS" in row["health"].get("proc_status_kib", {})
        ]
        deltas = Counter()
        oom_deltas = Counter()
        for source in {row["source"] for row in snapshots}:
            health_series = [row["health"] for row in snapshots if row["source"] == source]
            series = [row.get("ib_counters", {}) for row in health_series]
            for name, value in series[-1].items():
                if (
                    name in series[0]
                    and value != series[0][name]
                    and any(word in name for word in ("err", "discard", "link_downed", "constraint", "timeout"))
                ):
                    deltas[f"{source}:{name}"] = value - series[0][name]
            events = [
                dict(line.split() for line in (row.get("cgroup", {}).get("memory.events") or "").splitlines())
                for row in health_series
            ]
            for name in ("oom", "oom_kill", "oom_group_kill", "max"):
                if name in events[0] and name in events[-1]:
                    oom_deltas[f"{source}:{name}"] = int(events[-1][name]) - int(events[0][name])
        details.append(
            f"Memory evidence: minimum sampled MemAvailable={min(available) if available else 'N/A'} KiB; "
            f"maximum sampled process VmRSS={max(rss) if rss else 'N/A'} KiB. "
            f"Nonzero IB error-counter deltas (counts, final minus first per source): {dict(deltas)}."
        )
        details.append(
            f"Cgroup memory.events deltas (counts, final minus first): {dict(oom_deltas)}; "
            "an absent entry means the counter was unavailable."
        )
        replicas = [replica for row in selected if row["event"] == "replicas" for replica in row["replicas"]]
        details.append(
            f"Replica coverage: {len(replicas)} descriptor records; endpoints "
            f"{sorted({row['endpoint'] for row in replicas})}; "
            f"address range {min((row['address'] for row in replicas), default='N/A')} to "
            f"{max((row['address'] + row['size'] for row in replicas), default='N/A')} (exclusive end)."
        )
        native_counts = Counter()
        for path in (logs / case).glob("*.err"):
            for line in path.open(errors="replace"):
                for pattern in (
                    "transport retry counter exceeded",
                    "remote access error",
                    "Connection timed out",
                    "Cannot redispatch slice",
                    "LEASE_EXPIRED",
                    "Failed to complete transfers",
                ):
                    if pattern in line:
                        native_counts[pattern] += 1
        details.append(f"Native error text counts from `{case}/*.err`: {dict(native_counts)}.")
    text.extend(details)
    text.extend(inventory_summary(args.run_dir))
    exceptional = [record for record in records if record["event"] in ("fatal", "worker_error", "malformed_record")]
    probes = [item for record in records if record["event"] == "health" for item in record.get("probes", [])]
    external_probes = [record for record in records if record["event"] == "external_health" and "tcp_ok" in record]
    text.extend(
        [
            "",
            f"Fatal/worker/malformed records: {len(exceptional)}. "
            f"TCP endpoint probe failures: {sum(not item['ok'] for item in probes)}/{len(probes)}.",
            f"External controller probe failures: {sum(not item['tcp_ok'] for item in external_probes)}/{len(external_probes)}.",
            "TCP probes use only the P2P listener parsed from donor native stdout/stderr. "
            "They verify connection establishment, not RDMA success; store/replica endpoint names may use a different port.",
            "",
            "Sources: " + (", ".join(f"`{source}`" for source in sources) or "No JSONL files found."),
            "Replica records contain per-key address/size/endpoint. Compare them with donor registration base and "
            "max_mr_size in native logs before claiming a second MR or NUMA region was exercised. "
            "A 1100 GiB registration with a small working set does not establish that coverage.",
            "Health records contain local PID/RSS, cgroup memory/OOM data, node meminfo and raw IB counters; "
            "compare first/last values per process and NIC. Missing Linux counters are empty or null, not zeros.",
            "Mooncake's native setup holds the Python GIL, so local health sampling pauses during registration; "
            "startup samples do not continuously observe that interval. Kernel access failures are recorded in inventory JSON.",
            "Native stdout/stderr supplies RDMA errors and registration details; status files supply process exit codes. "
            "Missing completion/status records make a case incomplete even if recorded batches succeeded.",
            "No DSight/dashboard was used: per-call JSONL, TCP probes, native logs and local memory/IB counters "
            "directly test peer availability and transport correctness. "
            "The [slurm-job-sizing skill](../../../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md) "
            "guides the short bounded diagnostic; it does not turn this into a serving benchmark.",
        ]
    )
    if metadata:
        text.extend(["", "Run metadata (`metadata.json`):", "```json", metadata, "```"])
    status_files = (
        list(args.run_dir.rglob("*.exit"))
        + list(args.run_dir.rglob("*.unexpected-exit"))
        + list(args.run_dir.rglob("incomplete.txt"))
        + list(args.run_dir.rglob("teardown-unconfirmed.txt"))
    )
    for path in sorted(status_files):
        text.append(f"\nStatus `{path.relative_to(args.run_dir)}`: `{read_text(path)}`")
    (logs / "perf-analysis.md").write_text("\n".join(text) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, handler in (("inventory", inventory), ("donor", donor), ("client", client), ("report", report)):
        child = sub.add_parser(command)
        child.set_defaults(handler=handler)
        if command == "report":
            child.add_argument("--run-dir", type=Path, required=True)
            continue
        child.add_argument("--output", type=Path, required=True)
        if command == "inventory":
            child.add_argument("--scope", choices=("host", "container"), default="container")
            child.add_argument("--commands-timeout", type=float, default=3.0)
            continue
        child.add_argument("--master", required=True)
        if command == "donor":
            child.add_argument("--gib", type=int, required=True)
            child.add_argument("--ready", type=Path, required=True)
            child.add_argument(
                "--native-log", type=Path, help="Donor redirected stdout/stderr, used to discover actual P2P port"
            )
        else:
            child.add_argument("--case", required=True)
            child.add_argument("--seconds", type=float, required=True)
            child.add_argument("--batch-size", type=int, default=512)
            child.add_argument("--workers", type=int, default=2)
            child.add_argument("--donor-ready", type=Path)
    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception:
        if hasattr(args, "output") and args.command != "inventory":
            Jsonl(args.output, role=args.command, host=socket.gethostname()).emit("fatal", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
