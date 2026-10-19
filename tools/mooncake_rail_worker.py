#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent local perftest worker controlled by atomic JSON filesystem mailboxes."""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_NICS = "mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11"


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def capture(argv: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=3, check=False)
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout[-8192:],
            "stderr": result.stderr[-8192:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def cpu_list(value: str) -> set[int]:
    result = set()
    for part in value.split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        result.update(range(int(first), int(last or first) + 1))
    return result


def usable_ipv4(value: str) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
        return not (address.is_loopback or address.is_unspecified or address.is_multicast or address.is_reserved)
    except ipaddress.AddressValueError:
        return False


def cgroup_evidence() -> dict[str, Any]:
    membership = read_text(Path("/proc/self/cgroup"))
    result: dict[str, Any] = {"membership": membership, "scope": "visible process cgroup; ancestor limits not surveyed"}
    relative = next((line.split(":", 2)[2] for line in (membership or "").splitlines() if line.startswith("0::")), None)
    for line in (read_text(Path("/proc/self/mountinfo")) or "").splitlines():
        if " - cgroup2 " not in line or relative is None:
            continue
        fields = line.split()
        try:
            base = Path(fields[4]) / Path(relative).relative_to(fields[3])
        except ValueError:
            continue
        result["path"] = str(base)
        for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective", "cpuset.mems.effective"):
            result[name] = read_text(base / name)
        return result
    result["unavailable"] = "No matching visible cgroup v2 mount/membership"
    return result


def inventory(node: str) -> dict[str, Any]:
    errors = []
    tools = {}
    for name in ("ib_read_bw", "ib_write_bw", "numactl"):
        executable = shutil.which(name)
        tools[name] = {"path": executable, "version": capture([executable, "--version"]) if executable else None}
        if name.startswith("ib_"):
            tools[name]["help"] = capture([executable, "--help"]) if executable else None
        if not executable:
            errors.append(f"Required tool missing: {name}")
    status = dict(
        line.split(":", 1) for line in (read_text(Path("/proc/self/status")) or "").splitlines() if ":" in line
    )
    allowed_cpus = status.get("Cpus_allowed_list", "").strip()
    allowed_mems = status.get("Mems_allowed_list", "").strip()
    nics = {}
    for name in os.environ.get("MC_RAIL_NICS", DEFAULT_NICS).split(","):
        base = Path("/sys/class/infiniband") / name
        port = base / "ports/1"
        numa = read_text(base / "device/numa_node")
        values = {
            "sysfs_port": str(port),
            "state": read_text(port / "state"),
            "link_layer": read_text(port / "link_layer"),
            "numa_node": int(numa) if numa is not None and numa.isdigit() else -1,
            "gid": read_text(port / "gids/0"),
        }
        nics[name] = values
        if not (values["state"] or "").startswith("4:"):
            errors.append(f"{name}: port 1 is not ACTIVE: {values['state']}")
        if values["link_layer"] != "InfiniBand":
            errors.append(f"{name}: expected InfiniBand, got {values['link_layer']}")
        try:
            if not values["gid"] or ipaddress.IPv6Address(values["gid"]).is_unspecified:
                errors.append(f"{name}: GID index 0 is missing or all zero")
        except ipaddress.AddressValueError:
            errors.append(f"{name}: invalid GID index 0: {values['gid']}")
        if numa is None or not numa.isdigit():
            errors.append(f"{name}: unavailable NUMA node: {numa}")
    bind_checks = {}
    if tools["numactl"]["path"]:
        for numa in sorted({values["numa_node"] for values in nics.values() if values["numa_node"] >= 0}):
            check = capture([tools["numactl"]["path"], f"--cpunodebind={numa}", f"--membind={numa}", "/bin/true"])
            bind_checks[str(numa)] = check
            if check.get("returncode") != 0:
                errors.append(f"NUMA {numa}: actual CPU/memory bind failed: {check}")
    addresses = set()
    preferred = []
    for hostname in (node, socket.gethostname()):
        with contextlib.suppress(OSError):
            found = [item[4][0] for item in socket.getaddrinfo(hostname, None, socket.AF_INET)]
            preferred.extend(value for value in found if usable_ipv4(value))
            addresses.update(found)
    hostname_ip = capture(["hostname", "-i"])
    for value in hostname_ip.get("stdout", "").split():
        with contextlib.suppress(OSError):
            socket.inet_aton(value)
            addresses.add(value)
    numa_policy = capture([tools["numactl"]["path"], "--show"]) if tools["numactl"]["path"] else None
    ip = preferred[0] if preferred else next((value for value in sorted(addresses) if usable_ipv4(value)), None)
    if ip is None:
        errors.append("No usable non-loopback IPv4 address resolved for this node")
    return {
        "node": node,
        "hostname": socket.gethostname(),
        "ip": ip,
        "ipv4": sorted(addresses),
        "hostname_ip": hostname_ip,
        "tools": tools,
        "nics": nics,
        "cpus_allowed": allowed_cpus,
        "mems_allowed": allowed_mems,
        "numa_policy": numa_policy,
        "numa_bind_checks": bind_checks,
        "platform": {"uname": platform.uname()._asdict(), "os_release": read_text(Path("/etc/os-release"))},
        "cgroup": cgroup_evidence(),
        "errors": errors,
        "ready": not errors,
    }


def listener_owned(pid: int, port: int) -> bool:
    """Match an owned socket inode to LISTEN without opening a connection."""
    inodes = set()
    for descriptor in Path(f"/proc/{pid}/fd").glob("*"):
        with contextlib.suppress(OSError):
            match = re.fullmatch(r"socket:\[(\d+)\]", os.readlink(descriptor))
            if match:
                inodes.add(match[1])
    for name in ("tcp", "tcp6"):
        for line in (read_text(Path(f"/proc/{pid}/net/{name}")) or "").splitlines()[1:]:
            fields = line.split()
            if (
                len(fields) > 9
                and fields[3] == "0A"
                and fields[9] in inodes
                and int(fields[1].rsplit(":", 1)[1], 16) == port
            ):
                return True
    return False


def allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def build_command(command: dict[str, Any], info: dict[str, Any], port: int) -> list[str]:
    op, role, nic = command["op"], command["role"], command["nic"]
    if op not in ("read", "write") or role not in ("server", "client") or nic not in info["nics"]:
        raise ValueError("Invalid op, role, or NIC")
    numa = int(command["numa"])
    if numa != info["nics"][nic]["numa_node"]:
        raise ValueError(f"NUMA node {numa} does not match {nic}")
    size, duration = int(command.get("size", 2_006_016)), int(command.get("duration", 5))
    if size <= 0 or not 1 <= duration <= 300 or not 1 <= port <= 65535:
        raise ValueError("Invalid size, duration, or port")
    argv = [
        info["tools"]["numactl"]["path"],
        f"--cpunodebind={numa}",
        f"--membind={numa}",
        info["tools"][f"ib_{op}_bw"]["path"],
        "-d",
        nic,
        "-i",
        "1",
        "-x",
        "0",
        "-p",
        str(port),
        "-D",
        str(duration),
        "-f",
        "1",
        "-s",
        str(size),
        "-q",
        "2",
        "--report_gbits",
    ]
    if op == "read":
        argv.extend(["-o", "16"])
    if role == "client":
        peer = command["peer_ip"]
        socket.inet_pton(socket.AF_INET, peer)
        argv.append(peer)
    return argv


class Worker:
    def __init__(self, run_dir: Path, node: str, info: dict[str, Any]) -> None:
        self.run_dir, self.node, self.info = run_dir, node, info
        self.commands = run_dir / "logs/agents" / f"{node}.commands"
        self.results = run_dir / "logs/agents" / f"{node}.results"
        self.commands.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.child: dict[str, Any] | None = None
        self.stopping = False

    def finish(self, reason: str = "exit") -> dict[str, Any] | None:
        if self.child is None:
            return None
        child = self.child
        process = child["process"]
        if process.poll() is None:
            return None
        result = {key: value for key, value in child.items() if key not in ("process", "stream", "deadline", "started")}
        result.update(
            status="completed",
            returncode=process.returncode,
            runtime_s=time.monotonic() - child["started"],
            reason=reason,
        )
        child["stream"].close()
        if child["pending_ready"]:
            atomic_json(
                self.results / f"{child['command_id']}.json",
                {"status": "error", "error": "Server ended before LISTEN readiness", "reason": reason},
            )
        atomic_json(self.results / f"{child['command_id']}.done.json", result)
        self.child = None
        return result

    def stop_child(self, reason: str = "stop") -> dict[str, Any] | None:
        if self.child is None:
            return None
        process = self.child["process"]
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        return self.finish(reason)

    def handle(self, command_id: str, command: dict[str, Any]) -> dict[str, Any] | None:
        action = command["action"]
        if action in ("stop", "shutdown"):
            completed = self.stop_child(action)
            self.stopping = action == "shutdown"
            return {"status": "stopped", "completed": completed}
        if action != "start" or self.child is not None:
            raise ValueError("Unknown action or worker already busy")
        case_id = command["case_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id) or case_id in (".", ".."):
            raise ValueError("Invalid case_id")
        if command["role"] == "client" and not command.get("port"):
            raise ValueError("Client requires the server port")
        port = int(command.get("port") or allocate_port())
        argv = build_command(command, self.info, port)
        runtime_limit = float(command.get("timeout", int(command.get("duration", 5)) + 10))
        if not 1 <= runtime_limit <= 330:
            raise ValueError("Child timeout must be between 1 and 330 seconds")
        relative = Path("logs/cases") / f"{case_id}.{command['role']}.log"
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("w")
        try:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            stream.close()
            raise
        started = time.monotonic()
        self.child = {
            "process": process,
            "stream": stream,
            "command_id": command_id,
            "case_id": case_id,
            "pid": process.pid,
            "port": port,
            "argv": argv,
            "log_path": str(relative),
            "started": started,
            "pending_ready": command["role"] == "server",
            "deadline": started + runtime_limit,
        }
        if command["role"] == "server":
            return None
        return {"status": "started", "pid": process.pid, "port": port, "argv": argv, "log_path": str(relative)}

    def run(self) -> None:
        seen = set()
        try:
            while not self.stopping:
                if self.child is not None and time.monotonic() >= self.child["deadline"]:
                    self.stop_child("deadline")
                self.finish()
                if self.child is not None and self.child["pending_ready"]:
                    child = self.child
                    if listener_owned(child["pid"], child["port"]):
                        child["pending_ready"] = False
                        atomic_json(
                            self.results / f"{child['command_id']}.json",
                            {"status": "ready", **{key: child[key] for key in ("pid", "port", "argv", "log_path")}},
                        )
                    elif time.monotonic() - child["started"] >= 8:
                        self.stop_child("listener_not_ready")
                for path in sorted(self.commands.glob("*.json")):
                    if path.stem in seen:
                        continue
                    seen.add(path.stem)
                    try:
                        result = self.handle(path.stem, json.loads(path.read_text()))
                    except Exception as exc:
                        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    if result is not None:
                        atomic_json(self.results / path.name, result)
                    if self.stopping:
                        break
                time.sleep(0.05)
        finally:
            self.stop_child("worker_shutdown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--node", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.node) or args.node in (".", ".."):
        parser.error("Invalid node name")
    info = inventory(args.node)
    atomic_json(args.run_dir / "logs/agents" / f"{args.node}.inventory.json", info)
    if not info["ready"]:
        raise SystemExit(1)
    worker = Worker(args.run_dir, args.node, info)
    signal.signal(signal.SIGTERM, lambda *_: setattr(worker, "stopping", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(worker, "stopping", True))
    worker.run()


if __name__ == "__main__":
    main()
