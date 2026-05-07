# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight check: verify TCP ports are free on the host network namespace.

Designed to run on the bare host (no container, no third-party deps) before
SGLang prefill workers launch. Pyxis containers use host networking, so a
host-side scan sees both bare-host listeners and any containerized listener
bound to a host port.

Output is parsed by ``do_sweep.py`` preflight; keep the format stable.

Usage:
    python3 check_ports.py --ports 30000 30236 30237

Output (one line per port):
    PORT_OK    127.0.0.1:30000
    PORT_BUSY  127.0.0.1:30236  pid=12345 uid=1234 user=jullin name=python3 cmdline='python3 -m dynamo.sglang ...' state=R

Exit 0 if all ports OK, 1 if any port is busy.
"""

from __future__ import annotations

import argparse
import os
import pwd
import re
import shlex
import sys
from pathlib import Path

# Linux TCP socket states (see net/tcp_states.h). 0A == LISTEN.
TCP_LISTEN = "0A"


def _decode_local_address(field: str) -> tuple[str, int] | None:
    """Decode the ``local_address`` column from /proc/net/tcp{,6}.

    IPv4 form: ``0100007F:7530`` (LE-encoded address, hex port).
    IPv6 form: 32 hex chars + ``:`` + 4 hex port.
    """
    if ":" not in field:
        return None
    addr_hex, port_hex = field.rsplit(":", 1)
    try:
        port = int(port_hex, 16)
    except ValueError:
        return None
    if len(addr_hex) == 8:
        # IPv4: little-endian byte order
        try:
            packed = bytes.fromhex(addr_hex)
        except ValueError:
            return None
        ip = ".".join(str(b) for b in reversed(packed))
        return ip, port
    if len(addr_hex) == 32:
        # IPv6: 32 hex chars; for our purposes we just need to know whether
        # it's loopback or any. Translate the in6_addr struct (which the
        # kernel writes as four LE 32-bit words) to a colon-grouped form.
        # Simplification: we only care about ::1 (loopback) and :: (any) for
        # diagnostic clarity; otherwise fall through to "ipv6".
        try:
            words = [addr_hex[i : i + 8] for i in range(0, 32, 8)]
            decoded = b"".join(bytes.fromhex(w)[::-1] for w in words)
        except ValueError:
            return None
        if decoded == b"\x00" * 16:
            return "::", port
        if decoded == b"\x00" * 15 + b"\x01":
            return "::1", port
        return "ipv6", port
    return None


def _read_listening_inodes(proc_net_path: str, target_ports: set[int]) -> dict[int, list[tuple[str, int]]]:
    """Return ``{inode: [(local_ip, port), ...]}`` for each LISTEN entry on a target port.

    Multiple entries can map to the same inode if the kernel double-lists
    (rare); we keep all addresses for diagnostic output.
    """
    result: dict[int, list[tuple[str, int]]] = {}
    try:
        with open(proc_net_path) as fh:
            next(fh)  # skip header
            for line in fh:
                cols = line.split()
                # Columns: sl local_address rem_address st tx_q rx_q tr tm->when retrnsmt uid timeout inode
                if len(cols) < 12:
                    continue
                if cols[3] != TCP_LISTEN:
                    continue
                local = _decode_local_address(cols[1])
                if local is None or local[1] not in target_ports:
                    continue
                try:
                    inode = int(cols[9])
                except ValueError:
                    continue
                result.setdefault(inode, []).append(local)
    except FileNotFoundError:
        pass
    return result


_SOCKET_FD_RE = re.compile(r"^socket:\[(\d+)\]$")


def _find_owner_pid(target_inodes: set[int]) -> dict[int, int]:
    """Walk /proc/*/fd/* to map socket inodes to owning pids.

    Returns ``{inode: pid}``. Inodes not found are absent (caller falls back
    to placeholder fields). Tolerates EACCES/ENOENT silently.
    """
    if not target_inodes:
        return {}
    found: dict[int, int] = {}
    for entry in os.scandir("/proc"):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        pid = int(entry.name)
        fd_dir = Path("/proc") / entry.name / "fd"
        try:
            fd_iter = os.scandir(fd_dir)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        with fd_iter as it:
            for fd_entry in it:
                try:
                    target = os.readlink(fd_entry.path)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                m = _SOCKET_FD_RE.match(target)
                if not m:
                    continue
                inode = int(m.group(1))
                if inode in target_inodes and inode not in found:
                    found[inode] = pid
                    if len(found) == len(target_inodes):
                        return found
    return found


def _read_proc_field(pid: int, name: str) -> str | None:
    try:
        return (Path("/proc") / str(pid) / name).read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _process_info(pid: int) -> dict[str, str]:
    """Best-effort ``{uid, user, name, cmdline, state}`` for a pid."""
    info: dict[str, str] = {"pid": str(pid)}

    comm = _read_proc_field(pid, "comm")
    if comm is not None:
        info["name"] = comm.strip() or "?"

    cmdline = _read_proc_field(pid, "cmdline")
    if cmdline is not None:
        # NUL-separated argv; collapse to a shell-quoted string for log clarity.
        argv = [a for a in cmdline.split("\0") if a]
        info["cmdline"] = shlex.join(argv) if argv else "?"

    status = _read_proc_field(pid, "status")
    if status is not None:
        for line in status.splitlines():
            if line.startswith("Uid:"):
                parts = line.split()
                if len(parts) >= 2:
                    uid = parts[1]
                    info["uid"] = uid
                    try:
                        info["user"] = pwd.getpwuid(int(uid)).pw_name
                    except (KeyError, ValueError):
                        info["user"] = "?"
            elif line.startswith("State:"):
                # "State:\tR (running)" -> "R"
                parts = line.split()
                if len(parts) >= 2:
                    info["state"] = parts[1]
    return info


def _format_busy(addr: tuple[str, int], info: dict[str, str] | None) -> str:
    ip, port = addr
    fields = ["pid", "uid", "user", "name", "cmdline", "state"]
    if info is None:
        info = {}
    parts = []
    for f in fields:
        v = info.get(f, "?")
        if f == "cmdline":
            # Quote so spaces don't confuse downstream parsers.
            parts.append(f"cmdline={v!r}")
        else:
            parts.append(f"{f}={v}")
    return f"PORT_BUSY  {ip}:{port}  " + " ".join(parts)


def check_ports(ports: list[int]) -> int:
    """Check each port; print one line per port; return non-zero if any busy."""
    target_set = set(ports)

    # Aggregate listening inodes across IPv4 and IPv6 (host network namespace).
    inode_to_addrs: dict[int, list[tuple[str, int]]] = {}
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        for inode, addrs in _read_listening_inodes(path, target_set).items():
            inode_to_addrs.setdefault(inode, []).extend(addrs)

    # Build port -> (inode, addrs) map. A port can have multiple inodes
    # (IPv4+IPv6 dual-stack); we report each separately.
    port_to_entries: dict[int, list[tuple[int, tuple[str, int]]]] = {}
    for inode, addrs in inode_to_addrs.items():
        for addr in addrs:
            port_to_entries.setdefault(addr[1], []).append((inode, addr))

    # Resolve owner pids for any busy ports.
    busy_inodes = {inode for entries in port_to_entries.values() for inode, _ in entries}
    inode_to_pid = _find_owner_pid(busy_inodes)

    # Emit one line per requested port (in input order).
    any_busy = False
    for port in ports:
        entries = port_to_entries.get(port, [])
        if not entries:
            print(f"PORT_OK    127.0.0.1:{port}", flush=True)
            continue
        any_busy = True
        for inode, addr in entries:
            pid = inode_to_pid.get(inode)
            info = _process_info(pid) if pid is not None else None
            print(_format_busy(addr, info), flush=True)
    return 1 if any_busy else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=int, nargs="+", required=True, help="TCP ports to check")
    args = parser.parse_args(argv)
    return check_ports(args.ports)


if __name__ == "__main__":
    sys.exit(main())
