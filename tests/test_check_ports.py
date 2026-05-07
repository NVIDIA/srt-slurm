# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for srtctl/runtime_scripts/check_ports.py."""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from srtctl.runtime_scripts import check_ports

SCRIPT_PATH = Path(check_ports.__file__).resolve()


def _free_port() -> int:
    """Pick a port that is currently free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def listening_socket():
    """Bind+listen on a loopback port; tear down after the test."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    port = s.getsockname()[1]
    try:
        yield port
    finally:
        s.close()


def _run_script(ports: list[int]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--ports", *(str(p) for p in ports)],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestCheckPortsCLI:
    def test_all_free_returns_zero(self):
        # Pick two ports we're confident are free, then immediately query.
        # Tiny TOCTOU window but harmless for the test (the script doesn't bind).
        ports = [_free_port(), _free_port()]
        result = _run_script(ports)
        assert result.returncode == 0, result.stdout + result.stderr
        for p in ports:
            assert f"PORT_OK    127.0.0.1:{p}" in result.stdout

    def test_busy_port_emits_diagnostic(self, listening_socket):
        port = listening_socket
        free = _free_port()
        result = _run_script([port, free])
        assert result.returncode == 1, result.stdout + result.stderr
        assert f"PORT_OK    127.0.0.1:{free}" in result.stdout
        # The busy port line must include the test's own pid + diagnostic fields.
        busy_line = next(
            (line for line in result.stdout.splitlines() if line.startswith("PORT_BUSY")),
            None,
        )
        assert busy_line is not None, result.stdout
        assert f":{port}" in busy_line
        assert f"pid={os.getpid()}" in busy_line
        assert "user=" in busy_line
        assert "cmdline=" in busy_line
        assert "name=" in busy_line


class TestDecodeLocalAddress:
    def test_ipv4_loopback(self):
        # /proc/net/tcp encodes 127.0.0.1 as 0100007F (LE bytes)
        # and port 30236 as hex 7620.
        assert check_ports._decode_local_address("0100007F:7620") == ("127.0.0.1", 0x7620)

    def test_ipv4_any(self):
        assert check_ports._decode_local_address("00000000:1F90") == ("0.0.0.0", 8080)

    def test_ipv6_loopback(self):
        # ::1 encoded by the kernel as four LE 32-bit words: 00000000 00000000 00000000 01000000
        # (the trailing 01000000 is byte-reversed 00000001 -> ...01)
        assert check_ports._decode_local_address("00000000000000000000000001000000:1F90") == ("::1", 8080)

    def test_ipv6_any(self):
        assert check_ports._decode_local_address("00000000000000000000000000000000:1F90") == ("::", 8080)

    def test_invalid_returns_none(self):
        assert check_ports._decode_local_address("not-an-address") is None
        assert check_ports._decode_local_address("0100007F") is None  # missing port
        assert check_ports._decode_local_address("0100007F:GGGG") is None  # bad hex


class TestProcessInfo:
    def test_process_info_self(self):
        info = check_ports._process_info(os.getpid())
        # On Linux, the test runner has a comm and at least argv[0].
        assert info["pid"] == str(os.getpid())
        assert "name" in info
        assert info.get("uid") is not None
        assert info.get("user") not in (None, "")

    def test_process_info_missing_pid(self):
        # PID 1 may or may not be readable depending on container; pick a
        # guaranteed-not-running pid.
        info = check_ports._process_info(2**31 - 1)
        # Just the seed field; others are absent because /proc reads failed.
        assert info == {"pid": str(2**31 - 1)}


class TestFormatBusy:
    def test_format_busy_unknown_owner(self):
        # When pid AND uid resolution fail, the script must still emit a
        # PORT_BUSY row with all-placeholder fields rather than crashing.
        line = check_ports._format_busy(("127.0.0.1", 30236), None)
        assert line.startswith("PORT_BUSY")
        assert "127.0.0.1:30236" in line
        assert "pid=?" in line
        assert "user=?" in line
        assert "cmdline=" in line


class TestUidFallback:
    """Verify the cross-user fallback path emits user= even without pid."""

    def test_info_from_uid_resolves_username(self):
        # uid 0 is root on every Linux system.
        info = check_ports._info_from_uid(0)
        assert info["uid"] == "0"
        assert info["user"] == "root"

    def test_info_from_uid_missing_user_falls_back(self):
        # Pick a uid extremely unlikely to exist in /etc/passwd.
        info = check_ports._info_from_uid(2**31 - 2)
        # Either "?" if the lookup failed, or some unusual entry — accept both.
        assert info["uid"] == str(2**31 - 2)
        assert "user" in info

    def test_info_from_uid_negative_means_unknown(self):
        info = check_ports._info_from_uid(-1)
        assert info["uid"] == "?"
        assert "user" not in info  # No uid -> can't even attempt lookup.

    def test_busy_with_uid_only(self, monkeypatch, capsys, listening_socket):
        """Simulate the cross-user case: /proc/net/tcp shows the listener but
        we can't walk /proc/<pid>/fd to get the pid.

        We do this by monkeypatching ``_find_owner_pid`` to return {} (as if
        every fd directory was permission-denied). The script must still emit
        a PORT_BUSY line with user= populated from the uid recorded in
        /proc/net/tcp (which we own — it's our test process).
        """
        port = listening_socket
        free = _free_port()

        monkeypatch.setattr(check_ports, "_find_owner_pid", lambda inodes: {})
        rc = check_ports.check_ports([port, free])
        assert rc == 1
        out = capsys.readouterr().out
        assert f"PORT_OK    127.0.0.1:{free}" in out
        # Pid is "?" (we faked the lookup) but uid+user must be filled from
        # /proc/net/tcp's column 7 — which is our own uid.
        busy = next(line for line in out.splitlines() if line.startswith("PORT_BUSY"))
        assert f":{port}" in busy
        assert "pid=?" in busy
        assert f"uid={os.getuid()}" in busy
        # User name should resolve since the test runs under a real user.
        assert "user=?" not in busy
