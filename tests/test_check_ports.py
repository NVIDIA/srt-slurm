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
        # When pid resolution fails (e.g. another user's process), the script
        # must still emit a PORT_BUSY row with placeholder fields rather than
        # crashing. This is the diagnostic-fallback path.
        line = check_ports._format_busy(("127.0.0.1", 30236), None)
        assert line.startswith("PORT_BUSY")
        assert "127.0.0.1:30236" in line
        assert "pid=?" in line
        assert "user=?" in line
        assert "cmdline=" in line
