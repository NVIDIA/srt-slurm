# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the standalone diagnostic's data checks without Slurm or Mooncake."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import threading
from pathlib import Path

import pytest

from tools import diagnose_mooncake as diagnostic


class FakeStore:
    def __init__(self, failure: str = "") -> None:
        self.failure = failure
        self.values: dict[str, bytes] = {}
        self.registered: set[int] = set()
        self.lock = threading.Lock()
        self.closed = False

    def get_hostname(self) -> str:
        return "127.0.0.1:12345"

    def register_buffer(self, pointer: int, size: int) -> int:
        with self.lock:
            self.registered.add(pointer)
        return 0

    def unregister_buffer(self, pointer: int) -> int:
        with self.lock:
            self.registered.remove(pointer)
        return 0

    def batch_put_from_multi_buffers(
        self, keys: list[str], pointers: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        with self.lock:
            for key, pointer, size in zip(keys, pointers, sizes, strict=True):
                self.values[key] = ctypes.string_at(pointer[0], size[0])
        return [0] * len(keys)

    def batch_get_into_multi_buffers(
        self, keys: list[str], pointers: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        if self.failure == "transport":
            return [-800] * len(keys)
        with self.lock:
            for key, pointer, size in zip(keys, pointers, sizes, strict=True):
                ctypes.memmove(pointer[0], self.values[key], size[0])
                if self.failure == "corruption":
                    ctypes.c_ubyte.from_address(pointer[0]).value ^= 1
        return [size[0] for size in sizes]

    def close(self) -> int:
        assert not self.registered, "The store must outlive all registered buffers"
        self.closed = True
        return 0


def client_args(path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output=path,
        donor_ready=None,
        master="127.0.0.1:8700",
        case="synthetic",
        seconds=0.01,
        batch_size=2,
        workers=2,
    )


@pytest.mark.parametrize("failure", ["", "transport", "corruption"])
def test_client_verifies_restored_bytes_and_preserves_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    store = FakeStore(failure)
    monkeypatch.setattr(diagnostic, "PAGE_BYTES", 4096)
    monkeypatch.setattr(diagnostic, "setup_store", lambda *args: (store, {"fake": "0.3.13"}, 0.001))
    monkeypatch.setattr(diagnostic, "replica_samples", lambda *args: [])
    monkeypatch.setattr(diagnostic, "health", lambda: {})
    monkeypatch.setattr(diagnostic.signal, "signal", lambda *args: None)
    output = tmp_path / "client.jsonl"

    if failure:
        with pytest.raises(RuntimeError, match="diagnostic workers failed"):
            diagnostic.client(client_args(output))
    else:
        diagnostic.client(client_args(output))

    batches = [row for row in map(json.loads, output.read_text().splitlines()) if row["event"] == "get_batch"]
    assert batches
    assert store.closed
    if failure == "transport":
        assert all(row["errors"] == {"-800": 2} and row["verified_bytes"] == 0 for row in batches)
    elif failure == "corruption":
        assert all(row["verification_failures"] == 2 and row["verified_bytes"] == 0 for row in batches)
    else:
        assert all(row["ok"] and row["verified_bytes"] == 8192 for row in batches)


def test_report_preserves_missing_data_and_partial_records(tmp_path: Path) -> None:
    case = tmp_path / "logs" / "cn04-1100GiB"
    case.mkdir(parents=True)
    (case / "incomplete.txt").write_text("donor did not become ready\n")
    (case / "donor-health.jsonl").write_text('{"event":"health"}\n{"event":')

    diagnostic.report(argparse.Namespace(run_dir=tmp_path))

    report = (tmp_path / "logs" / "perf-analysis.md").read_text()
    assert "N/A (host-only byte transfer)" in report
    assert "donor did not become ready" in report
    assert "malformed" in report
    assert "does not establish" in report


def test_buffer_allocation_failure_releases_other_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore()
    real_mmap = diagnostic.mmap.mmap
    lock = threading.Lock()
    calls = 0

    def allocate(*args: object):
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        if first:
            raise OSError("synthetic memory allocation failure")
        return real_mmap(*args)

    monkeypatch.setattr(diagnostic, "PAGE_BYTES", 4096)
    monkeypatch.setattr(diagnostic, "setup_store", lambda *args: (store, {"fake": "0.3.13"}, 0.001))
    monkeypatch.setattr(diagnostic, "replica_samples", lambda *args: [])
    monkeypatch.setattr(diagnostic, "health", lambda: {})
    monkeypatch.setattr(diagnostic.signal, "signal", lambda *args: None)
    monkeypatch.setattr(diagnostic.mmap, "mmap", allocate)

    with pytest.raises(RuntimeError, match="diagnostic workers failed"):
        diagnostic.client(client_args(tmp_path / "client.jsonl"))

    assert calls == 2
    assert store.closed


def test_client_discovers_transport_probe_endpoint_from_replicas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FakeStore()
    watched: list[set[str]] = []

    def monitor(log, stop, endpoints, lock):
        watched.append(endpoints)
        stop.wait()

    monkeypatch.setattr(diagnostic, "PAGE_BYTES", 4096)
    monkeypatch.setattr(diagnostic, "setup_store", lambda *args: (store, {"fake": "0.3.13"}, 0.001))
    monkeypatch.setattr(diagnostic, "replica_samples", lambda *args: [{"endpoint": "127.0.0.1:15302"}])
    monkeypatch.setattr(diagnostic, "monitor", monitor)
    monkeypatch.setattr(diagnostic.signal, "signal", lambda *args: None)

    diagnostic.client(client_args(tmp_path / "client.jsonl"))

    assert watched == [{"127.0.0.1:15302"}]
    assert store.get_hostname() not in watched[0]


@pytest.mark.parametrize("failure,expected", [("permission", "error"), ("missing", "missing"), ("timeout", "timeout")])
def test_inventory_command_preserves_collection_failures(
    monkeypatch: pytest.MonkeyPatch, failure: str, expected: str
) -> None:
    def run(argv, *, stdout, stderr, timeout, check):
        stderr.write(b"Operation not permitted\n")
        if failure == "missing":
            raise FileNotFoundError("probe not installed")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    result = diagnostic.inventory_command(["read-only-probe"], 0.1)

    assert result["status"] == expected
    assert result["returncode"] != 0
    assert "Operation not permitted" in result["stderr"]


def test_inventory_command_bounds_retained_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv, *, stdout, stderr, timeout, check):
        stdout.write(b"x" * 70_000)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    result = diagnostic.inventory_command(["read-only-probe"], 0.1)

    assert result["status"] == "ok"
    assert result["stdout_truncated"] is True
    assert len(result["stdout"]) == 65_536


def test_inventory_report_uses_host_baseline_and_keeps_container_separate(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    after = logs / "cn05-512GiB"
    after.mkdir(parents=True)
    counter = "mlx5_0/ports/1/counters/port_rcv_errors"

    def snapshot(scope: str, stamp: str, count: int) -> dict:
        return {
            "scope": scope,
            "timestamp": stamp,
            "host": "container-hostname" if scope == "container" else "actual-node",
            "slurm_node": "vr-r42-cn05",
            "health": {"ib_counters": {counter: count}, "cgroup": {"membership": "0::/kubepods.slice/test"}},
            "interfaces": {"eth0": {"statistics": {"rx_dropped": str(count)}}},
            "platform": {"os_release": f'PRETTY_NAME="{scope}-OS"', "kernel": "test-kernel"},
            "commands": {"dmesg": {"status": "error"}},
            "kernel": {"status": "error", "stderr": "Operation not permitted", "messages": []},
        }

    (logs / "host-inventory-cn05.json").write_text(json.dumps(snapshot("host", "2026-10-19T06:00:00Z", 10)))
    (after / "host-inventory-cn05.json").write_text(json.dumps(snapshot("host", "2026-10-19T06:01:00Z", 13)))
    (after / "inventory-cn05.json").write_text(json.dumps(snapshot("container", "2026-10-19T06:01:00Z", 999)))

    diagnostic.report(argparse.Namespace(run_dir=tmp_path))
    report = (logs / "perf-analysis.md").read_text()

    assert "host-OS" in report and "container-OS" not in report
    assert f"'{counter}': 3" in report
    assert "'net/eth0/rx_dropped': 3" in report
    assert "Kubernetes Slurm worker context" in report
    assert "999" not in report
    assert "host-inventory-cn05.json" in report and "inventory-cn05.json" in report
    assert "vr-r42-cn05" in report
    assert "Unavailable probes do not establish a clean node" in report


@pytest.mark.parametrize(
    "gone_at,query_error,expected_signals,expected_rc",
    [
        (0, False, [], 0),
        (20, False, ["TERM"], 0),
        (30, False, ["TERM"], 0),
        (40, False, ["TERM", "KILL"], 0),
        (100, False, ["TERM", "KILL"], 1),
        (0, True, ["TERM", "KILL"], 1),
    ],
)
def test_step_teardown_allows_scheduler_drain_and_requires_confirmation(
    tmp_path: Path, gone_at: int, query_error: bool, expected_signals: list[str], expected_rc: int
) -> None:
    """Exercise real shell cleanup with a simulated clock and no Slurm commands."""
    source = (Path(__file__).parents[1] / "tools" / "diagnose_mooncake.sbatch").read_text()
    cleanup = source[source.index("stop_step() {") : source.index("stop_sampler() {")]
    harness = r"""
set -euo pipefail
case_dir=$1
SLURM_JOB_ID=synthetic
gone_at=$2
query_error=$3
SECONDS=0
event() { printf '%s\n' "$*" >> "$case_dir/events"; }
step_gone() {
    (( query_error == 0 )) || return 2
    (( SECONDS >= gone_at ))
}
timeout() { shift; "$@"; }
sleep() { SECONDS=$((SECONDS + $1)); }
scancel() {
    case $* in
        *--signal=TERM*) printf 'TERM\n' >> "$case_dir/signals" ;;
        *--signal=KILL*) printf 'KILL\n' >> "$case_dir/signals" ;;
    esac
}
pid_running() { return 1; }
true &
child=$!
wait "$child"
printf '9\n' > "$case_dir/step"
"""
    harness += cleanup
    harness += r"""
rc=0
stop_step "$child" "$case_dir/step" "$case_dir/exit" || rc=$?
exit "$rc"
"""
    result = subprocess.run(
        ["bash", "-c", harness, "_", str(tmp_path), str(gone_at), str(int(query_error))],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == expected_rc, result.stderr
    signals = tmp_path / "signals"
    assert (signals.read_text().splitlines() if signals.exists() else []) == expected_signals
    assert (tmp_path / "teardown-unconfirmed.txt").exists() == (expected_rc != 0)
    if gone_at == 0 and not query_error:
        assert "step_already_gone" in (tmp_path / "events").read_text()
