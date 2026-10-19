# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the standalone diagnostic's data checks without Slurm or Mooncake."""

from __future__ import annotations

import argparse
import ctypes
import json
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
