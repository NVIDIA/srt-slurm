# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the host-only NUMA workload without Mooncake, GPUs or Slurm."""

from __future__ import annotations

import ctypes
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import mooncake_numa_workload as workload


class FakeReplica:
    def __init__(self, endpoint: str, address: int, size: int, *, memory: bool = True) -> None:
        self.status = "COMPLETE"
        self.memory = memory
        self.descriptor = SimpleNamespace(transport_endpoint=endpoint, buffer_address=address, size=size)

    def is_memory_replica(self) -> bool:
        return self.memory

    def get_memory_descriptor(self) -> SimpleNamespace:
        return SimpleNamespace(buffer_descriptor=self.descriptor)


class FakeStore:
    """Copy actual pointed-to bytes and independently control API return values."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.registered: set[int] = set()
        self.endpoint = "192.0.2.5:12345"
        self.put_result: list[int] | None = None
        self.get_result: list[int] | None = None
        self.corrupt_interior = False
        self.replicas: dict[str, list[FakeReplica]] = {}
        self.calls: list[tuple[str, list[str]]] = []
        self.closed = False
        self.setup_config: dict[str, object] = {}
        self.mounted: dict[str, object] = {}

    def setup(self, **kwargs: object) -> int:
        self.setup_config = kwargs
        return 0

    def mount_segment(self, path: str, size: int, **kwargs: object) -> dict[str, object]:
        self.mounted = {"path": path, "size": size, **kwargs}
        return {"ret": 0, "segment_ids": ["donor-segment"]}

    def unmount_segment(self, segments: list[str]) -> int:
        return 0

    def get_hostname(self) -> str:
        return self.endpoint

    def register_buffer(self, pointer: int, size: int) -> int:
        self.registered.add(pointer)
        return 0

    def unregister_buffer(self, pointer: int) -> int:
        self.registered.remove(pointer)
        return 0

    def batch_put_from_multi_buffers(
        self, keys: list[str], pointers: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        self.calls.append(("put", list(keys)))
        for key, pointer, size in zip(keys, pointers, sizes, strict=True):
            self.values[key] = ctypes.string_at(pointer[0], size[0])
            self.replicas.setdefault(key, [FakeReplica(self.endpoint, pointer[0], size[0])])
        return self.put_result if self.put_result is not None else [0] * len(keys)

    def batch_get_into_multi_buffers(
        self, keys: list[str], pointers: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        self.calls.append(("get", list(keys)))
        for key, pointer, size in zip(keys, pointers, sizes, strict=True):
            ctypes.memmove(pointer[0], self.values[key], size[0])
            if self.corrupt_interior:
                ctypes.c_ubyte.from_address(pointer[0] + size[0] // 2).value ^= 1
        return self.get_result if self.get_result is not None else [size[0] for size in sizes]

    def batch_get_replica_desc(self, keys: list[str]) -> dict[str, list[FakeReplica]]:
        self.calls.append(("descriptors", list(keys)))
        return {key: self.replicas[key] for key in keys if key in self.replicas}

    def close(self) -> int:
        self.closed = True
        return 0


@pytest.fixture
def config(tmp_path: Path) -> dict[str, object]:
    ready = tmp_path / "donor-ready.json"
    ready.write_text(json.dumps({"status": "PASS", "endpoint": "192.0.2.5:12345"}))
    native_log = tmp_path / "donor.native.log"
    native_log.write_text("Transfer Engine RPC using P2P handshake, listening on 192.0.2.5:12345\n")
    return {
        "case_id": "unit-case",
        "key_prefix": "unit-case",
        "output": str(tmp_path / "result.json"),
        "donor_ready": str(ready),
        "native_log": str(native_log),
        "donor_verify_request": str(tmp_path / "verify-request.json"),
        "donor_verify_result": str(tmp_path / "verify-result.json"),
        "stop_file": str(tmp_path / "stop"),
        "page_bytes": 256,
        "batch_size": 4,
        "working_set_batches": 2,
        "seconds": 0.035,
        "warmup_seconds": 0,
        "pool_bytes": 4096,
        "numa": 1,
        "nics": ["mlx5_6", "mlx5_7"],
        "master": "192.0.2.1:8700",
        "affinity": True,
        "trace": False,
    }


@pytest.fixture
def buffer(config: dict[str, object]):
    allocation = workload.MemoryBuffer(int(config["page_bytes"]) * int(config["batch_size"]))
    try:
        yield allocation
    finally:
        allocation.close()


@pytest.fixture
def local_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workload, "_STOP_REQUESTED", False)
    monkeypatch.setattr(workload, "buffer_residency", lambda buffer, numa: {"all_expected": True})
    monkeypatch.setattr(workload, "health", lambda: {})
    monkeypatch.setattr(workload, "numa_maps", lambda config, label: f"numa-maps-{label}")


def populate(store: FakeStore, config: dict[str, object]) -> None:
    for index in range(int(config["batch_size"]) * int(config["working_set_batches"])):
        store.values[f"{config['key_prefix']}:{index}"] = bytes([index % 251 + 1]) * int(config["page_bytes"])


@pytest.mark.parametrize("affinity", [False, True])
def test_environment_uses_rdma_filters_and_unsets_disabled_affinity(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object], affinity: bool
) -> None:
    env = {
        name: "old"
        for name in (
            "MC_ENABLE_DEST_DEVICE_AFFINITY",
            "MC_ENABLE_HCA_PEER_AFFINITY",
            "MC_NIC_PEER_AFFINITY",
            "MC_STORE_USE_HUGEPAGE",
            "MC_STORE_MEMCPY",
        )
    }
    monkeypatch.setattr(workload.os, "environ", env)

    workload.configure_environment({**config, "affinity": affinity})

    assert env["MC_TE_FILTERS"] == env["MOONCAKE_RDMA_DEVICES"] == "mlx5_6,mlx5_7"
    assert env["MC_STORE_MEMCPY"] == "0"
    assert env["MC_MS_AUTO_DISC"] == "0"
    assert (env.get("MC_ENABLE_DEST_DEVICE_AFFINITY") == "1") == affinity
    if not affinity:
        assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in env
    assert not {"MC_ENABLE_HCA_PEER_AFFINITY", "MC_NIC_PEER_AFFINITY", "MC_STORE_USE_HUGEPAGE"} & env.keys()


@pytest.mark.parametrize(
    "statuses,returncode,expected", [([1, 1], 0, True), ([1, 0], 0, False), ([1, -13], 0, False), ([1, 1], -1, False)]
)
def test_residency_queries_actual_page_status_without_migration(
    monkeypatch: pytest.MonkeyPatch, statuses: list[int], returncode: int, expected: bool
) -> None:
    calls: list[list[int]] = []

    class MovePages:
        def __call__(self, pid, count, pages, nodes, result, flags):
            assert pid == 0 and nodes is None and flags == 0
            calls.append([pages[index] for index in range(count)])
            for index, status in enumerate(statuses):
                result[index] = status
            ctypes.set_errno(13 if returncode else 0)
            return returncode

    monkeypatch.setattr(workload.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(numa_move_pages=MovePages()))
    monkeypatch.setattr(workload.ctypes.util, "find_library", lambda name: "synthetic-libnuma")
    page = workload.os.sysconf("SC_PAGE_SIZE")
    addresses = [page + 12, 2 * page + 5, 2 * page + 99]
    if expected:
        result = workload.query_residency(addresses, 1)
        assert result["observed_counts"] == {"1": 2} and result["all_expected"]
    else:
        with pytest.raises(RuntimeError, match="Physical NUMA residency verification failed"):
            workload.query_residency(addresses, 1)
    assert calls == [[page, 2 * page]]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "owner", "incomplete", "nonmemory", "size"])
def test_descriptor_contract_requires_one_complete_memory_replica_on_the_donor(fault: str) -> None:
    store = FakeStore()
    replica = FakeReplica(store.endpoint, 4096, 256)
    store.replicas["key"] = [replica]
    if fault == "missing":
        store.replicas.clear()
    elif fault == "duplicate":
        store.replicas["key"].append(replica)
    elif fault == "owner":
        replica.descriptor.transport_endpoint = "192.0.2.9:12345"
    elif fault == "incomplete":
        replica.status = "PROCESSING"
    elif fault == "nonmemory":
        replica.memory = False
    else:
        replica.descriptor.size = 0

    with pytest.raises(RuntimeError):
        workload.descriptor_samples(store, ["key"], store.endpoint)


def test_descriptor_residency_checks_both_object_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore()
    store.replicas["key"] = [FakeReplica(store.endpoint, 4096, 8192)]
    calls = []
    monkeypatch.setattr(
        workload, "query_residency", lambda addresses, numa: calls.append((addresses, numa)) or {"all_expected": True}
    )

    result = workload.descriptor_samples(store, ["key"], store.endpoint, query_numa=1)

    assert calls == [([4096, 12287], 1)]
    assert result[0]["residency"]["all_expected"]


@pytest.mark.parametrize("put_result", [[0], [0, 0, 0, 0, 0], [0, 0, -800, 0], [0, 0, 1, 0]])
def test_seed_rejects_missing_extra_and_nonzero_put_statuses(
    config: dict[str, object], buffer, local_runtime, put_result: list[int]
) -> None:
    store = FakeStore()
    store.put_result = put_result
    with pytest.raises(RuntimeError, match="Cache PUT failed"):
        workload.run_seed(config, store, buffer)


def test_seed_verifies_every_key_owner_including_an_unsampled_key(
    config: dict[str, object], buffer, local_runtime
) -> None:
    store = FakeStore()
    store.replicas["unit-case:1"] = [FakeReplica("192.0.2.99:12345", buffer.address, 256)]
    with pytest.raises(RuntimeError, match="unexpected owner"):
        workload.run_seed(config, store, buffer)


def test_successful_seed_stores_full_payloads_and_checks_all_owners(
    config: dict[str, object], buffer, local_runtime
) -> None:
    store = FakeStore()

    result = workload.run_seed(config, store, buffer)

    assert result["status"] == "PASS" and result["verified_owner_keys"] == 8
    assert len(result["descriptor_samples"]) == 3
    assert store.calls[-1] == ("descriptors", [f"unit-case:{index}" for index in range(8)])
    assert store.values == {f"unit-case:{index}": bytes([index + 1]) * 256 for index in range(8)}


@pytest.mark.parametrize(
    "get_result",
    [[256], [256] * 5, [256, 256, -800, 256], [256, 256, 0, 256], [256, 256, 255, 256], [256, 256, 257, 256]],
)
def test_read_requires_an_exact_page_byte_count_for_every_get(
    config: dict[str, object], buffer, local_runtime, get_result: list[int]
) -> None:
    store = FakeStore()
    populate(store, config)
    store.get_result = get_result
    with pytest.raises(RuntimeError, match="Cache GET failed"):
        workload.run_read(config, store, buffer)


def test_payload_validation_detects_interior_corruption(config: dict[str, object], buffer, local_runtime) -> None:
    store = FakeStore()
    populate(store, config)
    store.corrupt_interior = True
    with pytest.raises(RuntimeError, match="Corrupt payload"):
        workload.run_read({**config, "trace": True}, store, buffer)


def test_trace_reads_only_two_keys_without_warmup_or_measurement(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object], buffer, local_runtime
) -> None:
    store = FakeStore()
    populate(store, config)
    monkeypatch.setattr(workload.time, "perf_counter", lambda: pytest.fail("Trace must not enter timing or warmup"))

    result = workload.run_read({**config, "trace": True}, store, buffer)

    assert store.calls == [("get", ["unit-case:0", "unit-case:1"])]
    assert result["count"] == 2 and result["bytes"] == 512 and result["errors"] == 0
    assert "goodput_gb_s" not in result and "latency_ms" not in result


def test_timed_goodput_excludes_setup_and_full_payload_verification(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object], buffer, local_runtime
) -> None:
    store = FakeStore()
    populate(store, config)
    clock = SimpleNamespace(now=0.0)
    real_get, real_validate = store.batch_get_into_multi_buffers, workload.validate_payload

    def get(*args):
        clock.now += 0.01
        return real_get(*args)

    def validate(*args):
        real_validate(*args)
        clock.now += 5

    def health():
        clock.now += 7
        return {}

    store.batch_get_into_multi_buffers = get
    monkeypatch.setattr(workload.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(workload, "validate_payload", validate)
    monkeypatch.setattr(workload, "health", health)

    result = workload.run_read(config, store, buffer)

    assert result["status"] == "PASS" and result["errors"] == 0
    assert result["count"] == 4 and result["bytes"] == 4096
    assert result["elapsed_s"] == pytest.approx(0.04)
    assert result["samples"] == pytest.approx([10.0] * 4)
    assert all(value == pytest.approx(10.0) for value in result["latency_ms"].values())
    assert math.isfinite(result["goodput_gb_s"]) and result["goodput_gb_s"] > 0
    assert result["goodput_gb_s"] == pytest.approx(4096 / 0.04 / 1e9)
    assert clock.now > 20  # Expensive setup and verification are outside the measured interval.


def test_store_setup_has_no_automatic_pool_and_uses_only_rdma(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object]
) -> None:
    store = FakeStore()
    native = ModuleType("mooncake.store")
    native.MooncakeDistributedStore = lambda: store
    monkeypatch.setitem(sys.modules, "mooncake", ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", native)
    monkeypatch.setattr(workload, "versions", lambda: {"mooncake-transfer-engine": "0.3.13"})
    monkeypatch.setattr(workload, "host_ip", lambda master: "192.0.2.3")

    result, _ = workload.open_store(config)

    assert result is store
    assert store.setup_config["global_segment_size"] == 0
    assert store.setup_config["protocol"] == "rdma"
    assert store.setup_config["rdma_devices"] == "mlx5_6,mlx5_7"


def test_donor_mounts_the_existing_shared_mapping_on_the_requested_numa(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object], local_runtime
) -> None:
    store = FakeStore()
    buffer = SimpleNamespace(path="/proc/self/fd/99", size=4096, segment_ids=[])
    monkeypatch.setattr(workload, "_STOP_REQUESTED", True)

    result = workload.run_donor(config, store, buffer)

    assert store.mounted == {"path": "/proc/self/fd/99", "size": 4096, "protocol": "rdma", "location": "cpu:1"}
    assert result["status"] == "PASS" and buffer.segment_ids == ["donor-segment"]


@pytest.mark.parametrize("role", ["read", "donor"])
def test_main_releases_native_store_before_closing_its_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, object], role: str
) -> None:
    events: list[str] = []
    store = FakeStore()

    class Buffer:
        def __init__(self, size: int, *, shared: bool = False) -> None:
            assert shared == (role == "donor")
            self.address, self.size, self.registered = 4096, size, False
            self.segment_ids = []

        def close(self):
            assert store.closed
            events.append("mapping_close")

    def execute(config, actual_store, buffer):
        assert actual_store is store
        if role == "donor":
            buffer.segment_ids = ["mounted-segment"]
        events.append("run")
        return {"status": "PASS"}

    def close():
        events.append("store_close")
        store.closed = True
        return 0

    monkeypatch.setattr(workload.os, "environ", dict(workload.os.environ))
    monkeypatch.setattr(workload, "open_store", lambda config: (store, {"fake": "0.3.13"}))
    monkeypatch.setattr(workload, "MemoryBuffer", Buffer)
    monkeypatch.setattr(workload, f"run_{role}", execute)
    monkeypatch.setattr(workload.signal, "signal", lambda *args: None)
    monkeypatch.setattr(store, "unregister_buffer", lambda pointer: events.append("unregister") or 0)
    monkeypatch.setattr(store, "unmount_segment", lambda segments: events.append("unmount") or 0)
    monkeypatch.setattr(store, "close", close)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    assert workload.main(["--config", str(path), "--role", role]) == 0

    assert events == ["run", "unmount" if role == "donor" else "unregister", "store_close", "mapping_close"]
    assert json.loads(Path(config["output"]).read_text())["status"] == "PASS"


@pytest.mark.parametrize("close_raises", [False, True])
def test_failed_native_close_preserves_the_mapping_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, object], close_raises: bool
) -> None:
    store = FakeStore()
    buffers = []

    class Buffer:
        def __init__(self, size: int) -> None:
            self.address, self.size, self.registered = 4096, size, False
            self.segment_ids = []
            buffers.append(self)

        def close(self):
            pytest.fail("A failed native close must not unmap memory that native threads may still access")

    def close():
        if close_raises:
            raise RuntimeError("synthetic native cleanup failure")
        return -1

    monkeypatch.setattr(workload.os, "environ", dict(workload.os.environ))
    monkeypatch.setattr(workload, "_UNRELEASED_RESOURCES", [])
    monkeypatch.setattr(workload, "open_store", lambda config: (store, {"fake": "0.3.13"}))
    monkeypatch.setattr(workload, "MemoryBuffer", Buffer)
    monkeypatch.setattr(workload, "run_read", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(workload.signal, "signal", lambda *args: None)
    monkeypatch.setattr(store, "close", close)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    assert workload.main(["--config", str(path), "--role", "read"]) == 1

    resources = workload._UNRELEASED_RESOURCES
    assert resources == [(store, buffers[0])]
    result = json.loads(Path(config["output"]).read_text())
    assert result["status"] == "FAIL" and result["cleanup_errors"]
    assert "store.close" in result["cleanup_errors"][0]
