# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify remote-memory evidence, corruption handling and reports without CUDA."""

from __future__ import annotations

import argparse
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tools import check_mooncake_egm as probe


def replica(endpoint: str = "10.0.0.2:25000", *, status: str = "COMPLETE", memory: bool = True) -> SimpleNamespace:
    descriptor = SimpleNamespace(size=32, buffer_address=4096, transport_endpoint=endpoint)
    return SimpleNamespace(
        status=f"ReplicaStatus.{status}",
        is_memory_replica=lambda: memory,
        get_memory_descriptor=lambda: SimpleNamespace(buffer_descriptor=descriptor),
    )


def test_client_configuration_has_no_store_capacity_or_host_staging() -> None:
    config = probe.client_store_config("master:50051", "10.0.0.1")
    assert config["protocol"] == "nvlink"
    assert config["rdma_devices"] == ""
    assert config["global_segment_size"] == config["local_buffer_size"] == "0"
    assert config["enable_egm_store_pool"] == "false"


@pytest.mark.parametrize("value", ["0,0", "-1", "0,x", ""])
def test_invalid_csv(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        probe.parse_csv(value)


def test_remote_memory_descriptor_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "host_addresses", lambda host: {host})
    evidence = probe.validate_replicas([replica()], {"10.0.0.2"}, {"10.0.0.1"}, 32)
    assert evidence[0]["transport_endpoint"] == "10.0.0.2:25000"
    assert evidence[0]["size_bytes"] == 32


@pytest.mark.parametrize(
    "replicas, expected_size, message",
    [
        ([], 32, "No Store replica"),
        ([replica("10.0.0.1:25000")], 32, "not exclusively on the remote"),
        ([replica("10.0.0.3:25000")], 32, "not exclusively on the remote"),
        ([replica(status="PROCESSING")], 32, "COMPLETE memory replica"),
        ([replica(memory=False)], 32, "COMPLETE memory replica"),
        ([replica()], 64, "differs from payload size"),
        ([replica("10.0.0.2")], 32, "Unexpected replica transport endpoint"),
    ],
)
def test_invalid_replica_evidence_fails(
    monkeypatch: pytest.MonkeyPatch, replicas: list[Any], expected_size: int, message: str
) -> None:
    monkeypatch.setattr(probe, "host_addresses", lambda host: {host})
    with pytest.raises(RuntimeError, match=message):
        probe.validate_replicas(replicas, {"10.0.0.2"}, {"10.0.0.1"}, expected_size)


def test_summary_excludes_warmup_and_weights_byte_throughput_by_time() -> None:
    samples = [
        {"gpu_id": 0, "size_bytes": 1024**3, "warmup": warmup, "put_seconds": delay, "get_seconds": delay * 2}
        for warmup, delay in [(True, 100.0), (False, 1.0), (False, 3.0)]
    ]
    put, get = probe.summarize(samples)
    assert put["iterations"] == get["iterations"] == 2
    assert put["latency_ms"] == {"p50": 2000.0, "p95": 2900.0, "p99": 2980.0}
    assert put["payload_GiB_per_second"] == 0.5
    assert get["payload_GiB_per_second"] == 0.25


def test_report_requires_native_nvlink_log_and_honors_launcher_failure(tmp_path: Path) -> None:
    probe.write_json(tmp_path / "logs/client-results.json", {"status": "passed", "samples": []})
    probe.write_report(tmp_path, 0)
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "FAIL / INCOMPLETE" in report
    assert "Native NVLink memcpy log observed: `False`" in report
    (tmp_path / "logs/client.log").write_text("NvlinkTransport: using cudaMemcpyBatchAsync for GPU memory\n")
    probe.write_report(tmp_path, 0)
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "FAIL / INCOMPLETE" in report
    assert "Requested verified samples complete: `False`" in report
    probe.write_report(tmp_path, 7)
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "FAIL / INCOMPLETE" in report
    assert "launcher exit code: `7`" in report


def test_report_without_client_results_is_cpu_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_runtime_import(name: str) -> None:
        raise AssertionError(f"Report imported a runtime dependency: {name}")

    monkeypatch.setattr(probe.importlib, "import_module", no_runtime_import)
    assert probe.main(["report", "--run-dir", str(tmp_path), "--exit-code", "9"]) == 0
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "FAIL / INCOMPLETE" in report
    assert "TTFT p50/p95/p99" in report and "ITL p50/p99" in report
    assert "agentic Pareto/SLO metrics: **N/A**" in report
    assert "no measured successful sample" in report
    assert "slurm-job-sizing" in report


@pytest.mark.parametrize("arguments", [["--iterations", "0"], ["--warmup", "-1"], ["--sizes", "0"]])
def test_client_rejects_invalid_work_before_imports(tmp_path: Path, arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        probe.main(
            ["client", "--master", "master:1", "--run-dir", str(tmp_path), "--provider-hostname", "provider"]
            + arguments
        )
    assert exc.value.code == 2


class FakeTensor:
    pointers: weakref.WeakValueDictionary[int, FakeTensor] = weakref.WeakValueDictionary()

    def __init__(self, array: np.ndarray, *, device: str = "cpu", pinned: bool = True) -> None:
        self.array = array
        self.device = SimpleNamespace(type=device)
        self.pinned = pinned

    def __getitem__(self, item: slice) -> FakeTensor:
        return FakeTensor(self.array[item], device=self.device.type, pinned=self.pinned)

    def is_pinned(self) -> bool:
        return self.pinned

    def random_(self, low: int, high: int, *, generator: FakeTorch.Generator) -> None:
        assert self.device.type == generator.device == "cpu"
        self.array[:] = np.random.default_rng(generator.seed).integers(low, high, self.array.shape, dtype=np.uint8)

    def fill_(self, value: int) -> None:
        self.array.fill(value)

    def data_ptr(self) -> int:
        pointer = self.array.ctypes.data
        self.pointers[pointer] = self
        return pointer


class FakeTorch:
    __version__ = "test-torch"
    uint8 = np.uint8
    version = SimpleNamespace(cuda="13.0")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda gpu: f"test-gpu-{gpu}",
        set_device=lambda gpu: None,
        synchronize=lambda gpu: None,
    )

    def __init__(self, failure: str | None = None) -> None:
        self.roots: list[weakref.ReferenceType[FakeTensor]] = []
        self.failure = failure

    def empty(self, size: int, *, dtype: Any, device: str, pin_memory: bool) -> FakeTensor:
        assert device == "cpu" and pin_memory  # Reject accidental CUDA or pageable payload allocations.
        tensor = FakeTensor(
            np.empty(size, dtype=dtype),
            device="cuda" if self.failure == "cuda_tensor" else device,
            pinned=self.failure != "unpinned",
        )
        self.roots.append(weakref.ref(tensor))
        return tensor

    class Generator:
        def __init__(self, *, device: str) -> None:
            assert device == "cpu"
            self.device = device
            self.seed = 0

        def manual_seed(self, seed: int) -> FakeTorch.Generator:
            self.seed = seed
            return self

    @staticmethod
    def equal(left: FakeTensor, right: FakeTensor) -> bool:
        assert left.device.type == right.device.type == "cpu"
        return bool(np.array_equal(left.array, right.array))


def fake_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None = None
) -> tuple[probe.ClientConfig, dict[str, Any]]:
    torch = FakeTorch(failure)
    state: dict[str, Any] = {"closed": False, "removed": [], "registered": []}

    class CudaRuntime:
        cudaError_t = SimpleNamespace(cudaSuccess=0)
        cudaMemoryType = SimpleNamespace(
            cudaMemoryTypeUnregistered=0, cudaMemoryTypeHost=1, cudaMemoryTypeDevice=2, cudaMemoryTypeManaged=3
        )

        @staticmethod
        def cudaPointerGetAttributes(pointer: int) -> tuple[int, SimpleNamespace]:
            assert FakeTensor.pointers[pointer].device.type == "cpu"
            memory_type = {"unregistered": 0, "device_memory": 2, "managed_memory": 3}.get(failure or "", 1)
            return (1 if failure == "pointer_attributes" else 0), SimpleNamespace(type=memory_type)

    class Store:
        def __init__(self) -> None:
            self.objects: dict[str, np.ndarray] = {}
            self.read_count = 0

        def setup(self, config: dict[str, str]) -> int:
            state["config"] = config
            return 0

        def get_hostname(self) -> str:
            return "10.0.0.1:1234"

        def register_buffer(self, pointer: int, size: int) -> int:
            state["registered"].append((pointer, size))
            return 0

        def batch_put_from_multi_buffers(
            self, keys: list[str], pointers: list[list[int]], sizes: list[list[int]]
        ) -> list[int]:
            if failure == "put":
                return [-9]
            self.objects[keys[0]] = FakeTensor.pointers[pointers[0][0]].array[: sizes[0][0]].copy()
            return [0]

        def get_replica_desc(self, key: str) -> list[SimpleNamespace]:
            item = replica()
            item.get_memory_descriptor().buffer_descriptor.size = len(self.objects[key])
            return [item]

        def batch_get_into_multi_buffers(
            self, keys: list[str], pointers: list[list[int]], sizes: list[list[int]]
        ) -> list[int]:
            tensor = FakeTensor.pointers[pointers[0][0]]
            tensor.array[:] = self.objects[keys[0]]
            self.read_count += 1
            if failure == "corruption" or (failure == "late_corruption" and self.read_count == 3):
                tensor.array[0] ^= 1
            return [sizes[0][0] - 1 if failure == "get" else sizes[0][0]]

        def remove(self, key: str, *, force: bool) -> int:
            assert force
            state["removed"].append(key)
            self.objects.pop(key, None)
            return 0

        def close(self) -> int:
            state["closed"] = True
            state["buffers_live_at_close"] = all(reference() is not None for reference in torch.roots)
            state["remaining_keys"] = list(self.objects)
            return -8 if failure == "close" else 0

    original_import = probe.importlib.import_module

    def runtime_import(name: str) -> Any:
        if name == "configs.mooncake_egm_donor":
            return SimpleNamespace(configure_environment=lambda: None, load_store_type=lambda: Store)
        if name == "torch":
            return torch
        if name == "cuda.bindings.runtime":
            return CudaRuntime
        return original_import(name)

    monkeypatch.setattr(probe.importlib, "import_module", runtime_import)
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: "test-wheel")
    monkeypatch.setattr(probe.socket, "gethostname", lambda: "client")
    monkeypatch.setattr(probe.socket, "gethostbyname", lambda host: "10.0.0.1")
    monkeypatch.setattr(probe, "host_addresses", lambda host: {"10.0.0.1" if host == "client" else "10.0.0.2"})
    probe.write_json(
        tmp_path / "logs/provider.ready.json",
        {
            "protocol": "nvlink",
            "enable_egm_store_pool": "true",
            "global_segment_size": "4294967296",
            "local_buffer_size": "0",
            "local_hostname": "10.0.0.2",
        },
    )
    (tmp_path / "logs/client.log").write_text("NvlinkTransport: using cudaMemcpyBatchAsync\n")
    return probe.ClientConfig("master:50051", tmp_path, "provider", (0, 1), (16, 32), 2, 1), state


def test_full_host_workflow_preserves_buffers_and_cleans_owned_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, state = fake_runtime(tmp_path, monkeypatch)
    assert probe.run_client(config) == 0
    result = json.loads((tmp_path / "logs/client-results.json").read_text())
    assert result["status"] == "passed"
    assert len(result["samples"]) == 12
    assert len({sample["key"] for sample in result["samples"]}) == 12
    assert all(sample["verified_bytes"] == sample["size_bytes"] for sample in result["samples"])
    assert result["buffer_memory"] == "pinned_host"
    assert len(result["buffers"]) == len(state["registered"]) == 4  # two host buffers per submission device
    assert {(item["gpu_id"], item["role"]) for item in result["buffers"]} == {
        (gpu, role) for gpu in config.gpu_ids for role in ("source", "destination")
    }
    assert all(
        item["device"] == "cpu" and item["pinned"] and item["cuda_memory_type"] == "cudaMemoryTypeHost"
        for item in result["buffers"]
    )
    assert state["closed"] and state["buffers_live_at_close"]
    assert len(state["removed"]) == 12 and not state["remaining_keys"]
    assert "Status: **PASS**" in (tmp_path / "logs/perf-analysis.md").read_text()


@pytest.mark.parametrize(
    "failure, message",
    [
        ("cuda_tensor", "must be pinned CPU memory"),
        ("unpinned", "must be pinned CPU memory"),
        ("device_memory", "must have cudaMemoryTypeHost"),
        ("managed_memory", "must have cudaMemoryTypeHost"),
        ("unregistered", "must have cudaMemoryTypeHost"),
        ("pointer_attributes", "cudaPointerGetAttributes failed"),
    ],
)
def test_nonhost_or_unverified_payload_buffers_fail_before_transfers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str
) -> None:
    config, state = fake_runtime(tmp_path, monkeypatch, failure)
    assert probe.run_client(config) == 1
    result = json.loads((tmp_path / "logs/client-results.json").read_text())
    assert result["status"] == "failed" and message in result["error"]
    assert not result["samples"] and not state["registered"]
    assert state["closed"] and state["buffers_live_at_close"]
    assert "FAIL / INCOMPLETE" in (tmp_path / "logs/perf-analysis.md").read_text()


@pytest.mark.parametrize(
    "missing, field, value",
    [
        (True, None, None),
        (False, "device", "cuda"),
        (False, "pinned", False),
        (False, "cuda_memory_type", "cudaMemoryTypeManaged"),
        (False, "size_bytes", 1),
        (False, "pointer", 0),
        (False, "role", "source"),
    ],
)
def test_report_rejects_missing_or_wrong_host_buffer_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool, field: str | None, value: Any
) -> None:
    config, _ = fake_runtime(tmp_path, monkeypatch)
    assert probe.run_client(config) == 0
    path = tmp_path / "logs/client-results.json"
    result = json.loads(path.read_text())
    if missing:
        result.pop("buffers")
    else:
        result["buffers"][1][field] = value
    probe.write_json(path, result)
    probe.write_report(tmp_path, 0)
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "FAIL / INCOMPLETE" in report
    assert "Pinned host source/destination buffers verified: `False`" in report


@pytest.mark.parametrize(
    "failure, message", [("put", "put failed"), ("get", "get failed"), ("corruption", "verification failed")]
)
def test_native_failure_or_corruption_fails_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str
) -> None:
    config, state = fake_runtime(tmp_path, monkeypatch, failure)
    assert probe.run_client(config) == 1
    result = json.loads((tmp_path / "logs/client-results.json").read_text())
    assert result["status"] == "failed" and message in result["error"]
    assert not result["samples"]
    assert state["closed"] and state["buffers_live_at_close"]
    assert state["removed"] and not state["remaining_keys"]
    assert "FAIL / INCOMPLETE" in (tmp_path / "logs/perf-analysis.md").read_text()


def test_partial_failure_preserves_only_completed_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, state = fake_runtime(tmp_path, monkeypatch, "late_corruption")
    assert probe.run_client(config) == 1
    result = json.loads((tmp_path / "logs/client-results.json").read_text())
    assert result["status"] == "failed"
    assert len(result["samples"]) == 2  # one warmup plus one measured pair
    assert all(row["iterations"] == 1 for row in result["summary"])
    assert state["closed"] and not state["remaining_keys"]
    report = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "FAIL / INCOMPLETE" in report and "1 measured pairs" in report
    assert "Payload GiB/s" in report


def test_store_close_error_invalidates_successful_payload_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, state = fake_runtime(tmp_path, monkeypatch, "close")
    assert probe.run_client(config) == 1
    result = json.loads((tmp_path / "logs/client-results.json").read_text())
    assert result["status"] == "failed"
    assert "Store close failed: -8" in result["cleanup_error"]
    assert state["buffers_live_at_close"] and not state["remaining_keys"]
    assert "FAIL / INCOMPLETE" in (tmp_path / "logs/perf-analysis.md").read_text()
