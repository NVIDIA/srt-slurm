# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile the real C++ reproducer against a deterministic CPU-only Driver API."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tools import cuda_host_numa_probe

SOURCE = Path(__file__).resolve().parents[1] / "tools/cuda_host_numa_repro.cpp"
GIB = 1024**3

CUDA_STUB_HEADER = r"""
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#define CUDA_VERSION 13000
using CUresult = int;
using CUdevice = int;
using CUmemGenericAllocationHandle = unsigned long long;
constexpr int CUDA_SUCCESS = 0;
enum CUdevice_attribute {
  CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
  CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
  CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
};
constexpr int CU_MEM_ALLOCATION_TYPE_PINNED = 1;
constexpr int CU_MEM_LOCATION_TYPE_HOST_NUMA = 3;
constexpr int CU_MEM_HANDLE_TYPE_FABRIC = 8;
constexpr int CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0;
struct CUmemLocation { int type; int id; };
struct CUmemAllocationProp {
  int type; int requestedHandleTypes; CUmemLocation location;
  unsigned char reserved[32];
};
struct CUuuid { char bytes[16]; };
inline unsigned long long next_handle = 0;
inline void event(const char* name, unsigned long long handle = 0, unsigned long long bytes = 0) {
  const char* path = std::getenv("CUDA_STUB_TRACE");
  if (!path) return;
  auto file = std::fopen(path, "a");
  std::fprintf(file, "%s %llu %llu\n", name, handle, bytes);
  std::fclose(file);
}
inline int env_int(const char* key) { const char* value = std::getenv(key); return value ? std::atoi(value) : 0; }
inline bool valid_properties(const CUmemAllocationProp* p) {
  if (p->type != CU_MEM_ALLOCATION_TYPE_PINNED || p->location.type != CU_MEM_LOCATION_TYPE_HOST_NUMA ||
      p->location.id != env_int("CUDA_STUB_NUMA") || p->requestedHandleTypes != CU_MEM_HANDLE_TYPE_FABRIC)
    return false;
  for (auto byte : p->reserved) if (byte) return false;
  return true;
}
inline CUresult cuInit(unsigned flags) { event("init", flags); return env_int("CUDA_STUB_INIT_ERROR"); }
inline CUresult cuDriverGetVersion(int* out) { *out = 13000; return 0; }
inline CUresult cuDeviceGetCount(int* out) { *out = 1; return 0; }
inline CUresult cuDeviceGet(CUdevice* out, int ordinal) { *out = ordinal; return ordinal == 0 ? 0 : 101; }
inline CUresult cuDeviceGetName(char* out, int size, CUdevice) { std::snprintf(out, size, "CPU stub device"); return 0; }
inline CUresult cuDeviceGetPCIBusId(char* out, int size, CUdevice) { std::snprintf(out, size, "0000:01:00.0"); return 0; }
inline CUresult cuDeviceGetUuid(CUuuid* out, CUdevice) { std::memset(out->bytes, 42, sizeof(out->bytes)); return 0; }
inline CUresult cuDeviceGetAttribute(int* out, CUdevice_attribute attribute, CUdevice) {
  *out = attribute == CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID ? 0 : 1;
  return 0;
}
inline CUresult cuMemGetAllocationGranularity(std::size_t* out, const CUmemAllocationProp* p, int flags) {
  if (!valid_properties(p) || flags != CU_MEM_ALLOC_GRANULARITY_MINIMUM) return 1;
  *out = 16ULL * 1024 * 1024;
  return 0;
}
inline CUresult cuMemCreate(CUmemGenericAllocationHandle* out, std::size_t size,
                            const CUmemAllocationProp* p, unsigned long long flags) {
  ++next_handle;
  event("create", next_handle, size);
  if (!valid_properties(p) || flags || size % (16ULL * 1024 * 1024)) return 1;
  if (env_int("CUDA_STUB_DELAY_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(env_int("CUDA_STUB_DELAY_MS")));
  if (static_cast<int>(next_handle) == env_int("CUDA_STUB_FAIL_CREATE")) return 2;
  *out = next_handle;
  return 0;
}
inline CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  event("release", handle);
  return static_cast<int>(handle) == env_int("CUDA_STUB_FAIL_RELEASE") ? 999 : 0;
}
inline CUresult cuGetErrorName(CUresult code, const char** out) {
  *out = code == 0 ? "CUDA_SUCCESS" : code == 2 ? "CUDA_ERROR_OUT_OF_MEMORY" :
         code == 100 ? "CUDA_ERROR_NO_DEVICE" : "CUDA_ERROR_UNKNOWN";
  return 0;
}
inline CUresult cuGetErrorString(CUresult code, const char** out) {
  *out = code == 0 ? "success" : code == 2 ? "out of memory" : "stub error";
  return 0;
}
"""


@pytest.fixture(scope="module")
def native_repro(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    build = tmp_path_factory.mktemp("cuda-driver-stub")
    (build / "cuda.h").write_text(CUDA_STUB_HEADER)
    binary = build / "cuda-host-numa-repro"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", f"-I{build}", str(SOURCE), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    return binary


def launch(
    binary: Path, tmp_path: Path, *, size: int = 250 * GIB, chunk: int = 0, **settings: str
) -> tuple[subprocess.Popen[str], Path, Path]:
    result = tmp_path / "result.json"
    ready = tmp_path / "ready.json"
    trace = tmp_path / "trace.txt"
    process = subprocess.Popen(
        [
            str(binary),
            "--device",
            "0",
            "--numa",
            "1",
            "--size-bytes",
            str(size),
            "--chunk-bytes",
            str(chunk),
            "--ready-file",
            str(ready),
            "--result-file",
            str(result),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "2",
            "CUDA_STUB_NUMA": "1",
            "CUDA_STUB_TRACE": str(trace),
            **settings,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while not ready.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            stdout, stderr = process.communicate(timeout=3)
            pytest.fail(f"native reproducer did not publish ready: {stdout}\n{stderr}")
        time.sleep(0.01)
    return process, result, trace


def stop(process: subprocess.Popen[str]) -> None:
    process.send_signal(signal.SIGTERM)
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=3)
        raise


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_balanced_chunks_are_host_only_retained_until_signal_then_released_in_reverse(
    native_repro: Path, tmp_path: Path
) -> None:
    process, result, trace = launch(native_repro, tmp_path, chunk=16 * GIB)
    try:
        state = load(result)
        assert state["status"] == "ready"
        assert state["allocation_success"] is True
        assert state["setup_complete"] is True
        assert state["cleanup_confirmed"] is False
        assert state["allocated_bytes"] == state["held_bytes"] == 250 * GIB
        assert state["chunk_bytes"] == 15.625 * GIB
        assert state["live_handles"] == 16
        assert state["cuda"]["visible_devices"] == "2"
        assert state["cuda"]["device_name"] == "CPU stub device"
        assert state["numa"] == 1
        assert state["handle_type"] == "FABRIC"
        assert not any(line.startswith("release") for line in trace.read_text().splitlines())
        assert process.poll() is None
    finally:
        stop(process)
    final = load(result)
    assert process.returncode == 0
    assert final["cleanup_confirmed"] is True
    assert final["held_bytes"] == final["live_handles"] == 0
    assert final["allocated_bytes"] == 250 * GIB
    assert final["allocation_status"] == "ready"
    assert [int(line.split()[1]) for line in trace.read_text().splitlines() if line.startswith("release")] == list(
        range(16, 0, -1)
    )


def test_partial_oom_keeps_successful_handles_resident_until_inspected(native_repro: Path, tmp_path: Path) -> None:
    process, result, trace = launch(native_repro, tmp_path, chunk=16 * GIB, CUDA_STUB_FAIL_CREATE="14")
    try:
        state = load(result)
        assert state["status"] == "allocation_failed"
        assert state["failed_api"] == "cuMemCreate"
        assert state["failed_result"] == 2
        assert state["allocated_bytes"] == state["held_bytes"] == 203.125 * GIB
        assert state["live_handles"] == 13
        assert state["calls"][-1]["name"] == "CUDA_ERROR_OUT_OF_MEMORY"
        assert not any(line.startswith("release") for line in trace.read_text().splitlines())
        assert process.poll() is None
    finally:
        stop(process)
    final = load(result)
    assert process.returncode == 2
    assert final["cleanup_confirmed"] is True
    assert final["allocation_status"] == "allocation_failed"
    assert final["allocated_bytes"] == 203.125 * GIB


def test_default_uses_one_create_and_aligns_size_to_granularity(native_repro: Path, tmp_path: Path) -> None:
    process, result, trace = launch(native_repro, tmp_path, size=17 * 1024**2)
    try:
        assert load(result)["effective_bytes"] == 32 * 1024**2
        creates = [line for line in trace.read_text().splitlines() if line.startswith("create")]
        assert creates == [f"create 1 {32 * 1024**2}"]
    finally:
        stop(process)
    assert process.returncode == 0


def test_initialization_error_is_distinct_from_host_allocation_failure(native_repro: Path, tmp_path: Path) -> None:
    process, result, trace = launch(native_repro, tmp_path, CUDA_STUB_INIT_ERROR="100")
    try:
        state = load(result)
        assert state["status"] == "setup_failed"
        assert state["failed_api"] == "cuInit"
        assert state["failed_result"] == 100
        assert state["held_bytes"] == 0
        assert trace.read_text() == "init 0 0\n"
    finally:
        stop(process)
    assert process.returncode == 2


def test_cleanup_failure_is_reported_and_other_handles_still_release(native_repro: Path, tmp_path: Path) -> None:
    process, result, trace = launch(native_repro, tmp_path, size=32 * GIB, chunk=16 * GIB, CUDA_STUB_FAIL_RELEASE="2")
    stop(process)
    final = load(result)
    assert process.returncode == 1
    assert final["status"] == "cleanup_failed"
    assert final["cleanup_confirmed"] is False
    assert final["live_handles"] == 1
    assert final["failed_api"] == "cuMemRelease"
    assert trace.read_text().splitlines()[-2:] == ["release 2 0", "release 1 0"]


def test_compiled_reproducer_integrates_with_supervisor_and_report(
    native_repro: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_STUB_NUMA", "0")
    run_dir = tmp_path / "run"
    settings = cuda_host_numa_probe.Settings(
        binary=native_repro,
        output_dir=run_dir / "logs/probe",
        size=32 * GIB,
        cases=("chunked",),
        allocation_timeout=5,
        budget=20,
        sample_interval=0.05,
        hold_seconds=0.05,
        chunk_size=16 * GIB,
    )
    assert cuda_host_numa_probe.run(settings) == 0
    summary = load(settings.output_dir / "summary.json")
    case = summary["cases"][0]
    assert case["outcome"] == "two_allocations_succeeded"
    assert case["cleanup_confirmed"] is True
    for rank in case["ranks"]:
        assert rank["observed"]["allocation_success"] is True
        assert rank["observed"]["held_bytes"] == 32 * GIB
        assert rank["final"]["cleanup_confirmed"] is True
        assert rank["final"]["held_bytes"] == 0
    assert cuda_host_numa_probe.report(run_dir, 0) == 0
    assert "**COMPLETE**" in (run_dir / "logs/perf-analysis.md").read_text()


@pytest.mark.parametrize("argument", ["--help", "--size-bytes", "-1", "--unknown"])
def test_usage_and_invalid_arguments_never_initialize_cuda(native_repro: Path, tmp_path: Path, argument: str) -> None:
    trace = tmp_path / "trace.txt"
    command = [str(native_repro), argument]
    result = subprocess.run(command, env={**os.environ, "CUDA_STUB_TRACE": str(trace)}, capture_output=True, text=True)
    assert result.returncode == (0 if argument == "--help" else 1)
    assert not trace.exists()
