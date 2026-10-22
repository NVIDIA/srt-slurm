# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify allocation diagnostics without CUDA, Slurm or native allocations."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import mooncake_egm_alloc_diagnostics as diagnostic


def test_cgroup_v2_mount_root_resolves_leaf_and_ancestors() -> None:
    groups = diagnostic.cgroup_paths(
        "0::/slurm/job_12/step_0/task_0\n",
        "10 9 0:28 /slurm/job_12 /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
    )
    assert groups[0]["ancestors"] == [
        "/sys/fs/cgroup/step_0/task_0",
        "/sys/fs/cgroup/step_0",
        "/sys/fs/cgroup",
    ]
    assert groups[0]["version"] == 2
    assert diagnostic.cgroup_paths("0::/\n", "10 9 0:28 /slurm/job_12 /sys/fs/cgroup rw - cgroup2 cgroup rw\n")[0][
        "ancestors"
    ] == ["/sys/fs/cgroup"]


def test_inventory_retains_ancestor_limits_and_missing_evidence(tmp_path: Path) -> None:
    fixture = {
        "proc/self/cgroup": "4:memory:/slurm/job_12/task_0\n3:cpuset:/slurm/job_12/task_0\n",
        "proc/self/mountinfo": (
            "10 9 0:28 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
            "11 9 0:29 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset\n"
        ),
        "sys/fs/cgroup/memory/slurm/job_12/memory.limit_in_bytes": "34359738368\n",
        "sys/fs/cgroup/memory/slurm/job_12/task_0/memory.limit_in_bytes": "9223372036854771712\n",
        "sys/fs/cgroup/cpuset/slurm/job_12/cpuset.mems": "0\n",
        "sys/devices/system/node/node0/meminfo": "Node 0 MemFree: 200000000 kB\n",
    }
    for name, text in fixture.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    result = diagnostic.inventory(tmp_path)
    assert result["files"]["/sys/fs/cgroup/memory/slurm/job_12/memory.limit_in_bytes"]["text"] == "34359738368\n"
    assert result["files"]["/sys/fs/cgroup/cpuset/slurm/job_12/cpuset.mems"]["text"] == "0\n"
    assert "Node 0 MemFree" in result["files"]["/sys/devices/system/node/node0/meminfo"]["text"]
    assert "FileNotFoundError" in result["files"]["/proc/self/limits"]["error"]


def test_cgroup_membership_outside_visible_mount_is_explicit() -> None:
    result = diagnostic.cgroup_paths("0::/other/job\n", "10 9 0:28 /slurm/job /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    assert "outside" in result[0]["error"]
    assert "ancestors" not in result[0]


def test_inventory_environment_only_retains_named_diagnostic_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in {
        "SLURM_JWT": "secret",
        "MC_AUTH_TOKEN": "secret",
        "CUDA_FUTURE_TOKEN": "secret",
        "SLURM_JOB_ID": "12",
        "MC_MAX_MR_SIZE": "4294967296",
        "CUDA_VISIBLE_DEVICES": "0",
    }.items():
        monkeypatch.setenv(key, value)
    environment = diagnostic.inventory(tmp_path)["environment"]
    assert environment["SLURM_JOB_ID"] == "12"
    assert environment["MC_MAX_MR_SIZE"] == "4294967296"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert not {"SLURM_JWT", "MC_AUTH_TOKEN", "CUDA_FUTURE_TOKEN"} & environment.keys()


def test_cuda_inventory_uses_binding_enum_and_visible_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    def count(pointer: ctypes.c_void_p) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0

    def get_device(pointer: ctypes.c_void_p, ordinal: int) -> int:
        assert ordinal == 0
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 7
        return 0

    def get_name(buffer: ctypes.Array[ctypes.c_char], size: int, device: int) -> int:
        assert device == 7
        buffer.value = b"Test GPU"
        return 0

    def get_pci(buffer: ctypes.Array[ctypes.c_char], size: int, device: int) -> int:
        assert device == 7
        buffer.value = b"0000:41:00.0"
        return 0

    def get_attribute(pointer: ctypes.c_void_p, attribute: int, device: int) -> int:
        assert (attribute, device) == (999, 7)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 3
        return 0

    cuda = SimpleNamespace(
        cuInit=Mock(return_value=0),
        cuDeviceGetCount=Mock(side_effect=count),
        cuDeviceGet=Mock(side_effect=get_device),
        cuDeviceGetName=Mock(side_effect=get_name),
        cuDeviceGetPCIBusId=Mock(side_effect=get_pci),
        cuDeviceGetAttribute=Mock(side_effect=get_attribute),
    )

    def bindings(module: str) -> SimpleNamespace:
        assert module == "cuda.bindings.driver"
        return SimpleNamespace(CUdevice_attribute=SimpleNamespace(CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID=999))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-uuid")
    monkeypatch.setattr(diagnostic.ctypes, "CDLL", lambda name: cuda)
    monkeypatch.setattr(diagnostic.importlib, "import_module", bindings)
    result = diagnostic.cuda_inventory()
    assert result == {
        "CUDA_VISIBLE_DEVICES": "GPU-test-uuid",
        "devices": [
            {"ordinal": 0, "device_result": 0, "name": "Test GPU", "pci_bus_id": "0000:41:00.0", "host_numa_id": 3}
        ],
    }


@pytest.mark.parametrize(
    "setup_result,close_result,status", [(0, 0, "passed"), (-1, 0, "setup_failed"), (0, -1, "cleanup_failed")]
)
def test_probe_requires_host_only_config_and_closes_failed_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup_result: int, close_result: int, status: str
) -> None:
    from configs import mooncake_egm_donor

    config_path = tmp_path / "config.json"
    config_path.write_text('{"master_server_address":"192.0.2.1:8700","local_hostname":"192.0.2.2"}')
    captured = {}

    class Store:
        def setup(self, config: dict[str, str]) -> int:
            captured.update(config)
            return setup_result

        def close(self) -> int:
            captured["closed"] = True
            return close_result

    monkeypatch.setattr(diagnostic, "inventory", lambda **kwargs: {"fake_inventory": True})
    donor = SimpleNamespace(
        configure_environment=lambda: None,
        provider_config=mooncake_egm_donor.provider_config,
        require_cuda_devices=lambda: None,
        load_store_type=lambda: Store,
    )
    output = tmp_path / "case.json"
    code = diagnostic.probe(config_path, 260 * 1024**3, "0", donor, output)
    result = json.loads(output.read_text())
    assert result["status"] == status
    assert result["cleanup_confirmed"] == (close_result == 0)
    assert code == (0 if status == "passed" else 1)
    assert captured["closed"] is True
    assert captured["global_segment_size"] == str(260 * 1024**3)
    assert captured["local_buffer_size"] == "0"
    assert captured["enable_egm_store_pool"] == "true"
    assert captured["protocol"] == "nvlink"


def test_timeout_is_recorded_and_reaped(tmp_path: Path) -> None:
    result = diagnostic.run_case(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path / "case.log",
        tmp_path / "case.json",
        0.05,
    )
    assert result["status"] == "timeout"
    assert result["cleanup_confirmed"] is False
    assert result["returncode"] is not None
    assert result["elapsed_seconds"] < 5


@pytest.mark.parametrize("cuda_code,name", [(2, "CUDA_ERROR_OUT_OF_MEMORY"), (999, "unknown CUDA error name")])
def test_killed_process_keeps_cuda_error_evidence(tmp_path: Path, cuda_code: int, name: str) -> None:
    command = f"import os,signal; print('cuMemCreate failed with CUDA result {cuda_code}', flush=True); os.kill(os.getpid(),signal.SIGKILL)"
    result = diagnostic.run_case([sys.executable, "-c", command], tmp_path / "case.log", tmp_path / "case.json", 2)
    assert result["status"] == "killed"
    assert result["cuda_errors"] == [{"code": cuda_code, "name": name}]


@pytest.mark.parametrize("payload", ["null", "13", '[["status", "passed"], ["cleanup_confirmed", true]]'])
def test_malformed_child_result_is_not_accepted(tmp_path: Path, payload: str) -> None:
    result_path = tmp_path / "case.json"
    result_path.write_text(payload)
    result = diagnostic.run_case(
        [sys.executable, "-c", "pass"],
        tmp_path / "case.log",
        result_path,
        2,
    )
    assert result["status"] == "child_failed"
    assert result["cleanup_confirmed"] is False
    assert "JSON object" in result["result_error"]


def fake_helper(tmp_path: Path) -> Path:
    helper = tmp_path / "fake_donor.py"
    helper.write_text(
        "import os\n"
        "def parse_size(value): return int(value.removesuffix('GiB')) * (1024**3 if value.endswith('GiB') else 1)\n"
        "def configure_environment(): pass\n"
        "def provider_config(path,size,node): return {'global_segment_size': str(size), 'local_buffer_size':'0', 'enable_egm_store_pool':'true', 'egm_numa_nodes':node}\n"
        "def require_cuda_devices(): pass\n"
        "def load_store_type(): return Store\n"
        "class Store:\n"
        " def setup(self, config):\n"
        "  print('GPU_SELECTOR=' + os.environ['CUDA_VISIBLE_DEVICES'], flush=True)\n"
        "  if int(config['global_segment_size']) > 4*1024**3:\n"
        "   print('EGM allocation failed: Memory: cuMemCreate failed with CUDA result 2', flush=True)\n"
        "   return -1\n"
        "  return 0\n"
        " def close(self): return 0\n"
    )
    return helper


def test_cli_continues_after_controlled_oom_and_remaps_single_gpu(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            diagnostic.__file__,
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--donor-helper",
            str(fake_helper(tmp_path)),
            "--gpu",
            "3",
            "--timeout-seconds",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert [case["status"] for case in summary["cases"]] == ["passed", "passed", "setup_failed", "setup_failed"]
    assert {case["numa_nodes"] for case in summary["cases"]} == {"0", "auto"}
    assert all(case["cleanup_confirmed"] for case in summary["cases"])
    for case in summary["cases"]:
        assert "GPU_SELECTOR=3" in (output / case["log"]).read_text()
        assert json.loads((output / case["result"]).read_text())["config"]["local_buffer_size"] == "0"
    assert summary["cases"][-1]["cuda_errors"] == [{"code": 2, "name": "CUDA_ERROR_OUT_OF_MEMORY"}]


def test_unknown_cleanup_stops_subsequent_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = fake_helper(tmp_path)
    calls = []

    def timeout(*args: object) -> dict[str, object]:
        calls.append(args)
        return {"status": "timeout", "cleanup_confirmed": False, "returncode": -9}

    monkeypatch.setattr(diagnostic, "run_case", timeout)
    monkeypatch.setattr(diagnostic, "inventory", lambda **kwargs: {})
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    output = tmp_path / "output"
    code = diagnostic.main(["--config", str(config_path), "--output-dir", str(output), "--donor-helper", str(helper)])
    assert code == 1
    assert len(calls) == 1
    assert json.loads((output / "summary.json").read_text())["status"] == "stopped_unsafe_cleanup"


@pytest.mark.parametrize("status,returncode", [("child_failed", 0), ("passed", 1), ("setup_failed", 0)])
def test_invalid_child_completion_stops_despite_close_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, returncode: int
) -> None:
    helper = fake_helper(tmp_path)
    monkeypatch.setattr(
        diagnostic, "run_case", lambda *args: {"status": status, "cleanup_confirmed": True, "returncode": returncode}
    )
    monkeypatch.setattr(diagnostic, "inventory", lambda **kwargs: {})
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    output = tmp_path / "output"
    assert (
        diagnostic.main(["--config", str(config_path), "--output-dir", str(output), "--donor-helper", str(helper)]) == 1
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "stopped_unsafe_cleanup"
    assert len(summary["cases"]) == 1
