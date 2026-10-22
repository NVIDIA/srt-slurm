#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record host-memory limits and isolate native EGM allocation failures.

Each probe creates only a HOST_NUMA Store pool, closes it immediately, and runs
in a fresh process. No device payload or ordinary pinned-host fallback is used.
Native configuration follows Mooncake 042ab8a18b0e6ccfe87782261a5556a3dd4446e4.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib
import importlib.util
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_V2_FILES = (
    "memory.current",
    "memory.max",
    "memory.peak",
    "memory.events",
    "memory.events.local",
    "memory.stat",
    "cpuset.cpus.effective",
    "cpuset.mems.effective",
    "cpuset.cpus",
    "cpuset.mems",
)
_V1_FILES = (
    "memory.usage_in_bytes",
    "memory.limit_in_bytes",
    "memory.max_usage_in_bytes",
    "memory.failcnt",
    "memory.oom_control",
    "memory.stat",
    "cpuset.effective_cpus",
    "cpuset.effective_mems",
    "cpuset.cpus",
    "cpuset.mems",
)
_DIAGNOSTIC_ENV = (
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_STEP_ID",
    "SLURM_STEPID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_NODELIST",
    "SLURM_JOB_NODELIST",
    "SLURMD_NODENAME",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_GPUS_PER_TASK",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_CPU_BIND",
    "SLURM_MEM_BIND",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "MC_MAX_MR_SIZE",
    "MC_MS_AUTO_DISC",
    "MC_USE_TENT",
    "MC_USE_TEV1",
    "MC_USE_NVLINK_IPC",
    "MC_TE_FILTERS",
    "MC_ENABLE_DEST_DEVICE_AFFINITY",
    "MC_STORE_MEMCPY",
    "MOONCAKE_STORE_BACKEND",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_text(path: Path) -> dict[str, str]:
    try:
        return {"text": path.read_text()}
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def cgroup_paths(cgroup: str, mountinfo: str) -> list[dict[str, Any]]:
    """Resolve membership through mount roots, including visible ancestors.

    Paths outside a container's mounted hierarchy are not accessible and are
    reported as such, rather than treating a visible leaf's limit as sufficient.
    """
    memberships = []
    for line in cgroup.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            memberships.append((set(filter(None, fields[1].split(","))), Path(fields[2])))
    resolved: list[dict[str, Any]] = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 5 or len(right) < 3 or right[0] not in ("cgroup", "cgroup2"):
            continue

        def decode(value: str) -> str:
            return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)

        mount_root, mountpoint = Path(decode(left[3])), Path(decode(left[4]))
        controllers = set(right[2].split(","))
        for member_controllers, member in memberships:
            unified = right[0] == "cgroup2" and not member_controllers
            if not unified and not (member_controllers & controllers & {"memory", "cpuset"}):
                continue
            row: dict[str, Any] = {
                "version": 2 if unified else 1,
                "membership": str(member),
                "mount_root": str(mount_root),
                "mountpoint": str(mountpoint),
            }
            try:
                # Cgroup namespaces may expose their mount root as membership /.
                relative = Path(".") if member == Path("/") else member.relative_to(mount_root)
            except ValueError:
                row["error"] = "Membership is outside the visible cgroup mount root"
                resolved.append(row)
                continue
            leaf = mountpoint / relative
            row["ancestors"] = [str(leaf), *(str(path) for path in leaf.parents if path.is_relative_to(mountpoint))]
            resolved.append(row)
    return resolved


def cuda_inventory() -> dict[str, Any]:
    """Query CUDA identities and HOST_NUMA IDs without creating device buffers."""
    evidence: dict[str, Any] = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "devices": []}
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuInit.argtypes, cuda.cuInit.restype = [ctypes.c_uint], ctypes.c_int
        cuda.cuDeviceGetCount.argtypes, cuda.cuDeviceGetCount.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        cuda.cuDeviceGet.argtypes, cuda.cuDeviceGet.restype = [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
        cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        cuda.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        status = cuda.cuInit(0)
        if status:
            raise RuntimeError(f"cuInit returned {status}")
        count = ctypes.c_int()
        status = cuda.cuDeviceGetCount(ctypes.byref(count))
        if status:
            raise RuntimeError(f"cuDeviceGetCount returned {status}")
        numa_attribute = None
        for module in ("cuda.bindings.driver", "cuda.cuda"):
            try:
                bindings = importlib.import_module(module)
                numa_attribute = int(bindings.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID)
                break
            except (ImportError, AttributeError) as exc:
                evidence["host_numa_attribute_error"] = str(exc)
        if numa_attribute is not None:
            evidence.pop("host_numa_attribute_error", None)
        for ordinal in range(count.value):
            device = ctypes.c_int()
            result = cuda.cuDeviceGet(ctypes.byref(device), ordinal)
            row: dict[str, Any] = {"ordinal": ordinal, "device_result": result}
            if result == 0:
                for key, function in (("name", cuda.cuDeviceGetName), ("pci_bus_id", cuda.cuDeviceGetPCIBusId)):
                    buffer = ctypes.create_string_buffer(256)
                    result = function(buffer, len(buffer), device.value)
                    row[key] = buffer.value.decode(errors="replace") if result == 0 else {"cuda_error": result}
                if numa_attribute is not None:
                    node = ctypes.c_int()
                    result = cuda.cuDeviceGetAttribute(ctypes.byref(node), numa_attribute, device.value)
                    row["host_numa_id"] = node.value if result == 0 else {"cuda_error": result}
            evidence["devices"].append(row)
    except (OSError, RuntimeError, AttributeError) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def inventory(root: Path = Path("/"), *, include_cuda: bool = False) -> dict[str, Any]:
    """Retain raw evidence even when files are absent or permission restricted."""
    files = {
        name: read_text(root / name.lstrip("/"))
        for name in (
            "/proc/self/status",
            "/proc/self/limits",
            "/proc/self/cgroup",
            "/proc/self/mountinfo",
            "/proc/meminfo",
            "/proc/buddyinfo",
            "/sys/devices/system/node/online",
            "/sys/devices/system/node/possible",
        )
    }
    for node in sorted((root / "sys/devices/system/node").glob("node[0-9]*")):
        for name in ("meminfo", "numastat", "cpulist"):
            files["/" + str((node / name).relative_to(root))] = read_text(node / name)
    cgroups = cgroup_paths(files["/proc/self/cgroup"].get("text", ""), files["/proc/self/mountinfo"].get("text", ""))
    for group in cgroups:
        for ancestor in group.get("ancestors", []):
            for name in _V2_FILES if group["version"] == 2 else _V1_FILES:
                path = str(Path(ancestor) / name)
                files[path] = read_text(root / path.lstrip("/"))
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time_unix": time.time(),
        "files": files,
        "cgroups": cgroups,
        "environment": {key: os.environ[key] for key in _DIAGNOSTIC_ENV if key in os.environ},
    }
    if include_cuda:
        result["cuda"] = cuda_inventory()
    return result


def load_donor(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("mooncake_egm_diagnostic_donor", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load native donor helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(config_path: Path, segment_size: int, numa_nodes: str, donor: Any, result_path: Path) -> int:
    result: dict[str, Any] = {
        "status": "child_failed",
        "cleanup_confirmed": False,
        "before": inventory(include_cuda=True),
    }
    store = None
    try:
        donor.configure_environment()
        config = donor.provider_config(config_path, segment_size, numa_nodes)
        result["config"] = config
        write_json(result_path, result)
        donor.require_cuda_devices()
        store = donor.load_store_type()()
        result["setup_result"] = store.setup(config)
        result["status"] = "passed" if result["setup_result"] == 0 else "setup_failed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if store is not None:
            try:
                result["close_result"] = store.close()
                result["cleanup_confirmed"] = result["close_result"] == 0
                if not result["cleanup_confirmed"]:
                    result["status"] = "cleanup_failed"
            except Exception as exc:
                result.update(status="cleanup_failed", close_error=f"{type(exc).__name__}: {exc}")
        result["after"] = inventory()
        write_json(result_path, result)
    return 0 if result["status"] == "passed" else 1


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def run_case(command: list[str], log_path: Path, result_path: Path, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    interrupted = False
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

        def stop_on_signal(signum: int, frame: object) -> None:
            nonlocal interrupted
            interrupted = True
            stop_process(process)

        handlers = {number: signal.signal(number, stop_on_signal) for number in (signal.SIGINT, signal.SIGTERM)}
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process(process)
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
    result: dict[str, Any] = {"status": "child_failed", "cleanup_confirmed": False}
    if result_path.exists():
        try:
            child_result = json.loads(result_path.read_text())
            if not isinstance(child_result, dict):
                raise ValueError("Child result must be a JSON object")
            result.update(child_result)
        except (OSError, ValueError) as exc:
            result["result_error"] = str(exc)
    if timed_out or interrupted or process.returncode is not None and process.returncode < 0:
        result.update(status="timeout" if timed_out else "killed", cleanup_confirmed=False)
    result.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - started)
    result["cuda_errors"] = [
        {"code": int(code), "name": "CUDA_ERROR_OUT_OF_MEMORY" if code == "2" else "unknown CUDA error name"}
        for code in sorted(set(re.findall(r"CUDA (?:result|error)\s+(\d+)", log_path.read_text())))
    ]
    # The child result retains before/after snapshots; keep the summary compact.
    return {key: value for key, value in result.items() if key not in ("before", "after")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", default="4GiB,260GiB")
    parser.add_argument("--gpu", default="0", help="One physical CUDA index or UUID for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--numa-nodes", default="0,auto", help="Comma-separated independent NUMA modes")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--budget-seconds", type=float, default=150)
    parser.add_argument(
        "--donor-helper", type=Path, default=Path(__file__).resolve().parent.parent / "configs/mooncake_egm_donor.py"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        donor = load_donor(args.donor_helper)
        sizes = [donor.parse_size(item.strip()) for item in args.sizes.split(",")]
        nodes = [item.strip() for item in args.numa_nodes.split(",")]
        if not args.gpu or "," in args.gpu or any(node != "auto" and not node.isdecimal() for node in nodes):
            raise ValueError("Choose one GPU and nonnegative NUMA IDs or auto")
        if any(not math.isfinite(value) or value <= 0 for value in (args.timeout_seconds, args.budget_seconds)):
            raise ValueError("Timeout and budget must be positive")
        if len(set(sizes)) != len(sizes) or len(set(nodes)) != len(nodes):
            raise ValueError("Size and NUMA cases must be unique")
        donor.provider_config(args.config, sizes[0], nodes[0])
    except (OSError, ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.child:
        if args.result is None:
            parser.error("--child requires --result")
        return probe(args.config, sizes[0], nodes[0], donor, args.result)
    donor.configure_environment()
    started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "running",
        "hostname": socket.gethostname(),
        "gpu_selector": args.gpu,
        "inventory": "inventory.json",
        "parameters": {
            "sizes": sizes,
            "numa_nodes": nodes,
            "timeout_seconds": args.timeout_seconds,
            "budget_seconds": args.budget_seconds,
        },
        "cases": [],
    }
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    write_json(args.output_dir / "inventory.json", inventory(include_cuda=True))
    for size in sizes:
        for node in nodes:
            remaining = args.budget_seconds - (time.monotonic() - started)
            if remaining <= 0:
                summary["status"] = "budget_exhausted"
                break
            name = f"{size}-numa-{node}"
            log_path, result_path = args.output_dir / f"{name}.log", args.output_dir / f"{name}.json"
            result_path.unlink(missing_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--config",
                str(args.config),
                "--output-dir",
                str(args.output_dir),
                "--sizes",
                str(size),
                "--gpu",
                args.gpu,
                "--numa-nodes",
                node,
                "--donor-helper",
                str(args.donor_helper),
                "--result",
                str(result_path),
            ]
            case = run_case(command, log_path, result_path, min(args.timeout_seconds, remaining))
            case.update(
                name=name,
                segment_size_bytes=size,
                numa_nodes=node,
                gpu_selector=args.gpu,
                log=log_path.name,
                result=result_path.name,
            )
            summary["cases"].append(case)
            expected_returncodes = {"passed": 0, "setup_failed": 1}
            if (
                not case["cleanup_confirmed"]
                or case["status"] not in expected_returncodes
                or case.get("returncode") != expected_returncodes[case["status"]]
            ):
                summary["status"] = "stopped_unsafe_cleanup"
            write_json(summary_path, summary)
            if summary["status"] != "running":
                break
        if summary["status"] != "running":
            break
    if summary["status"] == "running":
        summary["status"] = "complete"
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output_dir / "inventory-after.json", inventory())
    write_json(summary_path, summary)
    print(json.dumps({"allocation_diagnostics": summary["status"], "summary": str(summary_path)}), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
