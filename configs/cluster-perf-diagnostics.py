#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect serving placement and measure MPI/NCCL latency without changing host settings."""

import argparse
import datetime
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ENV_KEYS = (
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "OMPI_COMM_WORLD_RANK",
    "CUDA_VISIBLE_DEVICES",
    "UCX_TLS",
    "UCX_NET_DEVICES",
    "MPI_UCX_TLS",
    "MPI_UCX_NET_DEVICES",
    "NCCL_IB_HCA",
    "NCCL_CROSS_NIC",
    "NCCL_MNNVL_ENABLE",
    "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE",
    "OMPI_MCA_pml",
    "OMPI_MCA_coll",
    "TLLM_NUMA_AWARE_WORKER_AFFINITY",
    "OMP_NUM_THREADS",
    "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS",
    "TRTLLM_LOW_M_GEMM_BACKEND",
)


def read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError) as error:
        return f"UNAVAILABLE: {error}"


def command(argv: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
        return {"argv": argv, "exit_status": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": str(error)}


def process(pid: int, proc: Path = Path("/proc")) -> dict[str, Any]:
    root = proc / str(pid)
    threads = {}
    for task in sorted((root / "task").glob("*")):
        status = read(task / "status")
        threads[task.name] = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in status.splitlines()
            if line.startswith(("Name:", "Cpus_allowed_list:", "Mems_allowed_list:"))
        }
    try:
        entries = (root / "environ").read_bytes().split(b"\0")
        environment = dict(entry.decode(errors="replace").split("=", 1) for entry in entries if b"=" in entry)
        filtered_env = {key: environment[key] for key in ENV_KEYS if key in environment}
    except OSError as error:
        filtered_env = {"error": str(error)}
    cgroup = read(root / "cgroup")
    cgroup_limits = {}
    for line in cgroup.splitlines():
        if line.startswith("0::"):
            relative = line.removeprefix("0::").lstrip("/")
            for base in (root / "root/sys/fs/cgroup", root / "root/sys/fs/cgroup" / relative):
                for name in (
                    "cpu.max",
                    "cpu.stat",
                    "memory.max",
                    "memory.current",
                    "memory.events",
                    "cpuset.cpus.effective",
                    "cpuset.mems.effective",
                ):
                    cgroup_limits[str(base / name)] = read(base / name)
    return {
        "comm": read(root / "comm"),
        "status": read(root / "status"),
        "threads": threads,
        "environment": filtered_env,
        "numa_maps": read(root / "numa_maps"),
        "schedstat": read(root / "schedstat"),
        "cgroup": cgroup,
        "cgroup_limits": cgroup_limits,
        "library_maps": [line for line in read(root / "maps").splitlines() if ".so" in line],
    }


def nic_snapshot(sysfs: Path = Path("/sys")) -> dict[str, str]:
    values = {}
    for device in sorted((sysfs / "class/infiniband").glob("*")):
        paths = [
            device / "fw_ver",
            device / "device/numa_node",
            device / "device/current_link_speed",
            device / "device/current_link_width",
        ]
        for port in sorted((device / "ports").glob("*")):
            paths.extend(port / name for name in ("state", "rate", "sm_lid", "lid"))
            for group in ("gids", "counters", "hw_counters"):
                paths.extend(sorted((port / group).glob("*")))
        for path in paths:
            values[str(path.relative_to(sysfs))] = read(path)
    return values


def counter_deltas(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    result = {}
    for key in before.keys() & after.keys():
        if "/counters/" not in key and "/hw_counters/" not in key:
            continue
        if before[key].isdigit() and after[key].isdigit():
            delta = int(after[key]) - int(before[key])
            result[key] = delta if delta >= 0 else "RESET_OR_WRAP"
    return result


def model_manifest(model: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(model), "metadata_sha256": {}, "weight_sizes": {}}
    for path in sorted(model.glob("*.json")):
        result["metadata_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for pattern in ("*.safetensors", "*.bin"):
        for path in sorted(model.glob(pattern)):
            result["weight_sizes"][path.name] = path.stat().st_size
    if not result["metadata_sha256"] and not result["weight_sizes"]:
        raise ValueError(f"No model metadata or weight files found in {model}")
    result["note"] = "Metadata hashes and weight sizes do not prove identical weight contents."
    return result


def code_hashes() -> dict[str, str]:
    try:
        dist = importlib.metadata.distribution("tensorrt-llm")
    except importlib.metadata.PackageNotFoundError:
        return {"error": "tensorrt-llm distribution unavailable"}
    result = {}
    for name in (
        "llmapi/utils.py",
        "_torch/distributed/ops.py",
        "_torch/distributed/communicator.py",
        "_torch/pyexecutor/py_executor.py",
        "_torch/pyexecutor/cuda_graph_runner.py",
        "_torch/modules/fused_moe/mega_moe/mega_moe_cute_dsl.py",
        "_torch/pyexecutor/connectors/mooncake_store/worker.py",
        "_torch/pyexecutor/connectors/mooncake_store/staging.py",
    ):
        path = Path(str(dist.locate_file("tensorrt_llm"))) / name
        try:
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            result[name] = f"UNAVAILABLE: {error}"
    return result


def collect(args: argparse.Namespace) -> dict[str, Any]:
    commands = {
        "gpu": ["nvidia-smi", "-q"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "numa": ["numactl", "--hardware"],
        "network": ["ip", "-details", "link", "show"],
        "routes": ["ip", "route", "show"],
        "ucx": ["ucx_info", "-v"],
        "mpi": ["ompi_info", "--version"],
        "torch_nccl": [
            sys.executable,
            "-c",
            "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())",
        ],
    }
    result: dict[str, Any] = {
        "format": 1,
        "requested_duration_s": args.duration,
        "sample_interval_s": args.interval,
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "code_sha256": code_hashes(),
        "commands": {key: command(argv) for key, argv in commands.items()},
        "processes_before": {str(pid): process(pid) for pid in args.pid},
        "nic_before": nic_snapshot(),
        "samples": [],
    }
    if args.model:
        result["model"] = model_manifest(args.model)
    query = "index,uuid,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used"
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        result["samples"].append(
            {
                "time": time.time(),
                "gpu": command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]),
                "pressure": {name: read(Path("/proc/pressure") / name) for name in ("cpu", "memory", "io")},
                "process_schedstat": {str(pid): read(Path(f"/proc/{pid}/schedstat")) for pid in args.pid},
            }
        )
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(args.interval, remaining))
    result["processes_after"] = {str(pid): process(pid) for pid in args.pid}
    result["nic_after"] = nic_snapshot()
    result["counter_deltas"] = counter_deltas(result["nic_before"], result["nic_after"])
    result["readable_counter_count"] = len(result["counter_deltas"])
    return result


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("No measured samples")
    return {
        "p50_us": statistics.median(ordered),
        "p99_us": ordered[math.ceil(0.99 * len(ordered)) - 1],
        "max_us": ordered[-1],
    }


def routed_address(hosts: list[str]) -> str:
    """Select rank zero's IPv4 source for a peer route without sending packets."""
    own_host = socket.gethostname()
    peers = list(dict.fromkeys(host for host in hosts if host != own_host)) or [own_host]
    for peer in peers:
        for _family, _kind, _protocol, _name, target in socket.getaddrinfo(peer, 9, socket.AF_INET, socket.SOCK_DGRAM):
            address = ipaddress.ip_address(target[0])
            if address.is_loopback or address.is_link_local or address.is_unspecified:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.connect(target)
                source = route.getsockname()[0]
            address = ipaddress.ip_address(source)
            if not (address.is_loopback or address.is_link_local or address.is_unspecified):
                return source
    raise RuntimeError("No usable peer route for NCCL rendezvous; override --master-addr")


def nccl_store(comm: Any, dist: Any, addr: str | None, port: int | None) -> tuple[Any, dict[str, Any]]:
    """Keep an OS-assigned TCPStore port bound while MPI publishes its endpoint."""
    hosts = comm.allgather(socket.gethostname())
    store, endpoint = None, None
    if comm.Get_rank() == 0:
        try:
            addr = addr or routed_address(hosts)
            store = dist.TCPStore(
                addr,
                port or 0,
                comm.Get_size(),
                True,
                timeout=datetime.timedelta(seconds=90),
                wait_for_workers=False,
            )
            endpoint = {"address": addr, "port": store.port, "host": hosts[0]}
        except Exception as error:
            endpoint = {"error": str(error)}
    endpoint = comm.bcast(endpoint, root=0)
    if "error" in endpoint:
        raise RuntimeError(f"NCCL rendezvous setup failed: {endpoint['error']}")
    if comm.Get_rank() != 0:
        store = dist.TCPStore(
            endpoint["address"],
            endpoint["port"],
            comm.Get_size(),
            False,
            timeout=datetime.timedelta(seconds=90),
        )
    return store, endpoint


def communication(args: argparse.Namespace) -> dict[str, Any] | None:
    # Dependencies are imported only for an explicitly requested microbenchmark.
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    if world < 2:
        raise ValueError("Launch communication with srun/mpirun and at least two ranks")
    metadata = comm.gather(
        {
            "rank": rank,
            "host": socket.gethostname(),
            "placement": process(os.getpid()),
            "mpi_library": MPI.Get_library_version(),
        },
        root=0,
    )
    results = []
    for count in (8, 128, 1024):
        send = np.full(count, rank, dtype=np.int64)
        recv = np.empty((world, count), dtype=np.int64)
        for _repeat in range(args.warmup):
            comm.Allgather(send, recv)
        comm.Barrier()
        timings = []
        for _repeat in range(args.iterations):
            started = time.perf_counter_ns()
            comm.Allgather(send, recv)
            timings.append((time.perf_counter_ns() - started) / 1000)
        if not np.all(recv == np.arange(world)[:, None]):
            raise RuntimeError("MPI buffer allgather correctness failure")
        gathered = comm.gather(timings, root=0)
        if rank == 0:
            results.append(
                {
                    "operation": "MPI_Allgather",
                    "bytes_per_rank": send.nbytes,
                    "slowest_rank_per_iteration": summarize([max(step) for step in zip(*gathered, strict=True)]),
                }
            )
    payload = {"rank": rank, "state": list(range(128))}
    for _repeat in range(args.warmup):
        comm.allgather(payload)
    comm.Barrier()
    timings = []
    for _repeat in range(args.iterations):
        started = time.perf_counter_ns()
        output = comm.allgather(payload)
        timings.append((time.perf_counter_ns() - started) / 1000)
    if [item["rank"] for item in output] != list(range(world)):
        raise RuntimeError("MPI object allgather correctness failure")
    gathered = comm.gather(timings, root=0)
    if rank == 0:
        results.append(
            {
                "operation": "MPI_object_allgather",
                "payload": "rank + 128 integers",
                "slowest_rank_per_iteration": summarize([max(step) for step in zip(*gathered, strict=True)]),
            }
        )
    rendezvous = None
    if args.nccl:
        import torch
        import torch.distributed as dist

        local = comm.Split_type(MPI.COMM_TYPE_SHARED)
        device = local.Get_rank() if torch.cuda.device_count() > 1 else 0
        torch.cuda.set_device(device)
        store, rendezvous = nccl_store(comm, dist, args.master_addr, args.master_port)
        if rank == 0:
            print(f"NCCL rendezvous: {rendezvous}", flush=True)
        dist.init_process_group(
            "nccl",
            store=store,
            rank=rank,
            world_size=world,
            timeout=datetime.timedelta(seconds=90),
        )
        for count in (8, 16384, 262144):
            send_gpu = torch.full((count,), rank, dtype=torch.float32, device="cuda")
            recv_gpu = torch.empty(world * count, dtype=torch.float32, device="cuda")
            for _repeat in range(args.warmup):
                dist.all_gather_into_tensor(recv_gpu, send_gpu)
            torch.cuda.synchronize()
            comm.Barrier()
            timings = []
            for _repeat in range(args.iterations):
                started = time.perf_counter_ns()
                dist.all_gather_into_tensor(recv_gpu, send_gpu)
                torch.cuda.synchronize()
                timings.append((time.perf_counter_ns() - started) / 1000)
            expected = torch.arange(world, device="cuda", dtype=torch.float32)[:, None]
            if not torch.all(recv_gpu.view(world, count) == expected).item():
                raise RuntimeError("NCCL allgather correctness failure")
            gathered = comm.gather(timings, root=0)
            if rank == 0:
                results.append(
                    {
                        "operation": "NCCL_allgather",
                        "bytes_per_rank": count * 4,
                        "slowest_rank_per_iteration": summarize([max(step) for step in zip(*gathered, strict=True)]),
                    }
                )
        dist.destroy_process_group()
    if rank == 0:
        return {
            "format": 1,
            "world_size": world,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "ranks": metadata,
            "nccl_rendezvous": rendezvous,
            "results": results,
            "note": "Back-to-back microbenchmarks; not TRT-LLM graph replay or fused MegaMoE dispatch.",
        }
    return None


def differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            item
            for key in sorted(left.keys() | right.keys())
            for item in differences(left.get(key), right.get(key), f"{path}.{key}".lstrip("."))
        ]
    return [] if left == right else [{"field": path, "left": left, "right": right}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    capture = sub.add_parser("collect", help="Read-only snapshot; run inside each serving node/container")
    capture.add_argument("--pid", type=int, action="append", default=[])
    capture.add_argument("--model", type=Path)
    capture.add_argument("--duration", type=float, default=20)
    capture.add_argument("--interval", type=float, default=1)
    capture.add_argument("--output", type=Path, required=True)
    bench = sub.add_parser("communication", help="MPI allgather latency, optionally NCCL; use an idle allocation")
    bench.add_argument("--iterations", type=int, default=200)
    bench.add_argument("--warmup", type=int, default=20)
    bench.add_argument("--nccl", action="store_true")
    bench.add_argument("--master-addr")
    bench.add_argument("--master-port", type=int)
    bench.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare", help="Compare two snapshots; timestamps/hostnames may differ naturally")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--section", action="append", help="Compare only these top-level sections")
    args = parser.parse_args()
    if args.mode == "collect" and (args.duration < 0 or args.interval <= 0):
        parser.error("duration must be nonnegative and interval positive")
    if args.mode == "communication":
        if args.iterations <= 0 or args.warmup < 0:
            parser.error("iterations must be positive and warmup nonnegative")
        if args.master_port is not None and not 1 <= args.master_port <= 65535:
            parser.error("--master-port must be in 1..65535; omit it for automatic allocation")
    if args.mode == "compare":
        left, right = json.loads(args.left.read_text()), json.loads(args.right.read_text())
        if args.section:
            left = {key: left.get(key) for key in args.section}
            right = {key: right.get(key) for key in args.section}
        print(json.dumps(differences(left, right), indent=2))
        return
    result = collect(args) if args.mode == "collect" else communication(args)
    if result is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
