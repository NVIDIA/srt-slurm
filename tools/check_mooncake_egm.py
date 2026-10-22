#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify remote EGM Store transfers using registered pinned host buffers.

Native APIs and local host-buffer support were checked against Mooncake commit
042ab8a18b0e6ccfe87782261a5556a3dd4446e4. This is a correctness diagnostic,
not an inference or bandwidth qualification benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import socket
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

LOG = logging.getLogger("check-mooncake-egm")
UPSTREAM_COMMIT = "042ab8a18b0e6ccfe87782261a5556a3dd4446e4"


@dataclass(frozen=True)
class ClientConfig:
    master: str
    run_dir: Path
    provider_hostname: str
    gpu_ids: tuple[int, ...]
    sizes: tuple[int, ...]
    iterations: int
    warmup: int


def write_json(path: Path, value: Any) -> None:
    """Preserve completed samples even if a later native call hangs or fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def parse_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from exc
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Use unique, nonnegative comma-separated integers")
    return values


def client_store_config(master: str, local_hostname: str) -> dict[str, str]:
    return {
        "local_hostname": local_hostname,
        "metadata_server": "P2PHANDSHAKE",
        "master_server_addr": master,
        "protocol": "nvlink",
        "rdma_devices": "",
        "global_segment_size": "0",
        "local_buffer_size": "0",
        "enable_egm_store_pool": "false",
    }


def load_cuda_runtime() -> Any:
    """Use CUDA Python's public pointer-attribute API, without ctypes layouts."""
    try:
        return importlib.import_module("cuda.bindings.runtime")
    except ModuleNotFoundError:
        return importlib.import_module("cuda.cudart")


def host_buffer_evidence(tensor: Any, cudart: Any, gpu_id: int, role: str, size: int) -> dict[str, Any]:
    """Reject device, managed, pageable or otherwise unverified payload memory."""
    if tensor.device.type != "cpu" or not tensor.is_pinned():
        raise RuntimeError(f"Submission GPU {gpu_id} {role}: payload buffer must be pinned CPU memory")
    pointer = tensor.data_ptr()
    # cuda.bindings 13.0.1 documents the (status, cudaPointerAttributes)
    # return value and its `type` field; no CUDA struct layout is assumed.
    status, attributes = cudart.cudaPointerGetAttributes(pointer)
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Submission GPU {gpu_id} {role}: cudaPointerGetAttributes failed: {status}")
    if attributes.type != cudart.cudaMemoryType.cudaMemoryTypeHost:
        raise RuntimeError(
            f"Submission GPU {gpu_id} {role}: CUDA pointer must have cudaMemoryTypeHost, received {attributes.type}"
        )
    return {
        "gpu_id": gpu_id,
        "role": role,
        "pointer": pointer,
        "size_bytes": size,
        "device": tensor.device.type,
        "pinned": True,
        "cuda_memory_type": "cudaMemoryTypeHost",
    }


def host_addresses(host: str) -> set[str]:
    return {str(row[4][0]) for row in socket.getaddrinfo(host, None, family=socket.AF_INET)}


def validate_replicas(
    replicas: list[Any],
    provider_addresses: set[str],
    client_addresses: set[str],
    expected_size: int,
) -> list[dict[str, Any]]:
    """Confirm Store metadata places every replica on the other node's memory."""
    if not replicas:
        raise RuntimeError("No Store replica metadata returned after successful put")
    evidence = []
    for replica in replicas:
        status = str(replica.status)
        if status.rsplit(".", 1)[-1] != "COMPLETE" or not replica.is_memory_replica():
            raise RuntimeError(f"Expected a COMPLETE memory replica, received status={status}")
        descriptor = replica.get_memory_descriptor().buffer_descriptor
        endpoint = descriptor.transport_endpoint
        host, separator, port = endpoint.rpartition(":")
        if not separator or not port.isdigit():
            raise RuntimeError(f"Unexpected replica transport endpoint: {endpoint!r}")
        addresses = host_addresses(host)
        if addresses & client_addresses or not addresses & provider_addresses:
            raise RuntimeError(f"Replica {endpoint!r} is not exclusively on the remote provider")
        if descriptor.size != expected_size:
            raise RuntimeError(f"Replica size {descriptor.size} differs from payload size {expected_size}")
        evidence.append(
            {
                "status": status,
                "is_memory_replica": True,
                "size_bytes": descriptor.size,
                "buffer_address": descriptor.buffer_address,
                "transport_endpoint": endpoint,
                "resolved_addresses": sorted(addresses),
            }
        )
    return evidence


def percentile(values: list[float], percent: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    cases = sorted({(sample["gpu_id"], sample["size_bytes"]) for sample in samples if not sample["warmup"]})
    for gpu_id, size in cases:
        measured = [
            sample
            for sample in samples
            if not sample["warmup"] and sample["gpu_id"] == gpu_id and sample["size_bytes"] == size
        ]
        for operation in ("put", "get"):
            latencies = [sample[f"{operation}_seconds"] for sample in measured]
            rows.append(
                {
                    "gpu_id": gpu_id,
                    "size_bytes": size,
                    "operation": operation,
                    "iterations": len(measured),
                    "latency_ms": {f"p{p}": percentile(latencies, p) * 1000 for p in (50, 95, 99)},
                    "payload_GiB_per_second": size * len(measured) / sum(latencies) / 1024**3,
                }
            )
    return rows


def transfer_case(
    store: Any,
    torch: Any,
    config: ClientConfig,
    result: dict[str, Any],
    buffers: tuple[Any, Any],
    gpu_id: int,
    size: int,
    provider_addresses: set[str],
    client_addresses: set[str],
) -> None:
    torch.cuda.set_device(gpu_id)
    source, destination = (tensor[:size] for tensor in buffers)
    active_keys: set[str] = set()
    failure: BaseException | None = None
    try:
        for iteration in range(config.warmup + config.iterations):
            key = f"egm-smoke-{result['test_id']}-gpu{gpu_id}-{size}-{iteration}"
            # A seeded full-length random pattern detects stale data and moved
            # aligned chunks; a repeated short byte pattern could hide both.
            seed = (size * 31 + gpu_id * 1_000_003 + iteration * 17) % (2**63 - 1)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            source.random_(0, 256, generator=generator)
            destination.fill_(255)
            torch.cuda.synchronize(gpu_id)
            active_keys.add(key)
            start = time.perf_counter()
            put_status = store.batch_put_from_multi_buffers([key], [[source.data_ptr()]], [[size]])
            torch.cuda.synchronize(gpu_id)
            put_seconds = time.perf_counter() - start
            if put_status != [0]:
                raise RuntimeError(f"GPU {gpu_id}, {size} B, iteration {iteration}: put failed: {put_status}")
            evidence = validate_replicas(store.get_replica_desc(key), provider_addresses, client_addresses, size)
            start = time.perf_counter()
            get_status = store.batch_get_into_multi_buffers([key], [[destination.data_ptr()]], [[size]])
            torch.cuda.synchronize(gpu_id)
            get_seconds = time.perf_counter() - start
            if get_status != [size]:
                raise RuntimeError(f"GPU {gpu_id}, {size} B, iteration {iteration}: get failed: {get_status}")
            if not torch.equal(source, destination):
                raise RuntimeError(f"GPU {gpu_id}, {size} B, iteration {iteration}: full-payload verification failed")
            sample = {
                "gpu_id": gpu_id,
                "size_bytes": size,
                "iteration": iteration,
                "warmup": iteration < config.warmup,
                "payload_seed": seed,
                "key": key,
                "put_seconds": put_seconds,
                "get_seconds": get_seconds,
                "verified_bytes": size,
                "replicas": evidence,
            }
            result["samples"].append(sample)
            write_json(config.run_dir / "logs/client-results.json", result)
            LOG.info("GPU %d %d B iteration %d: verified remote put/get", gpu_id, size, iteration)
            # This run owns every UUID-prefixed key; transfers have completed,
            # so bypass the master's post-query lease before reclaiming it.
            status = store.remove(key, force=True)
            if status != 0:
                raise RuntimeError(f"Key removal failed for {key}: {status}")
            active_keys.remove(key)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_errors = []
        for key in active_keys:
            try:
                status = store.remove(key, force=True)
                if status != 0:
                    cleanup_errors.append(f"remove {key}: {status}")
            except Exception as exc:
                cleanup_errors.append(f"remove {key}: {exc}")
        if cleanup_errors:
            message = "; ".join(cleanup_errors)
            if failure is None:
                raise RuntimeError(f"Host buffer/key cleanup failed: {message}")
            LOG.error("Host buffer/key cleanup also failed: %s", message)


def run_client(config: ClientConfig) -> int:
    result: dict[str, Any] = {
        "status": "running",
        "test_id": uuid.uuid4().hex,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "provider_hostname": config.provider_hostname,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "platform": platform.platform(),
        "upstream_api_commit": UPSTREAM_COMMIT,
        "gpu_ids": list(config.gpu_ids),
        "sizes_bytes": list(config.sizes),
        "iterations_per_case": config.iterations,
        "warmup_per_case": config.warmup,
        "buffer_memory": "pinned_host",
        "buffers": [],
        "samples": [],
    }
    path = config.run_dir / "logs/client-results.json"
    write_json(path, result)
    store: Any = None
    buffers: dict[int, tuple[Any, Any]] = {}
    start = time.perf_counter()
    try:
        if config.provider_hostname == socket.gethostname():
            raise RuntimeError("Client and provider must run on different nodes")
        provider_addresses = host_addresses(config.provider_hostname)
        client_addresses = host_addresses(socket.gethostname())
        if provider_addresses & client_addresses:
            raise RuntimeError("Provider and client hostname addresses overlap")
        provider_path = config.run_dir / "logs/provider.ready.json"
        provider = json.loads(provider_path.read_text())
        if (
            provider["protocol"] != "nvlink"
            or provider["enable_egm_store_pool"] != "true"
            or int(provider["global_segment_size"]) <= 0
            or int(provider["local_buffer_size"]) != 0
        ):
            raise RuntimeError("Provider readiness metadata does not describe an NVLink EGM memory pool")
        provider_addresses |= host_addresses(provider["local_hostname"].split(":", 1)[0])
        result["provider"] = provider
        # Reuse the provider's native feature check and transport environment.
        repository_root = str(Path(__file__).resolve().parents[1])
        if repository_root not in sys.path:
            sys.path.insert(0, repository_root)
        donor = importlib.import_module("configs.mooncake_egm_donor")
        donor.configure_environment()
        torch = importlib.import_module("torch")
        cudart = load_cuda_runtime()
        store_type = donor.load_store_type()
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        if max(config.gpu_ids) >= torch.cuda.device_count():
            raise RuntimeError(f"Requested GPUs {config.gpu_ids}, visible count is {torch.cuda.device_count()}")
        result["torch_version"] = str(torch.__version__)
        result["cuda_version"] = torch.version.cuda
        result["gpu_names"] = {str(gpu): torch.cuda.get_device_name(gpu) for gpu in config.gpu_ids}
        result["wheel_versions"] = {}
        for name in ("mooncake-transfer-engine-cuda13", "mooncake-transfer-engine"):
            with contextlib.suppress(importlib.metadata.PackageNotFoundError):
                result["wheel_versions"][name] = importlib.metadata.version(name)
        native_config = client_store_config(config.master, socket.gethostbyname(socket.gethostname()))
        result["client_config"] = native_config
        LOG.info("Client setup: %s", json.dumps(native_config, sort_keys=True))
        store = cast(Any, store_type())
        status = store.setup(native_config)
        if status != 0:
            raise RuntimeError(f"NVLink client setup failed: {status}; no fallback transport is enabled")
        result["client_transport_endpoint"] = store.get_hostname()
        write_json(path, result)
        for gpu_id in config.gpu_ids:
            torch.cuda.set_device(gpu_id)
            # Pinned host regions are initiator-only local endpoints. Upstream
            # accepts them without publishing Fabric metadata, so
            # unregister_buffer cannot remove a corresponding metadata entry.
            # Register once, reuse subranges and retain tensors through close()
            # instead of unregistering/reallocating overlapping addresses.
            source = torch.empty(max(config.sizes), dtype=torch.uint8, device="cpu", pin_memory=True)
            destination = torch.empty(max(config.sizes), dtype=torch.uint8, device="cpu", pin_memory=True)
            buffers[gpu_id] = (source, destination)
            for role, tensor in zip(("source", "destination"), buffers[gpu_id], strict=True):
                evidence = host_buffer_evidence(tensor, cudart, gpu_id, role, max(config.sizes))
                result["buffers"].append(evidence)
                write_json(path, result)
                status = store.register_buffer(tensor.data_ptr(), max(config.sizes))
                if status != 0:
                    raise RuntimeError(f"GPU {gpu_id}: register_buffer failed: {status}")
            for size in config.sizes:
                transfer_case(
                    store, torch, config, result, buffers[gpu_id], gpu_id, size, provider_addresses, client_addresses
                )
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        LOG.exception("EGM diagnostic failed")
    finally:
        if store is not None:
            try:
                status = store.close()
                if status != 0:
                    raise RuntimeError(f"Store close failed: {status}")
            except Exception as exc:
                result["status"] = "failed"
                result["cleanup_error"] = str(exc)
                LOG.exception("Store cleanup failed")
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = time.perf_counter() - start
        result["summary"] = summarize(result["samples"])
        write_json(path, result)
    exit_code = 0 if result["status"] == "passed" else 1
    write_report(config.run_dir, exit_code)
    return exit_code


def write_report(run_dir: Path, exit_code: int) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    result_path = logs / "client-results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    client_log = logs / "client.log"
    transport_log_verified = client_log.exists() and "NvlinkTransport: using cudaMemcpyBatchAsync" in (
        client_log.read_text(errors="replace")
    )
    samples = result.get("samples", [])
    expected_samples = (
        len(result.get("gpu_ids", []))
        * len(result.get("sizes_bytes", []))
        * (result.get("iterations_per_case", 0) + result.get("warmup_per_case", 0))
    )
    measurements_complete = (
        bool(samples)
        and len(samples) == expected_samples
        and all(sample.get("verified_bytes") == sample["size_bytes"] and sample.get("replicas") for sample in samples)
    )
    buffers = result.get("buffers", [])
    expected_buffers = {(gpu, role) for gpu in result.get("gpu_ids", []) for role in ("source", "destination")}
    host_buffers_verified = (
        result.get("buffer_memory") == "pinned_host"
        and bool(expected_buffers)
        and len(buffers) == len(expected_buffers)
        and {(buffer.get("gpu_id"), buffer.get("role")) for buffer in buffers} == expected_buffers
        and all(
            buffer.get("device") == "cpu"
            and buffer.get("pinned") is True
            and buffer.get("cuda_memory_type") == "cudaMemoryTypeHost"
            and buffer.get("pointer", 0) > 0
            and buffer.get("size_bytes", 0) >= max(result.get("sizes_bytes", []), default=1)
            for buffer in buffers
        )
    )
    passed = (
        exit_code == 0
        and result.get("status") == "passed"
        and transport_log_verified
        and measurements_complete
        and host_buffers_verified
    )
    measured = [sample for sample in samples if not sample["warmup"]]
    rows = summarize(samples)
    lines = [
        "# Mooncake remote EGM diagnostic",
        "",
        f"Status: **{'PASS' if passed else 'FAIL / INCOMPLETE'}**; launcher exit code: `{exit_code}`.",
        f"Native NVLink memcpy log observed: `{transport_log_verified}` (source: `client.log`).",
        f"Requested verified samples complete: `{measurements_complete}` (source: `client-results.json`).",
        f"Pinned host source/destination buffers verified: `{host_buffers_verified}` "
        "(source: `client-results.json` CUDA pointer attributes).",
        f"Report generated: {datetime.now(timezone.utc).isoformat()}.",
        "",
        "## Goal and controls",
        "",
        "Test whether pinned host buffers can put/get and fully verify data in the other node's EGM host-memory "
        "Store pool using the explicitly selected Mooncake NVLink transport. The client owns zero global Store "
        "capacity and zero internal Store local-buffer capacity. Each returned replica must be COMPLETE memory on "
        "the provider node. Both client payload buffers are pinned CPU tensors and must have CUDA pointer type "
        "`cudaMemoryTypeHost`; managed and GPU payload buffers are rejected. GPU IDs select CUDA submission "
        "devices only. CUDA contexts, streams and mapping metadata may still use GPU resources; this test does "
        "not require zero driver/context HBM consumption.",
        "",
        "Baseline: none; this reduced correctness diagnostic does not compare transports or enforce a bandwidth SLO. "
        "The small payload sweep measures synchronous API latency, including RPC/control overhead; it does not "
        "measure NVLink saturation. No RDMA/TCP data transport is requested. TCP control-plane RPC is still used.",
        "",
        "## Configuration and hardware",
        "",
        f"- Slurm job: `{result.get('job_id', os.environ.get('SLURM_JOB_ID', 'unknown'))}`; two allocated nodes.",
        f"- Client: `{result.get('hostname', 'not reached')}`; provider: "
        f"`{result.get('provider_hostname', 'see provider/inventory logs')}`.",
        f"- CUDA submission GPU IDs/names: `{result.get('gpu_ids', [])}` / `{result.get('gpu_names', {})}`.",
        f"- Payload allocation: `{result.get('buffer_memory', 'unverified')}`; "
        f"verified buffer records: `{len(buffers)}` / `{len(expected_buffers)}`.",
        f"- PyTorch/CUDA: `{result.get('torch_version', 'unavailable')}` / "
        f"`{result.get('cuda_version', 'unavailable')}`; wheel packages: `{result.get('wheel_versions', {})}`.",
        f"- Payload sizes (B): `{result.get('sizes_bytes', [])}`; concurrency: one synchronous transfer; "
        f"warmup/measurement iterations per GPU/size: `{result.get('warmup_per_case', 'unknown')}` / "
        f"`{result.get('iterations_per_case', 'unknown')}`.",
        f"- Client window UTC: `{result.get('started_utc', 'not reached')}` to "
        f"`{result.get('finished_utc', 'incomplete')}`; wall duration: "
        f"`{result.get('duration_seconds', 'unavailable')}` s.",
        "",
        "## Measurements",
        "",
        "TTFT p50/p95/p99 (ms), ITL p50/p99 (ms/token), input/output/total throughput (tokens/s), "
        "and agentic Pareto/SLO metrics: **N/A**. This workload transfers bytes and runs no inference requests, "
        "model, tokenizer or agentic conversations.",
        "",
        f"Fully verified transfers: {len(samples)} put/get pairs including warmup; {len(measured)} measured pairs. "
        "Every successful sample compares the complete CPU destination with the deterministic CPU source. "
        "Payload generation and equality verification are outside timed intervals.",
        "",
    ]
    if rows:
        lines += [
            "| GPU | Payload B | API | Samples | p50 ms | p95 ms | p99 ms | Payload GiB/s |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            latency = row["latency_ms"]
            lines.append(
                f"| {row['gpu_id']} | {row['size_bytes']} | {row['operation']} | {row['iterations']} | "
                f"{latency['p50']:.3f} | {latency['p95']:.3f} | {latency['p99']:.3f} | "
                f"{row['payload_GiB_per_second']:.3f} |"
            )
        lines += [
            "",
            "Percentiles use linear interpolation across measured API latency samples. Payload GiB/s is "
            "successful payload bytes divided by summed API latency, using 2^30 B/GiB; it excludes warmup.",
        ]
    else:
        lines += ["Transfer latency and byte throughput: unavailable; no measured successful sample was preserved."]
    lines += [
        "",
        "## Evidence and interpretation",
        "",
        (
            "The configured NVLink path completed remote memory put/get operations with full payload verification "
            "for every requested submission GPU/size/iteration. CUDA pointer attributes establish pinned host "
            "source/destination allocations; replica descriptors prove remote memory placement. The native "
            "provider's EGM setup validates HOST_NUMA/Fabric support. This establishes this topology's functional "
            "path; it does not qualify TRT-LLM end-to-end behavior or peak throughput."
            if passed
            else "This run did not establish that all requested remote EGM transfers work. Inspect the failure and "
            "native provider/client logs before changing allocator, driver, topology or wheel configuration. "
            "Any preserved measurements describe completed cases only."
        ),
        "",
        f"Client error: `{result.get('error', 'none recorded; inspect launcher logs if incomplete')}`. "
        f"Cleanup error: `{result.get('cleanup_error', 'none recorded')}`.",
        "",
        "- [client-results.json](client-results.json): exact setup, package versions, submission GPU identities, "
        "pinned host allocation/pointer-type evidence, timestamps, "
        "per-iteration `samples` latencies/verified bytes and replica memory endpoints; raw source for this table.",
        "- [client.log](client.log), [provider.log](provider.log), [master.log](master.log): native setup, "
        "HOST_NUMA/Fabric allocation and transport errors. Missing files indicate setup did not reach that phase.",
        "- Inventory/configuration artifacts in this logs directory record node topology, GPU Fabric health, "
        "NUMA placement, wheel identity and launch controls; inspect these with the failure logs.",
        "- No DSight report or dashboard was collected: byte-transfer correctness and native replica metadata "
        "answer this diagnostic's goal without request traces.",
        "- The [slurm-job-sizing skill](https://github.com/NVIDIA/srt-slurm/blob/main/"
        "src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md) guided the short two-node allocation and reduced "
        "warmup/payload sweep; see the sbatch script for the whole-job budget. No full inference benchmark "
        "or campaign skill was used.",
        f"- [Mooncake native source at {UPSTREAM_COMMIT[:12]}](https://github.com/kvcache-ai/Mooncake/tree/"
        f"{UPSTREAM_COMMIT}): `mooncake-integration/store/store_py.cpp` supplies the native APIs; "
        "`mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp` accepts local host "
        "endpoints and copies directly to/from imported Fabric mappings. The descriptor API exposes a memory "
        "endpoint, not a transport-protocol or "
        "HOST_NUMA field; the transport/EGM evidence comes from strict native setup and logs.",
        "- [CUDA Python 13.0.1 pointer-attribute API](https://nvidia.github.io/cuda-python/cuda-bindings/"
        "13.0.1/module/runtime.html#cuda.bindings.runtime.cudaPointerGetAttributes) defines the status/attributes "
        "return pair and `cudaMemoryTypeHost` used to verify payload residence.",
        "",
    ]
    (logs / "perf-analysis.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    client = commands.add_parser("client", help="Verify pinned host to remote EGM Store transfers")
    client.add_argument("--master", required=True)
    client.add_argument("--run-dir", type=Path, required=True)
    client.add_argument("--provider-hostname", required=True)
    client.add_argument(
        "--gpu-ids", type=parse_csv, default=(0, 1, 2, 3), help="CUDA submission devices, not payload memory"
    )
    client.add_argument("--sizes", type=parse_csv, default=(65536, 1048576, 16777216, 67108864))
    client.add_argument("--iterations", type=int, default=10)
    client.add_argument("--warmup", type=int, default=2)
    report = commands.add_parser("report", help="Write the authoritative report, including setup failures")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if args.command == "report":
        write_report(args.run_dir, args.exit_code)
        return 0
    if args.iterations <= 0 or args.warmup < 0 or min(args.sizes) <= 0:
        parser.error("iterations and payload sizes must be positive; warmup must be nonnegative")
    return run_client(
        ClientConfig(
            args.master, args.run_dir, args.provider_hostname, args.gpu_ids, args.sizes, args.iterations, args.warmup
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
