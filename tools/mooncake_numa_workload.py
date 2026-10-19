#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Mooncake 0.3.13 cache GETs with verified host-memory NUMA placement.

Run each process under numactl CPU and memory binding. A memfd-backed donor
segment bypasses setup's automatic allocation across all NIC NUMA nodes.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import math
import mmap
import os
import re
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from tools.check_mooncake_rails import atomic_json, read_mailbox_json
from tools.diagnose_mooncake import PAGE_BYTES, health, host_ip, versions

_STOP_REQUESTED = False
_UNRELEASED_RESOURCES: list[Any] = []


def configure_environment(config: dict[str, Any]) -> None:
    """Apply configuration before importing the native store singleton."""
    devices = ",".join(config["nics"])
    os.environ.update(
        MC_MS_AUTO_DISC="0",
        MC_TE_FILTERS=devices,
        MOONCAKE_RDMA_DEVICES=devices,
        MC_STORE_MEMCPY="0",
        GLOG_logtostderr="1",
        GLOG_v="1" if config.get("trace", False) else "0",
        MC_LOG_RDMA_SLICE_AFFINITY="1" if config.get("trace", False) else "0",
    )
    for name in ("MC_ENABLE_HCA_PEER_AFFINITY", "MC_NIC_PEER_AFFINITY", "MC_STORE_USE_HUGEPAGE"):
        os.environ.pop(name, None)
    # 0.3.13 enables its local hot cache only for a positive SIZE value.
    # Remove inherited hot-cache settings so repeated GETs still reach the donor.
    for name in tuple(os.environ):
        if name.startswith("MC_STORE_LOCAL_HOT_"):
            os.environ.pop(name)
    if config.get("affinity", False):
        os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] = "1"
    else:
        os.environ.pop("MC_ENABLE_DEST_DEVICE_AFFINITY", None)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    defaults = dict(
        pool_bytes=4 * 1024**3,
        page_bytes=PAGE_BYTES,
        batch_size=512,
        working_set_batches=2,
        seconds=10,
        warmup_seconds=2,
        trace=False,
        affinity=True,
    )
    config = {**defaults, **config}
    for name in ("pool_bytes", "page_bytes", "batch_size", "working_set_batches"):
        config[name] = int(config[name])
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("seconds", "warmup_seconds"):
        config[name] = float(config[name])
        if not math.isfinite(config[name]) or config[name] < 0 or (name == "seconds" and config[name] == 0):
            raise ValueError(f"Invalid {name}")
    config["numa"] = int(config["numa"])
    if config["numa"] < 0 or not config["nics"] or not all(isinstance(nic, str) for nic in config["nics"]):
        raise ValueError("A nonnegative NUMA node and NIC list are required")
    return config


class MemoryBuffer:
    """Keep the mapping alive until native store cleanup has completed."""

    def __init__(self, size: int, *, shared: bool = False) -> None:
        self.size, self.fd = size, None
        self.registered = False
        self.segment_ids: list[str] = []
        self.path: str | None = None
        if shared:
            self.fd = os.memfd_create("mooncake-numa-pool", os.MFD_CLOEXEC)
            try:
                os.ftruncate(self.fd, size)
                self.mapping = mmap.mmap(self.fd, size, flags=mmap.MAP_SHARED)
            except BaseException:
                os.close(self.fd)
                raise
            self.path = f"/proc/self/fd/{self.fd}"
        else:
            self.mapping = mmap.mmap(-1, size)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        ctypes.memset(self.address, 0, size)

    def close(self) -> None:
        self.mapping.close()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def sample_addresses(address: int, size: int, count: int = 64) -> list[int]:
    page = os.sysconf("SC_PAGE_SIZE")
    first, last = address // page * page, (address + size - 1) // page * page
    pages = (last - first) // page + 1
    samples = min(count, pages)
    return [first + (index * (pages - 1) // max(1, samples - 1)) * page for index in range(samples)]


def query_residency(addresses: list[int], expected_numa: int) -> dict[str, Any]:
    """Query page residency; a null node array requests no page migration."""
    page = os.sysconf("SC_PAGE_SIZE")
    addresses = sorted({int(address) // page * page for address in addresses})
    if not addresses:
        raise ValueError("NUMA residency requires sample pages")
    library = ctypes.CDLL(ctypes.util.find_library("numa") or "libnuma.so.1", use_errno=True)
    move_pages = library.numa_move_pages
    move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    move_pages.restype = ctypes.c_int
    pages = (ctypes.c_void_p * len(addresses))(*addresses)
    statuses = (ctypes.c_int * len(addresses))()
    ctypes.set_errno(0)
    rc = move_pages(0, len(addresses), pages, None, statuses, 0)
    result = {
        "expected_numa": expected_numa,
        "sample_pages": len(addresses),
        "observed_counts": dict(Counter(str(value) for value in statuses)),
        "query_rc": rc,
        "errno": ctypes.get_errno(),
        "addresses": addresses,
        "all_expected": rc == 0 and all(value == expected_numa for value in statuses),
    }
    if not result["all_expected"]:
        raise RuntimeError(f"Physical NUMA residency verification failed: {json.dumps(result)}")
    return result


def buffer_residency(buffer: MemoryBuffer, numa: int) -> dict[str, Any]:
    return query_residency(sample_addresses(buffer.address, buffer.size), numa)


def descriptor_samples(
    store: Any, keys: list[str], expected_endpoint: str, *, query_numa: int | None = None
) -> list[dict[str, Any]]:
    descriptors = store.batch_get_replica_desc(keys)
    records = []
    for key in keys:
        replicas = descriptors.get(key, [])
        if len(replicas) != 1 or not replicas[0].is_memory_replica():
            raise RuntimeError(f"{key}: expected exactly one memory replica, got {len(replicas)}")
        replica = replicas[0]
        if str(replica.status).rsplit(".", 1)[-1] != "COMPLETE":
            raise RuntimeError(f"{key}: replica is not COMPLETE: {replica.status}")
        descriptor = replica.get_memory_descriptor().buffer_descriptor
        if descriptor.transport_endpoint != expected_endpoint:
            raise RuntimeError(
                f"{key}: unexpected owner {descriptor.transport_endpoint!r}; wanted {expected_endpoint!r}"
            )
        record = dict(
            key=key,
            endpoint=descriptor.transport_endpoint,
            address=int(descriptor.buffer_address),
            size=int(descriptor.size),
            status=str(replica.status),
        )
        if record["size"] <= 0:
            raise RuntimeError(f"{key}: invalid object size {record['size']}")
        if query_numa is not None:
            record["residency"] = query_residency(
                [record["address"], record["address"] + record["size"] - 1], query_numa
            )
        records.append(record)
    return records


def keys_for(config: dict[str, Any], batch: int) -> list[str]:
    start = batch * config["batch_size"]
    return [f"{config['key_prefix']}:{index}" for index in range(start, start + config["batch_size"])]


def describe_buffer(buffer: MemoryBuffer, page_bytes: int, count: int) -> tuple[list[list[int]], list[list[int]]]:
    return [[buffer.address + index * page_bytes] for index in range(count)], [[page_bytes] for _ in range(count)]


def validate_payload(buffer: MemoryBuffer, page_bytes: int, indices: list[int]) -> None:
    for slot, index in enumerate(indices):
        expected = bytes([index % 251 + 1]) * page_bytes
        if buffer.mapping[slot * page_bytes : (slot + 1) * page_bytes] != expected:
            raise RuntimeError(f"Corrupt payload for key index {index} in slot {slot}")


def donor_endpoint(config: dict[str, Any]) -> str:
    ready = read_mailbox_json(Path(config["donor_ready"]))
    if ready is None or ready.get("status") != "PASS":
        raise RuntimeError(f"Donor readiness is missing or failed: {ready}")
    return str(ready["endpoint"])


def counter_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    old, new = before.get("ib_counters", {}), after.get("ib_counters", {})
    return {name: new[name] - value for name, value in old.items() if name in new}


def numa_maps(config: dict[str, Any], label: str) -> str:
    output = Path(config["output"])
    path = output.with_name(f"{output.stem}.numa_maps.{label}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(Path("/proc/self/numa_maps").read_text())
    return str(path)


def file_present(path: Path) -> bool:
    try:
        with os.scandir(path.parent) as entries:
            return any(entry.name == path.name for entry in entries)
    except FileNotFoundError:
        return False


def transfer_endpoint(config: dict[str, Any], store: Any) -> str:
    """Read the native TE listener, whose port differs from the Store RPC port.

    Mooncake's own real_rdma_link_failover.sh uses this same startup message.
    The Python store exposes get_hostname(), but no TransferEngine endpoint getter.
    """
    if not config.get("native_log"):
        raise RuntimeError("Donor requires native_log to identify its actual P2P transfer endpoint")
    path = Path(config["native_log"])
    deadline = time.monotonic() + 2
    while True:
        try:
            text = path.read_text(errors="replace")
        except FileNotFoundError:
            text = ""
        listeners = re.findall(r"Transfer Engine RPC using P2P handshake, listening on (\S+)", text)
        if listeners:
            endpoint = listeners[-1]
            host, separator, port = endpoint.rpartition(":")
            expected_host = str(store.get_hostname()).rpartition(":")[0]
            if host != expected_host or not separator or not port.isdigit() or not 1 <= int(port) <= 65535:
                raise RuntimeError(f"Invalid donor transfer endpoint {endpoint!r} for Store host {expected_host!r}")
            return endpoint
        if time.monotonic() >= deadline:
            raise RuntimeError(f"No native P2P transfer-listener startup message in {path}")
        time.sleep(0.05)


def run_donor(config: dict[str, Any], store: Any, buffer: MemoryBuffer) -> dict[str, Any]:
    before = buffer_residency(buffer, config["numa"])
    mounted = store.mount_segment(buffer.path, buffer.size, protocol="rdma", location=f"cpu:{config['numa']}")
    if mounted.get("ret") != 0 or not mounted.get("segment_ids"):
        raise RuntimeError(f"mount_segment failed: {mounted}")
    buffer.segment_ids = list(mounted["segment_ids"])
    ready = dict(
        status="PASS",
        endpoint=transfer_endpoint(config, store),
        store_endpoint=store.get_hostname(),
        endpoint_source=config["native_log"],
        segment_ids=buffer.segment_ids,
        residency_before=before,
        residency_after=buffer_residency(buffer, config["numa"]),
        health=health(),
        numa_maps=numa_maps(config, "ready"),
    )
    atomic_json(Path(config["donor_ready"]), ready)
    seen = set()
    while not _STOP_REQUESTED and not file_present(Path(config["stop_file"])):
        request = read_mailbox_json(Path(config["donor_verify_request"]))
        if request is not None and request.get("id") not in seen:
            request_id = request["id"]
            seen.add(request_id)
            response: dict[str, Any] = {"id": request_id, "status": "PASS", "health": health()}
            try:
                if request["action"] == "verify":
                    expected = request.get("expected_endpoint", ready["endpoint"])
                    if expected != ready["endpoint"]:
                        raise RuntimeError("Verification request names a different donor endpoint")
                    response["descriptor_samples"] = descriptor_samples(
                        store, request["keys"], expected, query_numa=config["numa"]
                    )
                    response["residency"] = buffer_residency(buffer, config["numa"])
                elif request["action"] != "snapshot":
                    raise ValueError(f"Unknown donor request action: {request['action']}")
            except Exception as exc:
                response.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
            atomic_json(Path(config["donor_verify_result"]), response)
        time.sleep(0.05)
    return {**ready, "health_after": health(), "numa_maps_after": numa_maps(config, "after")}


def run_seed(config: dict[str, Any], store: Any, buffer: MemoryBuffer) -> dict[str, Any]:
    before = buffer_residency(buffer, config["numa"])
    pointers, sizes = describe_buffer(buffer, config["page_bytes"], config["batch_size"])
    keys = []
    for batch in range(config["working_set_batches"]):
        batch_keys = keys_for(config, batch)
        for slot in range(config["batch_size"]):
            index = batch * config["batch_size"] + slot
            ctypes.memset(pointers[slot][0], index % 251 + 1, config["page_bytes"])
        results = store.batch_put_from_multi_buffers(batch_keys, pointers, sizes)
        if len(results) != len(batch_keys) or any(value != 0 for value in results):
            raise RuntimeError(f"Cache PUT failed: {results}")
        keys.extend(batch_keys)
    descriptors = descriptor_samples(store, keys, donor_endpoint(config))
    selected = sorted({0, len(keys) // 2, len(keys) - 1})
    return dict(
        status="PASS",
        keys=keys,
        descriptor_samples=[descriptors[index] for index in selected],
        verified_owner_keys=len(descriptors),
        residency_before=before,
        residency_after=buffer_residency(buffer, config["numa"]),
        errors=0,
    )


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * percent / 100
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def run_read(config: dict[str, Any], store: Any, buffer: MemoryBuffer) -> dict[str, Any]:
    before = buffer_residency(buffer, config["numa"])
    page, batch_size = config["page_bytes"], config["batch_size"]
    pointers, sizes = describe_buffer(buffer, page, batch_size)

    def fetch(batch: int, count: int = batch_size) -> None:
        if _STOP_REQUESTED:
            raise InterruptedError("Reader stop requested")
        result = store.batch_get_into_multi_buffers(keys_for(config, batch)[:count], pointers[:count], sizes[:count])
        if len(result) != count or any(value != page for value in result):
            raise RuntimeError(f"Cache GET failed: {result}")

    if config["trace"]:
        count = min(2, batch_size)
        fetch(0, count)
        validate_payload(buffer, page, list(range(count)))
        return dict(
            status="PASS",
            trace=True,
            count=count,
            bytes=count * page,
            errors=0,
            residency_before=before,
            residency_after=buffer_residency(buffer, config["numa"]),
        )

    for batch in range(config["working_set_batches"]):
        fetch(batch)
        validate_payload(buffer, page, list(range(batch * batch_size, (batch + 1) * batch_size)))
    warmup_end = time.perf_counter() + config["warmup_seconds"]
    next_batch = 0
    while time.perf_counter() < warmup_end:
        fetch(next_batch % config["working_set_batches"])
        next_batch += 1
    ib_before = health()
    maps_before = numa_maps(config, "before")
    gc_was_enabled = gc.isenabled()
    gc.disable()
    latencies = []
    last_batch = 0
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < config["seconds"]:
            last_batch = next_batch % config["working_set_batches"]
            tick = time.perf_counter()
            fetch(last_batch)
            latencies.append((time.perf_counter() - tick) * 1000)
            next_batch += 1
    finally:
        elapsed = time.perf_counter() - started
        if gc_was_enabled:
            gc.enable()
    ib_after = health()
    if not latencies or elapsed <= 0:
        raise RuntimeError("No measured GET batches completed")
    validate_payload(buffer, page, list(range(last_batch * batch_size, (last_batch + 1) * batch_size)))
    transferred = len(latencies) * batch_size * page
    return dict(
        status="PASS",
        trace=False,
        bytes=transferred,
        elapsed_s=elapsed,
        goodput_gb_s=transferred / elapsed / 1e9,
        latency_ms={f"p{percent}": percentile(latencies, percent) for percent in (50, 95, 99)},
        samples=latencies,
        count=len(latencies),
        errors=0,
        gc_disabled_during_measurement=True,
        residency_before=before,
        residency_after=buffer_residency(buffer, config["numa"]),
        ib_before=ib_before,
        ib_after=ib_after,
        counter_deltas=counter_deltas(ib_before, ib_after),
        numa_maps_before=maps_before,
        numa_maps_after=numa_maps(config, "after"),
    )


def open_store(config: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    installed = versions()
    if not installed or any(value.split("+", 1)[0].split(".post", 1)[0] != "0.3.13" for value in installed.values()):
        raise RuntimeError(f"Mooncake 0.3.13 is required; installed distributions: {installed}")
    from mooncake.store import MooncakeDistributedStore  # type: ignore[import-not-found]

    store = MooncakeDistributedStore()
    required = (
        "mount_segment",
        "unmount_segment",
        "register_buffer",
        "unregister_buffer",
        "batch_get_replica_desc",
        "batch_get_into_multi_buffers",
        "batch_put_from_multi_buffers",
        "close",
        "get_hostname",
    )
    missing = [name for name in required if not hasattr(store, name)]
    if missing:
        raise RuntimeError(f"Installed Mooncake 0.3.13 wheel lacks required APIs: {missing}")
    rc = store.setup(
        local_hostname=host_ip(config["master"]),
        metadata_server="P2PHANDSHAKE",
        global_segment_size=0,
        local_buffer_size=64 * 1024**2,
        protocol="rdma",
        rdma_devices=",".join(config["nics"]),
        master_server_addr=config["master"],
    )
    if rc != 0:
        store.close()
        raise RuntimeError(f"Mooncake setup failed: {rc}")
    return store, installed


def main(argv: list[str] | None = None) -> int:
    global _STOP_REQUESTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=("donor", "seed", "read"), required=True)
    args = parser.parse_args(argv)
    raw = json.loads(args.config.read_text())
    output = Path(raw["output"])
    result: dict[str, Any] = {"status": "FAIL", "role": args.role, "case_id": raw["case_id"]}
    store = buffer = None
    cleanup_errors = []
    try:
        config = load_config(args.config)
        configure_environment(config)
        result["effective_environment"] = {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(("MC_", "GLOG_", "MOONCAKE_")) or name == "NCCL_IB_GID_INDEX"
        }
        store, installed = open_store(config)
        _STOP_REQUESTED = False

        def request_stop(*_: Any) -> None:
            global _STOP_REQUESTED
            _STOP_REQUESTED = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        result.update(
            versions=installed, endpoint=store.get_hostname(), affinity=config["affinity"], numa=config["numa"]
        )
        if args.role == "donor":
            buffer = MemoryBuffer(config["pool_bytes"], shared=True)
            result.update(run_donor(config, store, buffer))
        else:
            buffer = MemoryBuffer(config["page_bytes"] * config["batch_size"])
            rc = store.register_buffer(buffer.address, buffer.size)
            if rc != 0:
                raise RuntimeError(f"register_buffer failed: {rc}")
            buffer.registered = True
            result.update((run_seed if args.role == "seed" else run_read)(config, store, buffer))
    except Exception as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", errors=1)
        traceback.print_exc()
    finally:
        closed = store is None
        if store is not None:
            if buffer is not None:
                for action, arguments in (
                    ("unmount_segment", (buffer.segment_ids,)) if buffer.segment_ids else ("", ()),
                    ("unregister_buffer", (buffer.address,)) if buffer.registered else ("", ()),
                ):
                    if action:
                        try:
                            rc = getattr(store, action)(*arguments)
                            if rc != 0:
                                cleanup_errors.append(f"{action} returned {rc}")
                        except Exception as exc:
                            cleanup_errors.append(f"{action}: {exc}")
            try:
                rc = store.close()
                result["close_rc"] = rc
                closed = rc == 0
                if not closed:
                    cleanup_errors.append(f"store.close returned {rc}")
            except Exception as exc:
                cleanup_errors.append(f"store.close: {exc}")
        if buffer is not None:
            if closed:
                try:
                    buffer.close()
                except Exception as exc:
                    cleanup_errors.append(f"mapping close: {exc}")
            else:
                _UNRELEASED_RESOURCES.append((store, buffer))
        if cleanup_errors:
            result.update(status="FAIL", cleanup_errors=cleanup_errors, errors=1)
        atomic_json(output, result)
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    sys.exit(main())
