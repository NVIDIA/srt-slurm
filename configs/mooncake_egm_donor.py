#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a native Mooncake EGM Store provider without importing TRT-LLM or MPI.

Requires the EGM Store implementation merged in upstream commit
a2c6c5b701df206f79c768658209385e9098ef55; ordinary Mooncake 0.3.13 lacks it.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib
import importlib.metadata
import json
import logging
import os
import re
import signal
import socket
import sys
import tempfile
import threading
from pathlib import Path
from typing import Protocol, cast

LOG = logging.getLogger("mooncake-egm-donor")
_SIZE_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
_REQUIRED_APIS = (
    "setup",
    "close",
    "register_buffer",
    "unregister_buffer",
    "batch_put_from_multi_buffers",
    "batch_get_into_multi_buffers",
)


class NativeStore(Protocol):
    def setup(self, config: dict[str, str]) -> int: ...

    def close(self) -> int: ...


def configure_environment() -> None:
    """Select the classic, explicitly configured Fabric transport before imports."""
    os.environ.update(
        MC_MS_AUTO_DISC="0",
        MOONCAKE_STORE_BACKEND="cpp",
        MC_USE_TENT="0",
        MC_USE_TEV1="0",
    )
    for name in ("MC_USE_NVLINK_IPC", "MC_TE_FILTERS", "MC_ENABLE_DEST_DEVICE_AFFINITY"):
        os.environ.pop(name, None)


def load_store_type() -> type[NativeStore]:
    """Inspect the native binding, bypassing the optional Python Store facade."""
    try:
        native = importlib.import_module("mooncake._store")
    except ImportError as exc:
        raise RuntimeError("An ARM64 CUDA13 Mooncake wheel with native EGM Store support is required") from exc
    store_type = getattr(native, "MooncakeDistributedStore", None)
    if not callable(store_type):
        raise RuntimeError("Mooncake native binding does not export MooncakeDistributedStore")
    missing = [name for name in _REQUIRED_APIS if not callable(getattr(store_type, name, None))]
    if missing:
        raise RuntimeError(f"Mooncake native Store is missing required APIs: {', '.join(missing)}")
    # Older dictionary setup silently ignores unrecognized keys, so accepting a
    # dict or reporting a newer version is not evidence that EGM is implemented.
    setup_doc = store_type.setup.__doc__ or ""
    if not all(key in setup_doc for key in ("enable_egm_store_pool", "egm_numa_nodes")):
        raise RuntimeError("Mooncake native setup lacks EGM Store support; install a wheel containing PR #3335")
    versions = {}
    for distribution in ("mooncake-transfer-engine-cuda13", "mooncake-transfer-engine"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[distribution] = importlib.metadata.version(distribution)
    LOG.info("Native EGM Store available: module=%s packages=%s", native.__file__, json.dumps(versions, sort_keys=True))
    return cast("type[NativeStore]", store_type)


def check_install() -> None:
    configure_environment()
    load_store_type()
    try:
        cli = importlib.import_module("mooncake.cli")
    except ImportError as exc:
        raise RuntimeError("The EGM wheel must also provide mooncake.cli to launch its matching master") from exc
    if not callable(getattr(cli, "main", None)):
        raise RuntimeError("The EGM wheel does not provide a usable mooncake.cli master launcher")


def require_cuda_devices() -> None:
    """Fail before pool allocation if the provider container has no CUDA device."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise RuntimeError("EGM provider requires the NVIDIA driver and visible GPUs inside its container") from exc
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuDeviceGetCount.restype = ctypes.c_int
    result = cuda.cuInit(0)
    if result != 0:
        raise RuntimeError(f"cuInit failed with CUDA error {result}")
    count = ctypes.c_int()
    result = cuda.cuDeviceGetCount(ctypes.byref(count))
    if result != 0 or count.value <= 0:
        raise RuntimeError(f"EGM provider has no usable CUDA devices: error={result}, count={count.value}")
    # Native EGM setup additionally validates VMM, Fabric handles and HOST_NUMA
    # support, and refuses to allocate if the actual transport is not NVLink.
    LOG.info("Visible CUDA devices: %s", count.value)


def parse_size(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)(B|KiB|MiB|GiB|TiB)?", value)
    if not match:
        raise argparse.ArgumentTypeError("Use a positive byte count or binary size, for example 1040GiB")
    return int(match[1]) * _SIZE_UNITS[match[2] or "B"]


def provider_config(path: Path, segment_size: int, numa_nodes: str) -> dict[str, str]:
    with path.open() as stream:
        config = json.load(stream)
    master = config.get("master_server_address")
    if not isinstance(master, str) or not master.strip():
        raise ValueError(f"{path} must contain the generated master_server_address")
    local_hostname = config.get("local_hostname") or socket.gethostbyname(socket.gethostname())
    if not isinstance(local_hostname, str) or not local_hostname.strip():
        raise ValueError("EGM provider local_hostname must be a nonempty string")
    return {
        "local_hostname": local_hostname,
        "metadata_server": "P2PHANDSHAKE",
        "master_server_addr": master,
        "protocol": "nvlink",
        "rdma_devices": "",
        "global_segment_size": str(segment_size),
        "local_buffer_size": "0",
        "enable_egm_store_pool": "true",
        "egm_numa_nodes": numa_nodes,
    }


def write_ready_file(path: Path, config: dict[str, str]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), **config}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def serve(config_path: Path, segment_size: int, ready_file: Path, numa_nodes: str) -> None:
    configure_environment()
    ready_file.unlink(missing_ok=True)
    config = provider_config(config_path, segment_size, numa_nodes)
    store_type = load_store_type()
    require_cuda_devices()
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        stop.set()

    store = store_type()
    previous_handlers = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    failure: BaseException | None = None
    try:
        LOG.info("EGM provider setup: %s", json.dumps(config, sort_keys=True))
        result = store.setup(config)
        if result != 0:
            raise RuntimeError(f"EGM provider setup failed with code {result}; no RDMA fallback is enabled")
        if not stop.is_set():
            write_ready_file(ready_file, config)
            LOG.info("EGM provider ready: %s", ready_file)
            stop.wait()
    except BaseException as exc:
        failure = exc
        raise
    finally:
        ready_file.unlink(missing_ok=True)
        try:
            result = store.close()
            if result != 0:
                raise RuntimeError(f"EGM provider close failed with code {result}")
        except Exception:
            if failure is None:
                raise
            LOG.exception("EGM provider cleanup also failed")
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check-install", help="Check native EGM APIs without initializing CUDA or connecting to a master"
    )
    provider = commands.add_parser("serve", help="Allocate and serve an EGM-backed Mooncake Store pool")
    provider.add_argument("--config", type=Path, required=True)
    provider.add_argument("--segment-size", type=parse_size, required=True)
    provider.add_argument("--ready-file", type=Path, required=True)
    provider.add_argument("--numa-nodes", default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        if args.command == "check-install":
            check_install()
        else:
            serve(args.config, args.segment_size, args.ready_file, args.numa_nodes)
    except Exception as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
