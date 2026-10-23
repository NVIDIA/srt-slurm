# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise EGM capability, allocation configuration and shutdown without CUDA."""

from __future__ import annotations

import argparse
import ctypes
import json
import signal
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from configs import mooncake_egm_donor as donor

SignalHandler = Callable[[int, object], None] | int | None


class FakeStore:
    def __init__(self) -> None:
        self.config: dict[str, str] = {}
        self.setup_result = 0
        self.close_result = 0
        self.closed = False

    def setup(self, config: dict[str, str]) -> int:
        """enable_egm_store_pool egm_numa_nodes"""
        self.config = config
        return self.setup_result

    def close(self) -> int:
        self.closed = True
        return self.close_result

    register_buffer = Mock()
    unregister_buffer = Mock()
    batch_put_from_multi_buffers = Mock()
    batch_get_into_multi_buffers = Mock()


def test_native_capability_gate_rejects_dict_setup_that_ignores_egm(monkeypatch: pytest.MonkeyPatch) -> None:
    class OldStore(FakeStore):
        def setup(self, config: dict[str, str]) -> int:
            """A dictionary overload that silently ignores unknown keys."""
            return 0

    monkeypatch.setattr(
        donor.importlib, "import_module", lambda name: SimpleNamespace(MooncakeDistributedStore=OldStore)
    )
    with pytest.raises(RuntimeError, match="lacks EGM Store support"):
        donor.load_store_type()


def test_native_binding_bypasses_facade_and_requires_direct_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    def native_module(name: str) -> SimpleNamespace:
        assert name == "mooncake._store"
        return SimpleNamespace(MooncakeDistributedStore=FakeStore, __file__="/wheel/mooncake/_store.so")

    monkeypatch.setattr(donor.importlib, "import_module", native_module)
    assert donor.load_store_type() is FakeStore
    monkeypatch.setattr(FakeStore, "batch_get_into_multi_buffers", None)
    with pytest.raises(RuntimeError, match="missing required APIs: batch_get_into_multi_buffers"):
        donor.load_store_type()


def test_environment_prevents_ipc_and_rdma_affinity_from_leaking(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MC_USE_NVLINK_IPC", "MC_TE_FILTERS", "MC_ENABLE_DEST_DEVICE_AFFINITY"):
        monkeypatch.setenv(name, "1")
    for name in ("MC_MS_AUTO_DISC", "MC_USE_TENT", "MC_USE_TEV1"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("MOONCAKE_STORE_BACKEND", "rs")
    donor.configure_environment()
    for name in ("MC_USE_NVLINK_IPC", "MC_TE_FILTERS", "MC_ENABLE_DEST_DEVICE_AFFINITY"):
        assert name not in donor.os.environ
    for name in ("MC_MS_AUTO_DISC", "MC_USE_TENT", "MC_USE_TEV1"):
        assert donor.os.environ[name] == "0"
    assert donor.os.environ["MOONCAKE_STORE_BACKEND"] == "cpp"


@pytest.mark.parametrize("value, expected", [("1040GiB", 1040 * 1024**3), ("1100GiB", 1100 * 1024**3), ("4096", 4096)])
def test_segment_size(value: str, expected: int) -> None:
    assert donor.parse_size(value) == expected


@pytest.mark.parametrize("value", ["0", "-1GiB", "1.5GiB", "1040GB", "bogus"])
def test_reject_invalid_segment_size(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        donor.parse_size(value)


def test_config_uses_generated_master_but_replaces_consumer_settings(tmp_path: Path) -> None:
    path = tmp_path / "consumer.json"
    path.write_text(
        json.dumps(
            {
                "master_server_address": "192.0.2.1:8700",
                "local_hostname": "192.0.2.2",
                "protocol": "rdma",
                "local_buffer_size": 1024,
            }
        )
    )
    config = donor.provider_config(path, 1040 * 1024**3, "auto")
    assert config == {
        "master_server_addr": "192.0.2.1:8700",
        "local_hostname": "192.0.2.2",
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "nvlink",
        "rdma_devices": "",
        "global_segment_size": str(1040 * 1024**3),
        "local_buffer_size": "0",
        "enable_egm_store_pool": "true",
        "egm_numa_nodes": "auto",
    }


def test_config_falls_back_to_provider_ip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "consumer.json"
    path.write_text('{"master_server_address": "192.0.2.1:8700"}')
    monkeypatch.setattr(donor.socket, "gethostbyname", lambda host: "192.0.2.3")
    assert donor.provider_config(path, 1024**3, "0,1")["local_hostname"] == "192.0.2.3"
    path.write_text("{}")
    with pytest.raises(ValueError, match="master_server_address"):
        donor.provider_config(path, 1024**3, "auto")


@pytest.mark.parametrize("count", [0, 4])
def test_cuda_visibility_gate(count: int, monkeypatch: pytest.MonkeyPatch) -> None:
    def get_count(pointer: ctypes.c_void_p) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = count
        return 0

    cuda = SimpleNamespace(cuInit=Mock(return_value=0), cuDeviceGetCount=Mock(side_effect=get_count))
    monkeypatch.setattr(donor.ctypes, "CDLL", lambda name: cuda)
    if count:
        donor.require_cuda_devices()
    else:
        with pytest.raises(RuntimeError, match="no usable CUDA devices"):
            donor.require_cuda_devices()


def service_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeStore, Path, Path, dict[int, SignalHandler]]:
    path = tmp_path / "consumer.json"
    path.write_text('{"master_server_address": "192.0.2.1:8700", "local_hostname": "192.0.2.2"}')
    ready = tmp_path / "donor.ready"
    ready.write_text("stale")
    store = FakeStore()
    handlers: dict[int, SignalHandler] = {}

    def set_handler(signum: int, handler: SignalHandler) -> SignalHandler:
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(donor, "configure_environment", lambda: None)
    monkeypatch.setattr(donor, "load_store_type", lambda: lambda: store)
    monkeypatch.setattr(donor, "require_cuda_devices", lambda: None)
    monkeypatch.setattr(donor.signal, "signal", set_handler)

    class StopEvent:
        stopped = False

        def set(self) -> None:
            self.stopped = True

        def is_set(self) -> bool:
            return self.stopped

        def wait(self) -> None:
            payload = json.loads(ready.read_text())
            assert payload["enable_egm_store_pool"] == "true"
            assert payload["protocol"] == "nvlink"
            assert not store.closed
            handler = handlers[signal.SIGTERM]
            assert handler is not None and not isinstance(handler, int)
            handler(signal.SIGTERM, None)
            assert self.stopped

    monkeypatch.setattr(donor.threading, "Event", StopEvent)
    return store, path, ready, handlers


def test_provider_becomes_ready_then_closes_on_sigterm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, path, ready, handlers = service_fixture(tmp_path, monkeypatch)
    donor.serve(path, 1040 * 1024**3, ready, "auto")
    assert store.closed
    assert not ready.exists()
    assert handlers == {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL}
    assert not list(tmp_path.glob(".donor.ready.*"))


def test_failed_setup_closes_and_never_leaves_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, path, ready, handlers = service_fixture(tmp_path, monkeypatch)
    store.setup_result = -5
    with pytest.raises(RuntimeError, match="setup failed with code -5"):
        donor.serve(path, 1040 * 1024**3, ready, "auto")
    assert store.closed
    assert not ready.exists()
    assert handlers[signal.SIGTERM] == signal.SIG_DFL


def test_close_failure_is_nonzero_and_removes_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, path, ready, handlers = service_fixture(tmp_path, monkeypatch)
    store.close_result = -7
    with pytest.raises(RuntimeError, match="close failed with code -7"):
        donor.serve(path, 1040 * 1024**3, ready, "auto")
    assert not ready.exists()
    assert handlers[signal.SIGTERM] == signal.SIG_DFL


def test_install_check_never_initializes_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(donor, "configure_environment", Mock())
    monkeypatch.setattr(donor, "load_store_type", Mock(return_value=FakeStore))
    cuda = Mock(side_effect=AssertionError("check-install must not initialize CUDA"))
    monkeypatch.setattr(donor, "require_cuda_devices", cuda)
    monkeypatch.setattr(donor.importlib, "import_module", lambda name: SimpleNamespace(main=lambda: None))
    assert donor.main(["check-install"]) == 0
    cuda.assert_not_called()


def test_install_check_requires_matching_master_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(donor, "configure_environment", Mock())
    monkeypatch.setattr(donor, "load_store_type", Mock(return_value=FakeStore))
    monkeypatch.setattr(donor.importlib, "import_module", Mock(side_effect=ImportError("missing mooncake.cli")))
    assert donor.main(["check-install"]) == 1
