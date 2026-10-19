# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store RPC identity must never substitute for the actual RDMA peer endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import mooncake_numa_workload as workload


def test_donor_publishes_native_transfer_endpoint_when_store_rpc_port_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "donor.native.log"
    native.write_text("I1019 Transfer Engine RPC using P2P handshake, listening on 172.29.110.5:16144\n")
    stop = tmp_path / "stop"
    stop.touch()
    config = {
        "numa": 0,
        "native_log": str(native),
        "stop_file": str(stop),
        "donor_ready": str(tmp_path / "donor.ready.json"),
    }
    store = SimpleNamespace(
        get_hostname=lambda: "172.29.110.5:13395",
        mount_segment=lambda *args, **kwargs: {"ret": 0, "segment_ids": ["segment-id"]},
    )
    buffer = SimpleNamespace(path="/proc/self/fd/7", size=65536, segment_ids=[])
    monkeypatch.setattr(workload, "buffer_residency", lambda *args: {"all_expected": True})
    monkeypatch.setattr(workload, "health", lambda: {})
    monkeypatch.setattr(workload, "numa_maps", lambda *args: "numa-maps.txt")

    result = workload.run_donor(config, store, buffer)
    ready = json.loads(Path(config["donor_ready"]).read_text())

    assert result["endpoint"] == ready["endpoint"] == "172.29.110.5:16144"
    assert ready["store_endpoint"] == "172.29.110.5:13395"
    assert ready["endpoint_source"] == str(native)
    assert ready["segment_ids"] == ["segment-id"]


def test_transfer_endpoint_requires_native_log_instead_of_guessing_from_store_hostname() -> None:
    store = SimpleNamespace(get_hostname=lambda: "172.29.110.5:13395")
    with pytest.raises(RuntimeError, match="requires native_log"):
        workload.transfer_endpoint({}, store)


@pytest.mark.parametrize("endpoint", ["172.29.110.6:16144", "172.29.110.5:0", "172.29.110.5:bad"])
def test_transfer_endpoint_rejects_another_host_or_invalid_port(tmp_path: Path, endpoint: str) -> None:
    native = tmp_path / "donor.native.log"
    native.write_text(f"Transfer Engine RPC using P2P handshake, listening on {endpoint}\n")
    store = SimpleNamespace(get_hostname=lambda: "172.29.110.5:13395")
    with pytest.raises(RuntimeError, match="Invalid donor transfer endpoint"):
        workload.transfer_endpoint({"native_log": str(native)}, store)


def test_transfer_endpoint_fails_when_no_native_listener_message_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "donor.native.log"
    native.write_text("A store RPC port is not transfer-listener evidence\n")
    timestamps = iter([0.0, 3.0])
    monkeypatch.setattr(workload.time, "monotonic", lambda: next(timestamps))
    store = SimpleNamespace(get_hostname=lambda: "172.29.110.5:13395")
    with pytest.raises(RuntimeError, match="No native P2P transfer-listener"):
        workload.transfer_endpoint({"native_log": str(native)}, store)


def test_inherited_local_hot_cache_is_disabled_before_native_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MC_STORE_LOCAL_HOT_CACHE_SIZE", "4294967296")
    monkeypatch.setenv("MC_STORE_LOCAL_HOT_CACHE_USE_SHM", "1")
    workload.configure_environment({"nics": ["mlx5_0"], "affinity": True})
    assert "MC_STORE_LOCAL_HOT_CACHE_SIZE" not in workload.os.environ
    assert "MC_STORE_LOCAL_HOT_CACHE_USE_SHM" not in workload.os.environ
    assert workload.os.environ["MC_STORE_MEMCPY"] == "0"
