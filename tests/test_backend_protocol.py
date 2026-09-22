# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every backend answers the BackendProtocol questions the stages ask; nothing duck-types a backend.

The stage mixins, schema validators, services, and dry-run read optional
features (Mooncake, failover, batched startup) through the protocol. A backend
without the feature says so with None or an empty dict, so consumers never
probe with getattr or hasattr.
"""

import re
from pathlib import Path

import pytest

from srtctl.backends import (
    MockerProtocol,
    MooncakeKVStoreConfig,
    SGLangProtocol,
    TRTLLMProtocol,
    VLLMFailoverConfig,
    VLLMProtocol,
)
from srtctl.core.topology import Process
from srtctl.ports import MOONCAKE_MASTER_PORT

BACKENDS = [SGLangProtocol, TRTLLMProtocol, VLLMProtocol, MockerProtocol]
SRC = Path(__file__).resolve().parents[1] / "src" / "srtctl"
DUCK_TYPED_BACKEND = re.compile(r"\b(getattr|hasattr)\((self\.|config\.|self\.config\.)?backend\b")


def _process() -> Process:
    return Process(
        node="n0",
        gpu_indices=frozenset([0]),
        sys_port=8081,
        http_port=30000,
        endpoint_mode="agg",
        endpoint_index=0,
        node_rank=0,
    )


@pytest.mark.parametrize("backend_cls", BACKENDS, ids=lambda cls: cls.__name__)
def test_optional_features_read_as_absent_by_default(backend_cls):
    backend = backend_cls()
    assert backend.mooncake_kv_store is None
    assert backend.failover is None
    assert backend.get_mooncake_worker_env("10.0.0.1", "10.0.0.2") == {}
    assert backend.get_failover_environment(_process(), "12345") == {}
    assert backend.get_srun_config().sequential_node_start == 0


def test_sglang_mooncake_env_reaches_workers_through_the_protocol():
    backend = SGLangProtocol(mooncake_kv_store=MooncakeKVStoreConfig(env={"MOONCAKE_PROTOCOL": "rdma"}))
    env = backend.get_mooncake_worker_env("10.0.0.1", "10.0.0.2")
    assert env["MOONCAKE_MASTER"] == f"10.0.0.1:{MOONCAKE_MASTER_PORT}"
    assert env["MOONCAKE_LOCAL_HOSTNAME"] == "10.0.0.2"
    assert env["MOONCAKE_PROTOCOL"] == "rdma"


def test_vllm_failover_env_reaches_workers_through_the_protocol():
    backend = VLLMProtocol(failover=VLLMFailoverConfig(shadow_engines=1, shared_dir="/dev/shm"))
    assert backend.failover is not None
    env = backend.get_failover_environment(_process(), "12345")
    assert env["ENGINE_ID"] == "0"
    assert env["GMS_SOCKET_DIR"].startswith("/dev/shm/")


def test_trtllm_batched_startup_rides_on_srun_config():
    assert TRTLLMProtocol(sequential_node_start=2).get_srun_config().sequential_node_start == 2


def test_no_module_probes_a_backend_with_getattr_or_hasattr():
    offenders = [
        f"{path.relative_to(SRC.parent.parent)}:{number}"
        for path in SRC.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if DUCK_TYPED_BACKEND.search(line)
    ]
    assert offenders == [], "read the member on BackendProtocol instead:\n" + "\n".join(offenders)
