# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang workers reach the node-local LMCache MP server (services[].type: lmcache-server)."""

import pytest

from srtctl.backends.sglang import SGLangProtocol
from srtctl.core.topology import Process
from srtctl.core.schema import RoleConfig


def _process(mode: str = "prefill") -> Process:
    return Process("node0", frozenset(range(8)), 7500, 6100, mode, 0)


def test_enable_lmcache_points_the_worker_at_the_node_local_server() -> None:
    backend = SGLangProtocol(roles={"prefill": RoleConfig(args={"enable-lmcache": True})})

    assert backend.get_process_environment(_process("prefill")) == {
        "LMCACHE_MP_HOST": "127.0.0.1",
        "LMCACHE_MP_PORT": "8750",
    }
    assert backend.get_process_environment(_process("decode")) == {}


@pytest.mark.parametrize(
    ("prefill", "environment"),
    [
        ({"enable_lmcache": True, "lmcache_config_file": "/configs/lmcache.yaml"}, {}),
        ({"enable-lmcache": True}, {"LMCACHE_MP_HOST": "10.0.0.5"}),
    ],
)
def test_recipe_owned_lmcache_address_is_left_alone(prefill: dict, environment: dict) -> None:
    backend = SGLangProtocol(roles={"prefill": RoleConfig(env=environment, args=prefill)})

    assert backend.get_process_environment(_process()) == {}
