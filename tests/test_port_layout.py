# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every port srtctl assigns stays below EPHEMERAL_FLOOR, and no two blocks overlap.

Discovered from ``srtctl.ports``: each ``PortKind`` spans ``base + (cap - 1) * stride``,
and each other ``*_PORT`` / ``*_PORT_BASE`` constant is one port. A new constant
above the floor fails here unless it is added to ``ABOVE_FLOOR`` with its reason.
"""

from itertools import combinations

from srtctl import ports
from srtctl.core.port_preflight import ephemeral_conflicts, process_ports
from srtctl.core.topology import Process
from srtctl.ports import EPHEMERAL_FLOOR, PortKind

ABOVE_FLOOR = {
    "TOKENSPEED_PORTS": "the engine scans 100-1000 ports above --port for NCCL and skips busy ones",
    "VLLM_SCAN_PORTS": "VLLM_PORT is the base of vLLM's get_open_port() scan, which skips busy ports",
    "VLLM_MASTER_PORT_BASE": "torch/vLLM's default rendezvous port (failover only), overridable with master-port",
    "DCGM_EXPORTER_PORT": "Prometheus exporter port that scrape configs know; overridable per exporter",
    "NODE_EXPORTER_PORT": "Prometheus exporter port that scrape configs know; overridable per exporter",
    "PROCESS_EXPORTER_PORT": "Prometheus exporter port that scrape configs know; overridable per exporter",
    "CPU_POWER_EXPORTER_PORT": "Prometheus exporter port that scrape configs know; overridable per exporter",
}
# Host daemons on GPU nodes that srtctl must not bind over: slurmd, DCGM's nv-hostengine.
HOST_LISTENERS = {"slurmd": 6818, "nv-hostengine": 5555}


def _blocks() -> dict[str, tuple[int, int]]:
    kinds = {name: kind for name, kind in vars(ports).items() if isinstance(kind, PortKind)}
    blocks = {name: (kind.base, kind.base + (kind.cap - 1) * kind.stride) for name, kind in kinds.items()}
    constants = {
        name: value
        for name, value in vars(ports).items()
        if isinstance(value, int) and name.endswith(("_PORT", "_PORT_BASE"))
    }
    for kind_name, kind in kinds.items():
        bases = [name for name, value in constants.items() if value == kind.base]
        assert len(bases) == 1, f"{kind_name} base {kind.base} is named by {bases}, expected exactly one constant"
        del constants[bases[0]]
    blocks.update({name: (value, value) for name, value in constants.items()})
    return blocks


def test_blocks_sit_between_1024_and_the_floor_unless_justified():
    blocks = _blocks()
    for name, (low, high) in blocks.items():
        if name in ABOVE_FLOOR:
            assert high >= EPHEMERAL_FLOOR, f"{name} is below the floor now; drop it from ABOVE_FLOOR"
        else:
            assert low >= 1024 and high < EPHEMERAL_FLOOR, f"{name} spans {low}-{high}; move it below {EPHEMERAL_FLOOR}"
    assert set(ABOVE_FLOOR) <= set(blocks)


def test_blocks_below_the_floor_do_not_overlap():
    blocks = {name: span for name, span in _blocks().items() if name not in ABOVE_FLOOR}
    blocks.update({name: (port, port) for name, port in HOST_LISTENERS.items()})
    for (a, (a_low, a_high)), (b, (b_low, b_high)) in combinations(sorted(blocks.items(), key=lambda kv: kv[1]), 2):
        assert a_high < b_low or b_high < a_low, f"{a} {a_low}-{a_high} overlaps {b} {b_low}-{b_high}"


def test_preflight_flags_unreserved_ports_inside_the_range():
    process = Process(
        node="n1",
        gpu_indices=frozenset({0}),
        sys_port=7500,
        http_port=6100,
        endpoint_mode="decode",
        endpoint_index=0,
        vllm_scan_port=20000,
        nccl_port=None,
    )
    assigned = [("n1", owner, port) for owner, port in process_ports(process)] + [("n1", "dcgm metrics", 9401)]
    probe = "n1 9000 65000 9100,9400-9401,50000\nunparsable line\n"

    assert [port for _, _, port in assigned] == [7500, 6100, 20000, 9401]
    assert ephemeral_conflicts(probe, assigned) == [
        "n1: port 20000 (decode_0 vllm_scan_port) is inside ip_local_port_range 9000-65000"
    ]
    assert ephemeral_conflicts("n1 32768 60999 -", assigned) == []
    assert ephemeral_conflicts("", assigned) == []
