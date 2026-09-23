# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine evidence dialects, independent of launch configuration and rendering.

Dialects decode individual log lines and describe selected NVTX and metric names.
Readers own file discovery, clocks, provenance, joins and limits. A dialect can
support only some sources; recognizing an annotation does not assign ownership.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineIdentity:
    server_id: str
    client_id: str
    disagg_id: str


@dataclass(frozen=True)
class EngineIteration:
    """Recorded iteration fields; local_time has no timezone and second precision."""

    iteration: int
    global_rank: int
    rank: int
    batch_requests: int
    kv_cache_util: float
    host_step_ms: float
    previous_device_step_ms: float
    local_time: str


@dataclass(frozen=True)
class EngineLogRecord:
    # A line can carry both observations. Keep them together so the reader owns
    # line-level filtering and provenance, including out-of-window iterations.
    iteration: EngineIteration | None = None
    identity: EngineIdentity | None = None


@dataclass(frozen=True)
class MetricDefinition:
    label: str
    unit: str


@dataclass(frozen=True)
class EngineDialect:
    name: str
    nvtx_prefixes: tuple[str, ...] = ()
    log_parser: Callable[[str], EngineLogRecord | None] | None = None
    metrics: tuple[tuple[str, MetricDefinition], ...] = ()


_TRT_MAP = re.compile(r"Engine ID map: request_id=(\S+) trtllm_client_id=(\S+) disagg_request_id=(\S+)")
_TRT_ITERATION = re.compile(
    r"iter = (\d+).*?global_rank = (\d+).*?rank = (\d+).*?num_scheduled_requests = (\d+).*?"
    r"kv_cache_util = ([\d.]+).*?host_step_time = ([\d.eE+-]+)ms.*?"
    r"prev_device_step_time = ([\d.eE+-]+)ms.*?timestamp = ([\d-]+ [\d:]+)"
)


def _trtllm(line: str) -> EngineLogRecord | None:
    iteration = None
    if "iter =" in line and (m := _TRT_ITERATION.search(line)):
        iteration = EngineIteration(
            iteration=int(m[1]),
            global_rank=int(m[2]),
            rank=int(m[3]),
            batch_requests=int(m[4]),
            kv_cache_util=float(m[5]),
            host_step_ms=float(m[6]),
            previous_device_step_ms=float(m[7]),
            local_time=m[8],
        )
    identity = None
    if m := _TRT_MAP.search(line):
        identity = EngineIdentity(server_id=m[1], client_id=m[2], disagg_id=m[3])
    return EngineLogRecord(iteration, identity) if iteration is not None or identity is not None else None


# Engine vocabulary lives here; source readers do not branch on engine names.
DIALECTS = (
    EngineDialect(
        "trtllm",
        nvtx_prefixes=(
            "[Executor]",
            "_schedule",
            "_forward_step",
            "_prepare_inputs",
            "_fetch_new_requests",
            "prepare_resources",
            "LLM.generate_async",
            "RpcWorker.submit",
        ),
        log_parser=_trtllm,
        metrics=(
            ("trtllm_num_requests_running", MetricDefinition("Running requests", "requests")),
            ("trtllm_num_requests_waiting", MetricDefinition("Waiting requests", "requests")),
            ("trtllm_kv_cache_utilization", MetricDefinition("KV cache utilization", "ratio")),
        ),
    ),
    EngineDialect(
        "sglang",
        nvtx_prefixes=("scheduler.",),
        metrics=(
            ("sglang:num_running_reqs", MetricDefinition("Running requests", "requests")),
            ("sglang:num_queue_reqs", MetricDefinition("Waiting requests", "requests")),
            ("sglang:token_usage", MetricDefinition("KV cache utilization", "ratio")),
        ),
    ),
)

_COMMON_NVTX = ("preprocess.", "route.", "router.", "tokenize", "detokenize", "kv_router.", "transport.", "compute_")
_NVTX_PREFIXES = _COMMON_NVTX + tuple(prefix for dialect in DIALECTS for prefix in dialect.nvtx_prefixes)


def parse_engine_log(line: str) -> EngineLogRecord | None:
    """Decode a recognized line without assigning timestamps or request ownership."""
    for dialect in DIALECTS:
        if dialect.log_parser and (record := dialect.log_parser(line)) is not None:
            return record
    return None


def select_nvtx(name: str, duration_ns: int) -> bool:
    """Select shared host annotations; short bare detokenize ranges are omitted."""
    if name == "detokenize" and duration_ns < 100_000:
        return False
    return name.startswith(_NVTX_PREFIXES)


def engine_metrics() -> dict[str, MetricDefinition]:
    """Map recorded metric names to their display labels and original units."""
    return {name: definition for dialect in DIALECTS for name, definition in dialect.metrics}
