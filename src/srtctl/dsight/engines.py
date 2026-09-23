# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine evidence dialects, independent of launch configuration and rendering.

Only this table knows engine vocabulary. Readers own clocks, identities, source
references and limits; dialects interpret individual observations without joins.
A dialect need not implement every source. Unknown measurements stay unknown.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricDefinition:
    label: str
    unit: str
    group: str = "worker"
    description: str = ""


@dataclass(frozen=True)
class EngineDialect:
    name: str
    nvtx_prefixes: tuple[str, ...] = ()
    nvtx_names: tuple[str, ...] = ()
    log_parser: Callable[[str], dict[str, Any] | None] | None = None
    metrics: tuple[tuple[str, MetricDefinition], ...] = ()


_TRT_MAP = re.compile(r"Engine ID map: request_id=(\S+) trtllm_client_id=(\S+) disagg_request_id=(\S+)")
_TRT_ITERATION = re.compile(
    r"iter = (\d+).*?global_rank = (\d+).*?rank = (\d+).*?num_scheduled_requests = (\d+).*?"
    r"kv_cache_util = ([\d.]+).*?host_step_time = ([\d.eE+-]+)ms.*?"
    r"prev_device_step_time = ([\d.eE+-]+)ms.*?timestamp = ([\d-]+ [\d:]+)"
)
_TOKEN_BATCH = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\s+ATTN TP RANK (\d+)\].*?"
    r"(Prefill|Decode) batch\. (.+)"
)


def _trtllm(line: str) -> dict[str, Any] | None:
    if "Engine ID map:" in line and (m := _TRT_MAP.search(line)):
        return {"kind": "engine_identity", "server_id": m[1], "client_id": m[2], "disagg_id": m[3]}
    if "iter =" in line and (m := _TRT_ITERATION.search(line)):
        return {
            "kind": "iteration",
            "iteration": int(m[1]),
            "global_rank": int(m[2]),
            "rank": int(m[2]),
            "local_rank": int(m[3]),
            "rank_kind": "global_rank",
            "batch_requests": int(m[4]),
            "kv_cache_util": float(m[5]),
            "host_step_ms": float(m[6]),
            "previous_device_step_ms": float(m[7]),
            "local_time": m[8],
            "time_resolution_s": 1.0,
        }
    return None


def _tokenspeed(line: str) -> dict[str, Any] | None:
    if "batch." not in line or not (m := _TOKEN_BATCH.search(line)):
        return None
    # A periodic scheduler snapshot is not a numbered forward iteration.
    record: dict[str, Any] = {
        "kind": "batch_snapshot",
        "iteration": None,
        "global_rank": None,
        "rank": int(m[2]),
        "rank_kind": "attention_tp",
        "batch_kind": m[3].lower(),
        "local_time": m[1].replace(",", "."),
        "time_resolution_s": 0.001,
        "batch_requests": None,
        "queued_requests": None,
        "kv_cache_util": None,
        "host_step_ms": None,
        "previous_device_step_ms": None,
    }
    fields = (
        ("#running-req", "batch_requests", int),
        ("#queue-req", "queued_requests", int),
        ("#new-seq", "new_sequences", int),
        ("#new-token", "new_tokens", int),
        ("#cached-token", "cached_tokens", int),
        ("page ratio", "page_ratio", float),
        ("gen throughput (token/s)", "generation_tokens_per_s", float),
        ("avg_accept_len", "average_accept_length", float),
        ("accept_rate", "accept_rate", float),
    )
    for label, key, convert in fields:
        if value := re.search(re.escape(label) + r":\s*([\d.eE+-]+)", m[4]):
            record[key] = convert(value[1])
    if pages := re.search(r"#pages\(active/cached/total\):\s*(\d+)/(\d+)/(\d+)", m[4]):
        record.update(zip(("active_pages", "cached_pages", "total_pages"), map(int, pages.groups()), strict=True))
    return record


# Adding an engine changes this module and its source fixtures, not the viewer.
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
        "tokenspeed",
        nvtx_prefixes=("forward_step ",),
        nvtx_names=(
            "graph_replay",
            "target_forward",
            "update_runtime_state",
            "sampling_prep",
            "pre_fill_setup",
            "input_prep_fill",
            "output_d2h",
            "loop:commit",
            "commit:sync",
            "reset_valid_cache_length",
            "reset_remote_prefill_cache_lengths",
            "zero_cache_pages",
        ),
        log_parser=_tokenspeed,
        metrics=(
            (
                "tokenspeed:num_requests_running",
                MetricDefinition(
                    "Running requests", "requests", description="Requests with scheduler-side generation state."
                ),
            ),
            (
                "tokenspeed:num_requests_waiting",
                MetricDefinition(
                    "Waiting requests", "requests", description="Requests waiting in the engine scheduler queue."
                ),
            ),
            (
                "tokenspeed:kv_cache_usage_perc",
                MetricDefinition(
                    "KV pages in use", "ratio", description="Fraction of device KV pages in use, from 0 to 1."
                ),
            ),
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


def parse_engine_log(line: str) -> dict[str, Any] | None:
    """Return one recorded engine observation; never assign its time or owner."""
    for dialect in DIALECTS:
        if dialect.log_parser and (record := dialect.log_parser(line)) is not None:
            return {**record, "backend": dialect.name}
    return None


def classify_nvtx(name: str, duration_ns: int) -> dict[str, str | None] | None:
    """Select host annotations, retaining their original names and semantics."""
    if name == "detokenize" and duration_ns < 100_000:
        return None
    if name.startswith(_COMMON_NVTX):
        return {"backend": None, "scope": "host", "description": "Shared host annotation; not request ownership."}
    for dialect in DIALECTS:
        if name in dialect.nvtx_names or name.startswith(dialect.nvtx_prefixes):
            return {
                "backend": dialect.name,
                "scope": "host",
                "description": "Shared host annotation; includes waits and launch overhead, not GPU execution duration.",
            }
    return None


def engine_metrics() -> dict[str, MetricDefinition]:
    return {name: definition for dialect in DIALECTS for name, definition in dialect.metrics}
