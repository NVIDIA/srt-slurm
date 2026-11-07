# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine evidence dialects, independent of launch configuration and rendering.

Dialects decode individual log lines and describe selected NVTX and metric names.
Readers own file discovery, clocks, provenance, joins and limits. A dialect can
support only some sources; recognizing an annotation does not assign ownership.
"""

from __future__ import annotations

import ast
import datetime as dt
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeAlias


@dataclass(frozen=True)
class EngineIdentity:
    server_id: str
    client_id: str
    disagg_id: str


IterationFieldValue: TypeAlias = int | float | str | tuple[Any, ...] | None


@dataclass(frozen=True)
class EngineIteration:
    """Recorded iteration fields; local_time has no timezone and second precision.

    fields holds every recorded key in log order, so keys added or removed upstream
    pass through without a parser change. The named attributes are the subset other
    sources share; None means the line omitted the key or recorded N/A.
    """

    iteration: int
    global_rank: int
    rank: int
    local_time: str
    batch_requests: int | None = None
    kv_cache_util: float | None = None
    host_step_ms: float | None = None
    previous_device_step_ms: float | None = None
    fields: tuple[tuple[str, IterationFieldValue], ...] = ()


@dataclass(frozen=True)
class EngineBatchSnapshot:
    """Periodic scheduler observations, without a forward-step counter or timer."""

    local_time: str
    rank: int
    batch_kind: str
    time_resolution_s: float
    batch_requests: int | None = None
    queued_requests: int | None = None
    new_sequences: int | None = None
    new_tokens: int | None = None
    cached_tokens: int | None = None
    page_ratio: float | None = None
    generation_tokens_per_s: float | None = None
    average_accept_length: float | None = None
    accept_rate: float | None = None
    active_pages: int | None = None
    cached_pages: int | None = None
    total_pages: int | None = None


@dataclass(frozen=True)
class EngineLogRecord:
    # A line can carry both observations. Keep them together so the reader owns
    # line-level filtering and provenance, including out-of-window iterations.
    iteration: EngineIteration | None = None
    identity: EngineIdentity | None = None
    snapshot: EngineBatchSnapshot | None = None
    backend: str | None = None


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
    log_parser: Callable[[str], EngineLogRecord | None] | None = None
    metrics: tuple[tuple[str, MetricDefinition], ...] = ()


_TRT_MAP = re.compile(r"Engine ID map: request_id=(\S+) trtllm_client_id=(\S+) disagg_request_id=(\S+)")
# PyExecutor's iteration log (TensorRT-LLM v1.3.0rc29, _torch/pyexecutor/profiling.py)
# joins `key = value` fields with ", ". A boundary is ", " before the next bare
# `name = `; the states dict repr quotes its keys and lists hold bare numbers, so
# neither can produce one.
_TRT_ITERATION_START = re.compile(r"\biter = ")
_TRT_FIELD_BOUNDARY = re.compile(r", (?=[A-Za-z_]\w* = )")
_TRT_PAIR = re.compile(r"(\d+)/(\d+)")


def _trtllm_iteration_value(text: str) -> IterationFieldValue | dict[str, Any]:
    """Convert by the value's shape, never by its key, so new keys need no change."""
    if text == "N/A":
        return None
    number = text.removesuffix("ms")  # Every "ms" field is a duration in milliseconds.
    for convert in (int, float):
        try:
            value = convert(number)
        except ValueError:
            continue
        return value if math.isfinite(value) else text
    if m := _TRT_PAIR.fullmatch(text):
        return (int(m[1]), int(m[2]))
    if text[:1] in "{[":
        try:
            literal = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(literal, dict):
            return literal
        return tuple(literal) if isinstance(literal, list) else text
    return text


def parse_trtllm_iteration(line: str) -> EngineIteration | None:
    """Decode every `key = value` field. Nested dicts flatten to `parent.child`."""
    if not (start := _TRT_ITERATION_START.search(line)):
        return None
    fields: dict[str, IterationFieldValue] = {}
    for item in _TRT_FIELD_BOUNDARY.split(line[start.start() :].rstrip()):
        key, _, text = item.partition(" = ")
        value = _trtllm_iteration_value(text)
        if isinstance(value, dict):
            for child, nested in value.items():
                fields[f"{key}.{child}"] = tuple(nested) if isinstance(nested, list) else nested
        else:
            fields[key] = value
    iteration, global_rank, rank, stamp = (fields.get(k) for k in ("iter", "global_rank", "rank", "timestamp"))
    if not (isinstance(iteration, int) and isinstance(global_rank, int) and isinstance(rank, int)):
        return None
    try:
        dt.datetime.strptime(str(stamp), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

    def numeric(key: str) -> Any:
        value = fields.get(key)
        return value if isinstance(value, int | float) else None

    return EngineIteration(
        iteration=iteration,
        global_rank=global_rank,
        rank=rank,
        local_time=str(stamp),
        batch_requests=numeric("num_scheduled_requests"),
        kv_cache_util=numeric("kv_cache_util"),
        host_step_ms=numeric("host_step_time"),
        previous_device_step_ms=numeric("prev_device_step_time"),
        fields=tuple(fields.items()),
    )


def _trtllm(line: str) -> EngineLogRecord | None:
    iteration = parse_trtllm_iteration(line) if "iter = " in line else None
    identity = None
    if m := _TRT_MAP.search(line):
        identity = EngineIdentity(server_id=m[1], client_id=m[2], disagg_id=m[3])
    return EngineLogRecord(iteration, identity) if iteration is not None or identity is not None else None


_TOKEN_BATCH = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\s+ATTN TP RANK (\d+)\].*?(Prefill|Decode) batch\. (.+)"
)


def parse_tokenspeed_log(line: str) -> EngineLogRecord | None:
    if "batch." not in line or not (m := _TOKEN_BATCH.search(line)):
        return None

    def integer(label: str) -> int | None:
        value = re.search(re.escape(label) + r":\s*(\d+)", m[4])
        return int(value[1]) if value else None

    def number(label: str) -> float | None:
        value = re.search(re.escape(label) + r":\s*([\d.eE+-]+)", m[4])
        return float(value[1]) if value else None

    pages = re.search(r"#pages\(active/cached/total\):\s*(\d+)/(\d+)/(\d+)", m[4])
    return EngineLogRecord(
        snapshot=EngineBatchSnapshot(
            local_time=m[1].replace(",", "."),
            rank=int(m[2]),
            batch_kind=m[3].lower(),
            time_resolution_s=10 ** -len(m[1].split(",")[1]),
            batch_requests=integer("#running-req"),
            queued_requests=integer("#queue-req"),
            new_sequences=integer("#new-seq"),
            new_tokens=integer("#new-token"),
            cached_tokens=integer("#cached-token"),
            page_ratio=number("page ratio"),
            generation_tokens_per_s=number("gen throughput (token/s)"),
            average_accept_length=number("avg_accept_len"),
            accept_rate=number("accept_rate"),
            active_pages=int(pages[1]) if pages else None,
            cached_pages=int(pages[2]) if pages else None,
            total_pages=int(pages[3]) if pages else None,
        )
    )


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
        log_parser=parse_tokenspeed_log,
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
    EngineDialect(
        "vllm",
        nvtx_prefixes=("gpu_model_runner: ", "ngram_proposer_gpu: "),
        # Per-step scheduler stages; "schedule: allocate_slots" opens once per running
        # request per step and would exhaust the per-report event limit.
        nvtx_names=(
            "schedule: get_num_common_prefix_blocks",
            "schedule: make_cached_request_data",
            "schedule: update_after_schedule",
        ),
        metrics=(
            ("vllm:num_requests_running", MetricDefinition("Running requests", "requests")),
            ("vllm:num_requests_waiting", MetricDefinition("Waiting requests", "requests")),
            ("vllm:kv_cache_usage_perc", MetricDefinition("KV cache utilization", "ratio")),
        ),
    ),
)

_COMMON_NVTX = ("preprocess.", "route.", "router.", "tokenize", "detokenize", "kv_router.", "transport.", "compute_")


def parse_engine_log(line: str) -> EngineLogRecord | None:
    """Decode a recognized line without assigning timestamps or request ownership."""
    for dialect in DIALECTS:
        if dialect.log_parser and (record := dialect.log_parser(line)) is not None:
            return replace(record, backend=dialect.name)
    return None


def select_nvtx(name: str, duration_ns: int) -> bool:
    """Select supported host annotations through the common dialect catalog."""
    return classify_nvtx(name, duration_ns) is not None


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


def default_metric(sampled: set[str]) -> str | None:
    """The first dialect's leading metric (its running-request count) that has samples."""
    return next((d.metrics[0][0] for d in DIALECTS if d.metrics and d.metrics[0][0] in sampled), None)


def engine_metrics() -> dict[str, MetricDefinition]:
    """Map recorded metric names to their display labels and original units."""
    return {name: definition for dialect in DIALECTS for name, definition in dialect.metrics}
