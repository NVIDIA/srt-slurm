# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed SGLang prefill and decode batch fields, scoped to the logged ranks."""

from __future__ import annotations

import math
import re

from ..sources import SourceIdentity
from .base import LogMetricDefinition, LogMetricEvent

_RANK = re.compile(r"(DP|PP|ATTN_CP|MOE_DP|TP|EP)(\d+)")
_PREFIX = r"\[(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)" rf"(?P<ranks>(?: {_RANK.pattern})*)\] "
_BATCH = re.compile(_PREFIX + r"(?P<phase>Prefill|Decode) batch(?: \[\d+\])?, (?P<fields>.*)")
_REQUEST = re.compile(_PREFIX + r"ReqTimeStats\((?P<meta>[^)]*)\): (?P<fields>.*)")
_DECIMAL = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")

# Counts are retained as exact integers. Ratios and rates retain the decimal
# precision written by the logger; no workload-wide values are derived here.
_FIELDS: dict[str, tuple[str, str, str, str]] = {
    "#new-seq": ("new_sequences", "New sequences", "requests", "Prefill batch #new-seq."),
    "#new-token": ("new_tokens", "New tokens", "tokens", "Prefill batch #new-token."),
    "#cached-token": ("cached_tokens", "Cached tokens", "tokens", "Prefill batch #cached-token."),
    "#running-req": ("running_requests", "Running requests", "requests", "Batch #running-req snapshot."),
    "#queue-req": ("queued_requests", "Queued requests", "requests", "Batch #queue-req snapshot."),
    "#pending-token": ("pending_tokens", "Pending tokens", "tokens", "Prefill batch #pending-token snapshot."),
    "#bootstrap-req": ("bootstrap_requests", "Bootstrap requests", "requests", "Prefill batch #bootstrap-req."),
    "#inflight-req": ("inflight_requests", "Inflight requests", "requests", "Prefill batch #inflight-req."),
    "#optimistic-req": ("optimistic_requests", "Optimistic requests", "requests", "Prefill batch #optimistic-req."),
    "token usage": ("token_usage", "Token usage", "ratio", "Batch token usage as logged."),
    "input throughput (token/s)": (
        "input_throughput_tokens_per_second",
        "Input throughput",
        "tokens/s",
        "Prefill batch input throughput as logged.",
    ),
    "#token": ("decode_tokens", "Decode tokens", "tokens", "Decode batch #token snapshot."),
    "accept len": ("accept_length", "Accept length", "tokens", "Decode batch accept len as logged."),
    "accept rate": ("accept_rate", "Accept rate", "ratio", "Decode batch accept rate as logged."),
    "pre-allocated usage": (
        "preallocated_usage",
        "Pre-allocated usage",
        "ratio",
        "Decode batch pre-allocated usage as logged.",
    ),
    "#prealloc-req": ("preallocated_requests", "Preallocated requests", "requests", "Decode batch #prealloc-req."),
    "#transfer-req": ("transfer_requests", "Transfer requests", "requests", "Decode batch #transfer-req."),
    "#retracted-req": ("retracted_requests", "Retracted requests", "requests", "Decode batch #retracted-req."),
    "gen throughput (token/s)": (
        "generation_throughput_tokens_per_second",
        "Generation throughput",
        "tokens/s",
        "Decode batch generation throughput as logged.",
    ),
    "cuda graph": ("cuda_graph_enabled", "CUDA graph enabled", "boolean", "Batch cuda graph flag (true=1, false=0)."),
}
_COUNTS = {
    "#new-seq",
    "#new-token",
    "#cached-token",
    "#running-req",
    "#queue-req",
    "#pending-token",
    "#bootstrap-req",
    "#inflight-req",
    "#optimistic-req",
    "#token",
    "#prealloc-req",
    "#transfer-req",
    "#retracted-req",
}
_PREFILL_ONLY = {
    "#new-seq",
    "#new-token",
    "#cached-token",
    "#pending-token",
    "#bootstrap-req",
    "#inflight-req",
    "#optimistic-req",
    "input throughput (token/s)",
}
_DECODE_ONLY = {
    "#token",
    "accept len",
    "accept rate",
    "pre-allocated usage",
    "#prealloc-req",
    "#transfer-req",
    "#retracted-req",
    "gen throughput (token/s)",
}
_REQUEST_DURATIONS = {
    "bootstrap_duration": "bootstrap",
    "bootstrap_queue_duration": "bootstrap_queue",
    "prealloc_queue_duration": "preallocation_queue",
    "queue_duration": "queue",
    "forward_duration": "forward",
    "alloc_wait_duration": "allocation_wait",
    "transfer_duration": "transfer",
}
_REQUEST_TRANSFER = {
    "transfer_speed": ("transfer_speed_gib_per_second", "Transfer speed", "GiB/s", " GB/s"),
    "transfer_total": ("transfer_total_mib", "Transfer total", "MiB", " MB"),
}


def _scoped_event(match: re.Match[str], phase: str, values: list[tuple[str, float | None]]) -> LogMetricEvent | None:
    rank_fields = _RANK.findall(match["ranks"])
    ranks = dict(rank_fields)
    if len(ranks) != len(rank_fields):
        return None
    dp = ranks.pop("DP", None)
    return LogMetricEvent(
        match["time"],
        tuple(values),
        rank=int(dp) if dp is not None else None,
        rank_kind="dp" if dp is not None else None,
        labels=(("phase", phase), *((name.lower(), rank) for name, rank in ranks.items())),
        time_resolution_s=10 ** -len(match["time"].partition(".")[2]),
    )


class SGLangLogMetrics:
    name = "sglang"
    definitions = tuple(
        LogMetricDefinition(
            f"log_sglang_{suffix}",
            title,
            unit,
            description + " Per logged rank and phase; samples are not summed.",
            # Default second-resolution logs can contain several distinct batches
            # at one timestamp. Keep each source line, including equal values.
            temporal="event",
        )
        for suffix, title, unit, description in _FIELDS.values()
    ) + (
        LogMetricDefinition(
            "log_sglang_request_input_tokens",
            "Request input tokens",
            "tokens",
            "ReqTimeStats input_len at request completion. Per-request observations are not summed across ranks.",
            temporal="event",
        ),
        LogMetricDefinition(
            "log_sglang_request_cached_input_tokens",
            "Request cached input tokens",
            "tokens",
            "ReqTimeStats cached_input_len at request completion; this is not a workload-wide cache hit rate.",
            temporal="event",
        ),
        LogMetricDefinition(
            "log_sglang_request_uncached_input_tokens",
            "Request uncached input tokens",
            "tokens",
            "input_len minus cached_input_len on the same completed request, only when 0 <= cached <= input.",
            temporal="event",
        ),
        LogMetricDefinition(
            "log_sglang_request_cached_input_fraction",
            "Request cached input fraction",
            "ratio",
            "cached_input_len / input_len on the same completed request, only when input_len > 0 and 0 <= cached <= input.",
            temporal="event",
        ),
        *(
            LogMetricDefinition(
                f"log_sglang_request_{suffix}_duration_ms",
                f"Request {suffix.replace('_', ' ')} duration",
                "ms",
                f"ReqTimeStats {field} at request completion. A logged zero can mean a missing timing endpoint; stages may overlap and are not additive.",
                temporal="event",
            )
            for field, suffix in _REQUEST_DURATIONS.items()
        ),
        *(
            LogMetricDefinition(
                f"log_sglang_request_{suffix}",
                title,
                unit,
                f"ReqTimeStats {field} at request completion. SGLang prints the binary quantity with a {literal.strip()} suffix; chunked transfer may cover only the last chunk.",
                temporal="event",
            )
            for field, (suffix, title, unit, literal) in _REQUEST_TRANSFER.items()
        ),
    )

    def parse_line(self, line: str, source: SourceIdentity) -> LogMetricEvent | None:
        if source.role not in {"prefill", "decode", "agg"}:
            return None
        if match := _REQUEST.search(line):
            return self._parse_request(match)
        if (match := _BATCH.search(line)) is None:
            return None
        phase = match["phase"].lower()
        values: list[tuple[str, float | None]] = []
        for field in match["fields"].split(", "):
            key, separator, raw = field.partition(": ")
            if not separator or key not in _FIELDS:
                continue
            if (key in _PREFILL_ONLY and phase != "prefill") or (key in _DECODE_ONLY and phase != "decode"):
                continue
            if key == "cuda graph":
                if raw not in {"True", "False"}:
                    continue
                value: float | None = int(raw == "True")
            elif key in _COUNTS:
                if not raw.isascii() or not raw.isdecimal():
                    continue
                value = int(raw)
            else:
                if _DECIMAL.fullmatch(raw) is None:
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    continue
            values.append((f"log_sglang_{_FIELDS[key][0]}", value))
        if not values:
            return None
        return _scoped_event(match, phase, values)

    def _parse_request(self, match: re.Match[str]) -> LogMetricEvent | None:
        metadata = dict(part.split("=", 1) for part in match["meta"].split(", ") if "=" in part)
        phase = metadata.get("type")
        if phase not in {"prefill", "decode"}:
            return None
        values: list[tuple[str, float | None]] = []

        def count(field: str, suffix: str) -> int | None:
            raw = metadata.get(field, "")
            if not raw.isascii() or not raw.isdecimal():
                return None
            value = int(raw)
            values.append((f"log_sglang_request_{suffix}", value))
            return value

        input_tokens = count("input_len", "input_tokens")
        cached_tokens = count("cached_input_len", "cached_input_tokens")
        if input_tokens is not None and cached_tokens is not None and cached_tokens <= input_tokens:
            values.append(("log_sglang_request_uncached_input_tokens", input_tokens - cached_tokens))
            if input_tokens > 0:
                values.append(("log_sglang_request_cached_input_fraction", cached_tokens / input_tokens))

        for part in match["fields"].split(", "):
            key, separator, raw = part.partition("=")
            if not separator:
                continue
            if key in _REQUEST_DURATIONS:
                suffix, expected_unit = f"{_REQUEST_DURATIONS[key]}_duration_ms", "ms"
            elif key in _REQUEST_TRANSFER:
                suffix, _, _, expected_unit = _REQUEST_TRANSFER[key]
            else:
                continue
            if not raw.endswith(expected_unit) or _DECIMAL.fullmatch(raw.removesuffix(expected_unit)) is None:
                continue
            value = float(raw.removesuffix(expected_unit))
            if math.isfinite(value):
                values.append((f"log_sglang_request_{suffix}", value))
        if not values:
            return None
        return _scoped_event(match, phase, values)
