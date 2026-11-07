# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM periodic engine statistics (vllm/v1/metrics/loggers.py LoggingStatLogger)."""

from __future__ import annotations

import re

from ..sources import SourceIdentity
from .base import LogMetricDefinition, LogMetricEvent, LogMetricGenerator

RUNNING = "log_vllm_running_requests"
WAITING = "log_vllm_waiting_requests"
KV_USAGE = "log_vllm_kv_cache_usage"
PREFIX_HIT = "log_vllm_prefix_cache_hit_rate"
EXTERNAL_HIT = "log_vllm_external_prefix_cache_hit_rate"
PROMPT_THROUGHPUT = "log_vllm_prompt_throughput"
GENERATION_THROUGHPUT = "log_vllm_generation_throughput"

# "(MainProcess pid=12) INFO 10-09 17:50:00 [loggers.py:320] Engine 000: Avg prompt throughput: ..."
# vLLM's default formatter prints local month-day time without a year; the reader resolves it.
_STATS = re.compile(
    r"(?:\((?P<process>[^)\s]+)[^)]*\)\s+)?INFO\s+(?P<time>\d\d-\d\d \d\d:\d\d:\d\d)(?:,(?P<ms>\d+))?\s+"
    r"\[[^\]]*\]\s+Engine (?P<engine>\d+): (?P<body>.+)"
)
_FIELDS = (
    (PROMPT_THROUGHPUT, re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s"), 1.0),
    (GENERATION_THROUGHPUT, re.compile(r"Avg generation throughput: ([\d.]+) tokens/s"), 1.0),
    (RUNNING, re.compile(r"Running: (\d+) reqs"), 1.0),
    (WAITING, re.compile(r"Waiting: (\d+) reqs"), 1.0),
    (KV_USAGE, re.compile(r"GPU KV cache usage: ([\d.]+)%"), 0.01),
    (PREFIX_HIT, re.compile(r"(?<!External )Prefix cache hit rate: ([\d.]+)%"), 0.01),
    (EXTERNAL_HIT, re.compile(r"External prefix cache hit rate: ([\d.]+)%"), 0.01),
)

_PERIODIC = "Periodic engine log line (every VLLM_LOG_STATS_INTERVAL, 10 s by default); idle intervals are not logged."


class VLLMLogMetrics(LogMetricGenerator):
    name = "vllm"
    definitions = (
        LogMetricDefinition(
            RUNNING, "Running requests (log)", "requests", f"Requests the scheduler is running. {_PERIODIC}"
        ),
        LogMetricDefinition(
            WAITING, "Waiting requests (log)", "requests", f"Requests waiting to be scheduled. {_PERIODIC}"
        ),
        LogMetricDefinition(
            KV_USAGE,
            "KV cache usage (log)",
            "ratio",
            f"GPU KV cache usage printed as a percent, stored as a 0-1 fraction. {_PERIODIC}",
        ),
        LogMetricDefinition(
            PREFIX_HIT,
            "Prefix cache hit rate (log)",
            "ratio",
            f"Local prefix cache hit rate as printed by vLLM, stored as a 0-1 fraction. {_PERIODIC}",
        ),
        LogMetricDefinition(
            EXTERNAL_HIT,
            "Connector cache hit rate (log)",
            "ratio",
            "Hit rate of the KV connector's external cache (for example a disaggregated or offload store), "
            f"stored as a 0-1 fraction. {_PERIODIC}",
        ),
        LogMetricDefinition(
            PROMPT_THROUGHPUT,
            "Prompt throughput (log)",
            "tokens/s",
            f"Prompt tokens processed per second, averaged over the logging interval. {_PERIODIC}",
        ),
        LogMetricDefinition(
            GENERATION_THROUGHPUT,
            "Generation throughput (log)",
            "tokens/s",
            f"Output tokens generated per second, averaged over the logging interval. {_PERIODIC}",
        ),
    )

    def parse_line(self, line: str, source: SourceIdentity) -> LogMetricEvent | None:
        if "Engine " not in line or "throughput" not in line or not (m := _STATS.search(line)):
            return None
        values = tuple(
            (name, float(field[1]) * scale) for name, pattern, scale in _FIELDS if (field := pattern.search(m["body"]))
        )
        if not values:
            return None
        ms = m["ms"]
        return LogMetricEvent(
            m["time"] + (f".{ms}" if ms else ""),
            values,
            rank=int(m["engine"]),
            rank_kind="dp",
            process=m["process"],
            time_resolution_s=10 ** -len(ms) if ms else 1.0,
        )
