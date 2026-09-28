# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo–TokenSpeed batch observations and per-scheduler configuration."""

from __future__ import annotations

import re

from ..engines import parse_tokenspeed_log
from ..sources import SourceIdentity
from .base import LogMetricDefinition, LogMetricEvent

ACTIVE_DECODE = "log_tokenspeed_active_decode_requests"
DECODE_LIMIT = "log_tokenspeed_decode_request_limit"
ACTIVE_PAGES = "log_tokenspeed_active_kv_pages"
POOL_PAGES = "log_tokenspeed_kv_pool_pages"

_CONFIG = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\s+ATTN TP RANK (\d+)\].*?Scheduler config: (.+)")
_MAX_BATCH = re.compile(r"(?:^|\s)max_batch_size=(\d+)(?=\s|$)")


class DynamoTokenSpeedLogMetrics:
    name = "dynamo-tokenspeed"
    definitions = (
        LogMetricDefinition(
            ACTIVE_DECODE,
            "Active decode batch",
            "requests",
            "Active decode requests reported by the batch logger (#running-req). "
            "Periodic snapshots, not the exported scheduler running-state count or a time-weighted average.",
            reference=DECODE_LIMIT,
            reference_label="Logged batch limit",
        ),
        LogMetricDefinition(
            DECODE_LIMIT,
            "Logged decode batch limit",
            "requests",
            "Scheduler max_batch_size, scoped to the recorded attention TP rank. "
            "Held from its log timestamp until the next configuration in the same log scope; "
            "not global max_num_seqs or benchmark concurrency.",
            temporal="setting",
        ),
        LogMetricDefinition(
            ACTIVE_PAGES,
            "Active KV pages",
            "pages",
            "Active pages in #pages(active/cached/total). Cached pages are not added. "
            "Periodic snapshots; ranks are not summed.",
            reference=POOL_PAGES,
            reference_label="KV pool size",
        ),
        LogMetricDefinition(
            POOL_PAGES,
            "KV page pool size",
            "pages",
            "Usable total pages reported in the same batch snapshot as active pages. "
            "Not num_device_pages (which may include reserved pages) and not token capacity.",
        ),
    )

    def parse_line(self, line: str, source: SourceIdentity) -> LogMetricEvent | None:
        if "Scheduler config:" in line and source.role in {"decode", "agg"} and (m := _CONFIG.search(line)):
            maximum = _MAX_BATCH.search(m[3])
            # A new config without a valid positive limit invalidates the old one.
            value = int(maximum[1]) if maximum and int(maximum[1]) > 0 else None
            return LogMetricEvent(
                m[1].replace(",", "."),
                ((DECODE_LIMIT, value),),
                rank=int(m[2]),
                rank_kind="attention_tp",
                time_resolution_s=10 ** -len(m[1].split(",")[1]),
            )
        record = parse_tokenspeed_log(line)
        if record is None or record.snapshot is None:
            return None
        batch = record.snapshot
        values: list[tuple[str, float | None]] = []
        if batch.batch_kind == "decode" and batch.batch_requests is not None:
            values.append((ACTIVE_DECODE, batch.batch_requests))
        if batch.active_pages is not None:
            values.append((ACTIVE_PAGES, batch.active_pages))
        if batch.total_pages is not None:
            values.append((POOL_PAGES, batch.total_pages))
        return (
            LogMetricEvent(
                batch.local_time,
                tuple(values),
                rank=batch.rank,
                rank_kind="attention_tp",
                time_resolution_s=batch.time_resolution_s,
            )
            if values
            else None
        )
