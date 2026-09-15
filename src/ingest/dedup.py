# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The idempotent (labels, value) fold shared by every schema-2 metrics processor."""

from __future__ import annotations


def _dedup(entries: list[dict]) -> list[dict]:
    """Drop exact (labels, value) duplicates within one metric, preserving first-seen order.

    Both frontend ports (e.g. :8333 and the DYN_SYSTEM_PORT :8082) serve identical metrics, so
    without this a series would be double-listed per timestamp."""
    seen, uniq = set(), []
    for e in entries:
        key = (tuple(sorted(e["labels"].items())), e["value"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq
