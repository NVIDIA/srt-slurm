# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request failure counts and the five-percent benchmark gate."""

from __future__ import annotations

from typing import Literal, TypedDict

MAX_FAILURE_RATE = 0.05


class BenchmarkOutcome(TypedDict):
    status: Literal["passed", "failed"]
    requested: int
    completed: int
    failed: int
    max_failure_rate: float


def benchmark_outcome(requested: int, completed: int) -> BenchmarkOutcome:
    if type(requested) is not int or type(completed) is not int or requested <= 0 or not 0 <= completed <= requested:
        raise ValueError("Benchmark request counts must satisfy 0 <= completed <= requested > 0")
    failed = requested - completed
    return {
        "status": "failed" if failed / requested > MAX_FAILURE_RATE else "passed",
        "requested": requested,
        "completed": completed,
        "failed": failed,
        "max_failure_rate": MAX_FAILURE_RATE,
    }
