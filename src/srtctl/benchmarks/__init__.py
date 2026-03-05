# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runners for srtctl."""

# Import runners to trigger registration
from srtctl.benchmarks import gpqa, gsm8k, longbenchv2, mmlu, mooncake_router, router, sa_bench, sglang_bench
from srtctl.benchmarks.base import (
    BenchmarkRunner,
    get_runner,
    list_benchmarks,
    register_benchmark,
)

__all__ = [
    "BenchmarkRunner",
    "get_runner",
    "list_benchmarks",
    "register_benchmark",
    # Runners
    "sa_bench",
    "sglang_bench",
    "mmlu",
    "gpqa",
    "gsm8k",
    "longbenchv2",
    "router",
    "mooncake_router",
]
