# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical enum definitions for the Status API contract."""

from enum import Enum


class JobStatus(str, Enum):
    """Job status values.

    These represent the current stage of execution, not readiness.
    We report when ENTERING a stage, not when it's complete.
    """

    SUBMITTED = "submitted"
    STARTING = "starting"
    WORKERS = "workers"
    FRONTEND = "frontend"
    BENCHMARK = "benchmark"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class JobStage(str, Enum):
    """Job execution stages matching do_sweep.py flow."""

    STARTING = "starting"
    HEAD_INFRASTRUCTURE = "head_infrastructure"
    PREFLIGHT = "preflight"
    WORKERS = "workers"
    FRONTEND = "frontend"
    BENCHMARK = "benchmark"
    CLEANUP = "cleanup"
