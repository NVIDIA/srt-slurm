# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared API contract for the Status API.

This package defines the canonical Pydantic models and enums for the Status API.
It has zero internal imports — only depends on pydantic.

Usage (reporter):
    from srtctl.contract import JobStatus, JobStage, JobCreatePayload, JobUpdatePayload

Usage (server):
    from srtctl.contract import JobStatus, JobStage, JobCreatePayload, JobUpdatePayload
    from srtctl.contract import JobResponse, JobSummary, JobDetail, JobListResponse
"""

from srtctl.contract.enums import JobStage, JobStatus
from srtctl.contract.requests import JobCreatePayload, JobUpdatePayload
from srtctl.contract.responses import JobDetail, JobListResponse, JobResponse, JobSummary

__all__ = [
    "JobCreatePayload",
    "JobDetail",
    "JobListResponse",
    "JobResponse",
    "JobStage",
    "JobStatus",
    "JobSummary",
    "JobUpdatePayload",
]
