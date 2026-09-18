# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response models for the Status API contract."""

from pydantic import BaseModel


class JobResponse(BaseModel):
    """Response model for job create/update operations."""

    job_id: str
    status: str


class JobSummary(BaseModel):
    """Summary model for job list."""

    job_id: str
    job_name: str
    status: str
    stage: str | None = None
    message: str | None = None
    cluster: str | None = None
    submitted_at: str
    completed_at: str | None = None
    updated_at: str
    exit_code: int | None = None


class JobDetail(BaseModel):
    """Detailed job model with full event history."""

    job_id: str
    job_name: str
    status: str
    stage: str | None = None
    cluster: str | None = None
    recipe: str | None = None
    message: str | None = None
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str
    exit_code: int | None = None
    logs_url: str | None = None
    benchmark_results: dict | None = None
    artifacts: dict | None = None
    metadata: dict | None = None
    events: list[dict] | None = None


class JobListResponse(BaseModel):
    """Response model for paginated job list."""

    jobs: list[JobSummary]
    total: int
    page: int
    per_page: int


class JobEventRecord(BaseModel):
    """One lifecycle event. ``id`` is the cursor for the incremental feeds."""

    id: int
    job_id: str
    status: str
    stage: str | None = None
    message: str | None = None
    created_at: str


class JobEventListResponse(BaseModel):
    """GET /api/jobs/{job_id}/events: events after a cursor for one job."""

    job_id: str
    events: list[JobEventRecord]
    next_cursor: int | None = None


class EventFeedResponse(BaseModel):
    """GET /api/events: events after a cursor across every job."""

    events: list[JobEventRecord]
    next_cursor: int | None = None
