# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request payload models for the Status API contract."""

from pydantic import BaseModel, Field, field_validator, model_validator


class JobCreatePayload(BaseModel):
    """Payload for POST /api/jobs."""

    job_id: str = Field(..., description="Unique job identifier (SLURM job ID)")
    job_name: str = Field(..., description="Human-readable job name")
    submitted_at: str = Field(..., description="ISO 8601 submission timestamp")
    cluster: str | None = Field(None, description="Cluster name")
    recipe: str | None = Field(None, description="Path to recipe/config file")
    metadata: dict | None = Field(None, description="Job metadata (may include 'tags' list)")


class JobUpdatePayload(BaseModel):
    """Payload for PUT /api/jobs/{job_id}."""

    status: str = Field(..., description="New job status")
    updated_at: str = Field(..., description="ISO 8601 update timestamp")
    stage: str | None = Field(None, description="Current execution stage")
    message: str | None = Field(None, description="Human-readable status message")
    started_at: str | None = Field(None, description="ISO 8601 job start timestamp")
    completed_at: str | None = Field(None, description="ISO 8601 job completion timestamp")
    exit_code: int | None = Field(None, description="Process exit code")
    logs_url: str | None = Field(None, description="URL where job logs were uploaded (S3 today)")
    benchmark_results: dict | None = Field(None, description="Parsed benchmark results")
    artifacts: dict | None = Field(None, description="Collector-side artifact pointers to merge")
    metadata: dict | None = Field(None, description="Additional metadata to merge")


class LogChunk(BaseModel):
    """New bytes of one file under the run's log directory."""

    file: str = Field(..., min_length=1, max_length=4096, description="Path relative to the run's log directory")
    offset: int = Field(..., ge=0, le=(1 << 63) - 1, description="Byte offset of the chunk in the file")
    size: int = Field(..., gt=0, le=1 << 20, description="Byte length of the chunk in the file")
    data: str = Field(..., min_length=1, description="The chunk decoded as UTF-8 (invalid bytes replaced)")

    @field_validator("file")
    @classmethod
    def relative_file(cls, value: str) -> str:
        if "\x00" in value or any(part in ("", ".", "..") for part in value.split("/")):
            raise ValueError("file must be a relative path without empty, '.' or '..' components")
        return value

    @model_validator(mode="after")
    def bounded_range(self) -> "LogChunk":
        if self.offset + self.size > (1 << 63) - 1:
            raise ValueError("chunk end exceeds the maximum byte offset")
        return self


class LogAppendPayload(BaseModel):
    """Payload for POST /api/jobs/{job_id}/logs."""

    chunks: list[LogChunk] = Field(..., min_length=1, description="Chunks to store; an exact resend is ignored")
    metadata: dict | None = Field(None, description="Additional metadata, including the source cluster")
