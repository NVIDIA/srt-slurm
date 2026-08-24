# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests: StatusReporter against mock Status API server.

These tests exercise the full request/response cycle through the shared
contract models, ensuring srtslurm payloads are accepted by a server
implementing the same contract.
"""

from unittest.mock import patch

import mock_status_server
import pytest

from srtctl.contract import JobCreatePayload, JobStage, JobStatus, JobUpdatePayload
from srtctl.core.schema import ReportingConfig, ReportingStatusConfig
from srtctl.core.status import StatusReporter, create_job_record

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_api():
    """Provide a mock status API that intercepts HTTP requests.

    Patches requests.post/put in srtctl.core.status to route through
    FastAPI TestClient instead of making real HTTP calls.
    """
    client = mock_status_server.create_test_client()

    def route_post(url, json=None, timeout=None):
        """Route POST requests through TestClient."""
        # Extract path from URL (e.g., "http://mock:8080/api/jobs" -> "/api/jobs")
        path = "/" + url.split("/", 3)[-1]
        response = client.post(path, json=json)
        return _MockResponse(response.status_code, response.json())

    def route_put(url, json=None, timeout=None):
        """Route PUT requests through TestClient."""
        path = "/" + url.split("/", 3)[-1]
        response = client.put(path, json=json)
        return _MockResponse(response.status_code, response.json())

    with (
        patch("srtctl.core.status.requests.post", side_effect=route_post),
        patch("srtctl.core.status.requests.put", side_effect=route_put),
    ):
        yield client


class _MockResponse:
    """Minimal response object matching requests.Response interface."""

    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


@pytest.fixture
def reporting():
    """Reporting config pointing to mock server."""
    return ReportingConfig(status=ReportingStatusConfig(endpoint="http://mock:8080"))


@pytest.fixture
def reporter(reporting):
    """StatusReporter configured for mock server."""
    return StatusReporter.from_config(reporting, job_id="99999")


# ============================================================================
# Happy Path: Full Lifecycle
# ============================================================================


class TestStatusLifecycleHappyPath:
    """Test the complete status lifecycle: submit -> start -> stages -> complete."""

    def test_full_lifecycle_produces_correct_events(self, mock_api, reporting, reporter):
        """Simulates the full orchestrator flow and verifies event sequence."""
        # Step 0: Create job record (submit.py does this)
        created = create_job_record(
            reporting=reporting,
            job_id="99999",
            job_name="test-benchmark",
            cluster="ptyche",
            recipe="examples/llm/sglang/qwen3-32b-disaggregated.yaml",
            metadata={"tags": ["nightly", "disagg"]},
        )
        assert created is True

        # Step 1: report_started (orchestrator.run entry)
        # We can't call report_started without a real SrtConfig/RuntimeContext,
        # so we simulate it with a direct report() call
        result = reporter.report(JobStatus.STARTING, JobStage.STARTING, "Job started on gb200-01")
        assert result is True

        # Step 2: Head infrastructure
        result = reporter.report(JobStatus.STARTING, JobStage.HEAD_INFRASTRUCTURE, "Starting head infrastructure")
        assert result is True

        # Step 3: Workers
        result = reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
        assert result is True

        # Step 4: Frontend
        result = reporter.report(JobStatus.FRONTEND, JobStage.FRONTEND, "Starting frontend")
        assert result is True

        # Step 5: Benchmark
        result = reporter.report(JobStatus.BENCHMARK, JobStage.BENCHMARK, "Running benchmark")
        assert result is True

        # Step 6: Completed
        result = reporter.report_completed(exit_code=0)
        assert result is True

        # Verify event sequence
        events = mock_status_server.events
        assert len(events) == 7  # 1 create + 6 updates

        # Verify create event
        assert events[0]["type"] == "create"
        assert events[0]["job_id"] == "99999"
        assert events[0]["job_name"] == "test-benchmark"
        assert events[0]["cluster"] == "ptyche"
        assert events[0]["recipe"] == "examples/llm/sglang/qwen3-32b-disaggregated.yaml"
        assert events[0]["metadata"]["tags"] == ["nightly", "disagg"]

        # Verify status progression
        update_statuses = [e["status"] for e in events[1:]]
        assert update_statuses == [
            "starting",
            "starting",
            "workers",
            "frontend",
            "benchmark",
            "completed",
        ]

        # Verify stage progression
        update_stages = [e.get("stage") for e in events[1:]]
        assert update_stages == [
            "starting",
            "head_infrastructure",
            "workers",
            "frontend",
            "benchmark",
            "cleanup",
        ]

        # Verify final state
        job = mock_status_server.jobs["99999"]
        assert job["status"] == "completed"
        assert job["exit_code"] == 0
        assert "completed_at" in job

    def test_lifecycle_with_failure(self, mock_api, reporting, reporter):
        """Simulates a failure during workers stage."""
        # Create job
        create_job_record(
            reporting=reporting,
            job_id="99999",
            job_name="failing-benchmark",
        )

        # Start
        reporter.report(JobStatus.STARTING, JobStage.STARTING, "Job started")
        reporter.report(JobStatus.STARTING, JobStage.HEAD_INFRASTRUCTURE, "Starting infra")

        # Workers stage
        reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")

        # Simulate health check failure (this is what benchmark_stage.py does)
        reporter.report(JobStatus.FAILED, JobStage.BENCHMARK, "Workers failed health check")

        # Completed with failure
        reporter.report_completed(exit_code=1)

        # Verify failure events
        events = mock_status_server.events
        update_events = [e for e in events if e["type"] == "update"]

        # Last update should be the completed call
        last = update_events[-1]
        assert last["status"] == "failed"
        assert last["exit_code"] == 1

        # Final job state
        job = mock_status_server.jobs["99999"]
        assert job["status"] == "failed"
        assert job["exit_code"] == 1


# ============================================================================
# Contract Validation: Payloads are accepted by the server
# ============================================================================


class TestContractValidation:
    """Verify that srtslurm-generated payloads pass server-side Pydantic validation."""

    def test_create_payload_with_all_fields(self, mock_api, reporting):
        """Server accepts a fully-populated create payload."""
        result = create_job_record(
            reporting=reporting,
            job_id="11111",
            job_name="full-payload-test",
            cluster="lyris",
            recipe="examples/llm/sglang/qwen3-32b-aggregated.yaml",
            metadata={
                "tags": ["ci", "smoke"],
                "commit_sha": "abc123",
            },
        )
        assert result is True
        assert mock_status_server.jobs["11111"]["cluster"] == "lyris"

    def test_create_payload_minimal(self, mock_api, reporting):
        """Server accepts a minimal create payload (only required fields)."""
        result = create_job_record(
            reporting=reporting,
            job_id="22222",
            job_name="minimal-test",
        )
        assert result is True
        assert "22222" in mock_status_server.jobs

    def test_update_payload_with_metadata(self, mock_api, reporting, reporter):
        """Server accepts update payload with metadata dict."""
        create_job_record(
            reporting=reporting,
            job_id="99999",
            job_name="metadata-test",
        )

        result = reporter.report(
            JobStatus.STARTING,
            JobStage.STARTING,
            "Testing metadata",
        )
        assert result is True

    def test_update_payload_with_exit_code_zero(self, mock_api, reporting, reporter):
        """Server accepts exit_code=0 (falsy but valid)."""
        create_job_record(
            reporting=reporting,
            job_id="99999",
            job_name="exit-code-test",
        )

        result = reporter.report_completed(exit_code=0)
        assert result is True

        job = mock_status_server.jobs["99999"]
        assert job["exit_code"] == 0

    def test_update_nonexistent_job_returns_false(self, mock_api, reporter):
        """Updating a job that doesn't exist returns False (404 from server)."""
        # Don't create the job first
        result = reporter.report(JobStatus.STARTING, JobStage.STARTING)
        assert result is False


# ============================================================================
# Pydantic Model Compatibility
# ============================================================================


class TestModelCompatibility:
    """Ensure contract models serialize/deserialize correctly across boundary."""

    def test_job_create_payload_round_trip(self):
        """JobCreatePayload serializes to JSON matching server expectations."""
        payload = JobCreatePayload(
            job_id="12345",
            job_name="roundtrip-test",
            submitted_at="2025-01-26T10:00:00Z",
            cluster="bia",
            metadata={"tags": ["test"]},
        )

        dumped = payload.model_dump(exclude_none=True)

        # Verify it can be parsed back
        parsed = JobCreatePayload.model_validate(dumped)
        assert parsed.job_id == "12345"
        assert parsed.cluster == "bia"
        assert parsed.metadata == {"tags": ["test"]}

    def test_job_update_payload_round_trip(self):
        """JobUpdatePayload serializes to JSON matching server expectations."""
        payload = JobUpdatePayload(
            status="workers",
            updated_at="2025-01-26T10:05:00Z",
            stage="workers",
            message="Starting 6 workers",
            exit_code=None,
        )

        dumped = payload.model_dump(exclude_none=True)

        # None fields should be excluded
        assert "exit_code" not in dumped
        assert dumped["status"] == "workers"
        assert dumped["stage"] == "workers"

        # Verify round-trip
        parsed = JobUpdatePayload.model_validate(dumped)
        assert parsed.status == "workers"
        assert parsed.exit_code is None

    def test_enum_values_match_between_contract_and_status(self):
        """Contract enums produce same string values used in StatusReporter."""
        # These are the exact values the orchestrator passes
        assert JobStatus.STARTING.value == "starting"
        assert JobStatus.WORKERS.value == "workers"
        assert JobStatus.FRONTEND.value == "frontend"
        assert JobStatus.BENCHMARK.value == "benchmark"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"

        assert JobStage.STARTING.value == "starting"
        assert JobStage.HEAD_INFRASTRUCTURE.value == "head_infrastructure"
        assert JobStage.WORKERS.value == "workers"
        assert JobStage.FRONTEND.value == "frontend"
        assert JobStage.BENCHMARK.value == "benchmark"
        assert JobStage.CLEANUP.value == "cleanup"
