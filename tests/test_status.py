# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for status reporting functionality."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from srtctl.contract import JobCreatePayload, JobStage, JobStatus, JobUpdatePayload
from srtctl.core.schema import ReportingConfig, ReportingStatusConfig
from srtctl.core.status import (
    StatusReporter,
    _resolve_endpoints,
    create_job_record,
)

# ============================================================================
# _resolve_endpoints Tests
# ============================================================================


class TestResolveEndpoints:
    """Test _resolve_endpoints() helper function."""

    def test_returns_empty_tuple_for_none(self):
        assert _resolve_endpoints(None) == ()

    def test_returns_single_endpoint(self):
        status = ReportingStatusConfig(endpoint="https://a.com")
        assert _resolve_endpoints(status) == ("https://a.com",)

    def test_returns_endpoints_list(self):
        status = ReportingStatusConfig(endpoints=["https://a.com", "https://b.com"])
        assert _resolve_endpoints(status) == ("https://a.com", "https://b.com")

    def test_merges_endpoint_and_endpoints(self):
        status = ReportingStatusConfig(endpoint="https://a.com", endpoints=["https://b.com"])
        assert _resolve_endpoints(status) == ("https://a.com", "https://b.com")

    def test_deduplicates(self):
        status = ReportingStatusConfig(endpoint="https://a.com", endpoints=["https://a.com", "https://b.com"])
        assert _resolve_endpoints(status) == ("https://a.com", "https://b.com")

    def test_strips_trailing_slashes(self):
        status = ReportingStatusConfig(endpoint="https://a.com/", endpoints=["https://b.com/"])
        assert _resolve_endpoints(status) == ("https://a.com", "https://b.com")

    def test_deduplicates_after_stripping_slashes(self):
        status = ReportingStatusConfig(endpoint="https://a.com/", endpoints=["https://a.com"])
        assert _resolve_endpoints(status) == ("https://a.com",)

    def test_empty_endpoint_and_none_endpoints(self):
        status = ReportingStatusConfig(endpoint=None, endpoints=None)
        assert _resolve_endpoints(status) == ()


# ============================================================================
# StatusReporter Tests
# ============================================================================


class TestStatusReporterFromConfig:
    """Test StatusReporter.from_config() factory method."""

    def test_creates_enabled_reporter_with_endpoint(self):
        """Reporter is enabled when endpoint is configured."""
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint="https://status.example.com"))

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.enabled is True
        assert reporter.job_id == "12345"
        assert reporter.api_endpoints == ("https://status.example.com",)

    def test_creates_disabled_reporter_without_endpoint(self):
        """Reporter is disabled when no endpoint configured."""
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint=None))

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.enabled is False
        assert reporter.api_endpoints == ()

    def test_creates_disabled_reporter_with_none_reporting(self):
        """Reporter is disabled when reporting config is None."""
        reporter = StatusReporter.from_config(None, job_id="12345")

        assert reporter.enabled is False
        assert reporter.api_endpoints == ()

    def test_creates_disabled_reporter_with_none_status(self):
        """Reporter is disabled when status config is None."""
        reporting = ReportingConfig(status=None)

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.enabled is False

    def test_strips_trailing_slash_from_endpoint(self):
        """Trailing slash is removed from endpoint URL."""
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint="https://status.example.com/"))

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.api_endpoints == ("https://status.example.com",)

    def test_creates_reporter_with_endpoints_list(self):
        """Reporter uses endpoints list when provided."""
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoints=["https://a.com", "https://b.com"]))

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.enabled is True
        assert reporter.api_endpoints == ("https://a.com", "https://b.com")

    def test_creates_reporter_with_both_endpoint_and_endpoints(self):
        """Reporter merges endpoint and endpoints, deduplicating."""
        reporting = ReportingConfig(
            status=ReportingStatusConfig(
                endpoint="https://a.com",
                endpoints=["https://b.com"],
            )
        )

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.api_endpoints == ("https://a.com", "https://b.com")


class TestStatusReporterReport:
    """Test StatusReporter.report() method."""

    def test_returns_false_when_disabled(self):
        """Report returns False when reporter is disabled."""
        reporter = StatusReporter(job_id="12345", api_endpoints=())

        result = reporter.report(JobStatus.STARTING)

        assert result is False

    @patch("srtctl.core.status.requests.put")
    def test_report_started_includes_cpu_allocation(self, mock_put):
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="31315", api_endpoints=("https://status.example.com",))
        config = SimpleNamespace(
            name="b300-agg-smoke",
            model=SimpleNamespace(path="/model", precision="fp8"),
            resources=SimpleNamespace(
                gpu_type="b300",
                gpus_per_node=8,
                num_prefill=0,
                num_decode=0,
                num_agg=1,
            ),
            benchmark=SimpleNamespace(type="sa-bench"),
            backend_type="vllm",
            frontend=SimpleNamespace(type="dynamo"),
        )
        runtime = SimpleNamespace(nodes=SimpleNamespace(head="b300-010"), log_dir="/lustre/outputs/31315/logs/run")
        snapshot = {
            "cpus": {"allocated_total": 2, "allocated_per_node": [2]},
            "cpu_check": {"status": "warning", "minimum_cpu_count": 4},
        }

        assert reporter.report_started(config, runtime, resource_snapshot=snapshot) is True

        payload = mock_put.call_args.kwargs["json"]
        assert payload["metadata"]["resources"]["cpu_allocation"]["allocated_total"] == 2
        assert payload["metadata"]["resources"]["cpu_check"]["status"] == "warning"
        # Identity rides along so a collector that missed the submit-time POST can still name the row.
        assert payload["metadata"]["job_name"] == "b300-agg-smoke"
        assert "cluster" in payload["metadata"]

    @patch("srtctl.core.status.requests.put")
    def test_sends_put_request_to_correct_url(self, mock_put):
        """Report sends PUT request to /api/jobs/{job_id}."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        reporter.report(JobStatus.WORKERS, stage=JobStage.WORKERS)

        mock_put.assert_called_once()
        call_args = mock_put.call_args
        assert call_args[0][0] == "https://status.example.com/api/jobs/12345"

    @patch("srtctl.core.status.requests.put")
    def test_returns_true_on_success(self, mock_put):
        """Report returns True on HTTP 200."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        result = reporter.report(JobStatus.STARTING)

        assert result is True

    @patch("srtctl.core.status.requests.put")
    def test_returns_false_on_http_error(self, mock_put):
        """Report returns False on non-200 status."""
        mock_put.return_value = MagicMock(status_code=500)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        result = reporter.report(JobStatus.STARTING)

        assert result is False

    @patch("srtctl.core.status.requests.put")
    def test_returns_false_on_request_exception(self, mock_put):
        """Report returns False on network error (fire-and-forget)."""
        import requests

        mock_put.side_effect = requests.exceptions.ConnectionError("Network error")
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        result = reporter.report(JobStatus.STARTING)

        assert result is False

    @patch("srtctl.core.status.requests.put")
    def test_sends_to_all_endpoints(self, mock_put):
        """Report sends PUT to every configured endpoint."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://a.com", "https://b.com"))

        result = reporter.report(JobStatus.STARTING)

        assert result is True
        assert mock_put.call_count == 2
        urls = [call.args[0] for call in mock_put.call_args_list]
        assert "https://a.com/api/jobs/12345" in urls
        assert "https://b.com/api/jobs/12345" in urls

    @patch("srtctl.core.status.requests.put")
    def test_one_endpoint_failing_does_not_block_others(self, mock_put):
        """If first endpoint fails (both attempts), second still gets called and result is True."""
        import requests as req

        mock_put.side_effect = [
            req.exceptions.ConnectionError("Network error"),  # a.com, attempt 1
            req.exceptions.ConnectionError("Network error"),  # a.com, attempt 2
            MagicMock(status_code=200),  # b.com
        ]
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://a.com", "https://b.com"))

        result = reporter.report(JobStatus.STARTING)

        assert result is True
        assert mock_put.call_count == 3


class TestStatusReporterCompleted:
    """Test StatusReporter.report_completed() method."""

    def test_returns_false_when_disabled(self):
        """report_completed returns False when disabled."""
        reporter = StatusReporter(job_id="12345", api_endpoints=())

        result = reporter.report_completed(exit_code=0)

        assert result is False

    @patch("srtctl.core.status.requests.put")
    def test_reports_completed_status_on_success(self, mock_put):
        """Exit code 0 reports COMPLETED status."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        reporter.report_completed(exit_code=0)

        call_args = mock_put.call_args
        payload = call_args[1]["json"]
        assert payload["status"] == "completed"
        assert payload["exit_code"] == 0

    @patch("srtctl.core.status.requests.put")
    def test_reports_failed_status_on_nonzero_exit(self, mock_put):
        """Non-zero exit code reports FAILED status."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        reporter.report_completed(exit_code=1)

        call_args = mock_put.call_args
        payload = call_args[1]["json"]
        assert payload["status"] == "failed"
        assert payload["exit_code"] == 1

    @patch("srtctl.core.status.requests.put")
    def test_report_completed_sends_to_all_endpoints(self, mock_put):
        """report_completed sends to all configured endpoints."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://a.com", "https://b.com"))

        result = reporter.report_completed(exit_code=0)

        assert result is True
        assert mock_put.call_count == 2

    @patch("srtctl.core.status.requests.put")
    def test_report_completed_includes_logs_url_when_provided(self, mock_put):
        """logs_url is forwarded in the completion PUT when supplied."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        reporter.report_completed(exit_code=0, logs_url="s3://bucket/prefix/12345/")

        payload = mock_put.call_args[1]["json"]
        assert payload["logs_url"] == "s3://bucket/prefix/12345/"

    @patch("srtctl.core.status.requests.put")
    def test_report_completed_omits_logs_url_when_none(self, mock_put):
        """logs_url is excluded from the payload when None (exclude_none)."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        reporter.report_completed(exit_code=0)

        payload = mock_put.call_args[1]["json"]
        assert "logs_url" not in payload


class TestStatusReporterArtifacts:
    """Test StatusReporter.report_artifacts() method."""

    def test_returns_false_when_disabled(self):
        """report_artifacts returns False when disabled."""
        reporter = StatusReporter(job_id="12345", api_endpoints=())

        assert reporter.report_artifacts(logs_url="s3://bucket/") is False

    def test_returns_false_when_logs_url_empty(self):
        """report_artifacts is a no-op when no logs_url is supplied."""
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        assert reporter.report_artifacts(logs_url="") is False

    @patch("srtctl.core.status.requests.put")
    def test_report_artifacts_sends_logs_url_without_completing(self, mock_put):
        """report_artifacts forwards logs_url on a benchmark-stage PUT."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoints=("https://status.example.com",))

        result = reporter.report_artifacts(logs_url="s3://bucket/prefix/12345/")

        assert result is True
        payload = mock_put.call_args[1]["json"]
        assert payload["logs_url"] == "s3://bucket/prefix/12345/"
        # Must NOT flip the lifecycle to completed; that's report_completed's job
        assert payload["status"] != "completed"
        # Stays in the benchmark lifecycle so the dashboard knows the job is
        # still active (AI analysis may still be running).
        assert payload["status"] == "benchmark"


# ============================================================================
# Payload Model Tests
# ============================================================================


class TestJobCreatePayload:
    """Test JobCreatePayload Pydantic model."""

    def test_model_dump_includes_required_fields(self):
        """model_dump includes all required fields."""
        payload = JobCreatePayload(
            job_id="12345",
            job_name="test-job",
            submitted_at="2025-01-26T10:00:00Z",
        )

        result = payload.model_dump(exclude_none=True)

        assert result["job_id"] == "12345"
        assert result["job_name"] == "test-job"
        assert result["submitted_at"] == "2025-01-26T10:00:00Z"

    def test_model_dump_excludes_none_values(self):
        """model_dump(exclude_none=True) excludes fields with None values."""
        payload = JobCreatePayload(
            job_id="12345",
            job_name="test-job",
            submitted_at="2025-01-26T10:00:00Z",
            cluster=None,
            recipe=None,
            metadata=None,
        )

        result = payload.model_dump(exclude_none=True)

        assert "cluster" not in result
        assert "recipe" not in result
        assert "metadata" not in result

    def test_model_dump_includes_optional_fields_when_set(self):
        """model_dump includes optional fields when they have values."""
        payload = JobCreatePayload(
            job_id="12345",
            job_name="test-job",
            submitted_at="2025-01-26T10:00:00Z",
            cluster="gpu-cluster",
            recipe="configs/test.yaml",
            metadata={"key": "value"},
        )

        result = payload.model_dump(exclude_none=True)

        assert result["cluster"] == "gpu-cluster"
        assert result["recipe"] == "configs/test.yaml"
        assert result["metadata"] == {"key": "value"}


class TestJobUpdatePayload:
    """Test JobUpdatePayload Pydantic model."""

    def test_model_dump_includes_required_fields(self):
        """model_dump includes required fields."""
        payload = JobUpdatePayload(
            status="starting",
            updated_at="2025-01-26T10:00:00Z",
        )

        result = payload.model_dump(exclude_none=True)

        assert result["status"] == "starting"
        assert result["updated_at"] == "2025-01-26T10:00:00Z"

    def test_model_dump_excludes_none_values(self):
        """model_dump(exclude_none=True) excludes fields with None values."""
        payload = JobUpdatePayload(
            status="starting",
            updated_at="2025-01-26T10:00:00Z",
            stage=None,
            message=None,
        )

        result = payload.model_dump(exclude_none=True)

        assert "stage" not in result
        assert "message" not in result

    def test_model_dump_includes_exit_code_when_set(self):
        """model_dump includes exit_code when set."""
        payload = JobUpdatePayload(
            status="completed",
            updated_at="2025-01-26T10:00:00Z",
            exit_code=0,
        )

        result = payload.model_dump(exclude_none=True)

        assert result["exit_code"] == 0


# ============================================================================
# create_job_record Tests
# ============================================================================


class TestCreateJobRecord:
    """Test create_job_record() standalone function."""

    def test_returns_false_when_no_reporting_config(self):
        """Returns False when reporting config is None."""
        result = create_job_record(
            reporting=None,
            job_id="12345",
            job_name="test-job",
        )

        assert result is False

    def test_returns_false_when_no_endpoint(self):
        """Returns False when no endpoint configured."""
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint=None))

        result = create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        assert result is False

    @patch("srtctl.core.status.requests.post")
    def test_sends_post_request_to_correct_url(self, mock_post):
        """Sends POST request to /api/jobs."""
        mock_post.return_value = MagicMock(status_code=201)
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint="https://status.example.com"))

        create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://status.example.com/api/jobs"

    @patch("srtctl.core.status.requests.post")
    def test_returns_true_on_201_created(self, mock_post):
        """Returns True on HTTP 201."""
        mock_post.return_value = MagicMock(status_code=201)
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint="https://status.example.com"))

        result = create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        assert result is True

    @patch("srtctl.core.status.requests.post")
    def test_returns_false_on_request_exception(self, mock_post):
        """Returns False on network error (fire-and-forget)."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Network error")
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint="https://status.example.com"))

        result = create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        assert result is False

    @patch("srtctl.core.status.requests.post")
    def test_sends_to_all_endpoints(self, mock_post):
        """Sends POST to every configured endpoint."""
        mock_post.return_value = MagicMock(status_code=201)
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoints=["https://a.com", "https://b.com"]))

        result = create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        assert result is True
        assert mock_post.call_count == 2
        urls = [call.args[0] for call in mock_post.call_args_list]
        assert "https://a.com/api/jobs" in urls
        assert "https://b.com/api/jobs" in urls

    @patch("srtctl.core.status.requests.post")
    def test_one_endpoint_failing_does_not_block_others(self, mock_post):
        """If first endpoint fails (both attempts), second still gets called."""
        import requests as req

        mock_post.side_effect = [
            req.exceptions.ConnectionError("Network error"),  # a.com, attempt 1
            req.exceptions.ConnectionError("Network error"),  # a.com, attempt 2
            MagicMock(status_code=201),  # b.com
        ]
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoints=["https://a.com", "https://b.com"]))

        result = create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        assert result is True
        assert mock_post.call_count == 3


# ============================================================================
# Enum Tests
# ============================================================================


class TestJobStatusEnum:
    """Test JobStatus enum values match API spec."""

    def test_status_values(self):
        """Status values match API spec."""
        assert JobStatus.SUBMITTED.value == "submitted"
        assert JobStatus.STARTING.value == "starting"
        assert JobStatus.WORKERS.value == "workers"
        assert JobStatus.FRONTEND.value == "frontend"
        assert JobStatus.BENCHMARK.value == "benchmark"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.TIMEOUT.value == "timeout"


class TestJobStageEnum:
    """Test JobStage enum values match API spec."""

    def test_stage_values(self):
        """Stage values match API spec."""
        assert JobStage.STARTING.value == "starting"
        assert JobStage.HEAD_INFRASTRUCTURE.value == "head_infrastructure"
        assert JobStage.PREFLIGHT.value == "preflight"
        assert JobStage.WORKERS.value == "workers"
        assert JobStage.FRONTEND.value == "frontend"
        assert JobStage.BENCHMARK.value == "benchmark"
        assert JobStage.CLEANUP.value == "cleanup"


class TestTachometerStreaming:
    """The collector owns capture decoding, deduplication and text decoding."""

    @staticmethod
    def _store(tmp_path):
        from srtctl.status_server.store import StatusStore

        store = StatusStore(tmp_path / "collector.sqlite3")
        store.init()
        return store

    @staticmethod
    def _capture(tmp_path, rows, suffix="arrow"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from pyarrow import ipc

        path = tmp_path / f"capture.{suffix}"
        table = pa.Table.from_pylist(rows)
        if suffix == "parquet":
            pq.write_table(table, path)
        else:
            with ipc.new_stream(path, table.schema) as writer:
                writer.write_table(table)
        return path.read_bytes()

    @staticmethod
    def _upload(store, data, file="tachometer/local/current.arrow", generation=None):
        from uuid import uuid4

        generation = generation or uuid4().hex
        result = None
        for offset in range(0, len(data), 127):
            chunk = data[offset : offset + 127]
            result = store.append_capture("8", file, generation, offset, len(data), chunk)
            assert result["next_offset"] == offset + len(chunk)
        assert result["complete"]
        return generation

    @staticmethod
    def _rows(store):
        import json

        text, _ = store.read_log("8", "tachometer_rows.jsonl")
        return [json.loads(line) for line in text.splitlines()]

    def test_snapshots_rotation_and_compaction_deduplicate_on_collector(self, tmp_path):
        store = self._store(tmp_path)
        rows = [
            {"timestamp_ns": 20, "metric_name": "z", "metric_value": 1.0},
            {"timestamp_ns": 20, "metric_name": "a", "metric_value": 2.0},
            {"timestamp_ns": 10, "metric_name": "a", "metric_value": 3.0},
            {"timestamp_ns": 30, "metric_name": "a", "metric_value": 4.0},
        ]
        for end in (1, 2):
            self._upload(store, self._capture(tmp_path, rows[:end]))
        self._upload(store, self._capture(tmp_path, rows[:3], "parquet"), "tachometer/local/out-1.parquet")
        self._upload(store, self._capture(tmp_path, rows[3:]))
        compacted = [{**row, "metric_name_clean": row["metric_name"]} for row in reversed(rows)]
        self._upload(store, self._capture(tmp_path, compacted, "parquet"), "tachometer/local/final.parquet")
        assert self._rows(store) == rows
        with store._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM capture_chunks WHERE data IS NOT NULL").fetchone()[0] == 0

    def test_exact_retry_after_lost_final_response_never_reprocesses(self, tmp_path):
        store = self._store(tmp_path)
        data = self._capture(tmp_path, [{"metric_name": "a", "metric_value": 1}])
        generation = "a" * 32
        result = store.append_capture("8", "current.arrow", generation, 0, len(data), data)
        with patch.object(type(store), "_process_capture", side_effect=AssertionError("Already processed")):
            assert store.append_capture("8", "current.arrow", generation, 0, len(data), data) == result
        assert len(self._rows(store)) == 1

    def test_incomplete_generation_is_never_decoded_and_is_pruned(self, tmp_path):
        store = self._store(tmp_path)
        data = self._capture(tmp_path, [{"metric_name": "a", "metric_value": 1}])
        first = store.append_capture("8", "current.arrow", "a" * 32, 0, len(data), data[:100])
        assert first == {"job_id": "8", "next_offset": 100, "complete": False}
        assert self._rows(store) == []
        store.append_capture("8", "current.arrow", "b" * 32, 0, len(data), data[:100])
        with store._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM job_captures").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM capture_chunks").fetchone()[0] == 1
        store.append_capture("8", "current.arrow", "b" * 32, 100, len(data), data[100:])
        assert len(self._rows(store)) == 1

    def test_capture_conflicts_and_out_of_order_chunks(self, tmp_path):
        import pytest

        from srtctl.status_server.store import LogChunkConflict

        store = self._store(tmp_path)
        args = ("8", "current.arrow", "a" * 32)
        store.append_capture(*args, 0, 1000, b"first")
        assert store.append_capture(*args, 0, 1000, b"first")["next_offset"] == 5
        for offset, total, data in ((0, 1000, b"other"), (0, 1001, b"first"), (9, 1000, b"gap"), (2, 1000, b"overlap")):
            with pytest.raises(LogChunkConflict):
                store.append_capture(*args, offset, total, data)
        assert self._rows(store) == []

    def test_decode_failure_retries_without_duplicate_output(self, tmp_path):
        import pyarrow as pa
        import pytest

        store = self._store(tmp_path)
        rows = [{"metric_name": name, "metric_value": 1} for name in ("a", "b")]
        data = self._capture(tmp_path, rows)
        args = ("8", "current.arrow", "a" * 32, 0, len(data), data)

        def failing_batches(path):
            yield pa.RecordBatch.from_pylist(rows[:1])
            raise OSError("temporary decoder failure")

        with patch("srtctl.dsight.metrics.batches", failing_batches), pytest.raises(OSError):
            store.append_capture(*args)
        assert self._rows(store) == rows[:1]
        store.append_capture(*args)
        assert self._rows(store) == rows

    def test_raw_log_decodes_split_utf8_and_final_invalid_tail(self, tmp_path):
        store = self._store(tmp_path)
        raw = "hello €!".encode()
        assert store.append_raw_log("8", "worker.log", 0, raw[:7]) == 1
        assert store.read_log("8", "worker.log") == ("hello ", 6)
        assert store.append_raw_log("8", "worker.log", 0, raw[:7]) == 0
        store.append_raw_log("8", "worker.log", 7, raw[7:] + b"\xe2")
        assert store.read_log("8", "worker.log", offset=6) == ("€!", len(raw))
        store.append_raw_log("8", "worker.log", len(raw) + 1, b"", final=True)
        assert store.read_log("8", "worker.log", offset=len(raw)) == ("�", len(raw) + 1)

    def test_raw_log_partial_reads_resume_within_chunk(self, tmp_path):
        store = self._store(tmp_path)
        raw = "a€b".encode()
        store.append_raw_log("8", "worker.log", 0, raw)
        assert store.read_log("8", "worker.log", max_bytes=3) == ("a", 1)
        assert store.read_log("8", "worker.log", offset=1, max_bytes=4) == ("€b", 5)

    def test_delete_removes_capture_state_and_observation_index(self, tmp_path):
        store = self._store(tmp_path)
        store.create_job("8", "test")
        self._upload(store, self._capture(tmp_path, [{"metric_name": "a", "metric_value": 1}]))
        store.append_raw_log("8", "worker.log", 0, b"done", final=True)
        assert store.delete_job("8")
        with store._connect() as conn:
            for table in ("job_captures", "capture_chunks", "metric_observations", "job_logs", "job_log_ends"):
                assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    def test_raw_routes_validate_bounds_and_paths(self, tmp_path):
        from urllib.parse import urlencode

        import pytest

        from srtctl.status_server.server import ApiError, _raw_upload

        store = self._store(tmp_path)
        params = {"file": "current.arrow", "generation": "a" * 32, "offset": 0, "total": 10}
        for invalid in (
            {"file": "../current.arrow"},
            {"file": "/current.arrow"},
            {"file": "worker.log"},
            {"generation": "bad"},
            {"offset": -1},
            {"offset": 10},
            {"total": 0},
            {"total": 1 << 63},
        ):
            with pytest.raises(ApiError) as error:
                _raw_upload(store, "/api/jobs/8/captures?" + urlencode({**params, **invalid}), b"data")
            assert error.value.status == 422

    def test_binary_http_upload_requires_write_token_and_serves_rows(self, tmp_path):
        import socket
        import threading
        from urllib.parse import urlencode

        import requests

        from srtctl.status_server.server import AuthPolicy, make_server

        store = self._store(tmp_path)
        server = make_server(store, port=0, auth=AuthPolicy(write_token="secret", read_token="viewer"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}/api/jobs/8"
            data = self._capture(tmp_path, [{"metric_name": "a", "metric_value": 1}])
            path = (
                base
                + "/captures?"
                + urlencode(
                    {
                        "file": "tachometer/local/current.arrow",
                        "generation": "a" * 32,
                        "offset": 0,
                        "total": len(data),
                        "cluster": "test-cluster",
                    }
                )
            )
            headers = {"Content-Type": "application/octet-stream"}
            assert requests.post(path, data=data, headers=headers, timeout=5).status_code == 401
            assert (
                requests.post(
                    path, data=data, headers={**headers, "Authorization": "Bearer viewer"}, timeout=5
                ).status_code
                == 403
            )
            # A disconnected sender must not leave a shorter acknowledged chunk.
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as connection:
                connection.sendall(
                    b"POST /api/jobs/8/logs?file=truncated.log&offset=0 HTTP/1.1\r\n"
                    b"Host: localhost\r\nAuthorization: Bearer secret\r\n"
                    b"Content-Type: application/octet-stream\r\nContent-Length: 10\r\n\r\nabc"
                )
                connection.shutdown(socket.SHUT_WR)
                assert b"400" in connection.recv(4096).split(b"\r\n")[0]
            assert store.read_log("8", "truncated.log") == ("", 0)
            headers["Authorization"] = "Bearer secret"
            result = requests.post(path, data=data, headers=headers, timeout=5)
            assert result.status_code == 200
            assert result.json() == {"job_id": "8", "next_offset": len(data), "complete": True}
            result = requests.get(base + "/logs?file=tachometer_rows.jsonl", headers=headers, timeout=5)
            assert result.status_code == 200
            assert '"metric_name":"a"' in result.json()["data"]
            log = base + "/logs?file=worker.log&offset=0&final=1"
            assert requests.post(log, data=b"raw log", headers=headers, timeout=5).json()["stored"] == 1
            assert requests.get(base + "/logs?file=worker.log", headers=headers, timeout=5).json()["data"] == "raw log"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
