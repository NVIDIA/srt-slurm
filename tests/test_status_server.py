# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for ``srtctl status-server``.

The real HTTP server is bound to an ephemeral loopback port and driven with
``requests`` and the real ``StatusReporter`` / ``create_job_record``, so these
tests prove the reporter's payloads, the contract models, and the SQLite store
agree with each other with nothing patched in between.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from srtctl.cli import submit as submit_cli
from srtctl.contract import JobStage, JobStatus, JobSummary
from srtctl.core.schema import ReportingConfig, ReportingStatusConfig
from srtctl.core.status import StatusReporter, create_job_record
from srtctl.status_server import StatusStore, make_server

NOW = "2026-01-01T00:00:00Z"


@pytest.fixture
def store(tmp_path: Path) -> StatusStore:
    store = StatusStore(tmp_path / "status.db")
    store.init()
    return store


@pytest.fixture
def base_url(store: StatusStore):
    server = make_server(store, host="127.0.0.1", port=0)
    # Short poll interval so server.shutdown() returns quickly at teardown.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def reporting(base_url: str) -> ReportingConfig:
    return ReportingConfig(status=ReportingStatusConfig(endpoint=base_url))


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(path="/models/llama", precision="fp8"),
        resources=SimpleNamespace(gpu_type="h100", gpus_per_node=8, num_prefill=1, num_decode=2, num_agg=0),
        benchmark=SimpleNamespace(type="sa-bench"),
        backend_type="sglang",
        frontend=SimpleNamespace(type="dynamo"),
    )


def _runtime(log_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(nodes=SimpleNamespace(head="node-01"), log_dir=log_dir)


def _get(base_url: str, path: str) -> requests.Response:
    return requests.get(f"{base_url}{path}", timeout=5)


def _post(base_url: str, body: dict) -> requests.Response:
    return requests.post(f"{base_url}/api/jobs", json=body, timeout=5)


def _put(base_url: str, job_id: str, body: dict) -> requests.Response:
    return requests.put(f"{base_url}/api/jobs/{job_id}", json={"updated_at": NOW, **body}, timeout=5)


def _create(base_url: str, job_id: str, **extra) -> requests.Response:
    return _post(base_url, {"job_id": job_id, "job_name": f"name-{job_id}", "submitted_at": NOW, **extra})


# ============================================================================
# Driven by the real reporter
# ============================================================================


class TestLifecycleThroughReporter:
    def test_healthy_sweep_lands_as_ordered_events(self, base_url, reporting, tmp_path):
        log_dir = tmp_path / "outputs" / "777" / "logs" / "777_1P_2D"
        assert create_job_record(
            reporting,
            job_id="777",
            job_name="llama-pd",
            cluster="h100-rack",
            recipe="examples/pd.yaml",
            metadata={"tags": ["suite:nightly"]},
        )
        reporter = StatusReporter.from_config(reporting, job_id="777")
        assert reporter.enabled
        assert reporter.report_started(_config(), _runtime(log_dir))
        # The sequence do_sweep.py and BenchmarkStageMixin emit on a healthy run.
        assert reporter.report(JobStatus.STARTING, JobStage.HEAD_INFRASTRUCTURE, "Starting head infrastructure")
        assert reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
        assert reporter.report(JobStatus.FRONTEND, JobStage.FRONTEND, "Starting frontend")
        assert reporter.report(JobStatus.FRONTEND, JobStage.FRONTEND, "Inference endpoint ready")
        assert reporter.report(JobStatus.BENCHMARK, JobStage.BENCHMARK, "Running benchmark")
        assert reporter.report_artifacts("s3://bucket/777/")
        assert reporter.report_completed(0, logs_url="s3://bucket/777/")

        job = _get(base_url, "/api/jobs/777").json()
        assert job["job_name"] == "llama-pd"
        assert job["cluster"] == "h100-rack"
        assert job["recipe"] == "examples/pd.yaml"
        assert job["status"] == "completed"
        assert job["stage"] == "cleanup"
        assert job["exit_code"] == 0
        assert job["logs_url"] == "s3://bucket/777/"
        assert job["started_at"] and job["completed_at"]
        # metadata from the POST and from report_started are merged, not replaced
        assert job["metadata"]["tags"] == ["suite:nightly"]
        assert job["metadata"]["head_node"] == "node-01"
        assert job["metadata"]["log_dir"] == str(log_dir)
        assert job["metadata"]["resources"]["decode_workers"] == 2
        assert [(e["status"], e["stage"], e["message"]) for e in job["events"]] == [
            ("submitted", None, None),
            ("starting", "starting", "Job started on node-01"),
            ("starting", "head_infrastructure", "Starting head infrastructure"),
            ("workers", "workers", "Starting workers"),
            ("frontend", "frontend", "Starting frontend"),
            ("frontend", "frontend", "Inference endpoint ready"),
            ("benchmark", "benchmark", "Running benchmark"),
            ("benchmark", "cleanup", "Artifacts uploaded"),
            ("completed", "cleanup", "Benchmark completed successfully"),
        ]
        assert [e["id"] for e in job["events"]] == sorted(e["id"] for e in job["events"])

    def test_failed_sweep(self, base_url, reporting):
        create_job_record(reporting, job_id="778", job_name="boom")
        reporter = StatusReporter.from_config(reporting, job_id="778")
        reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
        reporter.report(JobStatus.FAILED, JobStage.WORKERS, "Workers failed health check")
        reporter.report_completed(1)

        job = _get(base_url, "/api/jobs/778").json()
        assert job["status"] == "failed"
        assert job["exit_code"] == 1
        assert job["message"] == "Job failed with exit code 1"
        assert job["logs_url"] is None
        assert [e["status"] for e in job["events"]] == ["submitted", "workers", "failed", "failed"]

    def test_update_before_create_makes_placeholder(self, base_url, reporting, tmp_path):
        """A sweep whose submit-time POST never reached the collector still shows up."""
        reporter = StatusReporter.from_config(reporting, job_id="779")
        assert reporter.report_started(_config(), _runtime(tmp_path))

        job = _get(base_url, "/api/jobs/779").json()
        assert job["job_name"] == "job-779"
        assert job["status"] == "starting"
        assert job["submitted_at"] == job["started_at"]
        assert [e["status"] for e in job["events"]] == ["starting"]

        # A late POST is acknowledged but does not rewind the job.
        assert create_job_record(reporting, job_id="779", job_name="late")
        job = _get(base_url, "/api/jobs/779").json()
        assert job["status"] == "starting"
        assert len(job["events"]) == 1

    def test_multiple_endpoints_each_receive_everything(self, tmp_path):
        stores = [StatusStore(tmp_path / f"{i}.db") for i in range(2)]
        servers = [make_server(store, host="127.0.0.1", port=0) for store in stores]
        threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
        for store, thread in zip(stores, threads, strict=True):
            store.init()
            thread.start()
        try:
            urls = [f"http://127.0.0.1:{server.server_address[1]}" for server in servers]
            reporting = ReportingConfig(status=ReportingStatusConfig(endpoints=urls))
            assert create_job_record(reporting, job_id="1", job_name="dual")
            StatusReporter.from_config(reporting, job_id="1").report(JobStatus.WORKERS, JobStage.WORKERS, "go")
            for url in urls:
                job = _get(url, "/api/jobs/1").json()
                assert job["status"] == "workers"
                assert len(job["events"]) == 2
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()


# ============================================================================
# HTTP surface
# ============================================================================


class TestHttpApi:
    def test_health(self, base_url):
        response = _get(base_url, "/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_create_is_idempotent_and_returns_current_status(self, base_url):
        assert _create(base_url, "1", cluster="c1").status_code == 201
        assert _put(base_url, "1", {"status": "workers"}).status_code == 200

        second = _create(base_url, "1", job_name="ignored")
        assert second.status_code == 201
        assert second.json() == {"job_id": "1", "status": "workers"}

        job = _get(base_url, "/api/jobs/1").json()
        assert job["job_name"] == "name-1"
        assert job["cluster"] == "c1"
        assert len(job["events"]) == 2

    def test_same_status_new_message_is_an_event_but_metadata_patch_is_not(self, base_url):
        _create(base_url, "1")
        _put(base_url, "1", {"status": "benchmark", "stage": "benchmark", "message": "Running benchmark"})
        _put(
            base_url, "1", {"status": "benchmark", "stage": "benchmark", "message": "Running post-benchmark evaluation"}
        )
        # Same triple again, only metadata differs: no event.
        _put(
            base_url,
            "1",
            {
                "status": "benchmark",
                "stage": "benchmark",
                "message": "Running post-benchmark evaluation",
                "metadata": {"x": 1},
            },
        )

        job = _get(base_url, "/api/jobs/1").json()
        assert [e["message"] for e in job["events"]] == [None, "Running benchmark", "Running post-benchmark evaluation"]
        assert job["metadata"] == {"x": 1}

    def test_artifacts_and_metadata_merge_while_results_replace(self, base_url):
        _create(base_url, "1", metadata={"tags": ["a"]})
        _put(base_url, "1", {"status": "benchmark", "artifacts": {"rollup": "r.json"}, "benchmark_results": {"tp": 1}})
        _put(
            base_url,
            "1",
            {
                "status": "completed",
                "artifacts": {"dashboard": "d.html"},
                "metadata": {"model": "m"},
                "benchmark_results": {"tp": 2},
            },
        )

        job = _get(base_url, "/api/jobs/1").json()
        assert job["artifacts"] == {"rollup": "r.json", "dashboard": "d.html"}
        assert job["metadata"] == {"tags": ["a"], "model": "m"}
        assert job["benchmark_results"] == {"tp": 2}

    def test_rejects_unknown_status_and_stage_but_accepts_preflight(self, base_url):
        bad_status = _put(base_url, "1", {"status": "exploded"})
        assert bad_status.status_code == 422
        assert "exploded" in bad_status.json()["detail"]

        bad_stage = _put(base_url, "1", {"status": "workers", "stage": "lunch"})
        assert bad_stage.status_code == 422
        assert "lunch" in bad_stage.json()["detail"]

        # Rejected PUTs store nothing, not even a placeholder.
        assert _get(base_url, "/api/jobs/1").status_code == 404

        # preflight is an srtctl stage that older collectors did not know about.
        assert _put(base_url, "1", {"status": "starting", "stage": "preflight"}).status_code == 200
        job = _get(base_url, "/api/jobs/1").json()
        assert [(e["status"], e["stage"]) for e in job["events"]] == [("starting", "preflight")]

    def test_contract_validation_errors(self, base_url):
        assert _post(base_url, {}).status_code == 422
        assert _post(base_url, {"job_id": "1"}).status_code == 422
        # status and updated_at are both required on PUT
        assert requests.put(f"{base_url}/api/jobs/1", json={"status": "workers"}, timeout=5).status_code == 422
        assert requests.put(f"{base_url}/api/jobs/1", timeout=5).status_code == 422
        assert (
            requests.put(
                f"{base_url}/api/jobs/1", json={"status": "workers", "updated_at": NOW, "exit_code": "x"}, timeout=5
            ).status_code
            == 422
        )

    def test_bad_json_and_unknown_routes(self, base_url):
        headers = {"Content-Type": "application/json"}
        assert requests.post(f"{base_url}/api/jobs", data="not json", headers=headers, timeout=5).status_code == 400
        assert requests.post(f"{base_url}/api/jobs", data="[1, 2]", headers=headers, timeout=5).status_code == 400
        assert _get(base_url, "/api/nope").status_code == 404
        assert _get(base_url, "/api/jobs/missing").status_code == 404
        assert _get(base_url, "/api/jobs/missing/events").status_code == 404
        assert requests.delete(f"{base_url}/api/jobs/missing", timeout=5).status_code == 404
        assert requests.post(f"{base_url}/api/health", timeout=5).status_code == 404

    def test_list_jobs_filters_and_pages(self, base_url):
        _create(base_url, "1", cluster="a", submitted_at="2026-01-01T00:00:01Z")
        _create(base_url, "2", cluster="a", submitted_at="2026-01-01T00:00:02Z")
        _create(base_url, "3", cluster="b", submitted_at="2026-01-01T00:00:03Z")
        _put(base_url, "2", {"status": "failed"})

        everything = _get(base_url, "/api/jobs").json()
        assert everything["total"] == 3
        assert [job["job_id"] for job in everything["jobs"]] == ["3", "2", "1"]
        assert set(everything["jobs"][0]) == set(JobSummary.model_fields)

        assert _get(base_url, "/api/jobs?cluster=a").json()["total"] == 2
        assert [job["job_id"] for job in _get(base_url, "/api/jobs?status=failed").json()["jobs"]] == ["2"]

        page = _get(base_url, "/api/jobs?per_page=1&page=2").json()
        assert page == {"jobs": [page["jobs"][0]], "total": 3, "page": 2, "per_page": 1}
        assert page["jobs"][0]["job_id"] == "2"

        assert _get(base_url, "/api/jobs?per_page=0").status_code == 422
        assert _get(base_url, "/api/jobs?per_page=101").status_code == 422
        assert _get(base_url, "/api/jobs?page=abc").status_code == 422

    def test_event_feeds_support_cursors(self, base_url):
        assert _get(base_url, "/api/events").json() == {"events": [], "next_cursor": None}

        _create(base_url, "j1")
        _put(base_url, "j1", {"status": "starting"})
        _create(base_url, "j2")
        _put(base_url, "j1", {"status": "workers"})

        feed = _get(base_url, "/api/events").json()
        ids = [event["id"] for event in feed["events"]]
        assert [(e["job_id"], e["status"]) for e in feed["events"]] == [
            ("j1", "submitted"),
            ("j1", "starting"),
            ("j2", "submitted"),
            ("j1", "workers"),
        ]
        assert feed["next_cursor"] == ids[-1]

        resumed = _get(base_url, f"/api/events?after={ids[1]}").json()
        assert [e["status"] for e in resumed["events"]] == ["submitted", "workers"]

        quiet = _get(base_url, f"/api/events?after={ids[-1]}").json()
        assert quiet == {"events": [], "next_cursor": ids[-1]}

        assert [e["job_id"] for e in _get(base_url, "/api/events?job_id=j2").json()["events"]] == ["j2"]

        per_job = _get(base_url, "/api/jobs/j1/events").json()
        assert per_job["job_id"] == "j1"
        assert [e["status"] for e in per_job["events"]] == ["submitted", "starting", "workers"]
        limited = _get(base_url, "/api/jobs/j1/events?limit=1").json()
        assert [e["status"] for e in limited["events"]] == ["submitted"]
        assert limited["next_cursor"] == ids[0]
        assert (
            _get(base_url, f"/api/jobs/j1/events?after={limited['next_cursor']}").json()["events"][0]["status"]
            == "starting"
        )

        assert _get(base_url, "/api/events?after=-1").status_code == 422
        assert _get(base_url, "/api/events?limit=1001").status_code == 422

    def test_delete_job(self, base_url):
        _create(base_url, "1")
        _put(base_url, "1", {"status": "workers"})

        response = requests.delete(f"{base_url}/api/jobs/1", timeout=5)
        assert response.status_code == 200
        assert response.json() == {"deleted": True, "job_id": "1"}
        assert _get(base_url, "/api/jobs/1").status_code == 404
        assert _get(base_url, "/api/events").json()["events"] == []

    def test_trailing_slash_is_tolerated(self, base_url):
        assert _get(base_url, "/api/health/").status_code == 200
        assert _get(base_url, "/api/jobs/").status_code == 200

    def test_concurrent_writers(self, base_url):
        """Many sweeps PUT at once; every request succeeds and no event is lost."""
        _create(base_url, "shared")
        writers, updates = 8, 10

        def worker(index: int) -> list[int]:
            codes = []
            for step in range(updates):
                codes.append(
                    _put(base_url, "shared", {"status": "benchmark", "message": f"w{index}-{step}"}).status_code
                )
                codes.append(_put(base_url, f"own-{index}", {"status": "workers", "message": str(step)}).status_code)
            return codes

        with ThreadPoolExecutor(max_workers=writers) as pool:
            codes = [code for result in pool.map(worker, range(writers)) for code in result]

        assert set(codes) == {200}
        shared = _get(base_url, "/api/jobs/shared").json()
        assert len(shared["events"]) == 1 + writers * updates
        assert _get(base_url, "/api/jobs?per_page=100").json()["total"] == 1 + writers


# ============================================================================
# Store
# ============================================================================


class TestStore:
    def test_reopen_sees_persisted_rows_and_init_is_idempotent(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "status.db"
        first = StatusStore(path)
        first.init()
        first.create_job("1", "persisted")

        again = StatusStore(path)
        again.init()
        job = again.get_job("1")
        assert job is not None
        assert job["job_name"] == "persisted"
        assert [e["status"] for e in job["events"]] == ["submitted"]

    def test_update_returns_whether_an_event_was_appended(self, store):
        store.create_job("1", "a")
        first = store.update_job("1", {"status": "workers", "stage": "workers", "message": "go"})
        repeat = store.update_job("1", {"status": "workers", "stage": "workers", "message": "go", "metadata": {"k": 1}})
        assert first == {"job_id": "1", "status": "workers", "event": True}
        assert repeat == {"job_id": "1", "status": "workers", "event": False}

    def test_create_reports_whether_the_row_is_new(self, store):
        assert store.create_job("1", "a")["created"] is True
        assert store.create_job("1", "a")["created"] is False


# ============================================================================
# CLI wiring
# ============================================================================


class TestCli:
    def test_status_server_subcommand_forwards_flags(self, monkeypatch, tmp_path):
        captured: dict = {}
        monkeypatch.setattr(submit_cli, "serve_status_server", lambda **kwargs: captured.update(kwargs))
        monkeypatch.setattr(
            sys,
            "argv",
            ["srtctl", "status-server", "--host", "0.0.0.0", "--port", "9999", "--db", str(tmp_path / "x.db")],
        )
        submit_cli.main()
        assert captured == {"host": "0.0.0.0", "port": 9999, "db_path": tmp_path / "x.db"}

    def test_status_server_defaults(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(submit_cli, "serve_status_server", lambda **kwargs: captured.update(kwargs))
        monkeypatch.setattr(sys, "argv", ["srtctl", "status-server"])
        submit_cli.main()
        assert captured == {"host": "127.0.0.1", "port": 8080, "db_path": None}
