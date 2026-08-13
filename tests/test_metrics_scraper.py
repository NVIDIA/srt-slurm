# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-job RAW /metrics scraper."""

import http.server
import json
import socketserver
import threading
import time

import pytest

from srtctl.analysis.metrics_scraper import RawMetricsScraper, ScrapeTarget, try_start_raw_scraper
from srtctl.core.schema import SrtConfig

BASE_CONFIG = {
    "name": "test-job",
    "model": {"path": "/models/test-model", "container": "test.sqsh", "precision": "fp8"},
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
}

EXPOSITION = (
    b"# HELP trtllm_kv_cache_used_blocks Used KV blocks\n"
    b"# TYPE trtllm_kv_cache_used_blocks gauge\n"
    b'trtllm_kv_cache_used_blocks{model_name="m"} 512\n'
)


@pytest.fixture
def metrics_server():
    """A throwaway HTTP server serving Prometheus exposition text."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(EXPOSITION)))
            self.end_headers()
            self.wfile.write(EXPOSITION)

        def log_message(self, *args):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


class TestScraperConfig:
    def test_scraper_follows_enabled(self):
        cfg = SrtConfig.Schema().load({**BASE_CONFIG, "observability": {"enabled": True}})
        assert cfg.observability.scraper_enabled is True

    def test_scraper_can_be_opted_out_independently(self):
        cfg = SrtConfig.Schema().load(
            {**BASE_CONFIG, "observability": {"enabled": True, "scrape_metrics": False}}
        )
        assert cfg.observability.enabled is True
        assert cfg.observability.scraper_enabled is False

    def test_off_by_default(self):
        cfg = SrtConfig.Schema().load(BASE_CONFIG)
        assert cfg.observability.scraper_enabled is False

    def test_interval_round_trips(self):
        cfg = SrtConfig.Schema().load(
            {**BASE_CONFIG, "observability": {"enabled": True, "scrape_interval_seconds": 2.5}}
        )
        assert cfg.observability.scrape_interval_seconds == 2.5


class TestRawMetricsScraper:
    def test_emits_the_raw_l1_contract(self, metrics_server, tmp_path):
        target = ScrapeTarget(
            url=f"http://127.0.0.1:{metrics_server}/metrics", role="decode", worker_id="node9"
        )
        scraper = RawMetricsScraper(tmp_path, [target], interval_seconds=0.5)
        scraper.start(threading.Event())
        time.sleep(1.6)
        scraper.stop()

        lines = [
            json.loads(x)
            for x in (tmp_path / "raw_prometheus.jsonl").read_text().splitlines()
            if x.strip()
        ]
        assert lines, "scraper produced no output"
        record = lines[0]
        assert set(record) == {"timestamp_ns", "endpoint_url", "role", "worker_id", "text"}
        assert record["role"] == "decode"
        assert record["worker_id"] == "node9"
        # Capture is verbatim: parsing happens offline, so a parser fix never
        # requires re-running the job.
        assert "trtllm_kv_cache_used_blocks" in record["text"]

    def test_unreachable_endpoint_degrades_without_killing_the_sweep(self, metrics_server, tmp_path):
        good = ScrapeTarget(
            url=f"http://127.0.0.1:{metrics_server}/metrics", role="prefill", worker_id="ok"
        )
        dead = ScrapeTarget(url="http://127.0.0.1:1/metrics", role="decode", worker_id="dead")
        scraper = RawMetricsScraper(tmp_path, [good, dead], interval_seconds=0.5)
        scraper.start(threading.Event())
        time.sleep(1.6)
        scraper.stop()

        lines = [
            json.loads(x)
            for x in (tmp_path / "raw_prometheus.jsonl").read_text().splitlines()
            if x.strip()
        ]
        assert lines
        assert all(r["worker_id"] == "ok" for r in lines)
        assert scraper.sweeps >= 2

    def test_stop_is_independent_of_the_shared_event(self, metrics_server, tmp_path):
        """The benchmark stage's stop_event only fires on abort, not on normal
        completion, so the scraper must also stop on its own signal -- otherwise
        it outlives the client and keeps polling endpoints being torn down."""
        target = ScrapeTarget(
            url=f"http://127.0.0.1:{metrics_server}/metrics", role="frontend", worker_id=None
        )
        scraper = RawMetricsScraper(tmp_path, [target], interval_seconds=0.5)
        shared = threading.Event()
        scraper.start(shared)
        time.sleep(0.8)
        scraper.stop()

        assert not shared.is_set()
        assert scraper._thread is None or not scraper._thread.is_alive()

    def test_no_targets_is_a_noop(self, tmp_path):
        scraper = RawMetricsScraper(tmp_path, [], interval_seconds=0.5)
        scraper.start(threading.Event())
        scraper.stop()
        assert not (tmp_path / "raw_prometheus.jsonl").exists()

    def test_try_start_returns_none_when_disabled(self, tmp_path):
        cfg = SrtConfig.Schema().load(BASE_CONFIG)
        assert try_start_raw_scraper(tmp_path, [], cfg.observability, threading.Event()) is None
