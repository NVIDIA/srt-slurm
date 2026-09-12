# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU power collector lifecycle against fake cpu-power-exporter endpoints."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from srtctl.core.power.cpu_samples import read_cpu_samples
from srtctl.core.power.cpu_session import CpuPowerCollector, CpuPowerSessionSettings


def _dcgm_body(socket0=43.878, socket1=52.35):
    return (
        "# TYPE cpu_power_dcgm_watts gauge\n"
        f'cpu_power_dcgm_watts{{socket="0",source="dcgm"}} {socket0}\n'
        f'cpu_power_dcgm_watts{{socket="1",source="dcgm"}} {socket1}\n'
    )


def _closed_port() -> int:
    """A port nothing listens on, for a fast, deterministic connection refusal."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class FakeCpuExporter:
    def __init__(self, body):
        self.body = body
        exporter = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                data = exporter.body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.thread.join(timeout=5)


@patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value="127.0.0.1")
def test_collector_writes_rows_from_a_reachable_node(_mock_ip, tmp_path):
    exporter = FakeCpuExporter(_dcgm_body())
    try:
        settings = CpuPowerSessionSettings(
            power_dir=tmp_path / "cpu",
            sample_interval_seconds=0.05,
            request_timeout_seconds=1.0,
            collector_join_timeout_seconds=3.0,
            exporter_port=exporter.port,
        )
        collector = CpuPowerCollector(settings=settings, nodes=["node-a"])
        collector.start()
        time.sleep(0.2)
        collector.stop_and_finalize()
    finally:
        exporter.stop()

    rows, reasons = read_cpu_samples(tmp_path / "cpu" / "samples.csv")
    assert reasons == ()
    assert len(rows) >= 2
    assert {row.socket_id for row in rows} == {0, 1}
    assert all(row.hostname == "node-a" for row in rows)

    manifest = json.loads((tmp_path / "cpu" / "cpu_manifest.json").read_text())
    assert manifest["nodes"]["node-a"]["resolved_mode"] == "dcgm"
    assert manifest["nodes"]["node-a"]["scrape_count"] >= 1
    assert manifest["nodes"]["node-a"]["error_count"] == 0


@patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value="127.0.0.1")
def test_collector_tolerates_an_unreachable_node(_mock_ip, tmp_path):
    settings = CpuPowerSessionSettings(
        power_dir=tmp_path / "cpu",
        sample_interval_seconds=0.05,
        request_timeout_seconds=0.2,
        collector_join_timeout_seconds=3.0,
        exporter_port=_closed_port(),
    )
    collector = CpuPowerCollector(settings=settings, nodes=["node-a"])

    collector.start()
    time.sleep(0.2)
    collector.stop_and_finalize()  # must not raise

    manifest = json.loads((tmp_path / "cpu" / "cpu_manifest.json").read_text())
    assert manifest["nodes"]["node-a"]["error_count"] >= 1
    assert manifest["nodes"]["node-a"]["resolved_mode"] == "unknown"

    rows, _ = read_cpu_samples(tmp_path / "cpu" / "samples.csv")
    assert rows == ()


@patch("srtctl.core.power.cpu_session.get_hostname_ip", return_value=None)
def test_collector_tolerates_endpoint_resolution_failure(_mock_ip, tmp_path):
    settings = CpuPowerSessionSettings(
        power_dir=tmp_path / "cpu",
        sample_interval_seconds=0.05,
        request_timeout_seconds=0.2,
        collector_join_timeout_seconds=3.0,
        exporter_port=1,
    )
    collector = CpuPowerCollector(settings=settings, nodes=["node-a"])

    collector.start()  # must not raise even though no endpoint resolves
    collector.stop_and_finalize()  # must not raise

    assert (tmp_path / "cpu" / "cpu_manifest.json").is_file()
