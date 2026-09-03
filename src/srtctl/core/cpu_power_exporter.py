# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Prometheus exporter for Grace CPU power via ACPI hwmon.

Runs on the host (not inside a container) so /sys/class/hwmon is visible.
Exposes metrics on :9401/metrics by default, readable by AIPerf's
DCGMTelemetryCollector or a dedicated CPUPowerTelemetryCollector.

Metric names:
    cpu_power_acpi_watts{socket, oem_info, source}   -- per-channel power (W)

Usage:
    python3 -m srtctl.core.cpu_power_exporter --port 9401
    # or via srun (host step, no container):
    srun ... python3 -m srtctl.core.cpu_power_exporter --port 9401
"""

from __future__ import annotations

import argparse
import math
import re
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# Channels we surface; others (NVSwitch rails etc.) are collected but labelled separately.
_OEM_LABELS = {
    re.compile(r"CPU Power Socket (\d+)", re.IGNORECASE): "cpu",
    re.compile(r"Grace Power Socket (\d+)", re.IGNORECASE): "grace",
    re.compile(r"SysIO Power Socket (\d+)", re.IGNORECASE): "sysio",
}


def _find_power_meter_sensors(hwmon_root: Path = Path("/sys/class/hwmon")) -> list[dict[str, Any]]:
    sensors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hwmon_dir in sorted(hwmon_root.glob("hwmon*")):
        try:
            if (hwmon_dir / "name").read_text().strip() != "power_meter":
                continue
        except OSError:
            continue
        for root in (hwmon_dir / "device", hwmon_dir):
            for avg_path in sorted(root.glob("power*_average")):
                identity = str(avg_path.resolve()) if avg_path.exists() else str(avg_path)
                if identity in seen:
                    continue
                seen.add(identity)
                if not avg_path.is_file():
                    continue
                chan = avg_path.stem.removesuffix("_average")
                oem = _read_text(root / f"{chan}_oem_info") or _read_text(root / f"{chan}_label") or chan
                channel_type = "other"
                socket_id = ""
                for pattern, kind in _OEM_LABELS.items():
                    m = pattern.search(oem)
                    if m:
                        channel_type = kind
                        socket_id = m.group(1)
                        break
                sensors.append(
                    {
                        "path": avg_path,
                        "oem": oem,
                        "type": channel_type,
                        "socket": socket_id,
                    }
                )
    return sensors


def _read_text(p: Path) -> str | None:
    try:
        v = p.read_text().strip()
        return v or None
    except OSError:
        return None


def _read_watts(path: Path) -> float | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        w = float(raw) / 1_000_000.0
        return w if math.isfinite(w) and w >= 0 else None
    except ValueError:
        return None


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_metrics(sensors: list[dict[str, Any]]) -> str:
    lines = [
        "# HELP cpu_power_acpi_watts Grace CPU power rail reading from ACPI hwmon (W).",
        "# TYPE cpu_power_acpi_watts gauge",
    ]
    for s in sensors:
        watts = _read_watts(s["path"])
        if watts is None:
            continue
        labels = (
            f'type="{_escape(s["type"])}",'
            f'socket="{_escape(s["socket"])}",'
            f'oem_info="{_escape(s["oem"])}",'
            f'source="acpi"'
        )
        lines.append(f"cpu_power_acpi_watts{{{labels}}} {watts:.6f}")
    lines.append("")
    return "\n".join(lines)


class _Handler(BaseHTTPRequestHandler):
    sensors: list[dict[str, Any]] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/metrics", "/health"):
            self.send_response(404)
            self.end_headers()
            return
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        body = _build_metrics(self.sensors).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # suppress default access log noise


def serve(port: int) -> int:
    sensors = _find_power_meter_sensors()
    if not sensors:
        print("ERROR: no ACPI power_meter hwmon sensors found", file=sys.stderr)
        return 2
    print(f"Found {len(sensors)} sensor(s):", file=sys.stderr)
    for s in sensors:
        print(f"  [{s['type']}] socket={s['socket']}  {s['oem']}  ({s['path']})", file=sys.stderr)

    _Handler.sensors = sensors
    server = HTTPServer(("", port), _Handler)
    stop = threading.Event()

    def _shutdown(_sig: int, _frame: Any) -> None:
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"Listening on :{port}/metrics", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9401)
    args = parser.parse_args()
    return serve(args.port)


if __name__ == "__main__":
    raise SystemExit(main())
