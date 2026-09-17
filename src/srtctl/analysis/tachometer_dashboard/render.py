# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a portable dashboard without a JavaScript build or a running web server."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")


def write_dashboard(catalog: dict, data_dir: Path, output: Path) -> Path:
    """Embed the catalog and gzip metric payloads in one offline HTML artifact.

    Metric payloads remain compressed until the browser opens their panels. The
    renderer does not interpret measurements or consult any additional run files.
    """
    data_dir = Path(data_dir).resolve()
    output = Path(output)
    payloads = []
    for metric in catalog.get("metrics", []):
        path = (data_dir / metric["payload"]).resolve()
        if not path.is_relative_to(data_dir):
            raise ValueError(f"metric payload is outside data directory: {metric['payload']}")
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append((str(metric["id"]), path))

    serialized = json.dumps(catalog, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    # HTML parsers recognize closing script tags even in JSON string literals.
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (ASSETS / "dashboard.html").read_text()
    before, after = template.split("<!-- DASHBOARD_DATA -->", 1)
    license_text = (ASSETS / "uPlot.LICENSE").read_text().replace("--", "—")
    before = before.replace("</head>", f"<!-- uPlot 1.6.32 license\n{license_text}\n-->\n</head>")
    before = before.replace("/* UPLOT_CSS */", (ASSETS / "uPlot.min.css").read_text())
    before = before.replace("/* DASHBOARD_CSS */", (ASSETS / "dashboard.css").read_text())
    after = after.replace("/* UPLOT_JS */", (ASSETS / "uPlot.iife.min.js").read_text())
    after = after.replace("/* DASHBOARD_JS */", (ASSETS / "dashboard.js").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(before)
        stream.write(f'<script type="application/json" id="dashboard-catalog">{serialized}</script>\n')
        for metric_id, path in payloads:
            encoded_id = html.escape("payload-" + metric_id, quote=True)
            stream.write(f'<script type="application/octet-stream" id="{encoded_id}">')
            stream.write(base64.b64encode(path.read_bytes()).decode("ascii"))
            stream.write("</script>\n")
        stream.write(after)
    return output
