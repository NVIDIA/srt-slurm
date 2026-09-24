# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lazy browser artifact retains the normalized metric evidence exactly."""

import base64
import copy
import gzip
import json
import re

from srtctl.dsight.build import render_html


def _pack(data):
    return gzip.compress(json.dumps(data).encode(), mtime=0)


def _embedded(html, identifier):
    match = re.search(r'<script[^>]*id="' + identifier + r'"[^>]*>(.*?)</script>', html, re.S)
    assert match is not None
    return base64.b64decode(match[1].strip())


def test_browser_families_reconstruct_exact_normalized_evidence():
    data = {
        "metric_catalog": [{"name": "custom</script>µ"}, {"name": "empty"}, {"name": "counter_total"}],
        "metrics": [
            {
                "id": 0,
                "name": "custom</script>µ",
                "labels": {"host": "a", "metric.key": "a,b"},
                "points": [[0.123456789, 4.0, 8, 19], [0.123456789, 7.0, 9, 22]],
                "conflict_timestamps": [0.123456789],
            },
            {"id": 1, "name": "counter_total", "labels": {"host": "b"}, "points": [[2, 99, 8, 20]]},
            {"id": 2, "name": "counter_total", "labels": {"host": "c"}, "points": [[3, 0, 8, 21]]},
        ],
        "requests": [{"id": "request", "start": 0}],
        "sources": [{"id": 8, "path": "preserved.arrow"}],
    }
    original = copy.deepcopy(data)
    html = render_html(_pack(data), data=data)
    core = json.loads(gzip.decompress(_embedded(html, "tracePayload")))
    assert data == original  # Building must not strip evidence from the query artifact.
    assert all("points" not in series for series in core["metrics"])
    assert set(core["metric_payloads"]) == {"custom</script>µ", "counter_total"}
    for name, identifier in core.pop("metric_payloads").items():
        samples = json.loads(gzip.decompress(_embedded(html, identifier)))
        for series in core["metrics"]:
            if series["name"] == name:
                series["points"] = samples[str(series["id"])]
    assert core == original
    assert "<script src=" not in html
    assert "custom</script>µ" not in html


def test_legacy_embedded_gzip_is_byte_identical():
    compressed = _pack({"metrics": [{"name": "legacy", "points": [[1, 2, 3, 4]]}]})
    html = render_html(compressed)
    assert _embedded(html, "tracePayload") == compressed
    assert 'id="metricPayload0"' not in html
