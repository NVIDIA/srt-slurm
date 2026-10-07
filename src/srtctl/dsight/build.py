# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a complete offline artifact before replacing an earlier generation."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from .importer import Importer
from .point_buffer import PointBuffers
from .query import TraceDataset
from .storage import FILENAME, dumps, write_store


def _browser_payload(compressed: bytes, data: dict[str, Any] | None = None) -> tuple[bytes, str]:
    """Keep metric samples independently decodable without changing the data artifact.

    Legacy normalized reports retain their original embedded bytes. Catalog-aware
    reports embed the same series metadata in the core payload and put each
    family's complete source-backed points in its own inert, compressed element.
    """
    if data is None:
        data = json.loads(gzip.decompress(compressed))
    if "metric_catalog" not in data or "delivery" in data:
        return compressed, ""
    families: dict[str, dict[str, Any]] = {}
    metadata = []
    for series in data["metrics"]:
        families.setdefault(series["name"], {})[str(series["id"])] = series["points"]
        metadata.append({key: value for key, value in series.items() if key != "points"})
    elements = []
    payload_ids = {}
    for index, (name, samples) in enumerate(sorted(families.items())):
        identifier = f"metricPayload{index}"
        payload_ids[name] = identifier
        payload = json.dumps(samples, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        encoded = base64.b64encode(gzip.compress(payload, compresslevel=6, mtime=0)).decode()
        elements.append(f'<script type="application/octet-stream" id="{identifier}">{encoded}</script>')
    core = {**data, "metrics": metadata, "metric_payloads": payload_ids}
    payload = json.dumps(core, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return gzip.compress(payload, compresslevel=6, mtime=0), "\n".join(elements)


def render_html(compressed: bytes, *, data: dict[str, Any] | None = None) -> str:
    """Render preserved normalized data with the packaged, fully offline viewer."""
    assets = files("srtctl.dsight").joinpath("assets")
    html = assets.joinpath("explorer.html").read_text(encoding="utf-8")
    core, metric_elements = _browser_payload(compressed, data)
    html = html.replace("__TRACE_DATA_GZIP_BASE64__", base64.b64encode(core).decode())
    html = html.replace(
        '<script src="explorer.js"></script>', metric_elements + '\n<script src="explorer.js"></script>'
    )
    for name in ("uPlot.min.css", "metric-charts.css"):
        stylesheet = assets.joinpath(name).read_text(encoding="utf-8")
        if "</style" in stylesheet.lower():
            raise ValueError("Packaged CSS contains an unsafe style terminator")
        html = html.replace(f'<link rel="stylesheet" href="{name}">', "<style>\n" + stylesheet + "\n</style>")
    html = html.replace(
        '<script src="metric-data.js">', '<script src="detail-data.js"></script>\n<script src="metric-data.js">'
    )
    for name in ("uPlot.iife.min.js", "metric-charts.js", "detail-data.js", "metric-data.js", "explorer.js"):
        javascript = assets.joinpath(name).read_text(encoding="utf-8")
        if "</script" in javascript.lower():
            raise ValueError("Packaged JavaScript contains an unsafe script terminator")
        if name == "uPlot.iife.min.js":
            license_text = assets.joinpath("uPlot.LICENSE").read_text(encoding="utf-8")
            javascript = "/*\n" + license_text + "\n*/\n" + javascript
        html = html.replace(f'<script src="{name}"></script>', "<script>\n" + javascript + "\n</script>")
    return html


def _validate_output(output: Path) -> Path:
    if output.is_symlink():
        raise ValueError("Output must not be a symlink")
    output = output.resolve()
    if output.exists():
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if manifest.get("generator") != "srtctl-trace":
            raise ValueError("Output exists and is not a generated trace dashboard; choose a new directory")
        expected = set(manifest.get("files", ["index.html", "trace-data.json.gz", "manifest.json"]))
        expected |= {parent.as_posix() for name in expected for parent in Path(name).parents if parent != Path(".")}
        extra = {p.relative_to(output).as_posix() for p in output.rglob("*")} - expected
        if any(p.is_symlink() for p in output.rglob("*")):
            raise ValueError("Output contains symlinks; choose a new directory")
        if extra:
            raise ValueError(f"Output contains files DSight did not generate: {sorted(extra)}; choose a new directory")
    return output


def build_dashboard(logs: Path, output: Path, *, single_file: bool = False, **options: Any) -> dict[str, Any]:
    """Read preserved artifacts only. No Slurm, profiler, or network operations."""
    output = _validate_output(output)
    if output == logs.resolve() or output in logs.resolve().parents:
        raise ValueError("Output must not replace the input directory or its ancestors")
    if single_file:
        return write_dashboard(Importer(logs, **options).run(), output, single_file=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trace-metrics-", dir=output.parent) as scratch:
        buffers = PointBuffers(Path(scratch) / "metrics.sqlite")
        try:
            data = Importer(logs, point_buffers=buffers, **options).run()
            return write_dashboard(data, output)
        finally:
            buffers.close()


def write_dashboard(data: dict[str, Any], output: Path, *, single_file: bool = False) -> dict[str, Any]:
    """Publish one complete generation, including all lazy browser dependencies."""
    from .bundle import write_details

    output = _validate_output(output)
    dataset = TraceDataset(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".trace-build-", dir=output.parent))
    backup = staged.with_name(staged.name + "-previous")
    try:
        filename = "trace-data.json.gz" if single_file else FILENAME
        if single_file:
            core = data
        else:
            write_store(data, staged / filename)
            core = write_details(data, staged / "detail")
        compressed = gzip.compress(dumps(core).encode(), compresslevel=6, mtime=0)
        if single_file:
            (staged / filename).write_bytes(compressed)
        html = render_html(compressed, data=core)
        (staged / "index.html").write_text(html, encoding="utf-8")
        digest = hashlib.sha256()
        with (staged / filename).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest = {
            "generator": "srtctl-trace",
            **dataset.query("summary"),
            "delivery": "embedded" if single_file else "static-details/1",
            "data_sha256": digest.hexdigest(),
            "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
            "sources": data["sources"],
            "files": sorted(
                ["manifest.json", *[p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()]]
            ),
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(staged, output)
        except OSError:
            if backup.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return {
        "output": str(output),
        "html": str(output / "index.html"),
        "data": str(output / filename),
        "delivery": manifest["delivery"],
        **dataset.query("summary"),
    }
