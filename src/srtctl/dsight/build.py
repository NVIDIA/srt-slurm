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
from .query import TraceDataset


def build_dashboard(logs: Path, output: Path, **options: Any) -> dict[str, Any]:
    """Read preserved artifacts only. No Slurm, profiler, or network operations."""
    if output.is_symlink():
        raise ValueError("Output must not be a symlink")
    output = output.resolve()
    if output == logs.resolve() or output in logs.resolve().parents:
        raise ValueError("Output must not replace the input directory or its ancestors")
    if output.exists():
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get("generator") != "srtctl-trace":
            raise ValueError("Output exists and is not a generated trace dashboard; choose a new directory")
        extra = {p.name for p in output.iterdir()} - {"index.html", "trace-data.json.gz", "manifest.json"}
        if extra:
            raise ValueError(f"Output contains files DSight did not generate: {sorted(extra)}; choose a new directory")
    data = Importer(logs, **options).run()
    dataset = TraceDataset(data)
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    compressed = gzip.compress(payload, compresslevel=6, mtime=0)
    assets = files("srtctl.dsight").joinpath("assets")
    javascript = assets.joinpath("explorer.js").read_text(encoding="utf-8")
    if "</script" in javascript.lower():
        raise ValueError("Packaged JavaScript contains an unsafe script terminator")
    html = assets.joinpath("explorer.html").read_text(encoding="utf-8")
    html = html.replace("__TRACE_DATA_GZIP_BASE64__", base64.b64encode(compressed).decode())
    html = html.replace('<script src="explorer.js"></script>', "<script>\n" + javascript + "\n</script>")
    manifest = {
        "generator": "srtctl-trace",
        **dataset.query("summary"),
        "data_sha256": hashlib.sha256(compressed).hexdigest(),
        "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
        "sources": data["sources"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".trace-build-", dir=output.parent))
    backup = staged.with_name(staged.name + "-previous")
    try:
        (staged / "trace-data.json.gz").write_bytes(compressed)
        (staged / "index.html").write_text(html, encoding="utf-8")
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
        "data": str(output / "trace-data.json.gz"),
        **dataset.query("summary"),
    }
