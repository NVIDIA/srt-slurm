# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an offline dashboard exclusively from raw Tachometer Parquet/Arrow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from srtctl.analysis.tachometer_dashboard.catalog import COMPONENTS, describe_metric, is_counter

logger = logging.getLogger(__name__)


def decorate_catalog(catalog: dict[str, Any], title: str) -> dict[str, Any]:
    """Attach display semantics without enriching measurements from other sources."""
    catalog["title"] = title
    catalog["components"] = list(COMPONENTS)
    catalog["source_policy"] = "Raw Tachometer Parquet/Arrow only"
    source_warnings = list(catalog.get("source_warnings", []))
    sources = catalog.get("source_files", [])
    names = [
        Path(source if isinstance(source, str) else source.get("path", source.get("name", ""))).name
        for source in sources
    ]
    if "final.parquet" not in names:
        source_warnings.append("No final.parquet was selected; capture may be incomplete.")
    excluded = catalog.get("excluded_source_files", [])
    if excluded:
        source_warnings.append(
            f"{len(excluded)} source file(s) were excluded; inspect excluded_source_files for reasons."
        )
    catalog["source_warnings"] = list(dict.fromkeys(source_warnings))
    for metric in catalog["metrics"]:
        metric.update(describe_metric(metric["name"], metric.get("endpoints", ())))
        metric["aliases"] = []
        if metric["kind"] == "histogram":
            metric["aliases"] = [metric["name"] + suffix for suffix in ("_bucket", "_count", "_sum")]
            metric["description"] += (
                " Percentiles are estimates from recorded bucket deltas. Historical attached sums/counts are not used."
            )
    return catalog


def build_dashboard(
    raw_path: Path,
    output: Path,
    *,
    resolution_s: float = 10.0,
    title: str = "Tachometer capture",
    data_dir: Path | None = None,
) -> Path:
    """Build a portable HTML artifact and retain its reproducible reduced data."""
    from srtctl.analysis.tachometer_dashboard.reader import discover_sources, reduce_capture
    from srtctl.analysis.tachometer_dashboard.render import write_dashboard

    paths = discover_sources(raw_path)
    if not paths:
        raise FileNotFoundError(f"No raw Tachometer Parquet/Arrow files found in {raw_path}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir or output.with_suffix(".data")
    data_dir.mkdir(parents=True, exist_ok=True)
    catalog = reduce_capture(paths, data_dir, resolution_s=resolution_s, is_counter=is_counter, progress=logger.info)
    decorate_catalog(catalog, title)
    (data_dir / "catalog.json").write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    result = write_dashboard(catalog, data_dir, output)
    logger.info("Built %s: %s rows, %d metric families", result, f"{catalog['row_count']:,}", len(catalog["metrics"]))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_path", type=Path, help="Raw .parquet/.arrow file, Tachometer storage directory, or run directory"
    )
    parser.add_argument("--out", "-o", type=Path, required=True, help="Self-contained offline HTML output")
    parser.add_argument(
        "--resolution", type=float, default=10.0, metavar="SECONDS", help="Display bin width (default: 10 s)"
    )
    parser.add_argument("--title", default="Tachometer capture", help="Display title; does not supply measurement data")
    parser.add_argument("--data-dir", type=Path, help="Reduced data directory (default: output stem + .data)")
    args = parser.parse_args(argv)
    if not 0 < args.resolution < float("inf"):
        parser.error("--resolution must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        build_dashboard(args.raw_path, args.out, resolution_s=args.resolution, title=args.title, data_dir=args.data_dir)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Dashboard build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
