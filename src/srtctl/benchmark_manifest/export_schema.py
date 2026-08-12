# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit the JSON Schema for the benchmark manifest from the Pydantic models.

The Pydantic models in ``models.py`` are the source of truth; this writes their
JSON Schema so non-Python consumers can validate against the same contract.

    python -m srtctl.benchmark_manifest.export_schema        # prints to stdout
    python -m srtctl.benchmark_manifest.export_schema -o schema/benchmark-manifest.schema.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from srtctl.benchmark_manifest.models import BenchmarkManifest


def build_schema() -> dict:
    schema = BenchmarkManifest.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Portable Benchmark Manifest"
    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, help="Write to this path instead of stdout.")
    args = parser.parse_args(argv)

    text = json.dumps(build_schema(), indent=2, sort_keys=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
