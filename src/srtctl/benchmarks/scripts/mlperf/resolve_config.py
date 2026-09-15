#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fill in the parts of an inference-endpoint client config that only the cluster knows.

The client config is the MLPerf team's, not srt-slurm's: ~60 nested settings whose
shape moves with the client version. This rewrites the two values that cannot be
written down in advance and copies everything else through, which is the same
contract as the MLPerf launcher's own resolver (endpoints-launch,
NVIDIA/src/sflow/tools/generate_endpoint_yaml.py).

  endpoint_config.endpoints   frontend addresses, assigned by Slurm at run time
  report_dir                  so results land with the job's other logs

Everything else is left alone, including unresolved ``${VAR}`` placeholders --
the client expands those itself at load time, and a config that reads
``tokenizer_name: "${MODEL_DIR}"`` must still say that when the client opens it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def resolve(config: dict, endpoints: list[str], report_dir: str | None) -> dict:
    """Apply the cluster-dependent rewrites in place and return the config."""
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    # setdefault rather than requiring the key: the endpoints are the one thing
    # we always know, so a config that omits the section should still work.
    config.setdefault("endpoint_config", {})["endpoints"] = endpoints
    if report_dir is not None:
        config["report_dir"] = report_dir
    return config


def normalize_endpoints(raw: str, scheme: str = "http") -> list[str]:
    """Split a comma-separated list and add a scheme where one is missing."""
    endpoints = []
    for entry in raw.split(","):
        value = entry.strip()
        if not value:
            continue
        if not value.startswith(("http://", "https://")):
            value = f"{scheme}://{value}"
        endpoints.append(value)
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="the client config to read")
    parser.add_argument("--output", required=True, type=Path, help="where to write the resolved config")
    parser.add_argument("--endpoints", required=True, help="comma-separated frontend addresses")
    parser.add_argument("--report-dir", default=None, help="overrides report_dir when given")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: client config not found: {args.input}", file=sys.stderr)
        return 1

    config = yaml.safe_load(args.input.read_text())
    if not isinstance(config, dict):
        print(f"ERROR: expected a YAML mapping at the root of {args.input}", file=sys.stderr)
        return 1

    try:
        config = resolve(config, normalize_endpoints(args.endpoints), args.report_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    print(f"[mlperf] resolved config -> {args.output}")
    for endpoint in config["endpoint_config"]["endpoints"]:
        print(f"[mlperf]   endpoint {endpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
