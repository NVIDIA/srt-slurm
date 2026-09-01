#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MLPerf benchmarking via the MLPerf team's `inference-endpoint` client.
#
# Run as a `custom` benchmark, so srt-slurm carries no MLPerf schema:
#
#   benchmark:
#     type: custom
#     command: bash /srtctl-benchmarks/mlperf/bench.sh
#     env:
#       MLPERF_CLIENT_CONFIG: /configs/dsr1-interactive-submission.yaml
#
# The client's config is passed through untouched apart from the few values
# that cannot be known until the cluster is up. That is deliberate: the config
# has ~60 nested settings (model params, two datasets with accuracy scoring,
# load pattern, a ZeroMQ transport block, drain/warmup/early-stopping), its
# shape moves with the client version, and re-expressing any of it here would
# be both a losing race and lossy - anything not modelled becomes unsettable.
# The MLPerf team's own launcher takes the same approach in
# endpoints-launch: NVIDIA/src/sflow/tools/generate_endpoint_yaml.py rewrites
# exactly one key and leaves the rest, including unresolved ${VAR} placeholders
# that the client itself expands at load time.
#
# Start from a template in the client repo
# (src/inference_endpoint/config/templates/submission_template.yaml) or one of
# the ~45 point configs in endpoints-launch under
# NVIDIA/src/configs/<system>/<model>/point_*/client.yaml.

set -euo pipefail

CLIENT_CONFIG=${MLPERF_CLIENT_CONFIG:-}
MODE=${MLPERF_MODE:-both}          # both | performance | accuracy
CLIENT_BIN=${MLPERF_CLIENT_BIN:-inference-endpoint}

[[ -n "$CLIENT_CONFIG" ]] || { echo "ERROR: MLPERF_CLIENT_CONFIG is required (path to the inference-endpoint client config, mounted via extra_mount)" >&2; exit 1; }
[[ -f "$CLIENT_CONFIG" ]] || { echo "ERROR: MLPERF_CLIENT_CONFIG $CLIENT_CONFIG not found in container" >&2; exit 1; }

# The client ships pre-installed in its own image ("no editable install, no
# runtime sync"), so unlike a source-installed harness there is nothing to build
# here - only a check that we are in the right container.
command -v "$CLIENT_BIN" >/dev/null 2>&1 || {
  echo "ERROR: '$CLIENT_BIN' not found on PATH. This benchmark expects the MLPerf endpoint client image" >&2
  echo "       (endpoint_client_*.sqsh); set model.container or benchmark.container_image to it." >&2
  exit 1
}

# srt-slurm injects the frontend it stood up. There is no localhost default:
# silently benchmarking nothing is worse than failing here.
[[ -n "${SRT_FRONTEND_HOST:-}" ]] || { echo "ERROR: SRT_FRONTEND_HOST is not set - this script expects to run as a benchmark.type=custom step" >&2; exit 1; }

# One endpoint per frontend. MLPERF_ENDPOINTS overrides for the multi-frontend
# case the client is built for: it load-balances across the list itself, which
# is how MLPerf gets past the ~28k-connection ceiling of a single ip:port.
# srt-slurm exposes a single frontend today, so that override is the only way
# to hand the client more than one.
ENDPOINTS=${MLPERF_ENDPOINTS:-http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT:-8000}}

RESULTS_DIR=/logs/mlperf
mkdir -p "$RESULTS_DIR"
RESOLVED_CONFIG="$RESULTS_DIR/client_config_resolved.yaml"

# Rewrite only what the cluster decides. Mirrors generate_endpoint_yaml.py:
# load, replace, dump - no schema knowledge, no validation, no reformatting of
# anything else.
SRT_IN="$CLIENT_CONFIG" SRT_OUT="$RESOLVED_CONFIG" SRT_ENDPOINTS="$ENDPOINTS" SRT_REPORT_DIR="$RESULTS_DIR" \
python3 - <<'PY'
import os
import sys

import yaml

source, dest = os.environ["SRT_IN"], os.environ["SRT_OUT"]
config = yaml.safe_load(open(source))
if not isinstance(config, dict):
    sys.exit(f"ERROR: expected a YAML mapping at the root of {source}")

endpoints = [e.strip() for e in os.environ["SRT_ENDPOINTS"].split(",") if e.strip()]
endpoints = [e if e.startswith(("http://", "https://")) else f"http://{e}" for e in endpoints]

# endpoint_config may be absent in a hand-written config; create it rather than
# failing, since the endpoints are the one thing we always know and it never does.
config.setdefault("endpoint_config", {})["endpoints"] = endpoints
# Keep results with the rest of the job's logs so they are collected and uploaded.
config["report_dir"] = os.environ["SRT_REPORT_DIR"]

with open(dest, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f"[mlperf] resolved config -> {dest}")
for e in endpoints:
    print(f"[mlperf]   endpoint {e}")
PY

echo "[mlperf] running $CLIENT_BIN benchmark from-config --mode $MODE"
set +e
"$CLIENT_BIN" benchmark from-config -c "$RESOLVED_CONFIG" --mode "$MODE"
CLIENT_RC=$?
set -e
echo "[mlperf] client exited with $CLIENT_RC"

# srt-slurm's postprocess reads benchmark-rollup.json if it exists, whoever
# wrote it, so a custom benchmark can feed the existing rollup concept without
# srt-slurm needing to know anything about this client. Records what srtctl
# chose, which the client's own report does not.
#
# The per-run metrics are deliberately absent: this client does not use LoadGen
# and writes its own report format, which has not been observed here yet.
# Fabricating a parser for it would be worse than leaving the field out.
SRT_ROLLUP=/logs/benchmark-rollup.json SRT_REPORT_DIR="$RESULTS_DIR" \
SRT_ENDPOINTS="$ENDPOINTS" SRT_MODE="$MODE" SRT_CONFIG="$CLIENT_CONFIG" SRT_RC="$CLIENT_RC" \
python3 - <<'PY'
import json
import os
from pathlib import Path

report_dir = Path(os.environ["SRT_REPORT_DIR"])
record = {
    "benchmark_type": "mlperf",
    "client": "inference-endpoint",
    "runs": [
        {
            "mode": os.environ["SRT_MODE"],
            "endpoints": [e.strip() for e in os.environ["SRT_ENDPOINTS"].split(",") if e.strip()],
            "client_config": os.environ["SRT_CONFIG"],
            "exit_code": int(os.environ["SRT_RC"]),
            "report_dir": str(report_dir),
            "report_files": sorted(p.name for p in report_dir.iterdir() if p.is_file()),
        }
    ],
}
Path(os.environ["SRT_ROLLUP"]).write_text(json.dumps(record, indent=1))
print(f"[mlperf] wrote {os.environ['SRT_ROLLUP']}")
PY

exit $CLIENT_RC
