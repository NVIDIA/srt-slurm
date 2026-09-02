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
case "$CLIENT_CONFIG" in
  /configs/*)
    # srt-slurm mounts its own configs/ at /configs for the nats-server and etcd
    # binaries; an extra_mount onto that path shadows them and the job dies in
    # head-node setup with "NATS binary not found", long before this script runs.
    # Reaching here means the mount got through, but it is still the wrong place.
    echo "[mlperf] WARNING: $CLIENT_CONFIG is under /configs, which srt-slurm uses for its own" >&2
    echo "         infrastructure binaries. Mount client configs elsewhere." >&2
    ;;
esac

# The client ships pre-installed in its own image ("no editable install, no
# runtime sync"), so unlike a source-installed harness there is nothing to build
# here - only a matter of finding it. The image installs it into a venv that is
# NOT on the default PATH (measured: PATH is just the usual /usr/local/sbin../bin
# and the binary is at /opt/venv/bin/inference-endpoint), which is why the
# MLPerf templates source an env file before invoking it. Look there too rather
# than requiring the caller to know.
CLIENT_CANDIDATES=("$CLIENT_BIN" /opt/venv/bin/"$CLIENT_BIN" /usr/local/bin/"$CLIENT_BIN")
CLIENT_RESOLVED=""
for candidate in "${CLIENT_CANDIDATES[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1; then
    CLIENT_RESOLVED=$(command -v "$candidate")
    break
  fi
done
[[ -n "$CLIENT_RESOLVED" ]] || {
  echo "ERROR: could not find '$CLIENT_BIN'. Looked at:" >&2
  for candidate in "${CLIENT_CANDIDATES[@]}"; do echo "         $candidate" >&2; done
  echo "       This benchmark expects the MLPerf endpoint client image (endpoint_client_*.sqsh);" >&2
  echo "       set benchmark.container_image to it, or MLPERF_CLIENT_BIN to an absolute path." >&2
  exit 1
}
echo "[mlperf] client: $CLIENT_RESOLVED"

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

# The image keeps its packages in the venv, so the system python3 has no pyyaml
# and the helper below dies with ModuleNotFoundError. Use the interpreter that
# sits beside the client binary, which is what the MLPerf templates do for the
# same reason.
CLIENT_PYTHON="$(dirname "$CLIENT_RESOLVED")/python"
[[ -x "$CLIENT_PYTHON" ]] || CLIENT_PYTHON=$(command -v python3)

"$CLIENT_PYTHON" "$(dirname "$0")/resolve_config.py" \
  --input "$CLIENT_CONFIG" \
  --output "$RESOLVED_CONFIG" \
  --endpoints "$ENDPOINTS" \
  --report-dir "$RESULTS_DIR"

echo "[mlperf] running $CLIENT_RESOLVED benchmark from-config --mode $MODE"
set +e
"$CLIENT_RESOLVED" benchmark from-config -c "$RESOLVED_CONFIG" --mode "$MODE"
CLIENT_RC=$?
set -e
echo "[mlperf] client exited with $CLIENT_RC"

# srt-slurm's postprocess reads benchmark-rollup.json if it exists, whoever
# wrote it, so a custom benchmark can feed the existing rollup concept without
# srt-slurm needing to know anything about this client.
#
# The parser summarises the client's own report format (not LoadGen's). It never
# fails the step: a run whose report is missing or restructured still gets a
# rollup, just without metrics.
"$CLIENT_PYTHON" "$(dirname "$0")/parse_report.py" \
  --report-dir "$RESULTS_DIR" \
  --output /logs/benchmark-rollup.json \
  --mode "$MODE" \
  --endpoints "$ENDPOINTS" \
  --client-config "$CLIENT_CONFIG" \
  --exit-code "$CLIENT_RC"

exit $CLIENT_RC
