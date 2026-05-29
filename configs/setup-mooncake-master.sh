#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Boot a Mooncake master (with embedded HTTP metadata server) on this node
# before sglang launches.  Used by recipes that set
# `--hicache-storage-backend mooncake` until srtctl gains a first-class
# infra service for it (see backends/sglang.py:MooncakeKVStoreConfig and
# cli/do_sweep.py:start_mooncake_master for the in-progress wiring).
#
# Recipes opt in via `setup_script: setup-mooncake-master.sh`.  The sglang
# Mooncake hicache backend connects via MOONCAKE_MASTER and
# MOONCAKE_TE_META_DATA_SERVER env vars; recipes set those in
# aggregated_environment / decode_environment.
set -eux

# --- locate the mooncake_master entrypoint ----------------------------------
if command -v mooncake_master >/dev/null 2>&1; then
  MM="mooncake_master"
elif python3 -c "import mooncake.mooncake_master" >/dev/null 2>&1; then
  MM="python3 -m mooncake.mooncake_master"
else
  echo "ERROR: neither mooncake_master binary nor python module found" >&2
  pip show mooncake-transfer-engine || true
  exit 1
fi

# --- start once per node ----------------------------------------------------
# IMPORTANT: use `pgrep -x` (exact process name match), NOT `pgrep -f`.
# `-f` matches the full command line, so `pgrep -f mooncake_master` matches
# the parent shell that's running this script (its argv contains the script
# path "setup-mooncake-master.sh"), and we'd skip startup forever.
if pgrep -x mooncake_master >/dev/null 2>&1; then
  echo "mooncake_master already running on $(hostname)"
else
  nohup ${MM} \
    --enable_http_metadata_server=true \
    --http_metadata_server_port=8080 \
    --port=50051 \
    --eviction_high_watermark_ratio=0.95 \
    > /logs/mooncake_master.log 2>&1 &
  disown || true
  echo "Started mooncake_master on $(hostname) (parent PID $!)"
fi

# --- wait for metadata HTTP up ---------------------------------------------
for i in $(seq 1 30); do
  if curl -fsS -o /dev/null http://127.0.0.1:8080/metadata 2>/dev/null \
     || curl -fsS http://127.0.0.1:8080/ 2>/dev/null | grep -q .; then
    echo "mooncake_master HTTP metadata up after ${i}s"
    break
  fi
  sleep 1
done

# Always show the tail so failures are visible in the worker log
echo "--- tail /logs/mooncake_master.log ---"
[ -f /logs/mooncake_master.log ] && tail -n 40 /logs/mooncake_master.log || echo "(no /logs/mooncake_master.log yet)"
echo "--- end mooncake_master log ---"

echo "=== setup-mooncake-master.sh complete ==="
