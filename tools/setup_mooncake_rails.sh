#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Invoked only by the writable, remapped-root container step in the batch job.
set -euo pipefail
run_dir=${1:?Shared run directory is required}
node=${2:?Slurm node name is required}
install_log=$run_dir/logs/agents/$node.install.log
mkdir -p "$run_dir/logs/agents"

fail_setup() {
    local reason=$1
    printf '%s; see %s\n' "$reason" "$install_log" >&2
    python3 - "$run_dir" "$node" "$reason" <<'PY'
import json
import sys
from pathlib import Path

run_dir, node, reason = sys.argv[1:]
path = Path(run_dir) / "logs/agents" / f"{node}.inventory.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "node": node,
    "ready": False,
    "errors": [reason],
    "setup_log": f"logs/agents/{node}.install.log",
}) + "\n")
temporary.replace(path)
PY
    exit 1
}

if command -v ib_read_bw >/dev/null && command -v ib_write_bw >/dev/null && command -v numactl >/dev/null; then
    printf 'Required tools already installed in the container.\n' > "$install_log"
else
    if [[ $(id -u) != 0 ]]; then
        printf 'Container uid must be 0 for apt installation.\n' > "$install_log"
        fail_setup 'Container setup needs --container-remap-root and --container-writable'
    fi
    printf 'Installing perftest and numactl inside the container on %s.\n' "$node"
    export DEBIAN_FRONTEND=noninteractive
    if ! timeout --signal=TERM --kill-after=10s 240s bash -c '
        set -euo pipefail
        apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 update
        apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 \
            install -y --no-install-recommends perftest numactl
    ' > "$install_log" 2>&1; then
        fail_setup 'apt installation inside the container failed or exceeded 240 seconds'
    fi
fi

if ! (command -v ib_read_bw && command -v ib_write_bw && command -v numactl &&
    ib_read_bw --version && ib_write_bw --version) >> "$install_log" 2>&1; then
    fail_setup 'Container perftest tools are missing or cannot execute after setup'
fi
exec python3 "$run_dir/mooncake_rail_worker.py" --run-dir "$run_dir" --node "$node"
