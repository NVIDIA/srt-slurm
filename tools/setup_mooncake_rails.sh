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

for tool in ib_read_bw ib_write_bw numactl; do
    if ! command -v "$tool" >> "$install_log"; then
        fail_setup "Container tool is missing after setup: $tool"
    fi
done
# Ubuntu perftest 6.20 prints its version, then exits 1 through return_error.
# https://github.com/linux-rdma/perftest/blob/24.01.0-0.38/src/read_bw.c#L69-L74
# Accept that informational exit only with a real version line; loader failures
# and actual RDMA benchmark failures must still fail their respective checks.
for tool in ib_read_bw ib_write_bw; do
    version_rc=0
    version_output=$(timeout 10s "$tool" --version 2>&1) || version_rc=$?
    printf '%s --version: exit %s\n%s\n' "$tool" "$version_rc" "$version_output" >> "$install_log"
    if [[ $version_rc -ne 0 && $version_rc -ne 1 ]] ||
        [[ ! $version_output =~ (^|$'\n')Version:[[:blank:]]+[0-9]+([.][0-9]+)*([[:blank:]]|$) ]]; then
        fail_setup "$tool version check failed (exit $version_rc): ${version_output:0:400}"
    fi
done
exec python3 "$run_dir/mooncake_rail_worker.py" --run-dir "$run_dir" --node "$node"
