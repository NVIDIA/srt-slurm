#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Invoked only by the writable, remapped-root container step in the batch job.
set -euo pipefail
run_dir=${1:?Shared run directory is required}
node=${2:?Slurm node name is required}
install_log=$run_dir/logs/agents/$node.install.log
setup_timeout=${MC_RAIL_SETUP_TIMEOUT:-240}
apt_updated=0
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

if [[ ! $setup_timeout =~ ^[1-9][0-9]*$ ]]; then
    printf 'MC_RAIL_SETUP_TIMEOUT must be a positive integer, got: %s\n' "$setup_timeout" > "$install_log"
    fail_setup "MC_RAIL_SETUP_TIMEOUT must be a positive integer, got: $setup_timeout"
fi

if command -v ib_read_bw >/dev/null && command -v ib_write_bw >/dev/null && command -v numactl >/dev/null; then
    printf 'Required tools already installed in the container.\n' > "$install_log"
else
    if [[ $(id -u) != 0 ]]; then
        printf 'Container uid must be 0 for apt installation.\n' > "$install_log"
        fail_setup 'Container setup needs --container-remap-root and --container-writable'
    fi
    printf 'Installing perftest and numactl inside the container on %s.\n' "$node"
    export DEBIAN_FRONTEND=noninteractive
    apt_rc=0
    timeout --signal=TERM --kill-after=10s "${setup_timeout}s" bash -c '
        set -euo pipefail
        apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 update
        apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 \
            install -y --no-install-recommends perftest numactl
    ' > "$install_log" 2>&1 || apt_rc=$?
    if [[ $apt_rc == 124 ]]; then
        fail_setup "apt installation inside the container timed out after $setup_timeout seconds"
    elif [[ $apt_rc != 0 ]]; then
        fail_setup "apt installation inside the container failed (exit $apt_rc)"
    fi
    apt_updated=1
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

# Topology evidence helps explain a blocked NIC matrix, but management tools
# and UMAD permissions are not required to exercise RDMA. Keep this best-effort
# setup separate from the required perftest checks above and inside the same
# writable container. Never install packages on the host.
if ! command -v ibstat >/dev/null || ! command -v ibv_devinfo >/dev/null || ! command -v iblinkinfo >/dev/null; then
    if [[ $(id -u) == 0 ]] && command -v apt-get >/dev/null; then
        printf '\nInstalling optional topology tools inside the container.\n' >> "$install_log"
        optional_rc=0
        timeout --signal=TERM --kill-after=5s 60s bash -c '
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            if [[ $1 != 1 ]]; then
                apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 update
            fi
            apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 \
                install -y --no-install-recommends ibverbs-utils infiniband-diags
        ' -- "$apt_updated" >> "$install_log" 2>&1 || optional_rc=$?
        printf 'Optional topology package setup exit code: %s; missing tools remain nonfatal.\n' \
            "$optional_rc" >> "$install_log"
    else
        printf '\nOptional topology tools unavailable; container apt/root access is absent. Continuing.\n' >> "$install_log"
    fi
fi
exec python3 "$run_dir/mooncake_rail_worker.py" --run-dir "$run_dir" --node "$node"
