#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Snapshot NVLink / MNNVL health on one node.
#
# Usage: collect_hwinfo.sh <parts_dir> <phase>
#
# Runs on the host (not in the job container) because the IMEX config and the
# MNNVL channel devices are host paths. Writes <parts_dir>/<hostname>.part; the
# orchestrator merges the parts into logs/hwinfo/<phase>.out.
#
# Everything here is best effort. A missing tool, a denied read or a hung driver
# call is recorded with its exit code and never fails the job, so this script
# must not use `set -e`.

set -u

PARTS_DIR="${1:?parts dir required}"
PHASE="${2:?phase required}"

# Bounds each command so a wedged driver cannot stall job startup or cleanup.
CMD_TIMEOUT="${HWINFO_CMD_TIMEOUT:-20}"

# SLURMD_NODENAME is SLURM's own name for this node, which is what the
# orchestrator merges by; hostname is only a fallback for direct invocation.
host="${SLURMD_NODENAME:-$(hostname -s 2>/dev/null || hostname)}"
part="${PARTS_DIR}/${host}.part"

mkdir -p "$PARTS_DIR"
{
    echo "===== ${host} | ${PHASE} | $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    echo ""
} > "$part"

section() {
    printf '# ---------- %s ----------\n\n' "$1" >> "$part"
}

record() {
    local cmd="$1"
    local out rc
    out="$(timeout "$CMD_TIMEOUT" bash -c "$cmd" 2>&1)"
    rc=$?
    {
        printf '$ %s\n' "$cmd"
        [ -n "$out" ] && printf '%s\n' "$out"
        if [ "$rc" -eq 124 ]; then
            printf '[timed out after %ss]\n' "$CMD_TIMEOUT"
        elif [ "$rc" -ne 0 ]; then
            printf '[exit %s]\n' "$rc"
        elif [ -z "$out" ]; then
            printf '[no output]\n'
        fi
        printf '\n'
    } >> "$part"
}

# Which GPU is which. Serials are what a hardware ticket needs, and the driver
# version tells you whether one node in the domain is out of step with the rest.
section "GPU inventory"
record 'nvidia-smi --query-gpu=index,name,serial,uuid,pci.bus_id --format=csv'
record 'nvidia-smi --version'

# Link state and, more importantly, the error counters. Comparing before.out
# against after.out shows which link degraded and by how much.
section "NVLink"
record 'nvidia-smi nvlink -s'
record 'nvidia-smi nvlink -e'
record 'nvidia-smi topo -m'

# Multi-node NVLink: the channel devices prove the kernel side is wired up, and
# nodes_config.cfg must list the nodes actually in the allocation.
section "MNNVL / IMEX"
record 'ls -al /dev/nvidia-caps-imex-channels/'
record 'cat /etc/nvidia-imex/nodes_config.cfg'
record 'cat /etc/nvidia-imex/config.cfg'
record 'systemctl is-active nvidia-imex'
record 'systemctl status nvidia-imex --no-pager -l 2>&1 | head -30'
# nvidia-imex-ctl parses /etc/nvidia-imex/config.cfg, whose LOG_FILE_NAME and
# STATS_FILE_NAME point at root-only paths; unprivileged runs then fail while
# parsing and report a misleading "invalid value". Point those at /tmp first,
# then ask the daemon for the domain state (-N) and the hosts in it (-H).
# Recording the sed separately keeps both commands in the snapshot short enough
# to copy-paste when reproducing by hand.
imex_cfg=/tmp/imex_hwinfo_config.cfg
record "sed -e 's|^LOG_FILE_NAME=.*|LOG_FILE_NAME=/tmp/imex_hwinfo.log|' -e 's|^STATS_FILE_NAME=.*|STATS_FILE_NAME=/tmp/imex_hwinfo_stats|' /etc/nvidia-imex/config.cfg > ${imex_cfg}"
record "nvidia-imex-ctl -c ${imex_cfg} -N -H"
rm -f "$imex_cfg"
# On NVL systems each GPU reports whether it joined the fabric (State/Status)
# and which clique it landed in. A GPU outside the clique cannot reach remote
# peers over NVLink even though its local links look healthy.
record 'nvidia-smi -q | grep -A8 -i "^ *Fabric"'

# Correlate an NVLink fault with what the driver logged, and check whether the
# GPU was already degrading before the run started.
section "Driver and GPU faults"
record 'dmesg -T | grep -iE "xid|nvlink|nvswitch|imex" | tail -60'
record 'nvidia-smi -q -d ROW_REMAPPER'

echo "[hwinfo] ${host}: ${PHASE} snapshot written to ${part}"
