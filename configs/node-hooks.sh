#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ordered pre-run / post-run command runner for srtctl, wired through the recipe's
# host_setup block (see examples/features/node-hooks.yaml):
#
#   host_setup:
#     commands: ["bash ${SRTCTL_SOURCE_DIR}/configs/node-hooks.sh pre"]
#     teardown: ["bash ${SRTCTL_SOURCE_DIR}/configs/node-hooks.sh post"]
#
# The commands themselves come from numbered HOOK_PRE_<n> and HOOK_POST_<n> variables,
# run in ascending <n>. Set them under the recipe's `environment:` block (the job script
# exports them, so every host srun inherits them) or pass KEY=VALUE arguments after the
# phase, which win over the environment:
#
#   environment:
#     HOOK_PRE_1:  "sudo -n nvidia-smi -lmc 2619,2619"
#     HOOK_PRE_2:  "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches"
#     HOOK_POST_1: "sudo -n nvidia-smi -rmc"
#
#   node-hooks.sh pre HOOK_PRE_3="echo extra >> ${SRTCTL_OUTPUT_DIR}/logs/hooks.log"
#
# Other settings:
#   HOOK_SNAPSHOT     1 (default) prints kernel, load, memory and GPU clocks after the commands
#   HOOK_PRE_STRICT   1 (default) makes a failing pre command exit non-zero, so host_setup
#                     fails the job (or warns with ignore_failure: true); 0 logs and continues
#
# Post commands are always best effort: teardown must never mask the job's real exit code.
set -uo pipefail

phase="${1:-}"
shift || true
case "${phase}" in
    pre|post) ;;
    *) echo "usage: $0 pre|post [HOOK_KEY=VALUE ...]" >&2; exit 2 ;;
esac

for kv in "$@"; do
    case "${kv}" in
        HOOK_*=*) export "${kv}" ;;
        *) echo "ignoring argument without HOOK_ prefix: ${kv}" >&2 ;;
    esac
done

: "${HOOK_SNAPSHOT:=1}"
: "${HOOK_PRE_STRICT:=1}"
node="$(hostname -s)"
log() { echo "[node-hooks ${phase} ${node}] $*"; }

snapshot() {
    [ "${HOOK_SNAPSHOT}" = "1" ] || return 0
    log "kernel: $(uname -r)  load: $(cut -d' ' -f1-3 /proc/loadavg)"
    log "memory: $(free -g | awk '/^Mem:/ {print $3 "G used / " $2 "G total, " $7 "G available"}')"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.draw,memory.used \
            --format=csv,noheader,nounits | while IFS= read -r line; do log "gpu: ${line}"; done
    fi
}

prefix="HOOK_$(echo "${phase}" | tr '[:lower:]' '[:upper:]')_"
mapfile -t names < <(compgen -A variable "${prefix}" | grep -E "^${prefix}[0-9]+$" | sort -t_ -k3,3n)

log "start $(date -Is)  job=${SLURM_JOB_ID:-?}  output=${SRTCTL_OUTPUT_DIR:-?}  commands=${#names[@]}"

failed=0
for name in "${names[@]}"; do
    cmd="${!name}"
    [ -n "${cmd}" ] || continue
    log "${name}: ${cmd}"
    rc=0
    bash -c "${cmd}" || rc=$?
    [ "${rc}" = "0" ] && continue
    log "${name} exited ${rc}"
    failed=1
    if [ "${phase}" = "pre" ] && [ "${HOOK_PRE_STRICT}" = "1" ]; then
        break
    fi
done

snapshot
log "done $(date -Is)"

if [ "${phase}" = "pre" ] && [ "${HOOK_PRE_STRICT}" = "1" ] && [ "${failed}" = "1" ]; then
    exit 1
fi
exit 0
