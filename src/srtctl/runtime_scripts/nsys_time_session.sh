#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run a serving worker in a named Nsight session, collect one bounded window,
# and keep the launch process tied to the real server lifetime. `nsys stop`
# finalizes the report without ending the server; SRT remains teardown owner.
set -euo pipefail

if [ "$#" -lt 8 ]; then
    echo "usage: $0 NSYS SESSION DELAY DURATION TRACE OUTPUT [START_ARGS...] -- APPLICATION [ARGS...]" >&2
    exit 2
fi

nsys_binary="$1"
session_seed="$2"
delay_seconds="$3"
duration_seconds="$4"
trace="$5"
output_file="$6"
shift 6

session_name="${session_seed}_${SLURM_PROCID:-0}_$(hostname)"

start_args=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    start_args+=("$1")
    shift
done
if [ "$#" -lt 2 ]; then
    echo "nsys-time session requires an application after --" >&2
    exit 2
fi
shift

launch_pid=""
cleanup() {
    if [ -n "$launch_pid" ] && kill -0 "$launch_pid" 2>/dev/null; then
        "$nsys_binary" shutdown --session="$session_name" --kill sigterm >/dev/null 2>&1 || true
    fi
}
terminate() {
    trap - EXIT
    cleanup
    exit 143
}
trap cleanup EXIT
trap terminate INT TERM

"$nsys_binary" launch \
    --session-new="$session_name" \
    --trace="$trace" \
    --cuda-graph-trace=node \
    "$@" &
launch_pid=$!

sleep "$delay_seconds"
if ! kill -0 "$launch_pid" 2>/dev/null; then
    wait "$launch_pid"
    exit $?
fi

start_command=(
    "$nsys_binary" start
    "--session=$session_name"
    "--sample=none"
    "--cpuctxsw=none"
    "--force-overwrite=true"
    "--output=$output_file"
)
if [ "${#start_args[@]}" -gt 0 ]; then
    start_command+=("${start_args[@]}")
fi

# A zero delay can race session registration. Retry only while the launch is
# alive; run once more without stderr suppression to retain a useful failure.
started=false
for ((attempt = 0; attempt < 50; attempt++)); do
    if "${start_command[@]}" 2>/dev/null; then
        started=true
        break
    fi
    if ! kill -0 "$launch_pid" 2>/dev/null; then
        wait "$launch_pid"
        exit $?
    fi
    sleep 0.1
done
if [ "$started" != true ]; then
    "${start_command[@]}"
fi
sleep "$duration_seconds"
"$nsys_binary" stop --session="$session_name"

wait "$launch_pid"
