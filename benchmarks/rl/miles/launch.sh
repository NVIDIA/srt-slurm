#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch a Miles (radixark/miles) RL run against the job's Ray service.
#
# srt-slurm runs this as the benchmark step of a services-only job:
#
#   frontend:  { type: none }
#   services:  [ { name: train, type: ray, nodes: N, preamble: "cd /root/miles" } ]
#   benchmark: { type: custom, command: /benchmarks/rl/miles/launch.sh, env: { MILES_...: ... } }
#
# The step is only the launcher and the log tail: the Miles launch script attaches to the
# Ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1) and runs `ray job submit -- python3 train.py`;
# the engines, router, rollout executor, reward and trainer ranks all run inside that Ray job,
# on the raylets the ray service started. See docs/miles.md.
#
# Reads, injected by srtctl for every custom benchmark command:
#   SRT_SERVICE_<NAME>_IPS, SRT_SERVICE_<NAME>_NODE_COUNT   the ray service's nodes; NAME is the service
#                                                            name upper-cased with non-alphanumerics as _
#   SRT_GPUS_PER_NODE                                        resources.gpus_per_node
# Reads, from benchmark.env:
#   MILES_RECIPE (required)          launch script, relative to MILES_ROOT or absolute in the container
#   MILES_RAY_SERVICE (train)        name of the services[] entry that is the Ray cluster
#   MILES_ROOT (/root/miles)         the Miles checkout inside the container
#   MILES_SUBCOMMAND                 positional for the multi-step launchers (train, prepare, full-train);
#                                    the single-command recipes such as run_qwen3_dense.py take none
#   MILES_RAY_PORT (6379)            must match the ray service's options.port
#   MILES_RAY_DASHBOARD_PORT (8265)  must match the ray service's options.dashboard_port
#   MILES_LAUNCH_DRY_RUN=1           print the resolved environment and command, then exit 0
#   MILES_SCRIPT_*                   Miles's own recipe options, passed through untouched: MODEL_NAME,
#                                    MODEL_DIR, DATA_DIR, OUTPUT_DIR, ENABLE_EVAL, EXTRA_ARGS, EXTRA_ENV_VARS, ...
# Exports, unless already set:
#   MILES_SCRIPT_EXTERNAL_RAY=1  MASTER_ADDR  RAY_ADDRESS  MILES_SCRIPT_NUM_NODES  MILES_SCRIPT_NUM_GPUS_PER_NODE
set -euo pipefail

service="${MILES_RAY_SERVICE:-train}"
key="$(printf '%s' "$service" | tr '[:lower:]-' '[:upper:]_' | tr -c 'A-Z0-9_' '_')"
ips_var="SRT_SERVICE_${key}_IPS"
count_var="SRT_SERVICE_${key}_NODE_COUNT"
ips="${!ips_var:-}"
if [ -z "$ips" ]; then
  echo "launch.sh: $ips_var is not set. Declare a services[] entry named '$service' with type: ray," \
       "or point MILES_RAY_SERVICE at the one you declared." >&2
  exit 2
fi
head_ip="${ips%%,*}"
: "${MILES_RECIPE:?launch.sh: MILES_RECIPE is required (the launch script, e.g. scripts/run_qwen3_dense.py)}"
: "${SRT_GPUS_PER_NODE:?launch.sh: SRT_GPUS_PER_NODE is not set (is this running as a custom benchmark?)}"

root="${MILES_ROOT:-/root/miles}"
recipe="$MILES_RECIPE"
case "$recipe" in /*) ;; *) recipe="$root/$recipe" ;; esac

# The launcher attaches to our cluster instead of starting one, and submits through the head's dashboard.
export MILES_SCRIPT_EXTERNAL_RAY="${MILES_SCRIPT_EXTERNAL_RAY:-1}"
export MASTER_ADDR="${MASTER_ADDR:-$head_ip}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://$head_ip:${MILES_RAY_DASHBOARD_PORT:-8265}}"
export MILES_SCRIPT_NUM_NODES="${MILES_SCRIPT_NUM_NODES:-${!count_var:-1}}"
export MILES_SCRIPT_NUM_GPUS_PER_NODE="${MILES_SCRIPT_NUM_GPUS_PER_NODE:-$SRT_GPUS_PER_NODE}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cmd=(python3 "$recipe")
if [ -n "${MILES_SUBCOMMAND:-}" ]; then
  cmd+=("$MILES_SUBCOMMAND")
fi

echo "miles launch: ray head $head_ip (service $service), ${MILES_SCRIPT_NUM_NODES} node(s) x ${MILES_SCRIPT_NUM_GPUS_PER_NODE} GPUs, recipe $recipe"
env | grep -E '^(MILES_|MASTER_ADDR=|RAY_ADDRESS=)' | sort
if [ "${MILES_LAUNCH_DRY_RUN:-0}" = "1" ]; then
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  echo
  exit 0
fi

cd "$root"
ray status --address "$head_ip:${MILES_RAY_PORT:-6379}" || true
exec "${cmd[@]}"
