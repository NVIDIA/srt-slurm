#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AgentX agentic-coding trace replay using the InferenceX harness.
# Expects: endpoint [infmax_workspace]
#
# srt-slurm owns the deployment. This script hands off to InferenceX's
# agentic_srt.sh, which resolves the trace corpus, builds an isolated AIPerf
# environment, and replays the agentic sessions. Its inputs arrive through the
# environment; see srtctl/benchmarks/agentx.py for what srtctl derives.

set -euo pipefail

ENDPOINT=$1
INFMAX_WORKSPACE=${2:-/infmax-workspace}

HARNESS="${INFMAX_WORKSPACE}/benchmarks/multi_node/agentic_srt.sh"

# Without the mount this is the first thing to fail, and it happens only after
# the workers have loaded. Say what is missing rather than leaving a bare
# "No such file or directory" from bash.
if [ ! -f "${HARNESS}" ]; then
    echo "ERROR: AgentX harness not found at ${HARNESS}" >&2
    echo "The AgentX client stack lives in InferenceX and is not bundled with srt-slurm." >&2
    echo "Clone it with submodules and point INFMAX_WORKSPACE at the checkout:" >&2
    echo "  git clone --recurse-submodules https://github.com/SemiAnalysisAI/InferenceX.git" >&2
    echo "  export INFMAX_WORKSPACE=<path to the checkout>   # before srtctl apply" >&2
    exit 1
fi

# The harness reads the workspace from its own variable, and reaches the
# frontend by port rather than by URL.
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-${INFMAX_WORKSPACE}}"
export PORT="${PORT:-$(echo "${ENDPOINT}" | sed -E 's|.*:([0-9]+).*|\1|')}"

echo "AgentX Config: endpoint=${ENDPOINT}; workspace=${INFMAX_WORKSPACE}; \
model=${MODEL:-unset}; framework=${FRAMEWORK:-unset}; conc=${CONC_LIST:-${CONC:-unset}}"

# Relative paths inside the harness resolve against the workspace.
cd "${INFMAX_WORKSPACE}"

exec bash "${HARNESS}"
