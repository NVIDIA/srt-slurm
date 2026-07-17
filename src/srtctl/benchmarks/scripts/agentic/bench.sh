#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AgentX / agentic-coding benchmark.
# srt-slurm owns server startup; this script runs the bundled InferenceX
# AgentX v1.0 client harness against the ready local frontend.

set -euo pipefail

ENDPOINT=$1
MODEL_NAME=$2
MODEL_PREFIX_ARG=$3
FRAMEWORK_ARG=$4
PRECISION_ARG=$5
CONCURRENCIES_ARG=$6
DURATION_ARG=$7
RESULT_FILENAME_ARG=$8
KV_OFFLOADING_ARG=$9
KV_OFFLOAD_BACKEND_ARG=${10}
TOTAL_CPU_DRAM_GB_ARG=${11}

PORT_FROM_ENDPOINT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')
if [[ -n "${SRT_FRONTEND_HOST:-}" ]]; then
  PORT_FROM_ENDPOINT="${SRT_FRONTEND_PORT:-$PORT_FROM_ENDPOINT}"
  ENDPOINT="http://${SRT_FRONTEND_HOST}:${PORT_FROM_ENDPOINT}"
fi
export PORT="${PORT:-$PORT_FROM_ENDPOINT}"

INFERENCEX_AGENTX_COMMIT="${INFERENCEX_AGENTX_COMMIT:-303669b0e16aa6a0c600b8a68b4f91b973a34127}"
# SemiAnalysisAI/aiperf#17, based on the AgentX v1 branch, adds native
# X-Dynamo-Session-ID and X-Dynamo-Parent-Session-ID support.
AIPERF_AGENTX_REF="${AIPERF_AGENTX_REF:-d14531b4a83e987c2477e82227ae2fd5184be755}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLED_INFMAX_WORKSPACE="${AGENTX_BUNDLED_INFMAX_WORKSPACE:-$SCRIPT_DIR/inferencex}"
BUNDLED_AIPERF_TARBALL="${AGENTX_BUNDLED_AIPERF_TARBALL:-$SCRIPT_DIR/third_party/aiperf-agentx-v1-src.tgz}"

if [[ "${AGENTX_USE_EXISTING_INFMAX_WORKSPACE:-0}" == "1" ]]; then
  WORKSPACE_ROOT="${INFMAX_CONTAINER_WORKSPACE:?AGENTX_USE_EXISTING_INFMAX_WORKSPACE=1 requires INFMAX_CONTAINER_WORKSPACE}"
  if [[ ! -f "$WORKSPACE_ROOT/benchmarks/multi_node/agentic_srt.sh" ]]; then
    echo "ERROR: existing INFMAX_CONTAINER_WORKSPACE lacks benchmarks/multi_node/agentic_srt.sh: $WORKSPACE_ROOT" >&2
    exit 1
  fi
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${AGENTX_WORKSPACE:-/tmp/inferencex-agentx-${SLURM_JOB_ID:-$$}}}"

if ! command -v tar >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends tar
fi

if [[ "${AGENTX_USE_EXISTING_INFMAX_WORKSPACE:-0}" != "1" ]]; then
  rm -rf "$WORKSPACE_ROOT"
  mkdir -p "$WORKSPACE_ROOT"

  if [[ "${AGENTX_USE_BUNDLED_INFMAX_WORKSPACE:-1}" == "1" && -f "$BUNDLED_INFMAX_WORKSPACE/benchmarks/multi_node/agentic_srt.sh" ]]; then
    echo "Using bundled InferenceX AgentX harness: $BUNDLED_INFMAX_WORKSPACE"
    cp -a "$BUNDLED_INFMAX_WORKSPACE/." "$WORKSPACE_ROOT/"
  else
    if ! command -v curl >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates
    fi
    echo "Downloading InferenceX AgentX harness: ${INFERENCEX_AGENTX_COMMIT}"
    curl -L --fail --retry 5 --retry-delay 2 \
      "https://codeload.github.com/SemiAnalysisAI/InferenceX/tar.gz/${INFERENCEX_AGENTX_COMMIT}" \
      | tar -xz --strip-components=1 -C "$WORKSPACE_ROOT"
  fi

  if [[ -n "${AIPERF_DIR:-}" ]]; then
    echo "Using caller-provided AIPERF_DIR: $AIPERF_DIR"
  elif [[ -f "$BUNDLED_AIPERF_TARBALL" ]]; then
    echo "Using bundled AIPerf AgentX source: $BUNDLED_AIPERF_TARBALL"
    mkdir -p "$WORKSPACE_ROOT/utils/aiperf"
    tar -xzf "$BUNDLED_AIPERF_TARBALL" -C "$WORKSPACE_ROOT/utils/aiperf"
  else
    if ! command -v curl >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates
    fi
    echo "Downloading AIPerf AgentX submodule: ${AIPERF_AGENTX_REF}"
    rm -rf "$WORKSPACE_ROOT/utils/aiperf"
    mkdir -p "$WORKSPACE_ROOT/utils/aiperf"
    curl -L --fail --retry 5 --retry-delay 2 \
      "https://codeload.github.com/SemiAnalysisAI/aiperf/tar.gz/${AIPERF_AGENTX_REF}" \
      | tar -xz --strip-components=1 -C "$WORKSPACE_ROOT/utils/aiperf"
  fi
fi

if [[ -f "${AIPERF_DIR:-$WORKSPACE_ROOT/utils/aiperf}/pyproject.toml" && "${AIPERF_ALLOW_GITHUB_TRANSFORMERS:-0}" != "1" ]]; then
  AIPERF_TRANSFORMERS_SPEC="${AIPERF_TRANSFORMERS_SPEC:-transformers>=4.53.0,<5}"
  echo "Using AIPerf transformers dependency override: ${AIPERF_TRANSFORMERS_SPEC}"
  sed -i -E \
    "s|\"transformers @ git\\+https://github.com/huggingface/transformers.git\"[^,]*,|\"${AIPERF_TRANSFORMERS_SPEC}\",|" \
    "${AIPERF_DIR:-$WORKSPACE_ROOT/utils/aiperf}/pyproject.toml"
fi

export INFMAX_CONTAINER_WORKSPACE="$WORKSPACE_ROOT"

# InferenceX commit 303669b hardcodes --num-dataset-entries 393 in the helper.
# AIPerf's Weka loaders load the full corpus when this flag is omitted, which
# is the desired AgentX behavior here.
sed -i '/REPLAY_CMD+=" --num-dataset-entries /d' "$WORKSPACE_ROOT/benchmarks/benchmark_lib.sh"

if [[ -n "${AIPERF_SYNTHESIS_MAX_OSL:-}" ]]; then
  AIPERF_ROOT="${AIPERF_DIR:-$WORKSPACE_ROOT/utils/aiperf}"
  WEKA_TRACE="$AIPERF_ROOT/src/aiperf/dataset/loader/weka_trace.py"
  if [[ ! -f "$WEKA_TRACE" ]]; then
    echo "ERROR: AIPERF_SYNTHESIS_MAX_OSL is set but Weka trace loader was not found at $WEKA_TRACE" >&2
    exit 1
  fi

  python3 - "$WEKA_TRACE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
original = text

if "max_tokens=creq.output_length," in text:
    marker = 'source_kind="weka_subagent",'
    marker_idx = text.find(marker)
    token_idx = text.find("max_tokens=creq.output_length,", marker_idx)
    if marker_idx == -1 or token_idx == -1:
        raise SystemExit(
            f"found uncapped subagent max_tokens in {path}, but not in the expected Weka subagent block"
        )
    text = text[:token_idx] + text[token_idx:].replace(
        "max_tokens=creq.output_length,",
        "max_tokens=self._cap_output(creq),",
        1,
    )

if '"source_kind": "weka_subagent",\n                    "theoretical_hit_blocks": hit_blocks,' in text:
    text = text.replace(
        '"source_kind": "weka_subagent",\n                    "theoretical_hit_blocks": hit_blocks,',
        '"source_kind": "weka_subagent",\n'
        '                    "capped_output_length": self._cap_output(creq),\n'
        '                    "theoretical_hit_blocks": hit_blocks,',
        1,
    )

if text == original:
    print(f"Weka trace loader already honors synthesis OSL cap for subagent turns: {path}")
else:
    path.write_text(text)
    print(f"Patched Weka trace loader to honor synthesis OSL cap for subagent turns: {path}")
PY
fi

export MODEL="$MODEL_NAME"
export MODEL_PREFIX="$MODEL_PREFIX_ARG"
export FRAMEWORK="$FRAMEWORK_ARG"
export PRECISION="$PRECISION_ARG"
export CONC="${CONCURRENCIES_ARG%% *}"
export CONC_LIST="$CONCURRENCIES_ARG"
export DURATION="$DURATION_ARG"
export RESULT_FILENAME="$RESULT_FILENAME_ARG"
export RESULT_DIR="${RESULT_DIR:-/logs/agentic}"
export AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-/logs/agentic_agg}"
export KV_OFFLOADING="$KV_OFFLOADING_ARG"
export KV_OFFLOAD_BACKEND="$KV_OFFLOAD_BACKEND_ARG"
if [[ -n "$TOTAL_CPU_DRAM_GB_ARG" && "$TOTAL_CPU_DRAM_GB_ARG" != "0" ]]; then
  export TOTAL_CPU_DRAM_GB="$TOTAL_CPU_DRAM_GB_ARG"
fi

echo "=============================================="
echo "AgentX benchmark"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "Port: ${PORT}"
echo "Model: ${MODEL}"
echo "Model prefix: ${MODEL_PREFIX}"
echo "Framework: ${FRAMEWORK}"
echo "Precision: ${PRECISION}"
echo "Concurrencies: ${CONC_LIST}"
echo "Duration: ${DURATION}"
echo "KV offloading: ${KV_OFFLOADING}"
echo "KV offload backend: ${KV_OFFLOAD_BACKEND:-<none>}"
echo "Result dir: ${RESULT_DIR}"
echo "Aggregate output dir: ${AGENTIC_OUTPUT_DIR}"
echo "=============================================="

bash "$WORKSPACE_ROOT/benchmarks/multi_node/agentic_srt.sh"
