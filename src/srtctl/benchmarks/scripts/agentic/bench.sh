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

INFERENCEX_AGENTX_COMMIT="${INFERENCEX_AGENTX_COMMIT:-f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd}"
# InferenceX ToT's AIPerf pin includes the flattened warmup handoff clock and
# keeps the system/trajectory idle watchdogs active across phase barriers.
AIPERF_AGENTX_REF="${AIPERF_AGENTX_REF:-818c3a5a2922c535af6271ff296ed374e292b8e4}"
BUNDLED_AIPERF_REF="818c3a5a2922c535af6271ff296ed374e292b8e4"
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
  elif [[ -f "$BUNDLED_AIPERF_TARBALL" && "$AIPERF_AGENTX_REF" == "$BUNDLED_AIPERF_REF" ]]; then
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

AIPERF_ROOT="${AIPERF_DIR:-$WORKSPACE_ROOT/utils/aiperf}"

# The datasets library still resolves Hub metadata before consulting a cached
# Hub snapshot. Allow AgentX runs to consume a staged Weka JSONL directly so
# benchmarks are independent of Hugging Face availability and rate limits.
if [[ -n "${AIPERF_LOCAL_WEKA_DATASET:-}" ]]; then
  if [[ ! -r "$AIPERF_LOCAL_WEKA_DATASET" ]]; then
    echo "ERROR: AIPERF_LOCAL_WEKA_DATASET is not readable: $AIPERF_LOCAL_WEKA_DATASET" >&2
    exit 1
  fi

  LOCAL_WEKA_LOADER="$AIPERF_ROOT/src/aiperf/dataset/loader/semianalysis_cc_traces_weka.py"
  if [[ ! -f "$LOCAL_WEKA_LOADER" ]]; then
    echo "ERROR: staged Weka dataset requested but loader was not found: $LOCAL_WEKA_LOADER" >&2
    exit 1
  fi

  python3 - "$LOCAL_WEKA_LOADER" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
sentinel = "Loading staged local Weka dataset"

if sentinel in text:
    print(f"Local Weka dataset support already present: {path}")
    raise SystemExit(0)

marker = "    async def load_dataset(self) -> dict[str, list[WekaTrace]]:\n"
if text.count(marker) != 1:
    raise SystemExit(
        f"ERROR: expected one Weka load_dataset marker in {path}, "
        f"found {text.count(marker)}"
    )

method = '''    def _load_hf_dataset(self) -> Any:
        """Load a staged Weka JSONL without resolving Hugging Face Hub metadata."""
        import os
        from pathlib import Path

        from datasets import (
            Dataset,
            concatenate_datasets,
            load_dataset as hf_load_dataset,
            load_from_disk,
        )

        local_path = os.environ.get("AIPERF_LOCAL_WEKA_DATASET")
        if not local_path:
            return super()._load_hf_dataset()
        self.info(f"Loading staged local Weka dataset from '{local_path}'")
        if Path(local_path).is_dir():
            self.info("Using preprocessed Arrow dataset")
            arrow_files = sorted(Path(local_path).glob("*.arrow"))
            if arrow_files:
                datasets = [Dataset.from_file(str(item)) for item in arrow_files]
                return concatenate_datasets(datasets)
            return load_from_disk(local_path)
        return hf_load_dataset(
            "json",
            data_files={"train": local_path},
            split=self.hf_split,
            streaming=False,
            cache_dir=os.environ.get("HF_DATASETS_CACHE"),
        )

'''

path.write_text(text.replace(marker, method + marker, 1))
print(f"Patched Weka loader for staged local dataset: {path}")
PY

  echo "Staged Weka dataset: ${AIPERF_LOCAL_WEKA_DATASET}"
fi

python3 - \
  "$AIPERF_ROOT/src/aiperf/common/scenario/inferencex_agentx_mvp.py" \
  "$AIPERF_ROOT/src/aiperf/timing/replay_dependencies.py" \
  "$AIPERF_ROOT/src/aiperf/timing/strategies/agentic_replay.py" \
  "$AIPERF_ROOT/src/aiperf/common/config/loadgen_config.py" \
  "$AIPERF_ROOT/src/aiperf/config/phases.py" \
  "$AIPERF_ROOT/src/aiperf/timing/phase/runner.py" <<'PY'
from pathlib import Path
import re
import sys

scenario_path = Path(sys.argv[1])
dependencies_path = Path(sys.argv[2])
replay_path = Path(sys.argv[3])
legacy_loadgen_path = Path(sys.argv[4])
phases_path = Path(sys.argv[5])
phase_runner_path = Path(sys.argv[6])
if not scenario_path.is_file():
    raise SystemExit(
        f"ERROR: AIPerf AgentX scenario source is missing: {scenario_path}"
    )

scenario = scenario_path.read_text()
required_scenario = (
    "system_idle_gap_cap_seconds=10.0",
    "forbid_inter_turn_delay_cap=True",
)
missing = [item for item in required_scenario if item not in scenario]
if missing:
    raise SystemExit(
        "ERROR: stale AIPerf AgentX client: missing system-idle policy "
        + ", ".join(missing)
    )

if phases_path.is_file():
    # PR #31 moved phase configuration into config/phases.py. Its default
    # preserves recorded cross-lane phase-start spacing, while the new runtime
    # trace-idle watchdog remains an explicit opt-in.
    current_paths = (dependencies_path, replay_path, phase_runner_path)
    if not all(path.is_file() for path in current_paths):
        raise SystemExit(
            "ERROR: AIPerf AgentX PR #31 timing-policy sources are missing: "
            + ", ".join(str(path) for path in current_paths if not path.is_file())
        )
    dependencies = dependencies_path.read_text()
    replay = replay_path.read_text()
    phases = phases_path.read_text()
    runner = phase_runner_path.read_text()
    if "forbid_trace_idle_gap_cap=True" in scenario:
        raise SystemExit(
            "ERROR: stale AIPerf AgentX client: trace idle cap is still a loader-time forbidden option"
        )
    if "root_idle_gap_cap_seconds" not in dependencies or "idle_cap_expired" not in dependencies:
        raise SystemExit(
            "ERROR: stale AIPerf AgentX client: persistent trajectory-tree idle watchdog is missing"
        )
    required_replay = (
        "spread = not self._burst_phase_starts",
        "self.scheduler.set_drain_observer(self.enforce_system_idle_cap)",
        "_handoff_replay_offset_ms",
    )
    missing_replay = [item for item in required_replay if item not in replay]
    if missing_replay:
        raise SystemExit(
            "ERROR: stale AIPerf AgentX client: missing ToT replay behavior "
            + ", ".join(missing_replay)
        )
    burst_field = re.search(
        r"burst_phase_starts:.*?Field\(\s*default=False,",
        phases,
        re.DOTALL,
    )
    if burst_field is None:
        raise SystemExit(
            "ERROR: stale AIPerf PR #31 client: phase starts do not preserve "
            "recorded spacing by default"
        )
    if "trace_idle_gap_cap_seconds" not in runner or "whole-tree idle" not in runner:
        raise SystemExit(
            "ERROR: stale AIPerf PR #31 client: runtime tree-idle watchdog is missing"
        )
    print(
        "Verified AIPerf AgentX PR #31 policy and timing policy: preserved phase spacing, "
        "system-idle cap 10s, optional runtime tree-idle watchdog"
    )
elif legacy_loadgen_path.is_file():
    # The bundled pre-PR client uses its former burst-phase default and forbids
    # per-trace idle caps. Retain this path for offline fallback diagnostics.
    loadgen = legacy_loadgen_path.read_text()
    if "forbid_trace_idle_gap_cap=True" not in scenario:
        raise SystemExit(
            "ERROR: stale bundled AIPerf client: per-trace idle caps are not forbidden"
        )
    if re.search(r"burst_phase_starts:.*?\]\s*=\s*True", loadgen, re.DOTALL) is None:
        raise SystemExit(
            "ERROR: stale bundled AIPerf client: burst phase starts are not enabled "
            "by default"
        )
    print("Verified bundled AIPerf AgentX timing policy: system-idle cap 10s, burst phase starts")
else:
    raise SystemExit(
        "ERROR: AIPerf phase-policy source is missing; expected either "
        f"{phases_path} or {legacy_loadgen_path}"
    )
PY

if [[ -f "${AIPERF_DIR:-$WORKSPACE_ROOT/utils/aiperf}/pyproject.toml" && "${AIPERF_ALLOW_GITHUB_TRANSFORMERS:-0}" != "1" ]]; then
  AIPERF_TRANSFORMERS_SPEC="${AIPERF_TRANSFORMERS_SPEC:-transformers>=4.53.0,<5}"
  echo "Using AIPerf transformers dependency override: ${AIPERF_TRANSFORMERS_SPEC}"
  sed -i -E \
    "s|\"transformers @ git\\+https://github.com/huggingface/transformers.git\"[^,]*,|\"${AIPERF_TRANSFORMERS_SPEC}\",|" \
    "${AIPERF_DIR:-$WORKSPACE_ROOT/utils/aiperf}/pyproject.toml"
fi

export INFMAX_CONTAINER_WORKSPACE="$WORKSPACE_ROOT"

# Keep model weights in the cluster-wide HF_HOME, but isolate Hugging Face
# dataset locks and Arrow caches per Slurm job. Shared dataset lock files can
# be owned by another runner and make AIPerf fail before the benchmark starts.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORKSPACE_ROOT}/hf-datasets}"
mkdir -p "$HF_DATASETS_CACHE"
echo "Hugging Face dataset cache: ${HF_DATASETS_CACHE}"

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
