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

# The published direct TRT-LLM AgentX job does not scrape server metrics. Keep
# the normal srt-slurm default for other studies, but allow an exact public-job
# reproduction to opt out so the client does not add a polling workload.
if [[ "${AGENTX_DISABLE_SERVER_METRICS:-0}" == "1" ]]; then
  unset AIPERF_SERVER_METRICS_URLS
else
  FRONTEND_METRICS_URL="http://localhost:${PORT}/metrics"
  case ",${AIPERF_SERVER_METRICS_URLS:-}," in
    *",${FRONTEND_METRICS_URL},"*) ;;
    ",,") export AIPERF_SERVER_METRICS_URLS="$FRONTEND_METRICS_URL" ;;
    *) export AIPERF_SERVER_METRICS_URLS="$FRONTEND_METRICS_URL,$AIPERF_SERVER_METRICS_URLS" ;;
  esac
fi

PINNED_INFERENCEX_AGENTX_COMMIT="f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd"
PINNED_AIPERF_AGENTX_REF="754356e9a39acc6cc6afb242d123bb57c3fb6f75"
PINNED_AIPERF_ARCHIVE_SHA256="cfae0010ef60fda40df88132e701fbaf276c4305813a01a5357cf4c35ec1e680"
PINNED_BENCHMARK_LIB_SHA256="bb65f69ec8bec16c95e0f59779e94e02efff1ecc57b663d16e52e4121e3aae36"
PINNED_AGENTIC_SRT_SHA256="9f68b35323b11f2261c20b2f6fdc5df0902f81f2c2ec1d94840e1df7cde2b898"

INFERENCEX_AGENTX_COMMIT="${INFERENCEX_AGENTX_COMMIT:-$PINNED_INFERENCEX_AGENTX_COMMIT}"
# InferenceX's final-submission AIPerf pin includes the canonical timing policy
# and rejects runs whose TTFT or ITL observations cover less than 95% of the
# profiling phase.
AIPERF_AGENTX_REF="${AIPERF_AGENTX_REF:-$PINNED_AIPERF_AGENTX_REF}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLED_INFMAX_WORKSPACE="${AGENTX_BUNDLED_INFMAX_WORKSPACE:-$SCRIPT_DIR/inferencex}"
BUNDLED_AIPERF_TARBALL="${AGENTX_BUNDLED_AIPERF_TARBALL:-$SCRIPT_DIR/third_party/aiperf-agentx-v1-src.tgz}"

if [[ "$INFERENCEX_AGENTX_COMMIT" != "$PINNED_INFERENCEX_AGENTX_COMMIT" ]]; then
  echo "ERROR: InferenceX must be pinned to ToT commit $PINNED_INFERENCEX_AGENTX_COMMIT, got $INFERENCEX_AGENTX_COMMIT" >&2
  exit 1
fi
if [[ "$AIPERF_AGENTX_REF" != "$PINNED_AIPERF_AGENTX_REF" ]]; then
  echo "ERROR: AIPerf must be pinned to ToT commit $PINNED_AIPERF_AGENTX_REF, got $AIPERF_AGENTX_REF" >&2
  exit 1
fi

for forbidden_env in \
  AIPERF_DIR \
  AIPERF_LOCAL_WEKA_DATASET \
  AIPERF_TOKENIZER \
  AIPERF_APPLY_CHAT_TEMPLATE \
  AIPERF_SYNTHESIS_MAX_OSL \
  AIPERF_TRANSFORMERS_SPEC \
  AIPERF_ALLOW_GITHUB_TRANSFORMERS \
  AGENTX_USE_EXISTING_INFMAX_WORKSPACE; do
  if [[ -n "${!forbidden_env:-}" ]]; then
    echo "ERROR: $forbidden_env is not allowed in exact InferenceX ToT mode" >&2
    exit 1
  fi
done

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${AGENTX_WORKSPACE:-/tmp/inferencex-agentx-${SLURM_JOB_ID:-$$}}}"
case "$WORKSPACE_ROOT" in
  /tmp/inferencex-agentx-*|/tmp/inferencex-agentic-*) ;;
  *)
    echo "ERROR: exact ToT workspace must be an isolated /tmp/inferencex-agentx-* path, got $WORKSPACE_ROOT" >&2
    exit 1
    ;;
esac

if ! command -v tar >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends tar
fi

if [[ ! -f "$BUNDLED_INFMAX_WORKSPACE/benchmarks/benchmark_lib.sh" || \
      ! -f "$BUNDLED_INFMAX_WORKSPACE/benchmarks/multi_node/agentic_srt.sh" ]]; then
  echo "ERROR: exact bundled InferenceX ToT harness is incomplete: $BUNDLED_INFMAX_WORKSPACE" >&2
  exit 1
fi
if [[ ! -f "$BUNDLED_AIPERF_TARBALL" ]]; then
  echo "ERROR: exact bundled AIPerf archive is missing: $BUNDLED_AIPERF_TARBALL" >&2
  exit 1
fi

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: exact ToT source checksum mismatch for $path: expected $expected, got $actual" >&2
    exit 1
  fi
}

check_sha256 "$BUNDLED_INFMAX_WORKSPACE/benchmarks/benchmark_lib.sh" "$PINNED_BENCHMARK_LIB_SHA256"
check_sha256 "$BUNDLED_INFMAX_WORKSPACE/benchmarks/multi_node/agentic_srt.sh" "$PINNED_AGENTIC_SRT_SHA256"
check_sha256 "$BUNDLED_AIPERF_TARBALL" "$PINNED_AIPERF_ARCHIVE_SHA256"

rm -rf "$WORKSPACE_ROOT"
mkdir -p "$WORKSPACE_ROOT/utils/aiperf"
cp -a "$BUNDLED_INFMAX_WORKSPACE/." "$WORKSPACE_ROOT/"
tar -xzf "$BUNDLED_AIPERF_TARBALL" -C "$WORKSPACE_ROOT/utils/aiperf"
echo "Verified exact InferenceX ToT harness $PINNED_INFERENCEX_AGENTX_COMMIT"
echo "Verified exact AIPerf source $PINNED_AIPERF_AGENTX_REF"

AIPERF_ROOT="$WORKSPACE_ROOT/utils/aiperf"

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
    "minimum_profile_metric_coverage_ratio=0.95",
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

export INFMAX_CONTAINER_WORKSPACE="$WORKSPACE_ROOT"

# Keep model weights in the cluster-wide HF_HOME, but isolate Hugging Face
# dataset locks and Arrow caches per Slurm job. Shared dataset lock files can
# be owned by another runner and make AIPerf fail before the benchmark starts.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORKSPACE_ROOT}/hf-datasets}"
mkdir -p "$HF_DATASETS_CACHE"
echo "Hugging Face dataset cache: ${HF_DATASETS_CACHE}"

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
