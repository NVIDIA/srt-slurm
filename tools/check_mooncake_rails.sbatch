#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit from the repository root: sbatch tools/check_mooncake_rails.sbatch
# Keep the two Python helpers and setup_mooncake_rails.sh beside this script.
#SBATCH --job-name=mooncake-rails
#SBATCH --account=root
#SBATCH --partition=batch-r42
#SBATCH --nodelist=vr-r42-cn[03,05]
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:25:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=mooncake-rails-%j.out

set -euo pipefail
# Both sockets must be available to the one node agent so numactl can bind
# each selected NIC's CPU and memory locally. No GPUs are requested.
export ENROOT_MAX_PROCESSORS=${ENROOT_MAX_PROCESSORS:-8}
export MC_RAIL_IMAGE=${MC_RAIL_IMAGE:-/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh}
export MC_RAIL_NICS=${MC_RAIL_NICS:-mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11}
export MC_RAIL_BUDGET=${MC_RAIL_BUDGET:-1320}
export MC_RAIL_INVENTORY_TIMEOUT=${MC_RAIL_INVENTORY_TIMEOUT:-420}
RUN_DIR=${MC_RAIL_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR}/outputs}/mooncake-rails-${SLURM_JOB_ID}
LOG_DIR=$RUN_DIR/logs
CONTROLLER_PID= AGENTS_PID=
if [[ -e $RUN_DIR ]]; then
    printf 'Refusing to reuse existing run directory: %s\n' "$RUN_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_DIR/agents" "$LOG_DIR/cases"
cat > "$LOG_DIR/perf-analysis.md" <<'REPORT'
# Mooncake cross-rail diagnostic: pending or incomplete

Goal: explicitly test reader NIC x donor NIC RDMA read/write connectivity.
No completed test is established yet. Inspect controller.log and agent output.
TTFT p50/p95/p99, ITL p50/p99, token throughput and agentic Pareto/SLO metrics
are N/A: this diagnostic serves no model and requests no GPUs.
The slurm-job-sizing skill bounds diagnostic traffic and reserves time for
cleanup/reporting. Inspect the submitted batch script and controller budget.
No DSight report/dashboard is required.
REPORT

finish() {
    local rc=$? deadline agent_rc=0
    trap - EXIT
    trap '' INT TERM USR1
    set +e
    # Signal only this job's two Slurm agent tasks, which terminate owned test
    # process groups before exiting. No pkill, SSH, or job-wide cancellation.
    if [[ -n $CONTROLLER_PID ]] && kill -0 "$CONTROLLER_PID" 2>/dev/null; then
        kill -TERM "$CONTROLLER_PID" 2>/dev/null
        deadline=$((SECONDS + 20))
        while kill -0 "$CONTROLLER_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        kill -KILL "$CONTROLLER_PID" 2>/dev/null || true
        wait "$CONTROLLER_PID" 2>/dev/null
    fi
    if [[ -n $AGENTS_PID ]]; then
        # The controller posted shutdown commands. Allow tasks and Slurm's
        # container cleanup to finish before escalating their launcher.
        deadline=$((SECONDS + 30))
        while kill -0 "$AGENTS_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        if kill -0 "$AGENTS_PID" 2>/dev/null; then
            kill -TERM "$AGENTS_PID" 2>/dev/null
            deadline=$((SECONDS + 20))
            while kill -0 "$AGENTS_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
            if kill -0 "$AGENTS_PID" 2>/dev/null; then kill -KILL "$AGENTS_PID" 2>/dev/null; fi
        fi
        wait "$AGENTS_PID" 2>/dev/null || agent_rc=$?
        printf '\nAgent Slurm launcher exit code: %s.\n' "$agent_rc" >> "$LOG_DIR/perf-analysis.md"
        if (( agent_rc != 0 && rc == 0 )); then rc=1; fi
    fi
    timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-final.txt" 2>&1
    printf '\nBatch exit code: %s. See controller.log and agents/ for termination/preflight details.\n' "$rc" \
        >> "$LOG_DIR/perf-analysis.md"
    printf 'Artifacts: %s\nReport: %s/perf-analysis.md\n' "$RUN_DIR" "$LOG_DIR"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 124' USR1
exec > >(tee -a "$LOG_DIR/controller.log") 2>&1

# --exclusive reserves both sockets. Use the full allocated CPU count rather
# than assuming either cluster's node size.
MC_RAIL_CPUS_PER_TASK=$(python3 - "${SLURM_JOB_CPUS_PER_NODE:?Missing Slurm CPU allocation}" <<'PY'
import re
import sys

counts = []
for item in sys.argv[1].split(","):
    match = re.fullmatch(r"([1-9][0-9]*)(?:\(x([1-9][0-9]*)\))?", item.strip())
    if not match:
        raise SystemExit(f"Cannot parse SLURM_JOB_CPUS_PER_NODE={sys.argv[1]!r}")
    counts.extend([int(match[1])] * int(match[2] or 1))
if len(counts) != 2 or counts[0] != counts[1]:
    raise SystemExit("Two homogeneous full-node CPU allocations are required")
print(counts[0])
PY
)
export MC_RAIL_CPUS_PER_TASK

SOURCE_DIR=${MC_RAIL_TOOL_DIR:-${SLURM_SUBMIT_DIR}/tools}
cp "$SOURCE_DIR/check_mooncake_rails.py" "$RUN_DIR/"
cp "$SOURCE_DIR/mooncake_rail_worker.py" "$RUN_DIR/"
cp "$SOURCE_DIR/setup_mooncake_rails.sh" "$RUN_DIR/"
cp "${BASH_SOURCE[0]}" "$RUN_DIR/check_mooncake_rails.sbatch"
if [[ -n ${MC_RAIL_LAUNCHER_SOURCE:-} ]]; then
    cp "$MC_RAIL_LAUNCHER_SOURCE" "$RUN_DIR/submitted.sbatch"
fi
timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-initial.txt"
timeout 5s scontrol show node "$SLURM_JOB_NODELIST" > "$LOG_DIR/slurm-nodes.txt"
NODES=()
while IFS= read -r node; do NODES+=("$node"); done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if (( ${#NODES[@]} != 2 )); then
    printf 'This diagnostic requires exactly two nodes.\n' >&2
    exit 2
fi
READER=${MC_RAIL_READER:-${NODES[0]}}
DONOR=${MC_RAIL_DONOR:-${NODES[1]}}
if [[ $READER == "$DONOR" || " ${NODES[*]} " != *" $READER "* || " ${NODES[*]} " != *" $DONOR "* ]]; then
    printf 'Reader and donor must be distinct nodes in this allocation.\n' >&2
    exit 2
fi
printf 'Cluster=%s reader=%s donor=%s NICs=%s segment-size=%s CPUs/task=%s ENROOT_MAX_PROCESSORS=%s image=%s\n' \
    "${MC_RAIL_CLUSTER:-unspecified}" "$READER" "$DONOR" "$MC_RAIL_NICS" "${SLURM_JOB_SEGMENT_SIZE:-unset}" \
    "$MC_RAIL_CPUS_PER_TASK" "$ENROOT_MAX_PROCESSORS" "$MC_RAIL_IMAGE"

# Persistent agents avoid 256 container launches and repeated Slurm teardown.
# The shared filesystem carries only commands/results; RDMA carries payloads.
# Package installation executes inside each writable container, never on the node.
srun --overlap --exact --mpi=none --cpu-bind=none --kill-on-bad-exit=1 --export=ALL \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="$MC_RAIL_CPUS_PER_TASK" \
    --container-image="$MC_RAIL_IMAGE" --container-mounts="$RUN_DIR:/rail-run" \
    --container-workdir=/rail-run --container-remap-root --container-writable \
    --output="$LOG_DIR/agents/%N.out" --error="$LOG_DIR/agents/%N.err" \
    bash -c 'exec bash /rail-run/setup_mooncake_rails.sh /rail-run "$SLURMD_NODENAME"' &
AGENTS_PID=$!
python3 "$RUN_DIR/check_mooncake_rails.py" --run-dir "$RUN_DIR" \
    --reader "$READER" --donor "$DONOR" --nics "$MC_RAIL_NICS" \
    --duration "${MC_RAIL_DURATION:-5}" --case-timeout "${MC_RAIL_CASE_TIMEOUT:-15}" \
    --inventory-timeout "$MC_RAIL_INVENTORY_TIMEOUT" --budget "$MC_RAIL_BUDGET" &
CONTROLLER_PID=$!
rc=0
wait "$CONTROLLER_PID" || rc=$?
CONTROLLER_PID=
exit "$rc"
