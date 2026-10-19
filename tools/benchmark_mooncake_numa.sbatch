#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit from the repository root; select the partition with sbatch -p.
# Hecate: add --segment=2 to sbatch for one two-node topology segment.
#SBATCH --job-name=mooncake-numa
#SBATCH --account=root
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:25:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=mooncake-numa-%j.out

set -euo pipefail

# Enroot extracts the image before the container wrapper starts. Bound its
# host-side decompression parallelism independently of the full CPU allocation.
export ENROOT_MAX_PROCESSORS=${ENROOT_MAX_PROCESSORS:-8}
export MC_NUMA_IMAGE=${MC_NUMA_IMAGE:-/home/_shared/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh}
export MC_NUMA_CLUSTER=${MC_NUMA_CLUSTER:-raplab}
export MC_NUMA_ALLOW_CROSS_RAIL=${MC_NUMA_ALLOW_CROSS_RAIL:-0}
export MC_NUMA_ROUNDS=${MC_NUMA_ROUNDS:-3}
export MC_NUMA_SECONDS=${MC_NUMA_SECONDS:-10}
export MC_NUMA_WARMUP=${MC_NUMA_WARMUP:-2}
export MC_NUMA_BATCH_SIZE=${MC_NUMA_BATCH_SIZE:-512}
export MC_NUMA_PAGE_BYTES=${MC_NUMA_PAGE_BYTES:-2006016}
export MC_NUMA_POOL_GIB=${MC_NUMA_POOL_GIB:-4}
export MC_NUMA_WORKING_BATCHES=${MC_NUMA_WORKING_BATCHES:-2}
export MC_NUMA_BUDGET=${MC_NUMA_BUDGET:-1320}

RUN_DIR=${MC_NUMA_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR}/outputs}/mooncake-numa-${SLURM_JOB_ID}
LOG_DIR=$RUN_DIR/logs
LAUNCHER_PID= LAUNCHER_RC=
if [[ -e $RUN_DIR ]]; then
    printf 'Refusing stale run directory: %s\n' "$RUN_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_DIR/agents" "$RUN_DIR/tools"
cat > "$LOG_DIR/perf-analysis.md" <<'REPORT'
# Mooncake NUMA cache GET diagnostic: pending or incomplete

Goal: compare real Mooncake cache GETs with controlled host-memory placement
and destination-device affinity. No completed measurement is established yet.
Inspect controller.log, agents/ and the captured configuration before drawing
conclusions. TTFT p50/p95/p99 (ms), ITL p50/p99 (ms/token), input/output/total
throughput (tokens/s), and agentic Pareto/SLO metrics are N/A: no model or
inference requests are served. The slurm-job-sizing skill bounds this diagnostic
to 25 minutes, with a 22-minute controller budget and cleanup reserve.
REPORT

finish() {
    local rc=$? deadline
    trap - EXIT
    trap '' INT TERM USR1
    set +e
    if [[ -n $LAUNCHER_PID ]]; then
        # Signal only the launcher owned by this batch script. Slurm and the
        # controller own the two agents and their native child processes.
        kill -TERM "$LAUNCHER_PID" 2>/dev/null
        deadline=$((SECONDS + 90))
        while kill -0 "$LAUNCHER_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        if kill -0 "$LAUNCHER_PID" 2>/dev/null; then kill -KILL "$LAUNCHER_PID" 2>/dev/null; fi
        wait "$LAUNCHER_PID" 2>/dev/null
        LAUNCHER_RC=$?
    fi
    printf '\nAgent Slurm launcher exit code: %s. Batch exit code: %s.\n' \
        "${LAUNCHER_RC:-not launched}" "$rc" >> "$LOG_DIR/perf-analysis.md"
    timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-final.txt" 2>&1
    printf 'Artifacts: %s\nReport: %s/perf-analysis.md\n' "$RUN_DIR" "$LOG_DIR"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 124' USR1
exec > >(tee -a "$LOG_DIR/controller.log") 2>&1

if [[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}} != 2 ]]; then
    printf 'This diagnostic requires exactly two allocated nodes.\n' >&2
    exit 2
fi
# --exclusive reserves complete nodes. Ask each persistent task for its full
# allocation, rather than assuming the CPU count of either cluster's machines.
MC_NUMA_CPUS_PER_TASK=$(python3 - "${SLURM_JOB_CPUS_PER_NODE:?Missing Slurm CPU allocation}" <<'PY'
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
export MC_NUMA_CPUS_PER_TASK

SOURCE_DIR=${MC_NUMA_TOOL_DIR:-${SLURM_SUBMIT_DIR}/tools}
for tool in benchmark_mooncake_numa.py mooncake_numa_workload.py diagnose_mooncake.py check_mooncake_rails.py; do
    cp "$SOURCE_DIR/$tool" "$RUN_DIR/tools/$tool"
done
cp "${BASH_SOURCE[0]}" "$RUN_DIR/benchmark_mooncake_numa.sbatch"
timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-initial.txt"
timeout 5s scontrol show node "$SLURM_JOB_NODELIST" > "$LOG_DIR/slurm-nodes.txt"

cat > "$RUN_DIR/launch-node.sh" <<'WRAPPER'
#!/bin/bash
set -euo pipefail
run_dir=${1:?Run directory is required}
node=${SLURMD_NODENAME:?Missing Slurm node name}
install_log=$run_dir/logs/agents/$node.install.log
mkdir -p "$run_dir/logs/agents"
if command -v numactl >/dev/null; then
    printf 'numactl already installed; preserving the Mooncake installation.\n' > "$install_log"
else
    if [[ $(id -u) != 0 ]]; then
        printf 'numactl is missing and container uid is not 0.\n' > "$install_log"
        printf 'Container setup needs --container-remap-root and --container-writable; see %s\n' "$install_log" >&2
        exit 1
    fi
    export DEBIAN_FRONTEND=noninteractive
    if ! timeout --signal=TERM --kill-after=10s 240s bash -c '
        set -euo pipefail
        apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 update
        apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 \
            install -y --no-install-recommends numactl
    ' > "$install_log" 2>&1; then
        printf 'Container numactl installation failed or exceeded 240 seconds; see %s\n' "$install_log" >&2
        exit 1
    fi
    command -v numactl >> "$install_log"
fi
# Mooncake version/API preflight and all benchmark/report work belong to the
# controller. Do not install or upgrade its wheel in this wrapper.
case ${SLURM_PROCID:?Missing Slurm rank} in
    0) role=controller ;;
    1) role=agent ;;
    *) printf 'Unexpected Slurm rank: %s\n' "$SLURM_PROCID" >&2; exit 2 ;;
esac
exec python3 -m tools.benchmark_mooncake_numa "$role" --run-dir "$run_dir"
WRAPPER

printf 'Cluster=%s nodes=%s segment-size=%s CPUs/task=%s ENROOT_MAX_PROCESSORS=%s affinity-OFF-opt-in=%s image=%s\n' \
    "$MC_NUMA_CLUSTER" "$SLURM_JOB_NODELIST" "${SLURM_JOB_SEGMENT_SIZE:-unset}" \
    "$MC_NUMA_CPUS_PER_TASK" "$ENROOT_MAX_PROCESSORS" "$MC_NUMA_ALLOW_CROSS_RAIL" "$MC_NUMA_IMAGE"
srun --overlap --exact --mpi=none --cpu-bind=none --kill-on-bad-exit=1 --export=ALL \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="$MC_NUMA_CPUS_PER_TASK" \
    --container-image="$MC_NUMA_IMAGE" --container-mounts="$RUN_DIR:/bench-run" \
    --container-workdir=/bench-run --container-remap-root --container-writable \
    --output="$LOG_DIR/agents/%N.out" --error="$LOG_DIR/agents/%N.err" \
    bash /bench-run/launch-node.sh /bench-run &
LAUNCHER_PID=$!
rc=0
wait "$LAUNCHER_PID" || rc=$?
LAUNCHER_PID=
LAUNCHER_RC=$rc
exit "$rc"
