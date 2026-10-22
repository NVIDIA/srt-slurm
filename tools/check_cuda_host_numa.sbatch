#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit from the repository root; pass the account, partition and node to sbatch.
# See tools/check_cuda_host_numa.md. No serving framework or wheel is installed.
#SBATCH --job-name=cuda-host-numa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --no-requeue
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=cuda-host-numa-%j.out

set -euo pipefail
export ENROOT_MAX_PROCESSORS=${ENROOT_MAX_PROCESSORS:-8}
export CUDA_EGM_IMAGE=${CUDA_EGM_IMAGE:-/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh}
export CUDA_EGM_SIZE=${CUDA_EGM_SIZE:-250GiB}
export CUDA_EGM_CASES=${CUDA_EGM_CASES:-same-numa,cross-numa,reverse-gpu,chunked}
export CUDA_EGM_ALLOCATION_TIMEOUT=${CUDA_EGM_ALLOCATION_TIMEOUT:-60}
export CUDA_EGM_BUDGET=${CUDA_EGM_BUDGET:-600}
export CUDA_EGM_SAMPLE_INTERVAL=${CUDA_EGM_SAMPLE_INTERVAL:-1}
export CUDA_EGM_HOLD_SECONDS=${CUDA_EGM_HOLD_SECONDS:-5}
export CUDA_EGM_CHUNK_SIZE=${CUDA_EGM_CHUNK_SIZE:-16GiB}
export CUDA_EGM_BUILD_TIMEOUT=${CUDA_EGM_BUILD_TIMEOUT:-120}
# Sixteen minutes for compilation, inventories and serial trials; four minutes for preservation.
export CUDA_EGM_STEP_TIMEOUT=${CUDA_EGM_STEP_TIMEOUT:-960}

RUN_DIR=${CUDA_EGM_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}/outputs}/cuda-host-numa-${SLURM_JOB_ID:?Missing job ID}
LOG_DIR=$RUN_DIR/logs
LAUNCHER_PID=
if [[ -e $RUN_DIR ]]; then
    printf 'Refusing stale run directory: %s\n' "$RUN_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_DIR" "$RUN_DIR/tools"
cat > "$LOG_DIR/perf-analysis.md" <<'REPORT'
# CUDA HOST_NUMA allocation diagnostic: pending or incomplete

Goal: reproduce failure of a second concurrent host-memory pool directly through
the CUDA Driver API, independently of any serving framework or transfer engine.
Compare same-NUMA placement, separate NUMA nodes, reversed GPU order and smaller
cuMemCreate requests. This is allocation-only: no model, CUDA device allocation,
payload transfer, network service or wheel installation is used.
The allocation trials have not completed. Inspect controller.log, probe.log,
logs/probe, build.log, and the host/container inventories for failure evidence.
TTFT p50/p95/p99 (ms), ITL p50/p99 (ms/token), input/output/total throughput
(tokens/s) and agentic Pareto/SLO metrics are N/A: no model requests are served.
The slurm-job-sizing skill bounds this diagnostic to one exclusive four-GPU node
for 20 minutes, a 16-minute step and four minutes for cleanup and preservation.
No DSight report is collected; native CUDA calls, NUMA snapshots, cgroup counters,
and kernel reclaim/compaction counters answer the allocation question.
REPORT

host_inventory() {
    local phase=$1 snapshot_rc=0
    if [[ ! -f $RUN_DIR/tools/cuda_host_numa_probe.py ]]; then
        printf 'Host %s inventory unavailable: helper was not copied.\n' "$phase" \
            >> "$LOG_DIR/host-inventory-errors.log"
        return 0
    fi
    timeout --signal=TERM --kill-after=2s 15s \
        python3 "$RUN_DIR/tools/cuda_host_numa_probe.py" inventory --output "$LOG_DIR/host-$phase.json" \
        >> "$LOG_DIR/host-inventory-errors.log" 2>&1 || snapshot_rc=$?
    if (( snapshot_rc != 0 )); then
        printf 'Host %s inventory failed with code %s.\n' "$phase" "$snapshot_rc" \
            >> "$LOG_DIR/host-inventory-errors.log"
    fi
}

finish() {
    local rc=$? deadline report_rc=0
    trap - EXIT
    trap '' INT TERM USR1
    set +e
    if [[ -n $LAUNCHER_PID ]] && kill -0 "$LAUNCHER_PID" 2>/dev/null; then
        kill -TERM "$LAUNCHER_PID" 2>/dev/null
        deadline=$((SECONDS + 90))
        while kill -0 "$LAUNCHER_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        kill -KILL "$LAUNCHER_PID" 2>/dev/null || true
        wait "$LAUNCHER_PID" 2>/dev/null
    fi
    host_inventory after
    timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-final.txt" 2>&1
    if [[ -f $RUN_DIR/tools/cuda_host_numa_probe.py ]]; then
        timeout --signal=TERM --kill-after=2s 15s \
            python3 "$RUN_DIR/tools/cuda_host_numa_probe.py" report --run-dir "$RUN_DIR" --exit-code "$rc" \
            || report_rc=$?
        if (( report_rc != 0 )); then
            printf '\nDiagnostic/report incomplete or requires attention (code %s); batch code %s.\n' "$report_rc" "$rc" \
                >> "$LOG_DIR/perf-analysis.md"
            if (( rc == 0 )); then rc=1; fi
        fi
    else
        printf '\nBatch failed before copying the report helper; exit code %s.\n' "$rc" \
            >> "$LOG_DIR/perf-analysis.md"
    fi
    printf '\nBatch exit code: %s.\n' "$rc" >> "$LOG_DIR/perf-analysis.md"
    printf 'Artifacts: %s\nReport: %s/perf-analysis.md\n' "$RUN_DIR" "$LOG_DIR"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 124' USR1
exec > >(tee -a "$LOG_DIR/controller.log") 2>&1

if [[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}} != 1 ]]; then
    printf 'This diagnostic requires exactly one allocated node.\n' >&2
    exit 2
fi
CUDA_EGM_CPUS_PER_TASK=$(python3 - "${SLURM_JOB_CPUS_PER_NODE:?Missing CPU allocation}" <<'PY'
import re
import sys

match = re.fullmatch(r"([1-9][0-9]*)(?:\(x1\))?", sys.argv[1].strip())
if not match:
    raise SystemExit(f"One full-node CPU allocation is required: {sys.argv[1]!r}")
print(int(match[1]))
PY
)
export CUDA_EGM_CPUS_PER_TASK
python3 - <<'PY'
import os
import re

for name in ("CUDA_EGM_ALLOCATION_TIMEOUT", "CUDA_EGM_BUDGET", "CUDA_EGM_BUILD_TIMEOUT", "CUDA_EGM_STEP_TIMEOUT"):
    if not re.fullmatch(r"[1-9][0-9]*", os.environ[name]):
        raise SystemExit(f"{name} must be a positive integer number of seconds")
PY
TOOL_DIR=${CUDA_EGM_TOOL_DIR:-$SLURM_SUBMIT_DIR/tools}
cp "$TOOL_DIR/cuda_host_numa_probe.py" "$TOOL_DIR/cuda_host_numa_repro.cpp" "$RUN_DIR/tools/"
cp "${BASH_SOURCE[0]}" "$RUN_DIR/submitted.sbatch"
if [[ -f $TOOL_DIR/check_cuda_host_numa.md ]]; then cp "$TOOL_DIR/check_cuda_host_numa.md" "$RUN_DIR/README.md"; fi
host_inventory before
python3 - "$RUN_DIR" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
keys = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_CPUS_PER_NODE", "CUDA_EGM_IMAGE",
        "CUDA_EGM_SIZE", "CUDA_EGM_CASES", "CUDA_EGM_ALLOCATION_TIMEOUT", "CUDA_EGM_BUDGET",
        "CUDA_EGM_SAMPLE_INTERVAL", "CUDA_EGM_HOLD_SECONDS", "CUDA_EGM_CHUNK_SIZE",
        "CUDA_EGM_BUILD_TIMEOUT", "CUDA_EGM_STEP_TIMEOUT", "CUDA_EGM_CXX", "CUDA_EGM_CUDA_HOME",
        "CUDA_EGM_INCLUDE", "CUDA_EGM_LIBCUDA")
metadata = {key: os.environ.get(key) for key in keys}
metadata.update(allocation_memory="HOST_NUMA", diagnostic="CUDA_Driver_API_allocation_only", run_dir=str(run_dir),
                source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((run_dir / "tools").iterdir()) if path.is_file()})
(run_dir / "logs" / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-initial.txt"
timeout 5s scontrol show node "$SLURM_JOB_NODELIST" > "$LOG_DIR/slurm-nodes.txt"

cat > "$RUN_DIR/launch-node.sh" <<'WRAPPER'
#!/bin/bash
set -euo pipefail
run_dir=${1:?Run directory is required}
diagnostic_pid=
cleanup() {
    local rc=$? deadline cleanup_failed=0
    trap - EXIT
    trap '' INT TERM
    set +e
    if [[ -n $diagnostic_pid ]]; then
        kill -TERM "$diagnostic_pid" 2>/dev/null || true
        deadline=$((SECONDS + 60))
        while kill -0 "$diagnostic_pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        if kill -0 "$diagnostic_pid" 2>/dev/null; then
            printf 'Diagnostic did not stop within 60 seconds; killing owned PID %s.\n' "$diagnostic_pid" >&2
            kill -KILL "$diagnostic_pid" 2>/dev/null || true
            cleanup_failed=1
        fi
        wait "$diagnostic_pid" 2>/dev/null
    fi
    timeout --signal=TERM --kill-after=2s 15s \
        python3 "$run_dir/tools/cuda_host_numa_probe.py" inventory --output "$run_dir/logs/container-after.json" \
        >> "$run_dir/logs/container-inventory-errors.log" 2>&1
    timeout --signal=TERM --kill-after=2s 5s dmesg --ctime > "$run_dir/logs/kernel-after.txt" 2>&1
    if (( cleanup_failed != 0 && rc == 0 )); then rc=1; fi
    printf 'exit_code=%s cleanup_failed=%s\n' "$rc" "$cleanup_failed" > "$run_dir/logs/shutdown.txt"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Remove inherited MPI launch state; preserve allocation/cgroup evidence.
for name in $(compgen -e); do
    case "$name" in PMIX_*|PMI_*|OMPI_*|MPI_*|OPAL_*|PRTE_*|HYDRA_*|I_MPI_*) unset "$name" ;; esac
done
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
{
    hostname
    timeout --signal=TERM --kill-after=2s 10s \
        nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version --format=csv || true
    timeout --signal=TERM --kill-after=2s 10s nvidia-smi topo -m || true
    timeout --signal=TERM --kill-after=2s 10s nvidia-smi -q || true
    if command -v numactl >/dev/null; then timeout --signal=TERM --kill-after=2s 5s numactl --hardware || true; fi
    ls -l /dev/nvidia-caps-imex-channels/ || true
} > "$run_dir/logs/container-hardware.txt" 2>&1
timeout --signal=TERM --kill-after=2s 5s dmesg --ctime > "$run_dir/logs/kernel-before.txt" 2>&1 || true
timeout --signal=TERM --kill-after=2s 15s \
    python3 "$run_dir/tools/cuda_host_numa_probe.py" inventory --output "$run_dir/logs/container-before.json" \
    > "$run_dir/logs/container-inventory-errors.log" 2>&1 || \
    printf 'Container inventory failed; see container-inventory-errors.log.\n' >&2

build() {
    local compiler=${CUDA_EGM_CXX:-g++} cuda_home=${CUDA_EGM_CUDA_HOME:-/usr/local/cuda}
    local include=${CUDA_EGM_INCLUDE:-} candidate library=${CUDA_EGM_LIBCUDA:-}
    if ! command -v "$compiler" >/dev/null; then
        if [[ -z ${CUDA_EGM_CXX:-} ]] && command -v c++ >/dev/null; then compiler=c++
        else printf 'A C++17 compiler is required; set CUDA_EGM_CXX.\n' >&2; return 2; fi
    fi
    if [[ -z $include ]]; then
        for candidate in "$cuda_home/include" "$cuda_home/targets/$(uname -m)-linux/include" /usr/local/cuda*/include; do
            if [[ -f $candidate/cuda.h ]]; then include=$candidate; break; fi
        done
    fi
    if [[ -z $include || ! -f $include/cuda.h ]]; then
        printf 'CUDA 12.9+ driver headers are required (CUDA 13 preferred); set CUDA_EGM_INCLUDE or CUDA_EGM_CUDA_HOME.\n' >&2
        return 2
    fi
    if [[ -z $library ]]; then
        for candidate in /usr/lib/"$(uname -m)"-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1 \
                         /usr/local/nvidia/lib64/libcuda.so.1 /usr/local/nvidia/lib/libcuda.so.1 \
                         "$cuda_home/lib64/stubs/libcuda.so" "$cuda_home/targets/$(uname -m)-linux/lib/stubs/libcuda.so"; do
            if [[ -f $candidate ]]; then library=$candidate; break; fi
        done
    fi
    local command=("$compiler" -std=c++17 -O2 -Wall -Wextra -I "$include" \
        "$run_dir/tools/cuda_host_numa_repro.cpp" -o "$run_dir/cuda_host_numa_repro")
    if [[ -n $library ]]; then command+=("$library"); else command+=(-lcuda); fi
    "$compiler" --version
    printf 'CUDA headers: %s\nCompiler command:' "$include"
    printf ' %q' "${command[@]}"
    printf '\n'
    python3 - "$run_dir/logs/build.json" "${command[@]}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"argv": sys.argv[2:]}, indent=2) + "\n")
PY
    # CUDA stubs, when used, are link-only. Never add stubs to the runtime library path.
    timeout --signal=TERM --kill-after=5s "${CUDA_EGM_BUILD_TIMEOUT}s" "${command[@]}"
    timeout 5s ldd "$run_dir/cuda_host_numa_repro"
    timeout 5s "$run_dir/cuda_host_numa_repro" --help
}
build > "$run_dir/logs/build.log" 2>&1
diagnostic_rc=0
# A completed CUDA allocation failure is evidence. Timeouts/incomplete cleanup fail the job.
timeout --signal=TERM --kill-after=45s "$((CUDA_EGM_BUDGET + 30))s" \
    python3 "$run_dir/tools/cuda_host_numa_probe.py" run \
    --binary "$run_dir/cuda_host_numa_repro" --output-dir "$run_dir/logs/probe" \
    --size "$CUDA_EGM_SIZE" --cases "$CUDA_EGM_CASES" \
    --allocation-timeout "$CUDA_EGM_ALLOCATION_TIMEOUT" --budget "$CUDA_EGM_BUDGET" \
    --sample-interval "$CUDA_EGM_SAMPLE_INTERVAL" --hold-seconds "$CUDA_EGM_HOLD_SECONDS" \
    --chunk-size "$CUDA_EGM_CHUNK_SIZE" > "$run_dir/logs/probe.log" 2>&1 &
diagnostic_pid=$!
wait "$diagnostic_pid" || diagnostic_rc=$?
diagnostic_pid=
printf 'CUDA allocation diagnostic helper exit code: %s\n' "$diagnostic_rc"
exit "$diagnostic_rc"
WRAPPER

printf 'CUDA HOST_NUMA diagnostic: node=%s CPUs/task=%s GPUs=4 pool=%s cases=%s budget=%ss step=%ss image=%s\n' \
    "$SLURM_JOB_NODELIST" "$CUDA_EGM_CPUS_PER_TASK" "$CUDA_EGM_SIZE" "$CUDA_EGM_CASES" \
    "$CUDA_EGM_BUDGET" "$CUDA_EGM_STEP_TIMEOUT" "$CUDA_EGM_IMAGE"
srun_options=(--overlap --exact --mpi=none --cpu-bind=none --kill-on-bad-exit=1 --wait=90 --export=ALL \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task="$CUDA_EGM_CPUS_PER_TASK" \
    --output="$LOG_DIR/rank-%t.out" --error="$LOG_DIR/rank-%t.err")
node_run_dir=$RUN_DIR
if [[ $CUDA_EGM_IMAGE != host ]]; then
    srun_options+=(--container-image="$CUDA_EGM_IMAGE" --container-mounts="$RUN_DIR:/cuda-egm-run" \
        --container-workdir=/cuda-egm-run --container-remap-root --container-writable --no-container-mount-home)
    node_run_dir=/cuda-egm-run
fi
timeout --signal=TERM --kill-after=90s "${CUDA_EGM_STEP_TIMEOUT}s" \
    srun "${srun_options[@]}" bash "$node_run_dir/launch-node.sh" "$node_run_dir" &
LAUNCHER_PID=$!
rc=0
wait "$LAUNCHER_PID" || rc=$?
LAUNCHER_PID=
exit "$rc"
