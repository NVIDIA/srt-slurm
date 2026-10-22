#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit from the repository root; choose the cluster account and partition with sbatch.
# See tools/check_mooncake_egm.md for prerequisites and the pass criteria.
#SBATCH --job-name=mooncake-egm
#SBATCH --nodes=2
#SBATCH --segment=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --no-requeue
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=mooncake-egm-%j.out

set -euo pipefail
export ENROOT_MAX_PROCESSORS=${ENROOT_MAX_PROCESSORS:-8}
export MC_EGM_IMAGE=${MC_EGM_IMAGE:-/lustre/fsw/coreai_comparch_inferencex/rihuo/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh}
export MC_EGM_WHEEL_DIR=${MC_EGM_WHEEL_DIR:-/lustre/fsw/coreai_comparch_inferencex/rihuo/mooncake_egm_wheels}
export MC_EGM_POOL_SIZE=${MC_EGM_POOL_SIZE:-4GiB}
export MC_EGM_GPU_IDS=${MC_EGM_GPU_IDS:-0,1,2,3}
export MC_EGM_SIZES=${MC_EGM_SIZES:-65536,1048576,16777216,67108864}
export MC_EGM_ITERATIONS=${MC_EGM_ITERATIONS:-10}
export MC_EGM_WARMUP=${MC_EGM_WARMUP:-2}
export MC_EGM_SETUP_TIMEOUT=${MC_EGM_SETUP_TIMEOUT:-240}
export MC_EGM_READY_TIMEOUT=${MC_EGM_READY_TIMEOUT:-180}
export MC_EGM_CLIENT_TIMEOUT=${MC_EGM_CLIENT_TIMEOUT:-180}
# Reproduce large HOST_NUMA allocation failures before the small transfer test.
export MC_EGM_ALLOC_DIAGNOSTICS=${MC_EGM_ALLOC_DIAGNOSTICS:-1}
export MC_EGM_ALLOC_SIZES=${MC_EGM_ALLOC_SIZES:-4GiB,260GiB}
export MC_EGM_ALLOC_GPU=${MC_EGM_ALLOC_GPU:-0}
export MC_EGM_ALLOC_NUMA_NODES=${MC_EGM_ALLOC_NUMA_NODES:-0,auto}
export MC_EGM_ALLOC_TIMEOUT=${MC_EGM_ALLOC_TIMEOUT:-30}
export MC_EGM_ALLOC_BUDGET=${MC_EGM_ALLOC_BUDGET:-150}
# Twelve minutes for extraction/setup/allocation/traffic; reserve three for cleanup/reporting.
export MC_EGM_TIMEOUT=${MC_EGM_TIMEOUT:-720}

RUN_DIR=${MC_EGM_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}/outputs}/mooncake-egm-${SLURM_JOB_ID:?Missing job ID}
LOG_DIR=$RUN_DIR/logs
LAUNCHER_PID=
if [[ -e $RUN_DIR ]]; then
    printf 'Refusing stale run directory: %s\n' "$RUN_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_DIR" "$RUN_DIR/tools" "$RUN_DIR/configs"
touch "$RUN_DIR/configs/__init__.py"
cat > "$LOG_DIR/perf-analysis.md" <<'REPORT'
# Mooncake EGM/NVLink diagnostic: pending or incomplete

Goal: verify cross-node pinned-host writes to and reads from EGM host DRAM over
NVLink, with full payload checks and no local Store capacity on the client.
Both payload endpoints use host memory; GPUs submit the CUDA transfers.
Before transfer, compare small and large EGM allocations on one visible GPU,
using explicit NUMA 0 and CUDA host NUMA discovery. Inspect allocation-diagnostics
for same-container NUMA memory, process/cgroup limits and native failure logs.
No completed transfer is established yet. Inspect controller.log and node logs.
TTFT p50/p95/p99 (ms), ITL p50/p99 (ms/token), input/output/total throughput
(tokens/s), and agentic Pareto/SLO metrics are N/A: no model requests are served.
The slurm-job-sizing skill bounds this diagnostic to a 15-minute allocation,
a 12-minute step deadline, and a cleanup/reporting reserve. No DSight report is
collected; native provider/client logs and replica metadata answer this goal.
REPORT

finish() {
    local rc=$? deadline
    trap - EXIT
    trap '' INT TERM USR1
    set +e
    if [[ -n $LAUNCHER_PID ]] && kill -0 "$LAUNCHER_PID" 2>/dev/null; then
        # GNU timeout forwards this signal to the owned Slurm step. The provider
        # task stops its donor before its master; no job-wide pkill or scancel.
        kill -TERM "$LAUNCHER_PID" 2>/dev/null
        deadline=$((SECONDS + 60))
        while kill -0 "$LAUNCHER_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        kill -KILL "$LAUNCHER_PID" 2>/dev/null || true
        wait "$LAUNCHER_PID" 2>/dev/null
    fi
    if (( rc == 0 )) && [[ ! -f $RUN_DIR/client.done ]]; then
        printf 'The Slurm step ended without verified client completion.\n' >&2
        rc=1
    fi
    if [[ -f $RUN_DIR/tools/check_mooncake_egm.py ]]; then
        python3 "$RUN_DIR/tools/check_mooncake_egm.py" report --run-dir "$RUN_DIR" --exit-code "$rc"
        report_rc=$?
        if (( report_rc != 0 )); then
            printf '\nReport finalization failed with code %s; batch code %s.\n' "$report_rc" "$rc" \
                >> "$LOG_DIR/perf-analysis.md"
            if (( rc == 0 )); then rc=1; fi
        fi
    else
        printf '\nBatch failed before copying the report helper; exit code %s.\n' "$rc" \
            >> "$LOG_DIR/perf-analysis.md"
    fi
    timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-final.txt" 2>&1
    printf '\nBatch exit code: %s.\n' "$rc" >> "$LOG_DIR/perf-analysis.md"
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
MC_EGM_CPUS_PER_TASK=$(python3 - "${SLURM_JOB_CPUS_PER_NODE:?Missing CPU allocation}" <<'PY'
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
export MC_EGM_CPUS_PER_TASK

TOOL_DIR=${MC_EGM_TOOL_DIR:-$SLURM_SUBMIT_DIR/tools}
CONFIG_DIR=${MC_EGM_CONFIG_DIR:-$SLURM_SUBMIT_DIR/configs}
cp "$TOOL_DIR/check_mooncake_egm.py" "$RUN_DIR/tools/"
cp "$TOOL_DIR/mooncake_egm_alloc_diagnostics.py" "$RUN_DIR/tools/"
cp "$CONFIG_DIR/mooncake_egm_donor.py" "$CONFIG_DIR/setup_mooncake_egm.sh" "$RUN_DIR/configs/"
cp "${BASH_SOURCE[0]}" "$RUN_DIR/submitted.sbatch"
if [[ ! -d $MC_EGM_WHEEL_DIR ]]; then
    printf 'Missing EGM wheel directory: %s\n' "$MC_EGM_WHEEL_DIR" >&2
    exit 2
fi
MC_EGM_WHEEL_DIR=$(cd -- "$MC_EGM_WHEEL_DIR" && pwd -P)
export MC_EGM_WHEEL_DIR
shopt -s nullglob
wheels=("$MC_EGM_WHEEL_DIR"/mooncake_transfer_engine*.whl)
if (( ${#wheels[@]} != 1 )); then
    printf 'Keep exactly one EGM-enabled ARM64 CUDA 13 wheel in %s; found %s.\n' \
        "$MC_EGM_WHEEL_DIR" "${#wheels[@]}" >&2
    exit 2
fi
python3 - "$RUN_DIR" "${wheels[0]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

run_dir, wheel = map(Path, sys.argv[1:])
with wheel.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
if digest is None:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
keys = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "MC_EGM_IMAGE", "MC_EGM_POOL_SIZE", "MC_EGM_TIMEOUT",
        "MC_EGM_GPU_IDS", "MC_EGM_SIZES", "MC_EGM_ITERATIONS", "MC_EGM_WARMUP",
        "MC_EGM_ALLOC_DIAGNOSTICS", "MC_EGM_ALLOC_SIZES", "MC_EGM_ALLOC_GPU",
        "MC_EGM_ALLOC_NUMA_NODES", "MC_EGM_ALLOC_TIMEOUT", "MC_EGM_ALLOC_BUDGET", "MC_MAX_MR_SIZE")
metadata = {key: os.environ.get(key) for key in keys}
metadata.update(wheel=wheel.name, wheel_sha256=digest, buffer_memory="pinned_host")
(run_dir / "logs" / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-initial.txt"
timeout 5s scontrol show node "$SLURM_JOB_NODELIST" > "$LOG_DIR/slurm-nodes.txt"

cat > "$RUN_DIR/launch-node.sh" <<'WRAPPER'
#!/bin/bash
set -euo pipefail
run_dir=${1:?Run directory is required}
rank=${SLURM_PROCID:?Missing Slurm task rank}
master_pid= provider_pid= diagnostic_pid=

stop_child() {
    local pid=$1 label=$2 deadline child_rc=0
    [[ -n $pid ]] || return 0
    kill -TERM "$pid" 2>/dev/null || true
    deadline=$((SECONDS + 20))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
        printf '%s did not stop within 20 seconds; killing owned PID %s.\n' "$label" "$pid" >&2
        kill -KILL "$pid" 2>/dev/null || true
        cleanup_failed=1
    fi
    wait "$pid" 2>/dev/null || child_rc=$?
    if (( child_rc != 0 )) && [[ $label == provider || $child_rc != 143 ]]; then
        printf '%s exited with code %s during shutdown.\n' "$label" "$child_rc" >&2
        cleanup_failed=1
    fi
}
cleanup() {
    local rc=$?
    trap - EXIT
    trap '' INT TERM
    set +e
    cleanup_failed=0
    stop_child "$diagnostic_pid" diagnostics
    stop_child "$provider_pid" provider
    stop_child "$master_pid" master
    if (( cleanup_failed != 0 && rc == 0 )); then rc=1; fi
    printf 'rank=%s exit_code=%s cleanup_failed=%s\n' "$rank" "$rc" "$cleanup_failed" \
        > "$run_dir/logs/shutdown-$rank.txt"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# These are Store clients, not TRT-LLM/MPI workers. Keep all four node GPUs
# visible so provider auto discovery includes both host NUMA nodes.
for name in $(compgen -e); do
    case "$name" in PMIX_*|PMI_*|OMPI_*|MPI_*|OPAL_*|PRTE_*|HYDRA_*|I_MPI_*) unset "$name" ;; esac
done
unset MC_USE_NVLINK_IPC MC_TE_FILTERS MC_ENABLE_DEST_DEVICE_AFFINITY MC_FORCE_TCP
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MC_MS_AUTO_DISC=0 MC_USE_TENT=0 MC_USE_TEV1=0 MOONCAKE_STORE_BACKEND=cpp
export GLOG_logtostderr=1 GLOG_minloglevel=0 GLOG_v=0 PYTHONUNBUFFERED=1
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export SRT_MOONCAKE_EGM_WHEEL_DIR=/egm-wheels
export PYTHONPATH="$run_dir${PYTHONPATH:+:$PYTHONPATH}"

timeout --signal=TERM --kill-after=10s "${MC_EGM_SETUP_TIMEOUT}s" \
    bash "$run_dir/configs/setup_mooncake_egm.sh" --mooncake-only \
    > "$run_dir/logs/setup-$rank.log" 2>&1

{
    hostname
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version --format=csv || true
    nvidia-smi topo -m || true
    # Some driver versions reject '-d FABRIC'; the full query includes Fabric state.
    nvidia-smi -q || true
    ls -l /dev/nvidia-caps-imex-channels/ || true
    command -v numactl >/dev/null && numactl --hardware || true
    cat /proc/self/status /proc/self/limits /proc/self/cgroup || true
    cat /proc/meminfo /proc/buddyinfo || true
    for memory_info in /sys/devices/system/node/node*/meminfo; do
        [[ ! -f $memory_info ]] || cat "$memory_info" || true
    done
    dmesg --ctime 2>&1 | tail -n 100 || true
} > "$run_dir/logs/inventory-$rank.txt" 2>&1

if [[ $rank == 0 ]]; then
    # Allocate two currently unused master listeners on this exclusive node.
    python3 - "$run_dir" <<'PY'
import json
import os
import socket
import sys
from contextlib import ExitStack
from pathlib import Path

run_dir = Path(sys.argv[1])
with ExitStack() as stack:
    listeners = [stack.enter_context(socket.socket()) for _ in range(2)]
    for listener in listeners:
        listener.bind(("0.0.0.0", 0))
    rpc, metrics = [listener.getsockname()[1] for listener in listeners]
hostname = socket.gethostname()
config = {"master_server_address": f"{socket.gethostbyname(hostname)}:{rpc}",
          "provider_hostname": hostname, "metrics_port": metrics}
temporary = run_dir / ".master-config.tmp"
temporary.write_text(json.dumps(config, indent=2) + "\n")
os.replace(temporary, run_dir / "master-config.json")
(run_dir / "master-ports").write_text(f"{rpc} {metrics}\n")
PY
    read -r master_port metrics_port < "$run_dir/master-ports"
    python3 -m mooncake.cli --rpc_port="$master_port" --metrics_port="$metrics_port" \
        --default_kv_lease_ttl=60000 --eviction_ratio=0.02 \
        > "$run_dir/logs/master.log" 2>&1 &
    master_pid=$!
    # Wait for the actual RPC listener rather than publishing premature readiness.
    python3 - "$master_port" "$MC_EGM_READY_TIMEOUT" <<'PY'
import socket
import sys
import time

deadline = time.monotonic() + int(sys.argv[2])
while time.monotonic() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
            break
    except OSError:
        time.sleep(0.25)
else:
    raise SystemExit("Mooncake master RPC readiness timed out; see logs/master.log")
PY
    kill -0 "$master_pid"
    if [[ $MC_EGM_ALLOC_DIAGNOSTICS == 1 ]]; then
        # Each allocation runs in a fresh child with just MC_EGM_ALLOC_GPU visible.
        # Controlled CUDA allocation errors are recorded and do not suppress the
        # separate small transfer test. Incomplete/unsafe cleanup aborts this step.
        diagnostic_rc=0
        # The outer timeout also bounds inventory/driver calls; allow ten seconds
        # beyond the helper's probe budget to save evidence and stop its child.
        timeout --signal=TERM --kill-after=10s "$((MC_EGM_ALLOC_BUDGET + 10))s" \
            python3 "$run_dir/tools/mooncake_egm_alloc_diagnostics.py" \
            --config "$run_dir/master-config.json" \
            --output-dir "$run_dir/logs/allocation-diagnostics" \
            --sizes "$MC_EGM_ALLOC_SIZES" --gpu "$MC_EGM_ALLOC_GPU" \
            --numa-nodes "$MC_EGM_ALLOC_NUMA_NODES" \
            --timeout-seconds "$MC_EGM_ALLOC_TIMEOUT" --budget-seconds "$MC_EGM_ALLOC_BUDGET" \
            > "$run_dir/logs/allocation-diagnostics.log" 2>&1 &
        diagnostic_pid=$!
        wait "$diagnostic_pid" || diagnostic_rc=$?
        diagnostic_pid=
        if (( diagnostic_rc != 0 )); then
            printf 'Allocation diagnostics incomplete (code %s); see logs/allocation-diagnostics.\n' \
                "$diagnostic_rc" >&2
            exit "$diagnostic_rc"
        fi
        kill -0 "$master_pid"
    elif [[ $MC_EGM_ALLOC_DIAGNOSTICS != 0 ]]; then
        printf 'MC_EGM_ALLOC_DIAGNOSTICS must be 0 or 1.\n' >&2
        exit 2
    fi
    python3 "$run_dir/configs/mooncake_egm_donor.py" serve \
        --config "$run_dir/master-config.json" --segment-size "$MC_EGM_POOL_SIZE" \
        --numa-nodes auto --ready-file "$run_dir/logs/provider.ready.json" \
        > "$run_dir/logs/provider.log" 2>&1 &
    provider_pid=$!
    deadline=$((SECONDS + MC_EGM_READY_TIMEOUT))
    while [[ ! -f $run_dir/logs/provider.ready.json ]]; do
        kill -0 "$master_pid"
        kill -0 "$provider_pid"
        if (( SECONDS >= deadline )); then
            printf 'EGM provider readiness timed out; see logs/provider.log.\n' >&2
            exit 1
        fi
        sleep 1
    done
    cp "$run_dir/logs/provider.ready.json" "$run_dir/logs/provider-config.json"
    deadline=$((SECONDS + MC_EGM_SETUP_TIMEOUT + MC_EGM_CLIENT_TIMEOUT + MC_EGM_READY_TIMEOUT))
    while [[ ! -f $run_dir/client.done ]]; do
        kill -0 "$master_pid"
        kill -0 "$provider_pid"
        if (( SECONDS >= deadline )); then
            printf 'Client completion timed out; see rank-1 and client logs.\n' >&2
            exit 1
        fi
        sleep 1
    done
elif [[ $rank == 1 ]]; then
    allocation_budget=0
    if [[ $MC_EGM_ALLOC_DIAGNOSTICS == 1 ]]; then allocation_budget=$((MC_EGM_ALLOC_BUDGET + 20)); fi
    deadline=$((SECONDS + MC_EGM_SETUP_TIMEOUT + allocation_budget + 2 * MC_EGM_READY_TIMEOUT))
    while [[ ! -f $run_dir/logs/provider.ready.json ]]; do
        if (( SECONDS >= deadline )); then
            printf 'Remote provider readiness timed out; see rank-0 logs.\n' >&2
            exit 1
        fi
        sleep 1
    done
    read -r master_address provider_hostname < <(python3 - "$run_dir/master-config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    config = json.load(stream)
print(config["master_server_address"], config["provider_hostname"])
PY
    )
    timeout --signal=TERM --kill-after=10s "${MC_EGM_CLIENT_TIMEOUT}s" \
        python3 "$run_dir/tools/check_mooncake_egm.py" client \
        --master "$master_address" --provider-hostname "$provider_hostname" \
        --run-dir "$run_dir" --gpu-ids "$MC_EGM_GPU_IDS" --sizes "$MC_EGM_SIZES" \
        --iterations "$MC_EGM_ITERATIONS" --warmup "$MC_EGM_WARMUP" \
        > "$run_dir/logs/client.log" 2>&1
    if ! grep -Fq 'NvlinkTransport: using cudaMemcpyBatchAsync' "$run_dir/logs/client.log"; then
        printf 'Missing actual NVLink transfer evidence in logs/client.log.\n' >&2
        exit 1
    fi
    touch "$run_dir/client.done"
else
    printf 'Unexpected Slurm rank: %s\n' "$rank" >&2
    exit 2
fi
WRAPPER

printf 'Host-to-host EGM smoke test: nodes=%s, CPUs/task=%s, pool=%s, CUDA submission devices=%s, step budget=%ss, image=%s\n' \
    "$SLURM_JOB_NODELIST" "$MC_EGM_CPUS_PER_TASK" "$MC_EGM_POOL_SIZE" "$MC_EGM_GPU_IDS" \
    "$MC_EGM_TIMEOUT" "$MC_EGM_IMAGE"
timeout --signal=TERM --kill-after=60s "${MC_EGM_TIMEOUT}s" \
    srun --overlap --exact --mpi=none --cpu-bind=none --kill-on-bad-exit=1 --wait=60 --export=ALL \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task="$MC_EGM_CPUS_PER_TASK" \
    --container-image="$MC_EGM_IMAGE" --container-mounts="$RUN_DIR:/egm-run,$MC_EGM_WHEEL_DIR:/egm-wheels:ro" \
    --container-workdir=/egm-run --container-remap-root --container-writable --no-container-mount-home \
    --output="$LOG_DIR/rank-%t.out" --error="$LOG_DIR/rank-%t.err" \
    bash /egm-run/launch-node.sh /egm-run &
LAUNCHER_PID=$!
rc=0
wait "$LAUNCHER_PID" || rc=$?
LAUNCHER_PID=
exit "$rc"
