#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit from the repository root; select the account, partition and node with sbatch.
# See tools/check_mooncake_egm_reclaim.md for the allocation cases and evidence.
#SBATCH --job-name=mooncake-egm-reclaim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --no-requeue
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=mooncake-egm-reclaim-%j.out

set -euo pipefail
export ENROOT_MAX_PROCESSORS=${ENROOT_MAX_PROCESSORS:-8}
export MC_EGM_IMAGE=${MC_EGM_IMAGE:-/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh}
export MC_EGM_WHEEL_DIR=${MC_EGM_WHEEL_DIR:-/home/rihuo/mooncake_egm_wheels}
export MC_EGM_RECLAIM_POOL_SIZE=${MC_EGM_RECLAIM_POOL_SIZE:-250GiB}
export MC_EGM_RECLAIM_CASES=${MC_EGM_RECLAIM_CASES:-same-numa,cross-numa,reverse-gpu,chunked}
export MC_EGM_RECLAIM_TIMEOUT=${MC_EGM_RECLAIM_TIMEOUT:-60}
export MC_EGM_RECLAIM_BUDGET=${MC_EGM_RECLAIM_BUDGET:-600}
export MC_EGM_RECLAIM_SAMPLE_INTERVAL=${MC_EGM_RECLAIM_SAMPLE_INTERVAL:-1}
export MC_EGM_RECLAIM_HOLD_SECONDS=${MC_EGM_RECLAIM_HOLD_SECONDS:-2}
export MC_EGM_RECLAIM_CHUNK_SIZE=${MC_EGM_RECLAIM_CHUNK_SIZE:-16GiB}
export MC_EGM_SETUP_TIMEOUT=${MC_EGM_SETUP_TIMEOUT:-240}
export MC_EGM_READY_TIMEOUT=${MC_EGM_READY_TIMEOUT:-60}
# Sixteen minutes for setup and the serial cases, with four minutes for preservation.
export MC_EGM_STEP_TIMEOUT=${MC_EGM_STEP_TIMEOUT:-960}

RUN_DIR=${MC_EGM_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}/outputs}/mooncake-egm-reclaim-${SLURM_JOB_ID:?Missing job ID}
LOG_DIR=$RUN_DIR/logs
LAUNCHER_PID=
if [[ -e $RUN_DIR ]]; then
    printf 'Refusing stale run directory: %s\n' "$RUN_DIR" >&2
    exit 2
fi
mkdir -p "$LOG_DIR" "$RUN_DIR/tools" "$RUN_DIR/configs"
cat > "$LOG_DIR/perf-analysis.md" <<'REPORT'
# Mooncake EGM host-memory reclamation diagnostic: pending or incomplete

Goal: determine whether retaining the first 250GiB HOST_NUMA pool prevents a
second same-NUMA allocation, and compare another NUMA node, reversed GPU order
and smaller native allocation chunks. This is an allocation-only diagnostic:
no model, device payload, reader, RDMA fallback or inference traffic is used.
The allocation cases and sampled host/cgroup counters have not completed yet.
Inspect controller.log, reclaim.log, logs/reclaim and the node inventories.
TTFT p50/p95/p99 (ms), ITL p50/p99 (ms/token), input/output/total throughput
(tokens/s) and agentic Pareto/SLO metrics are N/A: no model requests are served.
The slurm-job-sizing skill bounds this diagnostic to one exclusive four-GPU
node for 20 minutes, a 16-minute step and a four-minute preservation reserve.
No DSight report is collected; native allocation logs, NUMA memory snapshots,
cgroup counters and kernel reclaim/compaction deltas answer this goal.
REPORT

host_inventory() {
    local phase=$1 snapshot_rc=0
    if [[ ! -f $RUN_DIR/tools/mooncake_egm_reclaim.py ]]; then
        printf 'Host %s inventory unavailable: helper was not copied.\n' "$phase" \
            >> "$LOG_DIR/host-inventory-errors.log"
        return 0
    fi
    timeout --signal=TERM --kill-after=2s 15s env PYTHONPATH="$RUN_DIR" \
        python3 -m tools.mooncake_egm_reclaim inventory --output "$LOG_DIR/host-$phase.json" \
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
        # Stop this owned step; its wrapper stops the diagnostic before the master.
        kill -TERM "$LAUNCHER_PID" 2>/dev/null
        deadline=$((SECONDS + 90))
        while kill -0 "$LAUNCHER_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        kill -KILL "$LAUNCHER_PID" 2>/dev/null || true
        wait "$LAUNCHER_PID" 2>/dev/null
    fi
    host_inventory after
    timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-final.txt" 2>&1
    if [[ -f $RUN_DIR/tools/mooncake_egm_reclaim.py ]]; then
        timeout --signal=TERM --kill-after=2s 15s env PYTHONPATH="$RUN_DIR" \
            python3 -m tools.mooncake_egm_reclaim report --run-dir "$RUN_DIR" --exit-code "$rc" \
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
MC_EGM_CPUS_PER_TASK=$(python3 - "${SLURM_JOB_CPUS_PER_NODE:?Missing CPU allocation}" <<'PY'
import re
import sys

match = re.fullmatch(r"([1-9][0-9]*)(?:\(x1\))?", sys.argv[1].strip())
if not match:
    raise SystemExit(f"One full-node CPU allocation is required: {sys.argv[1]!r}")
print(int(match[1]))
PY
)
export MC_EGM_CPUS_PER_TASK
python3 - <<'PY'
import os
import re

for name in ("MC_EGM_RECLAIM_TIMEOUT", "MC_EGM_RECLAIM_BUDGET", "MC_EGM_SETUP_TIMEOUT",
             "MC_EGM_READY_TIMEOUT", "MC_EGM_STEP_TIMEOUT"):
    value = os.environ[name]
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise SystemExit(f"{name} must be a positive integer number of seconds")
PY

TOOL_DIR=${MC_EGM_TOOL_DIR:-$SLURM_SUBMIT_DIR/tools}
CONFIG_DIR=${MC_EGM_CONFIG_DIR:-$SLURM_SUBMIT_DIR/configs}
cp "$TOOL_DIR/mooncake_egm_reclaim.py" "$TOOL_DIR/mooncake_egm_alloc_diagnostics.py" "$RUN_DIR/tools/"
cp "$CONFIG_DIR/mooncake_egm_donor.py" "$CONFIG_DIR/setup_mooncake_egm.sh" "$RUN_DIR/configs/"
if [[ -f $TOOL_DIR/__init__.py ]]; then
    cp "$TOOL_DIR/__init__.py" "$RUN_DIR/tools/"
else
    touch "$RUN_DIR/tools/__init__.py"
fi
if [[ -f $CONFIG_DIR/__init__.py ]]; then
    cp "$CONFIG_DIR/__init__.py" "$RUN_DIR/configs/"
else
    touch "$RUN_DIR/configs/__init__.py"
fi
cp "${BASH_SOURCE[0]}" "$RUN_DIR/submitted.sbatch"
host_inventory before
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
digest = hashlib.sha256()
with wheel.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
keys = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_CPUS_PER_NODE", "MC_EGM_IMAGE",
        "MC_EGM_RECLAIM_POOL_SIZE", "MC_EGM_RECLAIM_CASES", "MC_EGM_RECLAIM_TIMEOUT",
        "MC_EGM_RECLAIM_BUDGET", "MC_EGM_RECLAIM_SAMPLE_INTERVAL", "MC_EGM_RECLAIM_HOLD_SECONDS",
        "MC_EGM_RECLAIM_CHUNK_SIZE", "MC_EGM_SETUP_TIMEOUT", "MC_EGM_READY_TIMEOUT",
        "MC_EGM_STEP_TIMEOUT", "MC_MAX_MR_SIZE")
metadata = {key: os.environ.get(key) for key in keys}
metadata.update(wheel=wheel.name, wheel_sha256=digest.hexdigest(), allocation_memory="HOST_NUMA",
                diagnostic="allocation_only", run_dir=str(run_dir))
(run_dir / "logs" / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
timeout 5s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-initial.txt"
timeout 5s scontrol show node "$SLURM_JOB_NODELIST" > "$LOG_DIR/slurm-nodes.txt"

cat > "$RUN_DIR/launch-node.sh" <<'WRAPPER'
#!/bin/bash
set -euo pipefail
run_dir=${1:?Run directory is required}
master_pid= diagnostic_pid=

stop_child() {
    local pid=$1 label=$2 grace=$3 deadline child_rc=0
    [[ -n $pid ]] || return 0
    kill -TERM "$pid" 2>/dev/null || true
    deadline=$((SECONDS + grace))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
        printf '%s did not stop within %s seconds; killing owned PID %s.\n' "$label" "$grace" "$pid" >&2
        kill -KILL "$pid" 2>/dev/null || true
        cleanup_failed=1
    fi
    wait "$pid" 2>/dev/null || child_rc=$?
    if (( child_rc != 0 && child_rc != 143 )); then
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
    stop_child "$diagnostic_pid" diagnostics 60
    stop_child "$master_pid" master 10
    if (( cleanup_failed != 0 && rc == 0 )); then rc=1; fi
    printf 'exit_code=%s cleanup_failed=%s\n' "$rc" "$cleanup_failed" \
        > "$run_dir/logs/shutdown.txt"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Retain Slurm allocation variables for the cgroup/environment evidence.
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
    > "$run_dir/logs/setup.log" 2>&1
{
    hostname
    timeout --signal=TERM --kill-after=2s 10s \
        nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version --format=csv || true
    timeout --signal=TERM --kill-after=2s 10s nvidia-smi topo -m || true
    timeout --signal=TERM --kill-after=2s 10s nvidia-smi -q || true
    if command -v numactl >/dev/null; then
        timeout --signal=TERM --kill-after=2s 5s numactl --hardware || true
    fi
    ls -l /dev/nvidia-caps-imex-channels/ || true
} > "$run_dir/logs/container-hardware.txt" 2>&1
timeout --signal=TERM --kill-after=2s 15s \
    python3 -m tools.mooncake_egm_reclaim inventory --output "$run_dir/logs/container-before.json" \
    > "$run_dir/logs/container-inventory-errors.log" 2>&1 || \
    printf 'Container inventory failed; see container-inventory-errors.log.\n' >&2

# Allocate two ephemeral listeners on this exclusive node; no port-range scan.
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
python3 - "$master_port" "$MC_EGM_READY_TIMEOUT" "$master_pid" <<'PY'
import os
import socket
import sys
import time

deadline = time.monotonic() + int(sys.argv[2])
while time.monotonic() < deadline:
    os.kill(int(sys.argv[3]), 0)
    try:
        with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
            break
    except OSError:
        time.sleep(0.25)
else:
    raise SystemExit("Mooncake master RPC readiness timed out; see logs/master.log")
PY
kill -0 "$master_pid"
diagnostic_rc=0
# A controlled allocation error is diagnostic evidence; the helper distinguishes
# completed cases from timeouts, missing results or unconfirmed donor cleanup.
timeout --signal=TERM --kill-after=45s "$((MC_EGM_RECLAIM_BUDGET + 30))s" \
    python3 -m tools.mooncake_egm_reclaim run \
    --config "$run_dir/master-config.json" --output-dir "$run_dir/logs/reclaim" \
    --pool-size "$MC_EGM_RECLAIM_POOL_SIZE" --cases "$MC_EGM_RECLAIM_CASES" \
    --timeout-seconds "$MC_EGM_RECLAIM_TIMEOUT" --budget-seconds "$MC_EGM_RECLAIM_BUDGET" \
    --sample-interval "$MC_EGM_RECLAIM_SAMPLE_INTERVAL" --hold-seconds "$MC_EGM_RECLAIM_HOLD_SECONDS" \
    --chunk-size "$MC_EGM_RECLAIM_CHUNK_SIZE" > "$run_dir/logs/reclaim.log" 2>&1 &
diagnostic_pid=$!
wait "$diagnostic_pid" || diagnostic_rc=$?
diagnostic_pid=
kill -0 "$master_pid"
printf 'Reclamation diagnostic helper exit code: %s\n' "$diagnostic_rc"
exit "$diagnostic_rc"
WRAPPER

printf 'EGM host-memory allocation diagnostic: node=%s, CPUs/task=%s, GPUs=4, pool/donor=%s, cases=%s, budget=%ss, step=%ss, image=%s\n' \
    "$SLURM_JOB_NODELIST" "$MC_EGM_CPUS_PER_TASK" "$MC_EGM_RECLAIM_POOL_SIZE" "$MC_EGM_RECLAIM_CASES" \
    "$MC_EGM_RECLAIM_BUDGET" "$MC_EGM_STEP_TIMEOUT" "$MC_EGM_IMAGE"
timeout --signal=TERM --kill-after=90s "${MC_EGM_STEP_TIMEOUT}s" \
    srun --overlap --exact --mpi=none --cpu-bind=none --kill-on-bad-exit=1 --wait=60 --export=ALL \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task="$MC_EGM_CPUS_PER_TASK" \
    --container-image="$MC_EGM_IMAGE" --container-mounts="$RUN_DIR:/egm-run,$MC_EGM_WHEEL_DIR:/egm-wheels:ro" \
    --container-workdir=/egm-run --container-remap-root --container-writable --no-container-mount-home \
    --output="$LOG_DIR/rank-%t.out" --error="$LOG_DIR/rank-%t.err" \
    bash /egm-run/launch-node.sh /egm-run &
LAUNCHER_PID=$!
rc=0
wait "$LAUNCHER_PID" || rc=$?
LAUNCHER_PID=
exit "$rc"
