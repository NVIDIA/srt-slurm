#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit from the repository root: sbatch tools/diagnose_mooncake.sbatch
# Requires the adjacent diagnose_mooncake.py on the shared filesystem.
#SBATCH --job-name=mooncake-rdma-diag
#SBATCH --account=root
#SBATCH --partition=batch-r42
#SBATCH --nodelist=vr-r42-cn[01-03,05-13,15-18]
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=mooncake-diag-%j.out

set -euo pipefail
# Host-memory diagnostic: no model or GPU kernels. Reserve three minutes for
# cleanup/reporting. A small working set does not prove second-region coverage.
IMAGE=${MC_DIAG_IMAGE:-/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh}
HELPER=${MC_DIAG_HELPER:-${SLURM_SUBMIT_DIR}/tools/diagnose_mooncake.py}
RUN_DIR=${MC_DIAG_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR}/outputs}/mooncake-diag-${SLURM_JOB_ID}
LOG_DIR=$RUN_DIR/logs
CASE_SECONDS=${MC_DIAG_SECONDS:-120}
BATCH_SIZE=${MC_DIAG_BATCH_SIZE:-512}
CLIENT_WORKERS=${MC_DIAG_CLIENT_WORKERS:-2}
BODY_DEADLINE=$((SECONDS + 22 * 60))
CASE_BUDGET=$((CASE_SECONDS + 330))
master_pid= donor_pid= client_pid= sampler_pid= aux_pid= aux_step=
case_dir=$LOG_DIR
overall_rc=0 finalized=0
STEP_WRAPPER='if [[ $SLURM_PROCID == 0 ]]; then printf "%s\n" "$SLURM_STEP_ID" > "/diag-run/$1"; fi; shift; exec "$@"'

event() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_DIR/events.log"; }
pid_running() {
    local candidate
    while read -r candidate; do [[ $candidate != "$1" ]] || return 0; done < <(jobs -pr)
    return 1
}
run_step() {
    local node_list=$1 node_count=$2
    shift 2
    local -a command=(srun --overlap --exact --mpi=none --cpu-bind=none
        --nodes="$node_count" --ntasks="$node_count" --ntasks-per-node=1
        --cpus-per-task=64 --nodelist="$node_list" --container-image="$IMAGE"
        --container-mounts="$RUN_DIR:/diag-run" --container-workdir=/diag-run)
    # Background callers' $! must own srun, not a shell waiting for srun.
    if (( BASH_SUBSHELL > 0 )); then exec "${command[@]}" "$@"; else "${command[@]}" "$@"; fi
}
step_gone() {
    local step=$1 listing
    listing=$(timeout 2s squeue --steps --noheader -j "$SLURM_JOB_ID" --format=%i) || return 2
    [[ ! $'\n'"$listing"$'\n' == *$'\n'"$SLURM_JOB_ID.$step"$'\n'* ]]
}
stop_step() {
    local pid=$1 step_file=$2 status_file=$3 step= rc=0 confirmed=0 deadline
    [[ -n $pid ]] || return 0
    [[ ! -s $step_file ]] || read -r step < "$step_file" || true
    if [[ $step =~ ^[0-9]+$ ]]; then
        event "stop_step pid=$pid step=$SLURM_JOB_ID.$step"
        timeout 3s scancel --full --signal=TERM "$SLURM_JOB_ID.$step" || true
        deadline=$((SECONDS + 8))
        while (( SECONDS < deadline )); do
            if step_gone "$step"; then confirmed=1; break; fi
            sleep 1
        done
        if (( confirmed == 0 )); then
            event "kill_step step=$SLURM_JOB_ID.$step"
            timeout 3s scancel --full --signal=KILL "$SLURM_JOB_ID.$step" || true
            deadline=$((SECONDS + 8))
            while (( SECONDS < deadline )); do
                if step_gone "$step"; then confirmed=1; break; fi
                sleep 1
            done
        fi
    else
        event "missing_step_identity pid=$pid file=$step_file"
    fi
    # Reap the local launcher too; this alone cannot confirm remote task exit.
    if pid_running "$pid"; then
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        if pid_running "$pid"; then kill -KILL "$pid" 2>/dev/null || true; fi
    fi
    wait "$pid" || rc=$?
    printf '%s\n' "$rc" > "$status_file"
    if (( confirmed == 0 )); then
        printf 'Remote teardown unconfirmed; campaign must not advance.\n' > "$case_dir/teardown-unconfirmed.txt"
        event "teardown_unconfirmed pid=$pid step=$step"
        return 1
    fi
}
stop_sampler() {
    [[ -n $sampler_pid ]] || return 0
    if pid_running "$sampler_pid"; then
        kill -TERM "$sampler_pid" 2>/dev/null || true
        sleep 1
        if pid_running "$sampler_pid"; then kill -KILL "$sampler_pid" 2>/dev/null || true; fi
    fi
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=
}
check_components() {
    local role pid
    for role in master donor; do
        if [[ $role == master ]]; then pid=$master_pid; else pid=$donor_pid; fi
        if [[ -n $pid && ! -e $case_dir/$role.unexpected-exit ]] && ! pid_running "$pid"; then
            event "unexpected_component_exit case=$case_name role=$role pid=$pid"
            printf 'Exited before intentional shutdown; see %s.exit.\n' "$role" > "$case_dir/$role.unexpected-exit"
            overall_rc=1
        fi
    done
}
finish() {
    local rc=$?
    (( finalized == 0 )) || return
    finalized=1
    trap - EXIT
    trap '' INT TERM USR1
    set +e
    event "finalize exit=$rc"
    # Drain readers before withdrawing donated memory; touch only owned steps.
    stop_step "$client_pid" "$case_dir/client.step" "$case_dir/client.exit" || rc=1
    stop_step "$aux_pid" "$aux_step" "$case_dir/aux.exit" || rc=1
    stop_step "$donor_pid" "$case_dir/donor.step" "$case_dir/donor.exit" || rc=1
    stop_step "$master_pid" "$case_dir/master.step" "$case_dir/master.exit" || rc=1
    stop_sampler
    timeout 8s scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-final.txt" 2>&1
    timeout 8s sacct -j "$SLURM_JOB_ID" --parsable2 --format=JobID,JobName,NodeList,State,ExitCode,Start,End,MaxRSS > "$LOG_DIR/sacct.txt" 2>&1
    if ! timeout 20s python3 "$RUN_DIR/diagnose_mooncake.py" report --run-dir "$RUN_DIR"; then
        printf '\nReport generation failed; inspect controller.log, events.log, per-case files and sacct.txt.\n' >> "$LOG_DIR/perf-analysis.md"
    fi
    (( rc != 0 || overall_rc == 0 )) || rc=1
    event "artifacts=$RUN_DIR report=$LOG_DIR/perf-analysis.md exit=$rc"
    exit "$rc"
}
# Install cleanup before copying artifacts, validating nodes or starting steps.
trap finish EXIT
trap 'event interrupted; exit 130' INT
trap 'event terminated; exit 143' TERM
trap 'event walltime_preservation; exit 124' USR1
mkdir -p "$LOG_DIR"
cat > "$LOG_DIR/perf-analysis.md" <<'REPORT'
# Mooncake host-memory/RDMA diagnostic: pending or incomplete

Goal: compare cn05 and cn02 with 512-GiB and 1100-GiB registered host memory.
TTFT p50/p95/p99, ITL p50/p99, token throughput and agentic Pareto/SLO metrics
are N/A: no model or inference trajectories. Get/put latency, GiB/s, errors,
registration and memory/network health come from the saved case artifacts.
No performance improvement is established; no DSight report is built.
The slurm-job-sizing skill sizes this diagnostic to 22 minutes plus 3 minutes
for cleanup and preservation; inspect controller.log and events.log for status.
REPORT
exec > >(tee -a "$LOG_DIR/controller.log") 2>&1
cp "$HELPER" "$RUN_DIR/diagnose_mooncake.py"
cp "${BASH_SOURCE[0]}" "$RUN_DIR/diagnose_mooncake.sbatch"
export PYTHONUNBUFFERED=1 MC_ENABLE_PARALLEL_REG_MR=0 MC_MS_AUTO_DISC=0
export MC_TE_FILTERS=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11
export MC_ENABLE_DEST_DEVICE_AFFINITY=1 GLOG_logtostderr=1 GLOG_v=0
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_NODE=${ALL_NODES[0]}
READER_NODES=()
for node in "${ALL_NODES[@]}"; do
    if [[ $node != vr-r42-cn05 && $node != vr-r42-cn02 ]]; then READER_NODES+=("$node"); fi
done
[[ " ${ALL_NODES[*]} " == *" vr-r42-cn05 "* && " ${ALL_NODES[*]} " == *" vr-r42-cn02 "* ]]
(( ${#READER_NODES[@]} > 0 ))
printf -v READER_LIST '%s,' "${READER_NODES[@]}"
READER_LIST=${READER_LIST%,}
event "begin nodes=$SLURM_JOB_NODELIST account=root partition=batch-r42 image=$IMAGE"
scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/slurm-job-initial.txt"
for donor_node in vr-r42-cn05 vr-r42-cn02; do
    for gib in 512 1100; do
        mkdir -p "$LOG_DIR/$donor_node-${gib}GiB"
        printf 'Not started; see events.log for early failure or budget exclusion.\n' > "$LOG_DIR/$donor_node-${gib}GiB/incomplete.txt"
    done
done
inventory_snapshot() {
    local rel=$1 rc=0
    aux_step=$RUN_DIR/$rel/inventory.step
    run_step "$SLURM_JOB_NODELIST" "${#ALL_NODES[@]}" --time=00:02:00 \
        --output="$RUN_DIR/$rel/inventory-%N.out" --error="$RUN_DIR/$rel/inventory-%N.err" \
        bash -c "$STEP_WRAPPER" _ "$rel/inventory.step" bash -c \
        'exec python3 /diag-run/diagnose_mooncake.py inventory --output "/diag-run/$1/inventory-${SLURMD_NODENAME}.json"' _ "$rel" &
    aux_pid=$!
    wait "$aux_pid" || rc=$?
    if ! stop_step "$aux_pid" "$aux_step" "$RUN_DIR/$rel/inventory.exit"; then exit 1; fi
    aux_pid= aux_step=
    return "$rc"
}
inventory_snapshot logs
python3 - "$RUN_DIR/metadata.json" "$IMAGE" "$READER_LIST" "$CASE_SECONDS" "$BATCH_SIZE" "$CLIENT_WORKERS" <<'PY'
import json, os, sys
path, image, readers, seconds, batch, workers = sys.argv[1:]
with open(path, "w") as handle:
    json.dump({"job_id": os.environ["SLURM_JOB_ID"], "image": image,
               "nodes": os.environ["SLURM_JOB_NODELIST"], "readers": readers,
               "seconds": int(seconds), "batch_size": int(batch), "workers": int(workers),
               "page_bytes": 2006016, "walltime": "00:25:00", "body_budget_seconds": 1320,
               "sizing_skill": "src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md"}, handle, indent=2)
PY

for donor_node in vr-r42-cn05 vr-r42-cn02; do
    for gib in 512 1100; do
        case_name=${donor_node}-${gib}GiB
        case_rel=logs/$case_name
        case_dir=$RUN_DIR/$case_rel
        if (( BODY_DEADLINE - SECONDS < CASE_BUDGET )); then
            printf 'Skipped: remaining body budget below %s seconds.\n' "$CASE_BUDGET" > "$case_dir/incomplete.txt"
            event "case_skipped name=$case_name reason=remaining_budget"
            overall_rc=1
            continue
        fi
        rm "$case_dir/incomplete.txt"
        event "case_begin name=$case_name donor=$donor_node gib=$gib"
        # Select fresh OS ports for each case; strict teardown prevents stale
        # masters satisfying readiness. Sockets close immediately before startup.
        aux_step=$case_dir/bootstrap.step
        run_step "$MASTER_NODE" 1 --time=00:02:00 --output="$case_dir/bootstrap.json" --error="$case_dir/bootstrap.err" \
            bash -c "$STEP_WRAPPER" _ "$case_rel/bootstrap.step" python3 -c '
import json, socket
sockets = [socket.socket(), socket.socket()]
for sock in sockets: sock.bind(("", 0))
print(json.dumps({"ip": socket.gethostbyname(socket.gethostname()), "rpc_port": sockets[0].getsockname()[1], "metrics_port": sockets[1].getsockname()[1]}), flush=True)
' &
        aux_pid=$!
        bootstrap_rc=0
        wait "$aux_pid" || bootstrap_rc=$?
        stop_step "$aux_pid" "$aux_step" "$case_dir/bootstrap.exit" || exit 1
        aux_pid= aux_step=
        if (( bootstrap_rc != 0 )); then
            printf 'Port selection failed.\n' > "$case_dir/incomplete.txt"
            overall_rc=1
            continue
        fi
        read -r MASTER_IP MASTER_PORT METRICS_PORT < <(python3 - "$case_dir/bootstrap.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(data["ip"], data["rpc_port"], data["metrics_port"])
PY
)
        MASTER_ADDR=$MASTER_IP:$MASTER_PORT
        run_step "$MASTER_NODE" 1 --time=00:08:00 --output="$case_dir/master.out" --error="$case_dir/master.err" \
            bash -c "$STEP_WRAPPER" _ "$case_rel/master.step" mooncake_master --rpc_port="$MASTER_PORT" \
            --metrics_port="$METRICS_PORT" --default_kv_lease_ttl=60000 --eviction_ratio=0.02 --allocation_strategy=random &
        master_pid=$!
        master_ready=0
        ready_deadline=$((SECONDS + 45))
        while (( SECONDS < ready_deadline )); do
            pid_running "$master_pid" || break
            if [[ -s $case_dir/master.step ]] && python3 - "$MASTER_IP" "$MASTER_PORT" "$METRICS_PORT" <<'PY'
import socket, sys, urllib.request
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5): pass
    with urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[3]}/metrics", timeout=0.5) as response: response.read()
except OSError: raise SystemExit(1)
PY
            then
                sleep 1
                if pid_running "$master_pid"; then master_ready=1; fi
                break
            fi
            sleep 1
        done
        ready=0
        if (( master_ready == 1 )); then
            run_step "$donor_node" 1 --time=00:08:00 --output="$case_dir/donor.out" --error="$case_dir/donor.err" \
                bash -c "$STEP_WRAPPER" _ "$case_rel/donor.step" python3 /diag-run/diagnose_mooncake.py donor \
                --master "$MASTER_ADDR" --gib "$gib" --ready "/diag-run/$case_rel/donor.ready.json" \
                --native-log "/diag-run/$case_rel/donor.err" --output "/diag-run/$case_rel/donor-health.jsonl" &
            donor_pid=$!
            for ((attempt=0; attempt<120; attempt++)); do
                check_components
                if [[ -s $case_dir/donor.ready.json ]] && pid_running "$donor_pid"; then ready=1; break; fi
                pid_running "$donor_pid" && pid_running "$master_pid" || break
                sleep 1
            done
        fi
        if (( ready == 1 )); then
            python3 - "$case_dir" "$MASTER_IP" "$METRICS_PORT" "$SLURM_JOB_ID" <<'PY' &
import datetime, json, pathlib, socket, subprocess, sys, time, urllib.request
root = pathlib.Path(sys.argv[1])
endpoint = json.loads((root / "donor.ready.json").read_text()).get("handshake_endpoint")
with (root / "external-health.jsonl").open("a", buffering=1) as output, (root / "master-metrics.jsonl").open("a", buffering=1) as metrics:
    while True:
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record = {"time": stamp, "handshake_endpoint": endpoint}
        if endpoint:
            host, port = endpoint.rsplit(":", 1)
            try:
                with socket.create_connection((host, int(port)), timeout=1): record["tcp_ok"] = True
            except OSError as exc: record.update(tcp_ok=False, tcp_error=str(exc))
        try:
            with urllib.request.urlopen(f"http://{sys.argv[2]}:{sys.argv[3]}/metrics", timeout=1) as response:
                metrics.write(json.dumps({"time": stamp, "prometheus": response.read().decode()}) + "\n")
        except OSError as exc: record["metrics_error"] = str(exc)
        try:
            record["steps"] = subprocess.run(["squeue", "--steps", "-j", sys.argv[4], "--noheader"], capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.TimeoutExpired) as exc: record["slurm_error"] = str(exc)
        output.write(json.dumps(record) + "\n")
        time.sleep(5)
PY
            sampler_pid=$!
            event "traffic_begin case=$case_name readers=$READER_LIST"
            run_step "$READER_LIST" "${#READER_NODES[@]}" --time=00:05:00 \
                --output="$case_dir/client-%N.out" --error="$case_dir/client-%N.err" \
                bash -c "$STEP_WRAPPER" _ "$case_rel/client.step" bash -c \
                'exec timeout --signal=TERM --kill-after=10s "$7" python3 /diag-run/diagnose_mooncake.py client --master "$2" --case "$3" --seconds "$4" --batch-size "$5" --workers "$6" --donor-ready "/diag-run/$1/donor.ready.json" --output "/diag-run/$1/client-${SLURMD_NODENAME}.jsonl"' \
                _ "$case_rel" "$MASTER_ADDR" "$case_name" "$CASE_SECONDS" "$BATCH_SIZE" "$CLIENT_WORKERS" "$((CASE_SECONDS + 90))" &
            client_pid=$!
            while pid_running "$client_pid"; do check_components; sleep 1; done
            client_rc=0
            wait "$client_pid" || client_rc=$?
            stop_step "$client_pid" "$case_dir/client.step" "$case_dir/client.exit" || exit 1
            client_pid=
            printf '%s\n' "$client_rc" > "$case_dir/client.exit"
            (( client_rc == 0 )) || overall_rc=1
            event "traffic_end case=$case_name exit=$client_rc"
            stop_sampler
        else
            event "startup_failed case=$case_name master_ready=$master_ready donor_ready=$ready"
            printf 'Startup failed: master_ready=%s donor_ready=%s.\n' "$master_ready" "$ready" > "$case_dir/incomplete.txt"
            overall_rc=1
        fi
        check_components
        stop_step "$donor_pid" "$case_dir/donor.step" "$case_dir/donor.exit" || exit 1
        donor_pid=
        stop_step "$master_pid" "$case_dir/master.step" "$case_dir/master.exit" || exit 1
        master_pid=
        inventory_snapshot "$case_rel" || overall_rc=1
        event "case_end name=$case_name"
    done
done
exit "$overall_rc"
