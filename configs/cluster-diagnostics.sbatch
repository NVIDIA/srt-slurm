#!/usr/bin/env bash
#SBATCH --job-name=cluster-diagnostics
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=cluster-diagnostics-%j.out

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit from shared storage; account/partition/node constraints are sbatch options.
# sbatch --account=<account> --partition=<partition> configs/cluster-diagnostics.sbatch raplab
# sbatch --account=<account> --partition=<partition> configs/cluster-diagnostics.sbatch hecate
# --nodes=8 matches a 32-rank decode endpoint with RANKS_PER_NODE=4.
# See docs/cluster-performance-diagnostics.md for environment overrides.
set -euo pipefail
export PATH="$PATH:/usr/sbin:/sbin"

# Internal worker entrypoint uses staged helpers, independent of repository mounts.
if [[ ${1:-} == --collect ]]; then
    scope=$2
    bash "$LOG_DIR/hardware.sh" --collect "$scope"
    if [[ $scope == container ]]; then
        python3 "$LOG_DIR/cluster-perf-diagnostics.py" collect \
            --duration "$SAMPLE_DURATION" --interval 1 \
            --output "$LOG_DIR/$(hostname -s)/container/performance.json"
    fi
    exit
fi

case ${1:-} in
    raplab) default_image=/home/rihuo/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh ;;
    hecate) default_image=/lustre/share/coreai_comparch_inferencex/rihuo/trtllm-release-f1bdd25-user_imant_glm5-rubin-30428-arm64.sqsh ;;
    *) echo 'Usage: sbatch [allocation options] configs/cluster-diagnostics.sbatch raplab|hecate' >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Submit using sbatch}"
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-$default_image}
export LOG_DIR=${LOG_DIR:-${SLURM_SUBMIT_DIR}/cluster-diagnostics-${SLURM_JOB_ID}}
export RUN_COPY_BENCH=${RUN_COPY_BENCH:-1} COPY_MIB=${COPY_MIB:-64} COPY_ITERATIONS=${COPY_ITERATIONS:-5}
export INSTALL_TOOLS=${INSTALL_TOOLS:-0} ENROOT_REMAP_ROOT=yes
export SAMPLE_DURATION=${SAMPLE_DURATION:-10}
RUN_COMM_BENCH=${RUN_COMM_BENCH:-1}
RUN_NCCL=${RUN_NCCL:-0}
RANKS_PER_NODE=${RANKS_PER_NODE:-4}
MPI_PLUGIN=${MPI_PLUGIN:-pmix}
COMM_ITERATIONS=${COMM_ITERATIONS:-200}
COMM_WARMUP=${COMM_WARMUP:-20}
[[ $RANKS_PER_NODE =~ ^[1-9][0-9]*$ ]] || { echo 'RANKS_PER_NODE must be positive' >&2; exit 2; }
[[ -f $CONTAINER_IMAGE ]] || { echo "Missing image: $CONTAINER_IMAGE" >&2; exit 1; }
nodes=()
while IFS= read -r node; do nodes+=("$node"); done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} -gt 0 ]] || { echo 'No allocated nodes' >&2; exit 1; }
: "${SLURM_JOB_CPUS_PER_NODE:?Need allocated CPU counts}"
step_cpus=0
IFS=',' read -r -a cpu_groups <<< "$SLURM_JOB_CPUS_PER_NODE"
for cpu_group in "${cpu_groups[@]}"; do
    count=${cpu_group%%(*}
    [[ $count =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CPU count: $cpu_group" >&2; exit 2; }
    if [[ $step_cpus == 0 || $count -lt $step_cpus ]]; then step_cpus=$count; fi
done
rank_cpus=$((step_cpus / RANKS_PER_NODE))
if [[ $RUN_COMM_BENCH == 1 && $rank_cpus -lt 1 ]]; then
    echo 'Not enough allocated CPUs for RANKS_PER_NODE' >&2; exit 2
fi
# sbatch executes a spool copy; helper paths must come from shared submit storage.
source_dir=${DIAG_SOURCE_DIR:-${SLURM_SUBMIT_DIR}/configs}
mkdir -p "$LOG_DIR"
cp "${BASH_SOURCE[0]}" "$LOG_DIR/batch-runner.sh"
cp "$source_dir/raplab-cluster-diagnostics.sbatch" "$LOG_DIR/hardware.sh"
# The existing copy collector expects this filename.
cp "$LOG_DIR/hardware.sh" "$LOG_DIR/runner.sh"
cp "$source_dir/cluster-perf-diagnostics.py" "$source_dir/numa_cpu_bind.sh" \
    "$source_dir/numa_net_devices.py" "$LOG_DIR/"
scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/allocation.txt"
scontrol show node "$SLURM_JOB_NODELIST" > "$LOG_DIR/slurm-nodes.txt" 2>&1 || true
{
    printf 'cluster=%s\nimage=%s\nnodes=%s\nallocated_cpus=%s\n' "$1" "$CONTAINER_IMAGE" "${nodes[*]}" "$SLURM_JOB_CPUS_PER_NODE"
    printf 'ranks_per_node=%s\ncpus_per_rank=%s\nmpi_plugin=%s\n' "$RANKS_PER_NODE" "$rank_cpus" "$MPI_PLUGIN"
    printf 'copy=%s\ncommunication=%s\nnccl=%s\ninstall_tools=%s\n' "$RUN_COPY_BENCH" "$RUN_COMM_BENCH" "$RUN_NCCL" "$INSTALL_TOOLS"
    stat "$CONTAINER_IMAGE"
} > "$LOG_DIR/settings.txt"
common=(--exact --nodes="${#nodes[@]}" --cpu-bind=none --kill-on-bad-exit=1)
container=(--container-image="$CONTAINER_IMAGE" --no-container-mount-home
    --container-writable --container-remap-root
    --container-mounts="$LOG_DIR:$LOG_DIR,/dev/infiniband:/dev/infiniband")
single=(--mpi=none --ntasks="${#nodes[@]}" --ntasks-per-node=1 --cpus-per-task="$step_cpus")
failed=0
run_phase() {
    local name=$1 rc=0
    shift
    "$@" > "$LOG_DIR/$name.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "$LOG_DIR/$name.exit-status"
    [[ $rc == 0 ]] || failed=1
}
run_phase host-collection srun "${common[@]}" "${single[@]}" bash "$LOG_DIR/batch-runner.sh" --collect host
run_phase container-collection srun "${common[@]}" "${single[@]}" "${container[@]}" \
    bash "$LOG_DIR/batch-runner.sh" --collect container
if [[ $RUN_COMM_BENCH == 1 ]]; then
    comm_args=(communication --iterations "$COMM_ITERATIONS" --warmup "$COMM_WARMUP" --output "$LOG_DIR/communication.json")
    if [[ $RUN_NCCL == 1 ]]; then
        comm_args+=(--nccl)
        [[ -z ${MASTER_ADDR:-} ]] || comm_args+=(--master-addr "$MASTER_ADDR")
        [[ -z ${MASTER_PORT:-} ]] || comm_args+=(--master-port "$MASTER_PORT")
    fi
    # All node GPUs must remain visible; the NUMA wrapper maps local rank to GPU.
    # GPU-local CPU and memory binding matches the current decode recipe.
    run_phase communication srun "${common[@]}" --mpi="$MPI_PLUGIN" \
        --ntasks="$((${#nodes[@]} * RANKS_PER_NODE))" --ntasks-per-node="$RANKS_PER_NODE" \
        --cpus-per-task="$rank_cpus" "${container[@]}" \
        bash "$LOG_DIR/numa_cpu_bind.sh" --bind-memory \
        python3 "$LOG_DIR/cluster-perf-diagnostics.py" "${comm_args[@]}"
fi
cat > "$LOG_DIR/README.txt" <<'NOTES'
Host/container hardware checks and sequential copies reuse the existing collector.
performance.json describes an idle collector, not a serving worker. For serving
placement, run the Python collect command with actual worker PIDs inside their container.
communication.json records rank environments/placement and MPI timings, optionally NCCL.
Transport settings are inherited; the NUMA wrapper filters UCX_NET_DEVICES by GPU NUMA.
No serving processes are launched. Missing commands/counters are unknown, not zero errors.
Check all phase exit-status files, per-node COMPLETE markers, copy exit-status/stderr,
and container-setup.txt. Package installation is disabled unless INSTALL_TOOLS=1.
See docs/cluster-performance-diagnostics.md for comparisons and interpretation limits.
NOTES
archive="${LOG_DIR}.tar.gz"
tar -czf "$archive" -C "$(dirname "$LOG_DIR")" "$(basename "$LOG_DIR")"
printf 'Logs: %s\nBundle: %s\n' "$LOG_DIR" "$archive"
exit "$failed"
