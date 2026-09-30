#!/usr/bin/env bash
#SBATCH --job-name=mlx5-0-bandwidth
#SBATCH --account=root
#SBATCH --partition=batch-r42
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --segment=2
#SBATCH --time=00:30:00
#SBATCH --output=rdma-bandwidth-%j.out

# Submit from a directory shared by both nodes:
# sbatch configs/raplab-rdma-bandwidth.sbatch
# Optional: --nodelist=vr-r42-cn08,vr-r42-cn16 (scheduler must accept this with --segment=2).
# Uses the TRT-LLM image below through Pyxis; missing tools are installed
# inside a writable container once per node (no host package installation).
# Override via sbatch --export=ALL,REPEATS=3,DURATION=15,PERFTEST_DIR=/path/to/bin
# This measures host-memory RDMA, not GPU copies, Mooncake restore, or NVLink.
# WRITE payload flows client -> server; READ payload flows server -> client.
# NUMA_POLICY=local binds both CPU and test memory near the selected NIC.
# Use NUMA_POLICY=none for a separate unbound comparison.
set -euo pipefail

report_version() {
    # Some perftest releases print the version successfully but exit with 1.
    local output rc=0
    output=$("$1" --version 2>&1) || rc=$?
    printf '%s\n' "$output"
    [[ $rc -le 1 && $output == *"Version:"* ]] || {
        echo "Could not identify perftest version: $1 (exit $rc)" >&2
        return 1
    }
}

endpoint() {
    local role=$1 tool=$2 peer=$3 port=$4 size=$5 qps=$6 prefix=$7
    local root=/sys/class/infiniband/${RDMA_DEVICE}/ports/${RDMA_PORT}
    local numa binary rc
    binary=${PERFTEST_DIR:+${PERFTEST_DIR}/}${tool}
    command -v "$binary" >/dev/null || { echo "Missing $binary; install/load perftest on both nodes" >&2; return 1; }
    for prerequisite in numactl ss timeout; do command -v "$prerequisite" >/dev/null; done
    [[ -d $root && $(cat "$root/state") == *ACTIVE* ]] || { echo "$root is not ACTIVE" >&2; return 1; }
    numa=$(cat "/sys/class/infiniband/${RDMA_DEVICE}/device/numa_node")
    local -a bind=() args=(-d "$RDMA_DEVICE" -i "$RDMA_PORT" -p "$port" -s "$size" -q "$qps" -t 128 -D "$DURATION" --report_gbits -N)
    if [[ $NUMA_POLICY == local && $numa -ge 0 ]]; then
        bind=(numactl --cpunodebind="$numa" --membind="$numa")
    fi
    # For RoCE, select a verified common GID index explicitly via GID_INDEX.
    [[ -z ${GID_INDEX:-} ]] || args+=(-x "$GID_INDEX")
    {
        hostname
        report_version "$binary"
        echo "device=$RDMA_DEVICE port=$RDMA_PORT numa=$numa policy=$NUMA_POLICY memlock=$(ulimit -l)"
        for attr in state phys_state rate active_mtu link_layer; do
            [[ ! -f $root/$attr ]] || { echo "$attr:"; cat "$root/$attr"; }
        done
        for f in "$root"/gids/* "$root"/counters/{port_xmit_data,port_rcv_data,port_xmit_wait,port_rcv_errors,port_xmit_discards}; do
            [[ ! -f $f ]] || { echo "$f:"; cat "$f"; }
        done
        printf 'command:'; printf ' %q' "${bind[@]}" "$binary" "${args[@]}"; echo
    } > "$prefix.$role.info"
    if [[ $role == server ]]; then
        timeout "$TEST_TIMEOUT" "${bind[@]}" "$binary" "${args[@]}" > "$prefix.server.log" 2>&1 &
        local child=$!
        trap "kill $child 2>/dev/null || true" EXIT
        # Inspect the listening socket without connecting: a probe connection
        # would consume perftest's single client handshake.
        local ready=0
        for ((attempt=0; attempt<60; attempt++)); do
            kill -0 "$child" 2>/dev/null || break
            if ss -H -ltn "sport = :$port" | grep -q .; then ready=1; break; fi
            sleep 1
        done
        if (( !ready )); then cat "$prefix.server.log" >&2; return 1; fi
        touch "$prefix.ready"
        rc=0; wait "$child" || rc=$?
        trap - EXIT
    else
        rc=0
        timeout "$TEST_TIMEOUT" "${bind[@]}" "$binary" "${args[@]}" "$peer" > "$prefix.client.log" 2>&1 || rc=$?
    fi
    for f in "$root"/counters/{port_xmit_data,port_rcv_data,port_xmit_wait,port_rcv_errors,port_xmit_discards}; do
        [[ ! -f $f ]] || { echo "$f:"; cat "$f"; }
    done > "$prefix.$role.counters-after"
    return "$rc"
}

if [[ ${1:-} == --setup ]]; then
    if ! command -v ib_write_bw >/dev/null || ! command -v ib_read_bw >/dev/null || \
       ! command -v numactl >/dev/null || ! command -v ss >/dev/null || ! command -v timeout >/dev/null; then
        [[ $(id -u) == 0 ]] || { echo 'Container root remapping is required for apt-get' >&2; exit 1; }
        command -v apt-get >/dev/null || { echo 'Expected a Debian/Ubuntu container with apt-get' >&2; exit 1; }
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends --no-upgrade perftest numactl iproute2 coreutils
    fi
    hostname
    report_version ib_write_bw
    report_version ib_read_bw
    [[ -d /dev/infiniband && -d /sys/class/infiniband/${RDMA_DEVICE} ]]
    exit
fi
if [[ ${1:-} == --endpoint ]]; then shift; endpoint "$@"; exit; fi

: "${SLURM_JOB_ID:?Submit this file with sbatch}"
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/home/rihuo/trtllm-release-4dece7f-user_imant_glm5-rubin-arm64.sqsh}
export ENROOT_REMAP_ROOT=yes
export RDMA_DEVICE=${RDMA_DEVICE:-mlx5_0} RDMA_PORT=${RDMA_PORT:-1}
export DURATION=${DURATION:-10} REPEATS=${REPEATS:-2}
export NUMA_POLICY=${NUMA_POLICY:-local} PERFTEST_DIR=${PERFTEST_DIR:-} GID_INDEX=${GID_INDEX:-}
export TEST_TIMEOUT=$((DURATION + 90))
[[ $NUMA_POLICY == local || $NUMA_POLICY == none ]] || { echo 'NUMA_POLICY must be local or none' >&2; exit 1; }
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} == 2 ]] || { echo 'Expected exactly two nodes' >&2; exit 1; }
log_dir=${LOG_DIR:-${SLURM_SUBMIT_DIR}/rdma-bandwidth-${SLURM_JOB_ID}}
mkdir -p "$log_dir"
# Copy the batch script out of SLURM's node-local spool to shared storage.
runner="$log_dir/runner.sh"
cp "${BASH_SOURCE[0]}" "$runner"
scontrol show job "$SLURM_JOB_ID" > "$log_dir/allocation.txt"
echo "Nodes: ${nodes[*]}; endpoint: $RDMA_DEVICE/$RDMA_PORT; logs: $log_dir"
[[ -f $CONTAINER_IMAGE ]] || { echo "Missing image: $CONTAINER_IMAGE" >&2; exit 1; }
container_name="rdma-bandwidth-${SLURM_JOB_ID}"
container_args=(--container-image="$CONTAINER_IMAGE" --container-name="$container_name"
    --container-writable --container-remap-root --no-container-mount-home
    --container-mounts="$log_dir:$log_dir,/dev/infiniband:/dev/infiniband")
echo "Preparing image $CONTAINER_IMAGE on both nodes; container name: $container_name"
srun --mpi=none --exact -N2 -n2 --ntasks-per-node=1 -c8 "${container_args[@]}" \
    bash "$runner" --setup > "$log_dir/container-setup.log" 2>&1 || {
    cat "$log_dir/container-setup.log" >&2; exit 1;
}
# Named containers preserve the installed tools across the test steps.
# Pyxis retains these job-specific containers after the job for inspection.
server_step=''
trap '[[ -z $server_step ]] || kill "$server_step" 2>/dev/null || true' EXIT
case_id=0
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for tool in ib_write_bw ib_read_bw; do
        for qps in 1 8; do
            for size in 65536 4194304; do
                for direction in 0 1; do
                    server=${nodes[$direction]}; client=${nodes[$((1-direction))]}
                    prefix="$log_dir/r${repeat}-${tool}-q${qps}-s${size}-${client}-to-${server}"
                    port=$((18515 + case_id)); case_id=$((case_id + 1))
                    echo "Testing $tool client=$client server=$server size=$size QPs=$qps repeat=$repeat"
                    srun --mpi=none --exact -N1 -n1 -c8 -w "$server" "${container_args[@]}" bash "$runner" --endpoint server "$tool" '' "$port" "$size" "$qps" "$prefix" &
                    server_step=$!
                    ready=0
                    for ((attempt=0; attempt<90; attempt++)); do
                        [[ ! -f $prefix.ready ]] || { ready=1; break; }
                        kill -0 "$server_step" 2>/dev/null || break
                        sleep 1
                    done
                    ((ready)) || { echo "Server did not become ready; see $prefix.server.log" >&2; exit 1; }
                    srun --mpi=none --exact -N1 -n1 -c8 -w "$client" "${container_args[@]}" bash "$runner" --endpoint client "$tool" "$server" "$port" "$size" "$qps" "$prefix"
                    wait "$server_step"; server_step=''
                    cat "$prefix.client.log"
                    rm "$prefix.ready"
                done
            done
        done
    done
done
echo "Finished. Per-case client/server output, topology, and counters: $log_dir"
