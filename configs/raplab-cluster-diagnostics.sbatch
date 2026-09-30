#!/usr/bin/env bash
#SBATCH --job-name=cluster-diagnostics
#SBATCH --account=root
#SBATCH --partition=batch-r42
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --segment=2
#SBATCH --time=00:20:00
#SBATCH --output=cluster-diagnostics-%j.out

# Run from shared storage: sbatch configs/raplab-cluster-diagnostics.sbatch
# For another cluster, override --account, --partition and CONTAINER_IMAGE.
# Optional: --nodelist=<node1>,<node2>; --export=ALL,RUN_COPY_BENCH=0
# Missing diagnostic tools are installed only inside a writable container.
# Set INSTALL_TOOLS=0 to use the image unchanged; failures remain in the bundle.
# No host changes. Copy tests temporarily use 128 MiB
# of pinned host memory and 64 MiB of GPU memory per GPU, sequentially.
# This collects idle-node evidence; it does not reproduce Mooncake serving load.
set -euo pipefail
export PATH="$PATH:/usr/sbin:/sbin"

capture() {
    local name=$1 rc=0
    shift
    {
        printf 'command:'; printf ' %q' "$@"; printf '\n'
        if ! command -v "$1" >/dev/null; then
            printf 'UNAVAILABLE: %s is not installed or not on PATH\n' "$1"
            rc=127
        elif command -v timeout >/dev/null; then
            timeout 45 "$@" || rc=$?
        else
            printf 'NOTE: timeout unavailable; running this read-only check directly\n'
            "$@" || rc=$?
        fi
        printf '\nexit_status=%s\n' "$rc"
    } > "$out/$name.txt" 2>&1
}

prepare_container() {
    local tool package rc=0
    local -a packages=()
    python3 -m pip list --format=json > "$out/packages-before-setup.json" 2>&1 || true
    command -v dpkg-query >/dev/null && dpkg-query -W > "$out/os-packages-before-setup.txt" 2>&1 || true
    for mapping in timeout:coreutils lscpu:util-linux numactl:numactl numastat:numactl \
        lspci:pciutils rdma:iproute2 ibv_devinfo:ibverbs-utils ibstat:infiniband-diags \
        ps:procps lsmod:kmod; do
        tool=${mapping%%:*}; package=${mapping#*:}
        command -v "$tool" >/dev/null || packages+=("$package")
    done
    {
        printf 'install_tools=%s uid=%s\n' "${INSTALL_TOOLS:-1}" "$(id -u)"
        if [[ ${#packages[@]} == 0 ]]; then
            echo 'All required diagnostic tools are available.'
        elif [[ ${INSTALL_TOOLS:-1} != 1 ]]; then
            printf 'Installation disabled; missing-tool packages: %s\n' "${packages[*]}"
        elif [[ $(id -u) != 0 ]] || ! command -v apt-get >/dev/null; then
            echo 'Cannot install tools: need remapped container root and apt-get; continuing with available tools.'
        else
            # No upgrades, kernel modules, drivers or Python packages installed.
            # Record both package lists because adding dependencies can still
            # change the user-space environment used by diagnostic commands.
            export DEBIAN_FRONTEND=noninteractive
            if command -v timeout >/dev/null; then
                timeout 180 apt-get update || rc=$?
            else
                apt-get -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 \
                    -o Acquire::Retries=0 update || rc=$?
            fi
            if [[ $rc == 0 ]]; then
                if command -v timeout >/dev/null; then
                    timeout 300 apt-get install -y --no-install-recommends --no-upgrade "${packages[@]}" || rc=$?
                else
                    apt-get -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 \
                        -o Acquire::Retries=0 install -y --no-install-recommends --no-upgrade \
                        "${packages[@]}" || rc=$?
                fi
            fi
            echo "package_setup_exit_status=$rc; collection continues even if setup failed."
        fi
        for tool in timeout lscpu numactl lspci rdma ibv_devinfo ibstat ibdev2netdev \
            nvidia-smi ucx_info ompi_info; do
            command -v "$tool" || echo "UNAVAILABLE: $tool"
        done
    } > "$out/container-setup.txt" 2>&1
    command -v dpkg-query >/dev/null && dpkg-query -W > "$out/os-packages-after-setup.txt" 2>&1 || true
}

read_files() {
    local file
    for file in "$@"; do
        [[ -f $file ]] || continue
        printf '\n%s\n' "$file"
        cat "$file" 2>&1 || true
    done
}

nic_snapshot() {
    local target=$1 device port
    {
        date -u +%FT%TZ
        [[ -d /sys/class/infiniband ]] || echo 'UNAVAILABLE: /sys/class/infiniband; consult host/nic-*.txt'
        for device in /sys/class/infiniband/*; do
            [[ -d $device ]] || continue
            printf '\nDEVICE %s PCI %s\n' "${device##*/}" "$(readlink -f "$device/device")"
            read_files "$device"/{fw_ver,node_guid,sys_image_guid} \
                "$device/device"/{numa_node,local_cpulist,current_link_speed,current_link_width,max_link_speed,max_link_width} \
                "$device/device"/net/*/{address,mtu,speed}
            for port in "$device"/ports/*; do
                read_files "$port"/{state,phys_state,rate,link_layer,lid,sm_lid,active_mtu} \
                    "$port"/pkeys/* "$port"/gids/* "$port"/gid_attrs/types/* "$port"/gid_attrs/ndevs/* \
                    "$port"/counters/* "$port"/hw_counters/*
            done
        done
    } > "$out/$target.txt" 2>&1
}

copy_bench() {
    python3 - "$@" <<'PY'
import json
import os
import statistics
import sys
import time

import torch

policy = sys.argv[1]
size = int(os.environ.get("COPY_MIB", "64")) * 1024**2
iterations = int(os.environ.get("COPY_ITERATIONS", "5"))
if size <= 0 or iterations <= 0:
    raise ValueError("COPY_MIB and COPY_ITERATIONS must be positive")

print(json.dumps({"type": "metadata", "policy": policy, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "gpus": torch.cuda.device_count()}), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; copy benchmark cannot run")

for gpu in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(gpu)
        source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        destination = torch.empty_like(source, pin_memory=True)
        source.fill_(1)
        destination.zero_()
        device = torch.empty(size, dtype=torch.uint8, device=f"cuda:{gpu}")
        device.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        # Capture actual allocation placement, rather than inferring it from -m.
        with open("/proc/self/numa_maps") as handle:
            mappings = list(handle)
        with open("/proc/self/maps") as handle:
            regions = list(handle)
        placement = []
        for region in regions:
            bounds = region.split()[0]
            low, high = (int(x, 16) for x in bounds.split("-"))
            if any(low <= tensor.data_ptr() < high for tensor in (source, destination)):
                placement.extend(line.strip() for line in mappings if line.startswith(f"{low:x} "))
        print(json.dumps({"type": "placement", "gpu": gpu, "policy": policy,
                          "numa_maps": placement}), flush=True)
        for direction in ("H2D", "D2H"):
            src, dst = (source, device) if direction == "H2D" else (device, destination)
            for chunk in (size, 1024**2, 64 * 1024):
                # Prebuild views so timings measure submissions/copies, not slicing.
                views = [(dst[start:start + chunk], src[start:start + chunk])
                         for start in range(0, size, chunk)]
                samples = []
                for repeat in range(iterations + 2):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    start_event.record()
                    for target, origin in views:
                        target.copy_(origin, non_blocking=True)
                    end_event.record()
                    submitted = time.perf_counter()
                    end_event.synchronize()
                    finished = time.perf_counter()
                    if repeat >= 2:
                        samples.append({"submit_ms": 1000 * (submitted - started),
                                        "sync_wait_ms": 1000 * (finished - submitted),
                                        "wall_ms": 1000 * (finished - started),
                                        "cuda_ms": start_event.elapsed_time(end_event)})
                medians = {key: statistics.median(sample[key] for sample in samples) for key in samples[0]}
                print(json.dumps({"type": "copy", "gpu": gpu,
                                  "gpu_name": torch.cuda.get_device_name(gpu), "policy": policy,
                                  "direction": direction, "bytes": size, "chunk_bytes": chunk,
                                  "copies": len(views), "iterations": iterations,
                                  "wall_GiB_s": size / 1024**3 / (medians["wall_ms"] / 1000),
                                  **medians, "samples": samples}), flush=True)
        del views, src, dst, target, origin, source, destination, device
        torch.cuda.empty_cache()
    except Exception as exc:
        print(json.dumps({"type": "error", "gpu": gpu, "policy": policy,
                          "error": f"{type(exc).__name__}: {exc}"}), flush=True)
PY
}

collect() {
    local scope=$1 node pid task policy rc
    node=$(hostname -s)
    out="$LOG_DIR/$node/$scope"
    mkdir -p "$out"
    [[ $scope != container ]] || prepare_container
    capture identity bash -c 'date -u; hostname; uname -a; cat /etc/os-release; id; ulimit -a'
    {
        # cgroup v2 can expose a namespace root or the host-relative job path.
        local group_path='' controller hierarchy relative base
        while IFS=: read -r hierarchy controller relative; do
            [[ $hierarchy != 0 ]] || group_path=$relative
        done < /proc/self/cgroup 2>/dev/null || true
        for base in /sys/fs/cgroup "/sys/fs/cgroup$group_path"; do
            read_files "$base"/{cpu.max,cpu.stat,memory.max,memory.high,memory.current,memory.events,cpuset.cpus.effective,cpuset.mems.effective}
        done
    } > "$out/cgroup-limits.txt" 2>&1
    {
        read_files /proc/{cpuinfo,meminfo,vmstat,buddyinfo,zoneinfo,loadavg,modules,self/status,self/cgroup} \
            /proc/pressure/* /sys/devices/system/node/node*/meminfo \
            /sys/kernel/mm/transparent_hugepage/{enabled,defrag} \
            /sys/devices/system/cpu/cpu*/cpufreq/{scaling_governor,scaling_cur_freq,scaling_min_freq,scaling_max_freq} \
            /sys/class/net/*/{mtu,speed,duplex} /proc/sys/kernel/random/boot_id \
            /proc/sys/kernel/numa_balancing /proc/sys/vm/{zone_reclaim_mode,swappiness,nr_hugepages} \
            /sys/module/{nvidia,mlx5_core}/version /proc/driver/nvidia/version
    } > "$out/proc-sysfs.txt" 2>&1
    nic_snapshot nic-before
    if [[ $scope == host ]]; then
        capture versions bash -c 'enroot version; srun --version'
        # Host-only permission check. Remapped container root cannot grant
        # permission to read the host kernel ring buffer.
        capture kernel-messages dmesg --ctime --level=err,warn
        {
            for device in /sys/bus/pci/devices/*; do
                [[ -r $device/vendor ]] || continue
                case $(cat "$device/vendor") in
                    0x10de|0x15b3)
                        read_files "$device"/{vendor,device,class,numa_node,local_cpulist,current_link_speed,current_link_width,max_link_speed,max_link_width} ;;
                esac
            done
        } > "$out/pci-sysfs.txt" 2>&1
        # Only process/thread placement, not environments or command-line secrets.
        for pid in /proc/[0-9]*; do
            [[ -r $pid/comm ]] || continue
            case $(cat "$pid/comm" 2>/dev/null) in
                *python*|*trtllm*|*mooncake*|*dynamo*)
                    {
                        read_files "$pid"/{comm,status,limits,cgroup,numa_maps,schedstat}
                        for task in "$pid"/task/*; do
                            read_files "$task"/{comm,status,schedstat}
                        done
                    } > "$out/process-${pid##*/}.txt" 2>&1 ;;
            esac
        done
    else
        capture cpu lscpu
        capture numa numactl --hardware
        capture numa-policy numactl --show
        capture gpu nvidia-smi -q
        capture gpu-topology nvidia-smi topo -m
        capture nvlink nvidia-smi nvlink -s
        capture pci lspci -Dvv
        capture rdma-devices ibv_devinfo -v
        capture rdma-visibility bash -c 'ls -ld /sys/class/infiniband /dev/infiniband; ls -l /dev/infiniband'
        capture rdma-links rdma link show
        capture ib-network ibstat
        capture ib-to-netdev ibdev2netdev
        capture net ip -details link show
        capture routes ip route show
        capture modules lsmod
        capture versions bash -c 'modinfo mlx5_core; modinfo nvidia; ofed_info -s'
        capture memory free -h
        capture numa-memory numastat -m
        capture vm-sample vmstat 1 6
        capture processes ps -eLo pid,tid,psr,pcpu,pmem,nlwp,comm --sort=-pcpu
        capture ucx-devices ucx_info -d
        capture ucx-config ucx_info -f
        capture mpi-version ompi_info --version
        capture python-packages python3 -m pip list --format=json
        capture image-code python3 -c 'import hashlib, importlib.util, pathlib
spec = importlib.util.find_spec("tensorrt_llm")
root = pathlib.Path(next(iter(spec.submodule_search_locations)))
for relative in ("llmapi/utils.py", "_torch/pyexecutor/connectors/mooncake_store/worker.py", "_torch/pyexecutor/connectors/mooncake_store/staging.py"):
    path = root / relative
    print(path, hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING")'
        if [[ ${RUN_COPY_BENCH:-1} == 1 ]]; then
            for policy in unrestricted 0 1; do
                rc=0
                if [[ $policy == unrestricted ]]; then
                    timeout 240 bash "$LOG_DIR/runner.sh" --copy "$policy" \
                        > "$out/copy-$policy.jsonl" 2> "$out/copy-$policy.stderr" || rc=$?
                else
                    timeout 240 numactl --cpunodebind="$policy" --membind="$policy" \
                        bash "$LOG_DIR/runner.sh" --copy "$policy" \
                        > "$out/copy-$policy.jsonl" 2> "$out/copy-$policy.stderr" || rc=$?
                fi
                printf '%s\n' "$rc" > "$out/copy-$policy.exit-status"
            done
        fi
    fi
    nic_snapshot nic-after
    [[ $scope != container ]] || capture gpu-after nvidia-smi -q
    touch "$out/COMPLETE"
}

case ${1:-} in
    --collect) collect "$2"; exit ;;
    --copy) copy_bench "$2"; exit ;;
esac

: "${SLURM_JOB_ID:?Submit this script using sbatch}"
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/home/rihuo/trtllm-release-4dece7f-user_imant_glm5-rubin-arm64.sqsh}
export LOG_DIR=${LOG_DIR:-${SLURM_SUBMIT_DIR}/cluster-diagnostics-${SLURM_JOB_ID}}
export RUN_COPY_BENCH=${RUN_COPY_BENCH:-1} COPY_MIB=${COPY_MIB:-64} COPY_ITERATIONS=${COPY_ITERATIONS:-5}
export INSTALL_TOOLS=${INSTALL_TOOLS:-1} ENROOT_REMAP_ROOT=yes
nodes=()
while IFS= read -r node; do nodes+=("$node"); done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} -ge 1 && ${#nodes[@]} -le 2 ]] || {
    echo 'This diagnostic requires one or two nodes and is limited to two nodes' >&2; exit 1;
}
mkdir -p "$LOG_DIR"
cp "${BASH_SOURCE[0]}" "$LOG_DIR/runner.sh"
scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/allocation.txt"
scontrol show node "$SLURM_JOB_NODELIST" > "$LOG_DIR/slurm-nodes.txt" 2>&1 || true
# The exclusive allocation owns the whole nodes, but an exact eight-CPU step
# confines its tasks to NUMA 0 on these systems. Request the allocated CPU
# count per node so numactl can also select NUMA 1. For unequal nodes use the
# smallest allocation; the collected masks show what is actually available.
: "${SLURM_JOB_CPUS_PER_NODE:?Need allocated CPU counts for the diagnostic step}"
step_cpus=0
IFS=',' read -r -a cpu_groups <<< "$SLURM_JOB_CPUS_PER_NODE"
for cpu_group in "${cpu_groups[@]}"; do
    cpu_count=${cpu_group%%(*}
    [[ $cpu_count =~ ^[1-9][0-9]*$ ]] || { echo "Invalid allocated CPU count: $cpu_group" >&2; exit 1; }
    if [[ $step_cpus == 0 || $cpu_count -lt $step_cpus ]]; then step_cpus=$cpu_count; fi
done
{
    printf 'image=%s\nnodes=%s\ncopy_benchmark=%s\n' "$CONTAINER_IMAGE" "${nodes[*]}" "$RUN_COPY_BENCH"
    printf 'allocated_cpus_per_node=%s\nstep_cpus_per_task=%s\n' "$SLURM_JOB_CPUS_PER_NODE" "$step_cpus"
    stat "$CONTAINER_IMAGE" || true
    for variable in UCX_TLS UCX_NET_DEVICES MPI_UCX_TLS MPI_UCX_NET_DEVICES \
        MC_MS_AUTO_DISC MC_MS_FILTERS MC_TE_FILTERS MC_ENABLE_DEST_DEVICE_AFFINITY \
        MC_ENABLE_PARALLEL_REG_MR TLLM_NUMA_AWARE_WORKER_AFFINITY OMP_NUM_THREADS; do
        printf '%s=%s\n' "$variable" "${!variable-<unset>}"
    done
} > "$LOG_DIR/settings.txt"
cat > "$LOG_DIR/README.txt" <<'EOF'
Compare matching files for the two nodes and between clusters.
Start with gpu-topology, pci, numa, nic-before, gpu, versions, memory,
numa-policy and container/copy-*.jsonl. The copy test splits CPU submission
and synchronization wait, and reports actual pinned-buffer NUMA mappings.
NIC counters are cumulative; compare before/after deltas, not raw totals.
host/ contains host sysfs and counters even when container sysfs hides them.
Hardware utility commands run in container/. Missing diagnostic tools are
installed there with apt --no-upgrade, never on the host. Installation failures
are nonfatal; consult container-setup.txt. INSTALL_TOOLS=0 skips installation.
packages-before-setup.json describes the image's Python packages; OS packages
after setup are recorded separately. nvidia-smi, UCX and MPI use the image's
existing installation: the collector does not install replacement stacks.
Kernel messages may require host privileges. Container root remapping does not
grant those privileges. modinfo can fail when host module files are not mounted;
host/proc-sysfs.txt also records loaded driver versions directly.
Missing commands, permissions and NUMA-policy failures are recorded explicitly.
Package lists and source hashes help check image equivalence; image size/mtime
alone does not establish that the squashfs images are identical.
No serving processes are started. Idle measurements cannot exclude congestion
under serving load. Copy tests are sequential and do not reproduce Mooncake's
exact page geometry or CPU/GPU contention. Clocks are sampled, not changed.
settings.txt describes the diagnostic job environment, not a serving worker's
environment. Process affinity snapshots cover processes visible in this job.
Diagnostic steps request the CPUs already allocated to the exclusive job.
Check cgroup-limits.txt and copy metadata for access to both CPU NUMA nodes.
EOF
step_args=(--mpi=none --exact --nodes="${#nodes[@]}" --ntasks="${#nodes[@]}"
    --ntasks-per-node=1 --cpus-per-task="$step_cpus" --cpu-bind=none)
rc=0
srun "${step_args[@]}" bash "$LOG_DIR/runner.sh" --collect host \
    > "$LOG_DIR/host-collection.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$LOG_DIR/host-collection.exit-status"
rc=0
if [[ -f $CONTAINER_IMAGE ]]; then
    srun "${step_args[@]}" --container-image="$CONTAINER_IMAGE" --no-container-mount-home \
        --container-writable --container-remap-root \
        --container-mounts="$LOG_DIR:$LOG_DIR,/dev/infiniband:/dev/infiniband" \
        bash "$LOG_DIR/runner.sh" --collect container > "$LOG_DIR/container-collection.log" 2>&1 || rc=$?
else
    echo "Missing image: $CONTAINER_IMAGE" > "$LOG_DIR/container-collection.log"
    rc=1
fi
printf '%s\n' "$rc" > "$LOG_DIR/container-collection.exit-status"
archive="${LOG_DIR}.tar.gz"
tar -czf "$archive" -C "$(dirname "$LOG_DIR")" "$(basename "$LOG_DIR")"
echo "Logs: $LOG_DIR"
echo "Bundle: $archive"
echo 'Inspect collection exit-status files, COMPLETE markers, and copy stderr for unavailable checks.'
