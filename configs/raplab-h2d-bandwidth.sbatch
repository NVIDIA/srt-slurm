#!/usr/bin/env bash
#SBATCH --job-name=h2d-bandwidth
#SBATCH --account=root
#SBATCH --partition=batch-r42
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=h2d-bandwidth-%j.out
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit: sbatch configs/raplab-h2d-bandwidth.sbatch
# Overrides: --nodelist=<node>, --account=<account>, --partition=<partition>
# Environment: CONTAINER_IMAGE, LOG_DIR, COPY_MIB=64, COPY_ITERATIONS=5,
#              H2D_POLICIES="unrestricted memory0 memory1 bound0 bound1"
# Directly inside a GPU container: bash raplab-h2d-bandwidth.sbatch --copy manual
# Derived from raplab-cluster-diagnostics.sbatch; measures H2D only.
# GPUs run sequentially. This does not measure RDMA or serving contention.
set -euo pipefail

copy_bench() {
    python3 - "$1" <<'PY'
import json
import os
import socket
import statistics
import sys
import time

import torch

policy = sys.argv[1]
size = int(os.environ.get("COPY_MIB", "64")) * 1024**2
iterations = int(os.environ.get("COPY_ITERATIONS", "5"))
if size <= 0 or iterations <= 0:
    raise ValueError("COPY_MIB and COPY_ITERATIONS must be positive")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; run inside a GPU-enabled container")

print(json.dumps({"type": "metadata", "hostname": socket.gethostname(), "policy": policy,
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "cpu_affinity": sorted(os.sched_getaffinity(0)),
                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "gpus": torch.cuda.device_count()}), flush=True)

failed = False
for gpu in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(gpu)
        source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        source.fill_(1)
        device = torch.empty(size, dtype=torch.uint8, device=f"cuda:{gpu}")
        device.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        with open("/proc/self/numa_maps") as handle:
            mappings = list(handle)
        with open("/proc/self/maps") as handle:
            regions = list(handle)
        placement = []
        for region in regions:
            low, high = (int(x, 16) for x in region.split()[0].split("-"))
            if low <= source.data_ptr() < high:
                placement.extend(line.strip() for line in mappings if line.startswith(f"{low:x} "))
        print(json.dumps({"type": "placement", "gpu": gpu, "policy": policy,
                          "host_buffer": source.data_ptr(), "numa_maps": placement}), flush=True)
        for chunk in dict.fromkeys((size, min(size, 1024**2), min(size, 64 * 1024))):
            views = [(device[start:start + chunk], source[start:start + chunk])
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
            # CUDA events include idle gaps between submissions of small copies.
            print(json.dumps({"type": "copy", "gpu": gpu, "gpu_name": torch.cuda.get_device_name(gpu),
                              "policy": policy, "direction": "H2D", "bytes": size, "chunk_bytes": chunk,
                              "copies": len(views), "iterations": iterations,
                              "wall_GiB_s": size / 1024**3 / (medians["wall_ms"] / 1000),
                              **medians, "samples": samples}), flush=True)
        del views, target, origin, source, device
        torch.cuda.empty_cache()
    except Exception as exc:
        failed = True
        print(json.dumps({"type": "error", "gpu": gpu, "policy": policy,
                          "error": f"{type(exc).__name__}: {exc}"}), flush=True)
sys.exit(1 if failed else 0)
PY
}

run_tests() {
    # All tools run inside the writable container; no host packages are changed.
    if ! command -v numactl >/dev/null || ! command -v timeout >/dev/null; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends --no-upgrade numactl coreutils
    fi
    numactl --hardware > "$LOG_DIR/numa.txt"
    nvidia-smi topo -m > "$LOG_DIR/gpu-topology.txt"
    nvidia-smi -q > "$LOG_DIR/gpu-before.txt"
    local policy rc failed=0
    local -a prefix policies
    read -r -a policies <<< "$H2D_POLICIES"
    [[ ${#policies[@]} -gt 0 ]] || { echo 'H2D_POLICIES is empty' >&2; return 1; }
    for policy in "${policies[@]}"; do
        case "$policy" in
            unrestricted) prefix=(env) ;;
            memory0) prefix=(numactl --membind=0) ;;
            memory1) prefix=(numactl --membind=1) ;;
            bound0) prefix=(numactl --cpunodebind=0 --membind=0) ;;
            bound1) prefix=(numactl --cpunodebind=1 --membind=1) ;;
            *) echo "Unknown H2D policy: $policy" >&2; return 1 ;;
        esac
        echo "Running $policy"
        rc=0
        timeout 120 "${prefix[@]}" bash "$LOG_DIR/runner.sh" --copy "$policy" \
            > "$LOG_DIR/h2d-$policy.jsonl" 2> "$LOG_DIR/h2d-$policy.stderr" || rc=$?
        printf '%s\n' "$rc" > "$LOG_DIR/h2d-$policy.exit-status"
        [[ $rc == 0 ]] || failed=1
    done
    nvidia-smi -q > "$LOG_DIR/gpu-after.txt"
    python3 - "$LOG_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fields = ["policy", "gpu", "gpu_name", "chunk_bytes", "copies", "wall_GiB_s",
          "submit_ms", "sync_wait_ms", "wall_ms", "cuda_ms"]
with (root / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for path in sorted(root.glob("h2d-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["type"] == "copy":
                writer.writerow(row)
                print(f"{row['policy']:12} GPU {row['gpu']} chunk={row['chunk_bytes']:9} B "
                      f"H2D={row['wall_GiB_s']:.2f} GiB/s")
PY
    return "$failed"
}

case ${1:-} in
    --copy) copy_bench "${2:-manual}"; exit ;;
    --run-tests) run_tests; exit ;;
esac

: "${SLURM_JOB_ID:?Submit with sbatch, or use --copy inside a GPU container}"
[[ ${SLURM_JOB_NUM_NODES:-1} == 1 ]] || { echo 'This benchmark requires one node' >&2; exit 1; }
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/home/rihuo/tensorrt-llm-release-arm64-0c74fcd-user_imant_glm5-rubin-30427.sqsh}
export LOG_DIR=${LOG_DIR:-${SLURM_SUBMIT_DIR}/h2d-bandwidth-${SLURM_JOB_ID}}
export COPY_MIB=${COPY_MIB:-64} COPY_ITERATIONS=${COPY_ITERATIONS:-5}
export H2D_POLICIES=${H2D_POLICIES:-unrestricted memory0 memory1 bound0 bound1}
export ENROOT_REMAP_ROOT=yes
[[ -f $CONTAINER_IMAGE ]] || { echo "Missing image: $CONTAINER_IMAGE" >&2; exit 1; }
mkdir -p "$LOG_DIR"
cp "${BASH_SOURCE[0]}" "$LOG_DIR/runner.sh"
scontrol show job "$SLURM_JOB_ID" > "$LOG_DIR/allocation.txt"
# Request the whole allocation so the step's CPU cgroup permits both NUMA nodes.
: "${SLURM_JOB_CPUS_PER_NODE:?Need the allocated CPU count}"
step_cpus=${SLURM_JOB_CPUS_PER_NODE%%(*}
[[ $step_cpus =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid allocated CPU count' >&2; exit 1; }
rc=0
srun --mpi=none --exact --nodes=1 --ntasks=1 --cpus-per-task="$step_cpus" --cpu-bind=none \
    --container-image="$CONTAINER_IMAGE" --no-container-mount-home \
    --container-writable --container-remap-root --container-mounts="$LOG_DIR:$LOG_DIR" \
    bash "$LOG_DIR/runner.sh" --run-tests > "$LOG_DIR/container.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$LOG_DIR/container.exit-status"
tar -czf "${LOG_DIR}.tar.gz" -C "$(dirname "$LOG_DIR")" "$(basename "$LOG_DIR")"
echo "Results: $LOG_DIR/summary.csv"
echo "Bundle: ${LOG_DIR}.tar.gz"
exit "$rc"
