#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Optional stricter NUMA CPU affinity for TRT-LLM workers (see
# backends/trtllm.py numa_cpu_bind). Binds via `taskset -c` *before* exec so
# secondary threads spawned by Python/UCX/MPI/TRT-LLM inherit the mask too —
# TRT-LLM's own internal affinity logic only pins the leader thread, leaving
# the rest to land cross-socket.
# With --bind-memory, also restrict allocations to the GPU's NUMA node.
# With --no-bind-cpu, apply memory binding without changing CPU affinity.
# Network device filters are restricted to the same NUMA node before launch.
# Local memory exhaustion can fail allocations; existing/shared pages are not migrated.
#
# CPU range is discovered at runtime from the physical GPU this task owns,
# rather than a static SLURM_LOCALID -> cpu_range table. A static table
# assumes SLURM_LOCALID is a node-wide GPU ordinal, which only holds when a
# single srun step owns the whole node; TRTLLM launches one srun per
# endpoint, so two endpoints sharing a node each get their own LOCALID
# sequence starting at 0, and a static table would bind both to the same
# CPUs. Resolving via the actual GPU (CUDA_VISIBLE_DEVICES[LOCALID] when
# set, else LOCALID itself for a full-node endpoint) sidesteps that.
: "${SLURM_LOCALID:?SLURM_LOCALID is required}"

bind_memory=false
bind_cpu=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bind-memory) bind_memory=true; shift ;;
        --no-bind-cpu) bind_cpu=false; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    physical_gpu="${visible_gpus[${SLURM_LOCALID}]:-}"
else
    physical_gpu="${SLURM_LOCALID}"
fi

if [[ -z "${physical_gpu}" ]]; then
    echo "numa_cpu_bind.sh: no GPU resolved for SLURM_LOCALID=${SLURM_LOCALID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
    exit 2
fi

pci_bus_id="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${physical_gpu}")"
# nvidia-smi reports an 8-hex-digit domain (e.g. 00000000:19:00.0); sysfs
# paths use the standard 4-hex-digit form (0000:19:00.0).
domain="${pci_bus_id%%:*}"
rest="${pci_bus_id#*:}"
sysfs_addr="$(printf '%04x:%s' "0x${domain}" "${rest}" | tr '[:upper:]' '[:lower:]')"

numa_node="$(cat "/sys/bus/pci/devices/${sysfs_addr}/numa_node" 2>/dev/null || echo -1)"

if [[ ! "${numa_node}" =~ ^[0-9]+$ ]]; then
    if [[ "${bind_memory}" == true ]]; then
        echo "numa_cpu_bind.sh: cannot bind memory: GPU ${physical_gpu} (${sysfs_addr}) has no valid NUMA node (${numa_node})" >&2
        exit 2
    fi
    echo "numa_cpu_bind.sh: GPU ${physical_gpu} (${sysfs_addr}) reports no NUMA affinity (numa_node=${numa_node}); running without CPU binding" >&2
    exec "$@"
fi

network_env="$(python3 "$(dirname "${BASH_SOURCE[0]}")/numa_net_devices.py" "${numa_node}")"
while IFS=$'\t' read -r key value; do
    [[ -n "${key}" ]] || continue
    export "${key}=${value}"
    echo "numa_cpu_bind.sh: numa_node=${numa_node} ${key}=${value}" >&2
done <<< "${network_env}"

if [[ "${bind_cpu}" == true ]]; then
    cpu_list="$(cat "/sys/devices/system/node/node${numa_node}/cpulist" 2>/dev/null || true)"
    if [[ -z "${cpu_list}" ]]; then
        echo "numa_cpu_bind.sh: cannot bind CPUs: GPU ${physical_gpu} (${sysfs_addr}) NUMA node ${numa_node} has no readable, nonempty CPU list" >&2
        exit 2
    fi
    echo "numa_cpu_bind.sh: SLURM_LOCALID=${SLURM_LOCALID} gpu=${physical_gpu} (${sysfs_addr}) numa_node=${numa_node} bound to cpus=${cpu_list}" >&2
else
    echo "numa_cpu_bind.sh: SLURM_LOCALID=${SLURM_LOCALID} gpu=${physical_gpu} (${sysfs_addr}) numa_node=${numa_node} CPU affinity unchanged" >&2
fi

if [[ "${bind_memory}" == true ]]; then
    echo "numa_cpu_bind.sh: memory_policy=bind:${numa_node}" >&2
    if [[ "${bind_cpu}" == true ]]; then
        exec numactl --membind="${numa_node}" taskset -c "${cpu_list}" "$@"
    fi
    exec numactl --membind="${numa_node}" "$@"
fi
if [[ "${bind_cpu}" == true ]]; then
    exec taskset -c "${cpu_list}" "$@"
fi
exec "$@"
