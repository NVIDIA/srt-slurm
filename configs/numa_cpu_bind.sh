#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Set NUMA CPU affinity for TRT-LLM and vLLM workers (see numa_cpu_bind in
# backends/trtllm.py and backends/vllm.py) via `numactl --cpunodebind` before
# exec. The TRT-LLM backend defaults TRT-LLM's worker affinity to 0 to preserve
# this mask through worker initialization. Explicit recipe environment values
# can override that default. See docs/engines.md for behavior at specific
# TRT-LLM versions.
# Without a memory flag, preserve the inherited memory policy.
# With --bind-memory, also restrict allocations to the GPUs' NUMA node(s).
# With --prefer-memory, prefer that node while allowing allocations to fall back
# to other nodes when it is full. When the GPUs span several NUMA nodes the
# kernel's default local allocation already prefers the node of the CPU the
# thread runs on (which the CPU binding keeps within those nodes), so no
# memory flag is added.
# Local memory exhaustion can fail allocations; existing/shared pages are not migrated.
#
# The NUMA node is discovered at runtime from the physical GPUs this task owns,
# rather than a static SLURM_LOCALID -> cpu_range table.
#
# --gpus <i,j,...> names those GPUs (node-local nvidia-smi indices). The vLLM
# backend passes it: one srun task runs one worker process that owns all of
# the listed GPUs, and srtctl knows them when it builds the command, whether it
# exports a device mask or passes --device-ids. The task binds to the union of
# their NUMA nodes.
#
# Without --gpus the task owns one GPU (TRT-LLM's MPI launch: one rank per GPU,
# the same argv on every rank). A static table would assume SLURM_LOCALID is a
# node-wide GPU ordinal, which only holds when a single srun step owns the
# whole node; TRTLLM launches one srun per endpoint, so two endpoints sharing
# a node each get their own LOCALID sequence starting at 0, and a static table
# would bind both to the same CPUs. Resolving via the actual GPU
# (CUDA_VISIBLE_DEVICES[LOCALID] when set, else LOCALID itself for a
# full-node endpoint) sidesteps that.

memory_policy=none
gpu_list=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bind-memory) memory_policy=bind; shift ;;
        --prefer-memory) memory_policy=preferred; shift ;;
        --gpus)
            gpu_list="${2:-}"
            if [[ ! "${gpu_list}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
                echo "numa_cpu_bind.sh: --gpus needs a comma-separated list of GPU indices (got '${gpu_list}')" >&2
                exit 2
            fi
            shift 2
            ;;
        *) break ;;
    esac
done

if [[ -n "${gpu_list}" ]]; then
    IFS=',' read -ra physical_gpus <<< "${gpu_list}"
    task_label="gpus=${gpu_list}"
else
    : "${SLURM_LOCALID:?SLURM_LOCALID is required}"
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
    physical_gpus=("${physical_gpu}")
    task_label="SLURM_LOCALID=${SLURM_LOCALID} gpu=${physical_gpu}"
fi

# The NUMA node of each GPU, in GPU order; duplicates are dropped below.
numa_nodes=()
gpu_addrs=()
for gpu in "${physical_gpus[@]}"; do
    pci_bus_id="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${gpu}")"
    # nvidia-smi reports an 8-hex-digit domain (e.g. 00000000:19:00.0); sysfs
    # paths use the standard 4-hex-digit form (0000:19:00.0).
    domain="${pci_bus_id%%:*}"
    rest="${pci_bus_id#*:}"
    sysfs_addr="$(printf '%04x:%s' "0x${domain}" "${rest}" | tr '[:upper:]' '[:lower:]')"
    gpu_addrs+=("${sysfs_addr}")

    numa_node="$(cat "/sys/bus/pci/devices/${sysfs_addr}/numa_node" 2>/dev/null || echo -1)"

    if [[ ! "${numa_node}" =~ ^[0-9]+$ ]]; then
        if [[ "${memory_policy}" != none ]]; then
            echo "numa_cpu_bind.sh: cannot apply memory policy ${memory_policy}: GPU ${gpu} (${sysfs_addr}) has no valid NUMA node (${numa_node})" >&2
            exit 2
        fi
        echo "numa_cpu_bind.sh: GPU ${gpu} (${sysfs_addr}) reports no NUMA affinity (numa_node=${numa_node}); running without CPU binding" >&2
        exec "$@"
    fi
    numa_nodes+=("${numa_node}")
done

nodes="$(printf '%s\n' "${numa_nodes[@]}" | sort -n -u | paste -sd, -)"
gpu_label="${task_label} (${gpu_addrs[*]})"

cpu_lists=()
for node in ${nodes//,/ }; do
    cpu_list="$(cat "/sys/devices/system/node/node${node}/cpulist" 2>/dev/null || true)"
    if [[ -z "${cpu_list}" ]]; then
        echo "numa_cpu_bind.sh: cannot bind CPUs: ${gpu_label} NUMA node ${node} has no readable, nonempty CPU list" >&2
        exit 2
    fi
    cpu_lists+=("${cpu_list}")
done

echo "numa_cpu_bind.sh: ${gpu_label} binding CPUs in numa_node=${nodes} (node cpus=$(IFS=';'; echo "${cpu_lists[*]}"))" >&2
case "${memory_policy}" in
    bind)
        echo "numa_cpu_bind.sh: memory_policy=bind:${nodes}" >&2
        exec numactl --cpunodebind="${nodes}" --membind="${nodes}" "$@"
        ;;
    preferred)
        if [[ "${nodes}" != *,* ]]; then
            echo "numa_cpu_bind.sh: memory_policy=preferred:${nodes}" >&2
            exec numactl --cpunodebind="${nodes}" --preferred="${nodes}" "$@"
        fi
        echo "numa_cpu_bind.sh: memory_policy=preferred:${nodes} (default local allocation across the bound nodes)" >&2
        ;;
esac
exec numactl --cpunodebind="${nodes}" "$@"
