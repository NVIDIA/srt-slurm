#!/usr/bin/env bash
set -euo pipefail

# Decode-only NUMA CPU affinity fix (see backends/trtllm.py numa_cpu_bind).
# Binds via `taskset -c` *before* exec so secondary threads spawned by
# Python/UCX/MPI/TRT-LLM inherit the mask too — TRT-LLM's own internal
# affinity logic only pins the leader thread, leaving the rest to land
# cross-socket and stall the decode phase.
#
# NUMA_CPU_BIND_RANGES is a ';'-separated "localid=cpu_list" map set by
# srtctl from backend.numa_cpu_bind_ranges, e.g.
#   "0=0-87,176-263;1=0-87,176-263;2=88-175,264-351;3=88-175,264-351"
: "${NUMA_CPU_BIND_RANGES:?NUMA_CPU_BIND_RANGES is required}"
: "${SLURM_LOCALID:?SLURM_LOCALID is required}"

cpu_list=""
IFS=';' read -ra entries <<< "${NUMA_CPU_BIND_RANGES}"
for entry in "${entries[@]}"; do
    localid="${entry%%=*}"
    cpu_range="${entry#*=}"
    if [[ "${localid}" == "${SLURM_LOCALID}" ]]; then
        cpu_list="${cpu_range}"
        break
    fi
done

if [[ -z "${cpu_list}" ]]; then
    echo "numa_cpu_bind.sh: no CPU range configured for SLURM_LOCALID=${SLURM_LOCALID}" >&2
    exit 2
fi

echo "numa_cpu_bind.sh: SLURM_LOCALID=${SLURM_LOCALID} bound to cpus=${cpu_list}" >&2
exec taskset -c "${cpu_list}" "$@"
