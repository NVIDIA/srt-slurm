#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pins prefill-rank workers to their GPU-local RDMA NICs (UCX_NET_DEVICES),
# rather than leaving every rank free to register on all mlx5 devices on the
# node. On a disaggregated prefill/decode job, unpinned prefill ranks each
# register on every NIC, and that per-rank fan-out is what floods UCX's
# endpoint/registration state on nodes with many rails. Decode/agg workers
# stay unpinned, so their GPU memory is registered on whichever NIC a
# prefill peer happens to select.
#
# Sourced (NOT executed) as srtctl's cluster-level default_bash_preamble
# (see default_bash_preamble in srtslurm.yaml), so its exports land directly
# in the worker's own shell rather than a subshell.
#
# The cluster-level preamble runs BEFORE the worker's own `export
# CUDA_VISIBLE_DEVICES=...` (see core/slurm.py: the cluster preamble is
# spliced in ahead of the per-worker exports), so CUDA_VISIBLE_DEVICES is
# not yet a real env var here when the endpoint uses a subset of the node's
# GPUs. It's recovered from the upcoming command line (BASH_EXECUTION_STRING)
# instead. SLURM_LOCALID *is* a genuine SLURM-injected var and is already
# set at this point.
#
# Only prefill workers are pinned (matches trtllm.py's config filename
# convention: trtllm_config_prefill.yaml / trtllm_config_decode.yaml).
case "${BASH_EXECUTION_STRING:-}" in
    *trtllm_config_prefill*) ;;
    *)
        echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} role=non-prefill: leaving UCX_NET_DEVICES unset"
        return 0 2>/dev/null || true
        ;;
esac

_srt_cvd=$(printf '%s' "${BASH_EXECUTION_STRING:-}" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | head -1 | cut -d= -f2)
if [ -n "$_srt_cvd" ]; then
    IFS=',' read -r -a _srt_gpus <<< "$_srt_cvd"
    _srt_phys="${_srt_gpus[${SLURM_LOCALID:-0}]}"
else
    _srt_phys="${SLURM_LOCALID:-0}"
fi

# Resolve the GPU's PCI sysfs path the same way configs/numa_cpu_bind.sh
# does: nvidia-smi reports an 8-hex-digit domain, sysfs paths use 4.
_srt_bus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${_srt_phys}" 2>/dev/null)"
if [ -z "${_srt_bus}" ]; then
    echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} phys_gpu=${_srt_phys}: nvidia-smi unavailable, leaving UCX_NET_DEVICES unset"
    return 0 2>/dev/null || true
fi
_srt_domain="${_srt_bus%%:*}"
_srt_rest="${_srt_bus#*:}"
_srt_gpu_addr="$(printf '%04x:%s' "0x${_srt_domain}" "${_srt_rest}" | tr '[:upper:]' '[:lower:]')"
_srt_gpu_path="$(readlink -f "/sys/bus/pci/devices/${_srt_gpu_addr}" 2>/dev/null)"

if [ -z "${_srt_gpu_path}" ]; then
    echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} phys_gpu=${_srt_phys}: no sysfs entry for ${_srt_gpu_addr}, leaving UCX_NET_DEVICES unset"
    return 0 2>/dev/null || true
fi

# Rank each ACTIVE mlx5 device by how many PCIe path components it shares
# with the GPU's own sysfs path. The deepest shared ancestor is the closest
# common switch/root port -- the same relationship `nvidia-smi topo -m`
# reports as PIX/PXB -- so the device(s) tied for the longest shared prefix
# are the GPU-local NICs. Deriving this from sysfs at runtime (rather than a
# hardcoded per-cluster rail table) keeps it correct across different b300
# nodes/rail wirings without needing a measured table per site.
_srt_best_depth=-1
_srt_candidates=""
for _srt_nic_dir in /sys/class/infiniband/mlx5_*; do
    [ -e "${_srt_nic_dir}" ] || continue
    _srt_nic="$(basename "${_srt_nic_dir}")"

    case "$(cat "${_srt_nic_dir}/ports/1/state" 2>/dev/null)" in
        *ACTIVE*) ;;
        *)
            echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} dropping non-ACTIVE device ${_srt_nic}"
            continue
            ;;
    esac

    _srt_nic_path="$(readlink -f "${_srt_nic_dir}/device" 2>/dev/null)"
    [ -n "${_srt_nic_path}" ] || continue

    IFS='/' read -ra _srt_gpu_parts <<< "${_srt_gpu_path}"
    IFS='/' read -ra _srt_nic_parts <<< "${_srt_nic_path}"
    _srt_depth=0
    for _srt_i in "${!_srt_gpu_parts[@]}"; do
        [ "${_srt_gpu_parts[${_srt_i}]}" = "${_srt_nic_parts[${_srt_i}]:-}" ] || break
        _srt_depth=$((_srt_depth + 1))
    done

    if [ "${_srt_depth}" -gt "${_srt_best_depth}" ]; then
        _srt_best_depth="${_srt_depth}"
        _srt_candidates="${_srt_nic}:1"
    elif [ "${_srt_depth}" -eq "${_srt_best_depth}" ]; then
        _srt_candidates="${_srt_candidates},${_srt_nic}:1"
    fi
done

if [ -z "${_srt_candidates}" ]; then
    echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} phys_gpu=${_srt_phys}: no ACTIVE mlx5 device found, leaving UCX_NET_DEVICES unset"
    return 0 2>/dev/null || true
fi

# Every pinned rank still needs a universally-reachable fallback lane: two
# same-node prefill ranks pinned to disjoint NICs otherwise have no shared
# transport that supports UCX's peer error handling (self/sysv only), and
# UCX reports "no active messages transport". bond0 patches that
# reachability gap; it cannot steal the bulk KV path since
# UCX_RNDV_SCHEME=put_zcopy requires RDMA, which tcp does not provide.
if [ -e /sys/class/net/bond0 ]; then
    _srt_candidates="${_srt_candidates},bond0"
fi

export UCX_NET_DEVICES="${_srt_candidates}"
echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} phys_gpu=${_srt_phys} UCX_NET_DEVICES=${UCX_NET_DEVICES}"
