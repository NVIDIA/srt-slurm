#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-GPU UCX rail pinning, ported from configs/hca_pin.sh's default
# (no SRT_FABRIC_MODE) branch. Gated on TRTLLM_CTX_LOCAL_HCA_PIN=1 -- the
# same env var name BTK's own patch uses to gate its pinning on bia, kept
# here so a recipe that copies bia's environment verbatim also activates
# this. Pins the matching rank to its own physical GPU's 2 local rails from
# a measured, static table (not runtime PCIe discovery), plus a shared
# "common pair" and a bond0 fallback for same-node reachability -- see
# configs/hca_pin.sh for why: sa's fabric is rail-based with no cross-rail
# L3 routing, so two same-node ranks pinned to disjoint rails otherwise
# cannot reach each other at all, wedging KV transfers between them.
#
# Sourced (NOT executed) as srtctl's cluster-level default_bash_preamble
# (see default_bash_preamble in srtslurm.yaml), so its exports land directly
# in the worker's own shell rather than a subshell.
#
# The cluster-level preamble runs BEFORE the worker's own `export
# TRTLLM_CTX_LOCAL_HCA_PIN=...`/`export CUDA_VISIBLE_DEVICES=...` (see
# core/slurm.py: the cluster preamble is spliced in ahead of the per-worker
# exports), so neither is yet a real env var here. Both are recovered from
# the upcoming command line (BASH_EXECUTION_STRING) instead. SLURM_LOCALID
# *is* a genuine SLURM-injected var and is already set at this point.
case "${BASH_EXECUTION_STRING:-}" in
    *TRTLLM_CTX_LOCAL_HCA_PIN=1*) ;;
    *)
        echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} TRTLLM_CTX_LOCAL_HCA_PIN not set: leaving UCX_NET_DEVICES unset"
        return 0 2>/dev/null || true
        ;;
esac

_srt_cvd=$(printf '%s' "${BASH_EXECUTION_STRING:-}" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | head -1 | cut -d= -f2)
if [ -n "${_srt_cvd}" ]; then
    IFS=',' read -r -a _srt_g <<< "${_srt_cvd}"
    _srt_phys="${_srt_g[${SLURM_LOCALID:-0}]}"
else
    _srt_phys="${SLURM_LOCALID:-0}"
fi

# Measured on sa's b300 nodes (nvidia-smi topo -m PXB columns cross-checked
# against /sys/class/infiniband/*/device PCI ids). Rail-group based, not a
# naive PCIe-card pairing -- see configs/hca_pin.sh's own commentary if this
# table ever needs re-deriving; it is NOT the same as sysfs PCIe-switch
# locality and must not be "corrected" without re-measuring.
case "${_srt_phys}" in
    0) _srt_hca="mlx5_2:1,mlx5_3:1"   ;;
    1) _srt_hca="mlx5_8:1,mlx5_9:1"   ;;
    2) _srt_hca="mlx5_4:1,mlx5_5:1"   ;;
    3) _srt_hca="mlx5_0:1,mlx5_1:1"   ;;
    4) _srt_hca="mlx5_16:1,mlx5_17:1" ;;
    5) _srt_hca="mlx5_22:1,mlx5_23:1" ;;
    6) _srt_hca="mlx5_20:1,mlx5_21:1" ;;
    7) _srt_hca="mlx5_10:1,mlx5_11:1" ;;
    *)
        echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0}: no HCA mapping for physical GPU ${_srt_phys}, leaving UCX_NET_DEVICES unset"
        return 0 2>/dev/null || true
        ;;
esac

# Common pair: every rank not already on GPU3's own pair also gets
# mlx5_0:1,mlx5_1:1, so same-node ctx siblings pinned to otherwise-disjoint
# rails still share one mutually reachable RDMA path. Registration stays at
# 4 devices (2 own + 2 common) instead of the site's full rail count.
case "${_srt_hca}" in
    mlx5_0:*) ;;
    *) _srt_hca="${_srt_hca},mlx5_0:1,mlx5_1:1" ;;
esac

# Drop any device this node doesn't have, or whose port isn't ACTIVE --
# existence alone is not enough (measured on this same cluster: b300-007's
# mlx5_2/mlx5_3 and b300-017's mlx5_4/mlx5_5 are present in /sys but
# state="1: DOWN"). A dead entry in UCX_NET_DEVICES is not benign: UCX
# either errors out or silently falls back to a non-local device, which is
# exactly what this table exists to prevent.
_srt_keep=""
_srt_oIFS="${IFS}"; IFS=','
for _srt_d in ${_srt_hca}; do
    _srt_bare="${_srt_d%%:*}"
    if [ ! -d "/sys/class/infiniband/${_srt_bare}" ]; then
        echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} dropping absent device ${_srt_d}"
        continue
    fi
    case "$(cat "/sys/class/infiniband/${_srt_bare}/ports/1/state" 2>/dev/null)" in
        *ACTIVE*) _srt_keep="${_srt_keep:+${_srt_keep},}${_srt_d}" ;;
        *) echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} dropping non-ACTIVE device ${_srt_d}" ;;
    esac
done
IFS="${_srt_oIFS}"
_srt_hca="${_srt_keep}"

if [ -z "${_srt_hca}" ]; then
    echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} phys_gpu=${_srt_phys}: no live device left after filtering, leaving UCX_NET_DEVICES unset"
    return 0 2>/dev/null || true
fi

export UCX_NET_DEVICES="${_srt_hca}"
echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} phys_gpu=${_srt_phys} UCX_NET_DEVICES=${UCX_NET_DEVICES}"

# bond0 as an active-message fallback only: with per-rank IB pinning, two
# same-node ctx ranks can hold disjoint device sets and UCX finds no AM lane
# supporting NIXL's peer error handling ("no active messages transport"),
# killing the rank. bond0's tcp transport does support it. It cannot steal
# the bulk KV path (UCX_RNDV_SCHEME=put_zcopy requires RDMA, which tcp does
# not provide), so rendezvous still rides the two pinned IB rails.
case ",${UCX_NET_DEVICES:-}," in
    *,bond0,*) ;;
    ,,)        ;;
    *)         export UCX_NET_DEVICES="${UCX_NET_DEVICES},bond0" ;;
esac
echo "UCX_DEVICE_PIN localid=${SLURM_LOCALID:-0} phys_gpu=${_srt_phys} final UCX_NET_DEVICES=${UCX_NET_DEVICES}"
