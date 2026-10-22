#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# EGM requires the merged Mooncake stack at db195f7f7cad61a0a3c3141b8ecf6ca399a54fca
# (or a descendant), built for ARM64/CUDA 13 with WITH_STORE and USE_MNNVL enabled.
# Published 0.3.13 wheels do not provide the EGM Store pool.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    '') bash "${script_dir}/install_dynamo_wheel.sh" ;;
    --mooncake-only) ;;
    *) echo "Usage: $0 [--mooncake-only]" >&2; exit 2 ;;
esac

# Fabric mode checks this variable by presence, so even '0' selects the wrong
# mode. A child setup script cannot unset it in the worker's parent shell.
if [[ ${MC_USE_NVLINK_IPC+x} ]]; then
    echo 'EGM needs Fabric handles: unset MC_USE_NVLINK_IPC before launching this recipe.' >&2
    exit 1
fi

wheel_dir="${SRT_MOONCAKE_EGM_WHEEL_DIR:-/mooncake_egm_wheels}"
shopt -s nullglob
wheels=("${wheel_dir}"/mooncake_transfer_engine*.whl)
if (( ${#wheels[@]} > 1 )); then
    echo "Keep exactly one ARM64 CUDA 13 Mooncake wheel in ${wheel_dir}." >&2
    exit 1
fi

# Like the Dynamo installer, serialize writes to the container's Python prefix
# across ranks and key the sentinel on the wheel's content, not its filename.
prefix_dir="$(python3 -c 'import sys; print(sys.prefix)')"
(
    flock -x 200
    if (( ${#wheels[@]} == 1 )); then
        wheel_id="$(sha256sum "${wheels[0]}" | cut -d ' ' -f 1)"
        sentinel="${prefix_dir}/.srtctl_mooncake_egm.${wheel_id}.complete"
        if [[ ! -f "${sentinel}" ]]; then
            echo "Installing EGM Mooncake wheel: ${wheels[0]} SHA256=${wheel_id}"
            # This image already has the Store dependencies. Preserve its
            # custom TRT-LLM, Torch and CUDA stack when replacing Mooncake.
            python3 -m pip install --no-cache-dir --no-deps --force-reinstall "${wheels[0]}"
            python3 "${script_dir}/mooncake_egm_donor.py" check-install
            rm -f "${prefix_dir}"/.srtctl_mooncake_egm.*.complete
            touch "${sentinel}"
        fi
    else
        echo "No wheel in ${wheel_dir}; checking the Mooncake build already in the container."
    fi
    # Also check cached installs: another setup step may have replaced the wheel.
    if ! python3 "${script_dir}/mooncake_egm_donor.py" check-install; then
        echo "Supply an EGM-enabled ARM64 CUDA 13 wheel in ${wheel_dir}, or use an image containing it." >&2
        exit 1
    fi
) 200>"${prefix_dir}/.srtctl_mooncake_egm_install.lock"
