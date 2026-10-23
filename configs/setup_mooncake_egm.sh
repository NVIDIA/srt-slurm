#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# EGM requires the merged Mooncake stack at db195f7f7cad61a0a3c3141b8ecf6ca399a54fca
# (or a descendant), built for ARM64/CUDA 13 with WITH_STORE and USE_MNNVL enabled.
# Published 0.3.13 wheels do not provide the EGM Store pool.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wheel_dir="${SRT_MOONCAKE_EGM_WHEEL_DIR:-/home/rihuo/mooncake_egm_wheels}"

usage() {
    echo "Usage: $0 [--mooncake-only | --build-image SOURCE.sqsh DESTINATION.sqsh]"
    echo 'Set SRT_MOONCAKE_EGM_WHEEL_DIR to the directory containing one compatible wheel.'
}

# Run this mode on an allocated compute node, outside a container. Use the
# site's Enroot data path for the large unpacked rootfs, and a unique disposable
# container so existing Pyxis containers and the source image are never modified.
build_image() (
    local source_image=$1 destination_image=$2
    local destination_dir container_name name_dir publish_dir=''
    if [[ ! -f ${source_image} ]]; then
        echo "Source image does not exist: ${source_image}" >&2
        exit 1
    fi
    if [[ ${destination_image} != *.sqsh ]]; then
        echo 'The destination image must end in .sqsh.' >&2
        exit 1
    fi
    if [[ -e ${destination_image} || -L ${destination_image} ]]; then
        echo "Refusing to overwrite an existing image: ${destination_image}" >&2
        exit 1
    fi
    shopt -s nullglob
    local wheels=("${wheel_dir}"/mooncake_transfer_engine*.whl)
    if (( ${#wheels[@]} != 1 )); then
        echo "Image building requires exactly one ARM64 CUDA 13 Mooncake wheel in ${wheel_dir}." >&2
        exit 1
    fi
    command -v enroot >/dev/null || {
        echo 'Run --build-image on a cluster host with Enroot installed.' >&2
        exit 1
    }
    # Absolute paths remain valid across Enroot's container setup.
    source_image="$(cd -- "$(dirname -- "${source_image}")" && pwd)/$(basename -- "${source_image}")"
    wheel_dir="$(cd -- "${wheel_dir}" && pwd)"
    mkdir -p -- "$(dirname -- "${destination_image}")"
    destination_dir="$(cd -- "$(dirname -- "${destination_image}")" && pwd)"
    destination_image="${destination_dir}/$(basename -- "${destination_image}")"
    name_dir="$(mktemp -d "${TMPDIR:-/tmp}/srtctl-mooncake-egm-XXXXXXXX")"
    container_name=${name_dir##*/}
    cleanup_image_build() {
        enroot remove --force "${container_name}" >/dev/null 2>&1 || true
        [[ -z ${publish_dir} ]] || rm -rf -- "${publish_dir}"
        rm -rf -- "${name_dir}"
    }
    trap cleanup_image_build EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    echo "Creating temporary Enroot container from ${source_image}"
    enroot create --name "${container_name}" "${source_image}"
    enroot start --root --rw \
        --mount "${script_dir}:/srt-mooncake-configs:ro" \
        --mount "${wheel_dir}:/srt-mooncake-wheels:ro" \
        --env SRT_MOONCAKE_EGM_WHEEL_DIR=/srt-mooncake-wheels \
        "${container_name}" bash /srt-mooncake-configs/setup_mooncake_egm.sh --mooncake-only

    # Export on the destination filesystem, then publish with a hard link so
    # concurrent builders cannot replace an existing or partially written image.
    publish_dir="$(mktemp -d "${destination_dir}/.mooncake-egm-XXXXXXXX")"
    enroot export --output "${publish_dir}/image.sqsh" "${container_name}"
    ln -- "${publish_dir}/image.sqsh" "${destination_image}"
    echo "Saved Mooncake EGM image: ${destination_image}"
    echo 'Use this path as model.container; the original image is unchanged.'
)

case "${1:-}" in
    '') bash "${script_dir}/install_dynamo_wheel.sh" ;;
    --mooncake-only) [[ $# == 1 ]] || { usage >&2; exit 2; } ;;
    --build-image)
        [[ $# == 3 ]] || { usage >&2; exit 2; }
        build_image "$2" "$3"
        exit
        ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

# Fabric mode checks this variable by presence, so even '0' selects the wrong
# mode. A child setup script cannot unset it in the worker's parent shell.
if [[ ${MC_USE_NVLINK_IPC+x} ]]; then
    echo 'EGM needs Fabric handles: unset MC_USE_NVLINK_IPC before launching this recipe.' >&2
    exit 1
fi

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
