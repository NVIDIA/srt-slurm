#!/usr/bin/env bash
#SBATCH --job-name=vllm_e2fa2859_arm64
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/kylliang/srtslurm_studies/20260730_agentx_vllm_minimaxm3_eagle3_gqa_latest/logs/container_import_e2fa2859_%j.log

set -euo pipefail

container_dir="/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/kylliang/srtslurm_studies/downloaded_dockerhub"
image="vllm/vllm-openai:nightly-e2fa28594f7baad142a426b0b6a2cfe2c79201c7"
target="$container_dir/vllm_vllm-openai_nightly-e2fa28594f7baad142a426b0b6a2cfe2c79201c7_arm64.sqsh"
partial="${target}.partial.${SLURM_JOB_ID}"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Refusing to import on non-ARM64 host: $(uname -m)" >&2
    exit 1
fi

mkdir -p "$container_dir"

if unsquashfs -s "$target" >/dev/null 2>&1; then
    echo "Valid squashfs already present: $target"
else
    if [[ -e "$target" ]]; then
        mv "$target" "${target}.invalid.${SLURM_JOB_ID}"
    fi
    echo "Importing docker://$image on $(hostname) ($(uname -m))"
    enroot import -o "$partial" "docker://$image"
    unsquashfs -s "$partial"
    mv "$partial" "$target"
fi

srun --nodes=1 --ntasks=1 \
    --container-image="$target" \
    --no-container-entrypoint \
    /bin/uname -m

echo "Container import and ARM64 execution validation complete: $target"
