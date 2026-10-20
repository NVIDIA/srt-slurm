#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit from the repository root: sbatch tools/check_hecate_rails.sbatch
# Reuses the cross-rail controller and persistent container agents.
#SBATCH --job-name=hecate-rails
#SBATCH --account=root
#SBATCH --partition=batch-xdr
#SBATCH --nodes=2
#SBATCH --segment=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=hecate-rails-%j.out

set -euo pipefail
export MC_RAIL_CLUSTER=${MC_RAIL_CLUSTER:-hecate}
export MC_RAIL_IMAGE=${MC_RAIL_IMAGE:-/lustre/fsw/coreai_comparch_inferencex/rihuo/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh}
export MC_RAIL_DURATION=${MC_RAIL_DURATION:-3}
export MC_RAIL_CASE_TIMEOUT=${MC_RAIL_CASE_TIMEOUT:-8}
export MC_RAIL_BUDGET=${MC_RAIL_BUDGET:-1620}
export MC_RAIL_TOOL_DIR=${MC_RAIL_TOOL_DIR:-${SLURM_SUBMIT_DIR}/tools}
export MC_RAIL_LAUNCHER_SOURCE=${BASH_SOURCE[0]}
exec bash "$MC_RAIL_TOOL_DIR/check_mooncake_rails.sbatch"
