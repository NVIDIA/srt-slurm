#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage banners for benchmark scripts.
#
# A worker log is a continuous stream with nothing in it to say which
# concurrency level or which phase produced a given line. stage_banner writes a
# timestamped marker into the live worker logs so warmup, measured runs and
# concurrency levels can be told apart by eye and lined up with what the workers
# were doing at that moment:
#
#     ======== [12:34:10] cc=1024 warmup begin ========
#
# The job log dir is mounted at /logs in every container, and srun opens worker
# logs with --open-mode=append, so these appends interleave with worker output
# instead of being overwritten.

stage_banner() {
    local log_dir="${STAGE_BANNER_LOG_DIR:-/logs}"
    local line
    line="======== [$(date '+%H:%M:%S')] $* ========"

    local file wrote=0
    for file in "${log_dir}"/*_prefill_w*.out \
                "${log_dir}"/*_decode_w*.out \
                "${log_dir}"/*_agg_w*.out; do
        [ -e "$file" ] || continue
        # Blank lines around the marker are what make it findable while
        # scrolling through thousands of lines of worker output.
        printf '\n%s\n\n' "$line" >> "$file"
        wrote=1
    done

    # No worker logs to annotate (unexpected layout, or a run without workers):
    # keep the marker visible in the benchmark log rather than dropping it.
    [ "$wrote" -eq 1 ] || echo "$line"
}
