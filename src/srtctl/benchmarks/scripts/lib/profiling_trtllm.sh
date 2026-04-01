#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# No-op profiling library for TRTLLM workers.
#
# For TRTLLM, profiling is managed entirely at worker launch time:
#   - nsys (iteration-based): TLLM_PROFILE_START_STOP=<start>-<stop> env var causes
#     TRT-LLM's PyExecutor to call cudaProfilerStart/Stop at the right iterations.
#     The worker is wrapped with: nsys profile -c cudaProfilerApi ...
#   - nsys-time (time-based): worker is wrapped with nsys profile --delay N --duration M,
#     capturing the same wall-clock window on all workers simultaneously.
#
# No HTTP calls to /engine/start_profile are needed — the benchmark script just
# needs to generate traffic for long enough to cover the capture window.
#
# This library exposes the same function signatures as profiling.sh so bench.sh
# can source either library transparently based on PROFILING_BACKEND.

profiling__started=0

profiling_init_from_env() {
    PROFILE_TYPE="${PROFILE_TYPE:-none}"
    PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-}"
    PROFILE_BENCHMARK_DURATION_SECS="${PROFILE_BENCHMARK_DURATION_SECS:-300}"
}

start_all_profiling() {
    if [[ "${PROFILE_TYPE:-none}" == "none" ]]; then
        return 0
    fi
    echo ""
    echo "[profiling_trtllm] Profiling type=${PROFILE_TYPE}: managed by worker env vars at launch"
    echo "  - nsys: capture triggered by TLLM_PROFILE_START_STOP on worker (cudaProfilerApi)"
    echo "  - nsys-time: capture window defined by nsys --delay/--duration on worker process"
    echo "  No HTTP start_profile call needed."
    profiling__started=1
    return 0
}

stop_all_profiling() {
    if [[ "${profiling__started:-0}" != "1" ]]; then
        return 0
    fi
    echo ""
    echo "[profiling_trtllm] Profiling capture window closes automatically on the worker."
    if [[ -n "${PROFILE_OUTPUT_DIR}" ]]; then
        echo "  Profiling results will be saved to ${PROFILE_OUTPUT_DIR}"
    fi
    profiling__started=0
    return 0
}
