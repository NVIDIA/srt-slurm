#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Warmup must finish before start. Each call waits for every rank/frontend.
nsys_window_start() {
    if [[ -n "${SRT_NSYS_CONTROL_DIR:-}" ]]; then
        python3 "${SRT_NSYS_CONTROL_SCRIPT}" start
    fi
}

nsys_window_stop() {
    if [[ -n "${SRT_NSYS_CONTROL_DIR:-}" ]]; then
        python3 "${SRT_NSYS_CONTROL_SCRIPT}" stop
    fi
}
