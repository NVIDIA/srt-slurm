#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pip install msgpack

if [ -n "${DYNAMO_VERSION:-}" ] || [ -n "${DYNAMO_WHEEL_NAME:-}" ]; then
    if [ -f /srtctl-runtime/dynamo_wheels.py ]; then
        python3 /srtctl-runtime/dynamo_wheels.py install
    else
        echo "ERROR: /srtctl-runtime/dynamo_wheels.py not found for ai-dynamo wheel install" >&2
        exit 1
    fi
fi
