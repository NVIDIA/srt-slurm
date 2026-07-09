# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime helper scripts mounted into worker containers."""

from pathlib import Path

RUNTIME_SCRIPTS_HOST_DIR = Path(__file__).resolve().parent
RUNTIME_SCRIPTS_CONTAINER_DIR = Path("/srtctl-runtime")
FINGERPRINT_SCRIPT_NAME = "fingerprint.py"
FINGERPRINT_SCRIPT_HOST_PATH = RUNTIME_SCRIPTS_HOST_DIR / FINGERPRINT_SCRIPT_NAME
FINGERPRINT_SCRIPT_CONTAINER_PATH = RUNTIME_SCRIPTS_CONTAINER_DIR / FINGERPRINT_SCRIPT_NAME
