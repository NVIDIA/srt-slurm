#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Python packages sa-bench needs at runtime, and the module names that prove
# they are importable (they differ: Pillow imports as PIL).
#
# Single source of truth: bench.sh installs whatever is missing, and
# `srtctl bake-image --sa-bench` preinstalls the same list into a container
# image so that install becomes a no-op. Keeping both in one file is what stops
# a baked image from drifting away from what the benchmark checks for.

SA_BENCH_DEPS=(aiohttp numpy pandas datasets Pillow tqdm transformers huggingface_hub)
SA_BENCH_IMPORTS="aiohttp, numpy, pandas, datasets, PIL, tqdm, transformers, huggingface_hub"
