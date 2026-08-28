# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 3 (L3): render an ingest bundle into a single-file component dashboard.

``build_dynamo_bench_dash`` turns the bundle produced by :mod:`src.ingest.ingest`
into one self-contained HTML page with Overview / Router / Engine / Frontend tabs,
plus an optional Log-analysis tab built from the Dynamo frontend log alone.

    python3 -m src.visualization.build_dynamo_bench_dash <bundle_dir> out.html

See ``docs/component-dashboard.md`` for the end-to-end recipe against an srt-slurm
run directory, and for which tabs a given run can populate.
"""
