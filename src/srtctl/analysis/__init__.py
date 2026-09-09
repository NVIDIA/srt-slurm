# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-window analysis that is not a metrics scraper.

:mod:`.host_sampler` reads ``/proc`` for what no ``/metrics`` endpoint publishes
(host CPU saturation, fd headroom, per-process context switches);
:mod:`.perf_dashboard` drives the tachometer parquet and the other run artifacts
through ``src/ingest`` into the per-run HTML dashboard. All metrics scraping is
tachometer's.
"""
