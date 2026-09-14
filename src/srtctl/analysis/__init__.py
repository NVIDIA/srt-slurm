# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-window analysis that is not a metrics scraper.

:mod:`.host_sampler` reads ``/proc`` for what no ``/metrics`` endpoint publishes
(host CPU saturation, fd headroom, per-process context switches);
:mod:`.tachometer_dashboard` builds the TRT-LLM run UI from raw Tachometer
Parquet/Arrow. The legacy :mod:`.perf_dashboard` bridge combines other run
artifacts and is scheduled for deprecation. All metrics scraping is tachometer's.
"""
