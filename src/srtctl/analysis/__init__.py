# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-window analysis that is not a metrics scraper.

:mod:`.host_sampler` reads ``/proc`` for what no ``/metrics`` endpoint publishes
(host CPU saturation, fd headroom, per-process context switches).
:mod:`.metric_catalog` provides captured-metric presentation semantics for DSight.
All metrics scraping is tachometer's.
"""
