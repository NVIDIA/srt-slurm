# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw multinode GPU power artifacts for the ``dcgm-power`` telemetry provider.

The provider records watts per allocated GPU, optional utilization when the
exporter reports it, the srt-slurm topology needed to map devices to
``prefill``/``decode``/``agg``, and the exact formal benchmark window. Which
Prometheus exporter supplies the watts, and under which metric and labels, is
one row of ``profile.POWER_PROFILES`` (DCGM by default, rocm/device-metrics-exporter
for AMD); the artifact layout is the same for every row. It never integrates
power into energy or aggregates utilization; that belongs to consumers of the
artifact contract.
"""
