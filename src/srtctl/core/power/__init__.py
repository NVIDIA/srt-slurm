# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw multinode GPU power artifacts for the ``dcgm-power`` telemetry provider.

The provider records watts per allocated GPU, the srt-slurm topology needed to
map devices to ``prefill``/``decode``/``agg``, and the exact formal benchmark
window. It never integrates power into energy; that belongs to consumers of the
artifact contract.
"""
