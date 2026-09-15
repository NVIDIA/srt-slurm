# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process Dynamo router and worker logs into a small portable bundle."""

from srtctl.ruter.normalize import NormalizationReport, normalize_run

__all__ = ["NormalizationReport", "normalize_run"]
