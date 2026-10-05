# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types without dependencies on backends or orchestration."""

from typing import Literal, TypeAlias

WorkerMode: TypeAlias = Literal["prefill", "decode", "agg"]
