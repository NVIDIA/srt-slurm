# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-independent log metric generator contract; no I/O or dashboard state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ..sources import SourceIdentity


@dataclass(frozen=True)
class LogMetricDefinition:
    name: str
    title: str
    unit: str
    description: str
    # A setting remains effective until the next recorded setting in this scope.
    # Samples are observations only; neither their values nor their limits persist.
    temporal: Literal["sample", "setting", "event"] = "sample"
    reference: str | None = None
    reference_label: str | None = None


@dataclass(frozen=True)
class LogMetricEvent:
    """Values share a timestamp and scope. None explicitly clears a known setting.

    Timestamps may carry a UTC offset. Naive local timestamps require the caller's
    explicit timezone. Rank namespaces and process identities must be recorded,
    never inferred from another source's rank or process. Labels further scope a
    series (e.g. a DP instance). File identity is added by the shared reader.
    """

    time: str
    values: tuple[tuple[str, float | None], ...]
    rank: int | None = None
    rank_kind: str | None = None
    process: str | None = None
    labels: tuple[tuple[str, str], ...] = ()
    time_resolution_s: float = 0.001


class LogMetricGenerator(ABC):
    """One stateless parser per dialect; the reader owns clocks, scope and evidence.

    Definitions use log_<component>_<name>, where component identifies the producer
    of the consumed log (e.g. tokenspeed or dynamo_frontend), distinct from native
    exported metrics. A reference names another definition of the same unit; it is
    joined only in the exact same file/worker/rank/process/label scope. Unsupported
    lines return None. Event metrics retain distinct source lines even when multiple
    requests have the same timestamp and value. Invalid or missing observations are omitted; invalid
    settings can emit None to prevent
    carrying an earlier configuration through a restart/configuration record.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def definitions(self) -> tuple[LogMetricDefinition, ...]: ...

    @abstractmethod
    def parse_line(self, line: str, source: SourceIdentity) -> LogMetricEvent | None: ...
