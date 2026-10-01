# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One socket's CPU power at one instant: origin and per-rail readings kept apart, aggregate derived.

This is the single pivot shared by every CPU power producer. The head-node
scraper (``cpu_parser``) and the host collector's readers (``cpu_power``) each
classify their raw readings into :class:`RailReading` records and hand them to
:func:`pivot_socket_samples`; everything downstream -- CSV rows, node totals,
the energy report -- reads the derived ``power_w``/``rails`` off the
:class:`CpuSample` instead of re-deciding which reading is the socket's power.

Origin (``source``) decides which rail is *primary*:

* ``acpi`` -- the ``total`` envelope (``Grace Power Socket N``); the component
  rails ``cpu_rail``/``soc``/``dram`` ride along as reference breakdowns.
* ``dcgm`` -- field 1130, one already-aggregated value per socket, no rails.

A socket without its primary reading is not a sample: a component rail must
never stand in for the socket's power.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from srtctl.core.power.cpu_rails import COMPONENT_RAIL_KINDS, DCGM_KIND, TOTAL_KIND, sensor_name

PRIMARY_KIND_BY_SOURCE: dict[str, str] = {"acpi": TOTAL_KIND, "dcgm": DCGM_KIND}


def primary_kind(source: str) -> str:
    """The rail kind that *is* a socket's power for this origin."""
    try:
        return PRIMARY_KIND_BY_SOURCE[source]
    except KeyError as exc:
        raise ValueError(
            f"unknown CPU power source {source!r}; expected one of {sorted(PRIMARY_KIND_BY_SOURCE)}"
        ) from exc


@dataclass(frozen=True)
class RailReading:
    """One classified sensor value: which socket, which rail, from which sensor."""

    socket_id: int
    kind: str  # cpu_rails.TOTAL_KIND / DCGM_KIND / a COMPONENT_RAIL_KINDS member
    sensor: str
    watts: float


@dataclass(frozen=True)
class CpuSample:
    """One socket's readings for one instant, with the aggregate derived rather than stored.

    ``readings`` holds every classified rail for the socket, including the
    primary. ``power_w``, ``sensor`` and ``rails`` are views onto it so no
    caller can disagree with another about what the socket's power is.
    """

    source: str
    socket_id: int
    readings: tuple[RailReading, ...]

    def __post_init__(self) -> None:
        kind = primary_kind(self.source)
        if any(reading.socket_id != self.socket_id for reading in self.readings):
            raise ValueError(f"CpuSample for socket {self.socket_id} holds readings from another socket")
        if sum(reading.kind == kind for reading in self.readings) != 1:
            raise ValueError(f"CpuSample for {self.source} socket {self.socket_id} needs exactly one {kind!r} reading")

    @property
    def primary(self) -> RailReading:
        kind = primary_kind(self.source)
        return next(reading for reading in self.readings if reading.kind == kind)

    @property
    def power_w(self) -> float:
        """The socket's power: the primary rail for this origin."""
        return self.primary.watts

    @property
    def sensor(self) -> str:
        """Provenance: the sensor that fed ``power_w``."""
        return self.primary.sensor

    @property
    def rails(self) -> dict[str, float]:
        """Component rails only (never the primary), in canonical column order."""
        by_kind = {reading.kind: reading.watts for reading in self.readings}
        return {kind: by_kind[kind] for kind in COMPONENT_RAIL_KINDS if kind in by_kind}

    def reading(self, kind: str) -> RailReading | None:
        return next((reading for reading in self.readings if reading.kind == kind), None)

    @classmethod
    def from_columns(
        cls,
        *,
        source: str,
        socket_id: int,
        power_w: float,
        sensor: str | None = None,
        rails: Mapping[str, float] | None = None,
    ) -> CpuSample:
        """Rebuild a sample from the wide-CSV columns (``power_w`` + ``<rail>_w``)."""
        kind = primary_kind(source)
        readings = [RailReading(socket_id, kind, sensor or sensor_name(kind, socket_id), power_w)]
        for rail_kind, watts in (rails or {}).items():
            readings.append(RailReading(socket_id, rail_kind, sensor_name(rail_kind, socket_id), watts))
        return cls(source=source, socket_id=socket_id, readings=tuple(readings))


def pivot_socket_samples(source: str, readings: Iterable[RailReading]) -> tuple[CpuSample, ...]:
    """Group classified readings by socket; one :class:`CpuSample` per socket that has its primary.

    The first reading seen for a (socket, kind) wins; a later duplicate is a
    producer bug, not something to average away. Sockets lacking the primary
    rail are dropped -- see the module docstring.
    """
    kind = primary_kind(source)
    by_socket: dict[int, dict[str, RailReading]] = {}
    for reading in readings:
        by_socket.setdefault(reading.socket_id, {}).setdefault(reading.kind, reading)
    return tuple(
        CpuSample(source=source, socket_id=socket_id, readings=tuple(kinds.values()))
        for socket_id, kinds in sorted(by_socket.items())
        if kind in kinds
    )


def node_total_watts(samples: Iterable[CpuSample]) -> float | None:
    """Node aggregate: the sum of every socket's primary rail; None with no sockets."""
    watts = [sample.power_w for sample in samples]
    return sum(watts) if watts else None
