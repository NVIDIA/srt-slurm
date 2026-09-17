# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side per-socket CPU-side power collector.

The implementation follows BTK's CPU-power source ordering for NVIDIA Grace:
Linux ACPI ``power_meter`` socket totals first, then DCGM CPU entity field 1130.
It runs on the host (not in the model container) so sysfs and the host DCGM
installation remain visible.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import os
import signal
import socket
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from srtctl.core.power.cpu_rails import (
    COMPONENT_RAIL_KINDS,
    DCGM_KIND,
    RAIL_COLUMN_NAMES,
    TOTAL_KIND,
    classify_acpi_label,
    sensor_name,
)
from srtctl.core.power.cpu_sample import CpuSample, RailReading, node_total_watts, pivot_socket_samples

CPU_POWER_FIELD_ID = 1130
DCGM_PYTHON_BINDING_DIRS = (
    Path("/usr/share/datacenter-gpu-manager-4/bindings/python3"),
    Path("/usr/local/dcgm/bindings/python3"),
)


class CpuUtilizationField(NamedTuple):
    """One optional per-socket utilization column and the DCGM CPU entity field that feeds it."""

    column: str
    field_id: int
    field_name: str


# DCGM CPU entity utilization fields (same entities as power field 1130). DCGM
# reports them as fractions of the socket's CPU time; ACPI has no equivalent,
# so ACPI rows leave these columns blank. Tuple order defines the trailing
# SAMPLES_HEADER columns.
CPU_UTILIZATION_FIELDS: tuple[CpuUtilizationField, ...] = (
    CpuUtilizationField("cpu_util_total", 1100, "DCGM_FI_DEV_CPU_UTIL_TOTAL"),
    CpuUtilizationField("cpu_util_user", 1101, "DCGM_FI_DEV_CPU_UTIL_USER"),
    CpuUtilizationField("cpu_util_nice", 1102, "DCGM_FI_DEV_CPU_UTIL_NICE"),
    CpuUtilizationField("cpu_util_sys", 1103, "DCGM_FI_DEV_CPU_UTIL_SYS"),
    CpuUtilizationField("cpu_util_irq", 1104, "DCGM_FI_DEV_CPU_UTIL_IRQ"),
)
UTILIZATION_COLUMNS = tuple(field.column for field in CPU_UTILIZATION_FIELDS)
_UTILIZATION_COLUMN_BY_FIELD_ID = {field.field_id: field.column for field in CPU_UTILIZATION_FIELDS}

# v2 added timestamp_local; v3 appended the optional utilization columns; v4
# pivots to one row per socket with the ACPI component rails as columns
# (power_w is the socket "total" envelope, or the DCGM value). v3 wrote one
# row per rail, which left readers to work out which rows were one socket.
SAMPLES_SCHEMA_VERSION = 4
SAMPLES_HEADER_V2 = (
    "schema_version",
    "timestamp_unix",
    "timestamp_local",
    "hostname",
    "source",
    "sensor",
    "socket_id",
    "power_w",
    "total_power_w",
)
SAMPLES_HEADER_V3 = (*SAMPLES_HEADER_V2, *UTILIZATION_COLUMNS)
SAMPLES_HEADER = (
    "schema_version",
    "timestamp_unix",
    "timestamp_local",
    "hostname",
    "source",
    "sensor",  # the sensor that fed power_w (provenance only)
    "socket_id",
    "power_w",  # ACPI: the socket "total" envelope; DCGM: field 1130
    *RAIL_COLUMN_NAMES,  # cpu_rail_w, soc_w, dram_w -- ACPI only, blank for DCGM
    "total_power_w",  # node aggregate: sum of power_w over sockets
    *UTILIZATION_COLUMNS,
)


def format_local_timestamp(timestamp: float) -> str:
    """ISO 8601 local wall-clock time (with UTC offset) for a ``time.time()`` value."""
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


class CpuPowerSourceUnavailable(RuntimeError):
    """Raised when a CPU power source cannot be used on the current host."""


class CpuPowerReader(ABC):
    """CPU power source interface.

    Readers only *classify*: ``read_watts`` yields raw sensor values and
    ``classify_readings`` tags each with its socket and rail kind. The pivot
    into one :class:`CpuSample` per socket and the node aggregate are shared
    (``cpu_sample``), so neither reader decides what a socket's power is.
    """

    source_name: str

    @abstractmethod
    def read_watts(self) -> dict[str, float | None]:
        """Return watts by stable sensor name."""

    @abstractmethod
    def classify_readings(self, readings: dict[str, float | None]) -> list[RailReading]:
        """Tag each non-None reading with its socket and rail kind."""

    def socket_samples(self, readings: dict[str, float | None]) -> tuple[CpuSample, ...]:
        """One :class:`CpuSample` per socket that has its primary rail (shared pivot)."""
        return pivot_socket_samples(self.source_name, self.classify_readings(readings))

    def aggregate_watts(self, readings: dict[str, float | None]) -> float | None:
        """Node total: the sum of every socket's primary rail, never a component rail."""
        return node_total_watts(self.socket_samples(readings))

    def read_utilization(self) -> dict[int, dict[str, float]]:
        """Return utilization columns by socket id from the most recent ``read_watts``.

        Providers without utilization data (ACPI) return an empty mapping and
        the sample rows leave the columns blank.
        """
        return {}

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return source and sensor provenance."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""


class AcpiPowerMeterReader(CpuPowerReader):
    """Read ACPI ``power_meter`` socket totals and reference component rails.

    Label classification lives in :mod:`srtctl.core.power.cpu_rails` so the
    exporter, the head-node scraper, and this reader agree on every name.
    """

    source_name = "acpi"

    def __init__(self, hwmon_root: Path = Path("/sys/class/hwmon")) -> None:
        sensors: list[dict[str, Any]] = []
        available_domains: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        seen_domains: set[tuple[int, str]] = set()
        for hwmon_dir in sorted(hwmon_root.glob("hwmon*")):
            try:
                if (hwmon_dir / "name").read_text().strip() != "power_meter":
                    continue
            except OSError:
                continue
            for attribute_root in (hwmon_dir / "device", hwmon_dir):
                channels = {
                    path.stem.removesuffix(suffix)
                    for suffix in ("_average", "_input")
                    for path in attribute_root.glob(f"power*{suffix}")
                }
                for channel in sorted(channels):
                    average_path = attribute_root / f"{channel}_average"
                    input_path = attribute_root / f"{channel}_input"
                    value_path = average_path if average_path.is_file() else input_path
                    if not value_path.is_file():
                        continue
                    try:
                        identity = str(value_path.resolve())
                    except OSError:
                        identity = str(value_path)
                    if identity in seen_paths:
                        continue
                    seen_paths.add(identity)
                    domain = _read_optional_text(attribute_root / f"{channel}_oem_info")
                    label = _read_optional_text(attribute_root / f"{channel}_label")
                    display_name = domain or label or channel
                    available_domains.append({"name": display_name, "path": str(value_path)})
                    classified = classify_acpi_label(display_name)
                    if classified is None:
                        continue
                    domain_kind, socket_id = classified
                    identity_key = (socket_id, domain_kind)
                    if identity_key in seen_domains:
                        continue
                    seen_domains.add(identity_key)
                    sensors.append(
                        {
                            "name": sensor_name(domain_kind, socket_id),
                            "socket_id": socket_id,
                            "domain_kind": domain_kind,
                            "domain": display_name,
                            "label": label,
                            "path": value_path,
                            "accuracy_path": attribute_root / f"{channel}_accuracy",
                            "interval_path": attribute_root / f"{channel}_average_interval",
                        }
                    )
        self._sensors = sorted(sensors, key=lambda sensor: sensor["socket_id"])
        self._available_domains = available_domains
        if not any(sensor["domain_kind"] == TOTAL_KIND for sensor in self._sensors):
            domains = ", ".join(domain["name"] for domain in available_domains)
            suffix = f"; available domains: {domains}" if domains else ""
            raise CpuPowerSourceUnavailable(f"no ACPI socket-total power_meter channels under {hwmon_root}{suffix}")

    def read_watts(self) -> dict[str, float | None]:
        readings: dict[str, float | None] = {}
        for sensor in self._sensors:
            try:
                watts = float(sensor["path"].read_text().strip()) / 1_000_000.0
                readings[sensor["name"]] = watts if math.isfinite(watts) and watts >= 0 else None
            except (OSError, ValueError):
                readings[sensor["name"]] = None
        return readings

    def classify_readings(self, readings: dict[str, float | None]) -> list[RailReading]:
        return [
            RailReading(sensor["socket_id"], sensor["domain_kind"], sensor["name"], watts)
            for sensor in self._sensors
            if (watts := readings.get(sensor["name"])) is not None
        ]

    def aggregate_watts(self, readings: dict[str, float | None]) -> float | None:
        # Stricter than the shared default: a node total is only published
        # when EVERY discovered socket envelope read back, so a partial
        # total can never masquerade as the node's power.
        totals = [readings.get(sensor["name"]) for sensor in self._sensors if sensor["domain_kind"] == TOTAL_KIND]
        if not totals or any(watts is None for watts in totals):
            return None
        return super().aggregate_watts(readings)

    def metadata(self) -> dict[str, Any]:
        sensors: list[dict[str, Any]] = []
        for sensor in self._sensors:
            interval_ms: int | None = None
            with contextlib.suppress(OSError, ValueError):
                interval_ms = int(sensor["interval_path"].read_text().strip())
            sensors.append(
                {
                    "name": sensor["name"],
                    "socket_id": sensor["socket_id"],
                    "domain_kind": sensor["domain_kind"],
                    "domain": sensor["domain"],
                    "label": sensor["label"],
                    "path": str(sensor["path"]),
                    "accuracy": _read_optional_text(sensor["accuracy_path"]),
                    "average_interval_ms": interval_ms,
                }
            )
        return {
            "source": self.source_name,
            "driver": "Linux ACPI power_meter hwmon",
            "semantics": "firmware-reported average CPU-side socket total and component power in watts",
            "sensors": sensors,
            "available_power_domains": self._available_domains,
            "total_method": (
                "sum of recognized CPU-side socket-total domains only; component rails are reference breakdowns"
            ),
            "aggregate_scope": "cpu_side_socket_total",
        }

    def close(self) -> None:
        """ACPI sysfs reads hold no persistent resources."""


def _add_standard_dcgm_binding_path() -> Path | None:
    """Expose DCGM's distro-installed Python bindings when not site-packaged."""
    for binding_dir in DCGM_PYTHON_BINDING_DIRS:
        binding_path = str(binding_dir)
        if binding_path in sys.path:
            return binding_dir
        if (binding_dir / "dcgm_agent.py").is_file():
            sys.path.insert(0, binding_path)
            return binding_dir
    return None


class DcgmCpuPowerReader(CpuPowerReader):
    """Read per-Grace-CPU instantaneous power through DCGM field 1130."""

    source_name = "dcgm"

    def __init__(self) -> None:
        self._handle: Any = None
        self._group: Any = None
        self._field_group: Any = None
        _add_standard_dcgm_binding_path()
        try:
            dcgm_agent = importlib.import_module("dcgm_agent")
            dcgm_fields = importlib.import_module("dcgm_fields")
            dcgm_structs = importlib.import_module("dcgm_structs")
            pydcgm = importlib.import_module("pydcgm")
        except ImportError as exc:
            raise CpuPowerSourceUnavailable(f"DCGM Python bindings unavailable: {exc}") from exc
        self._agent = dcgm_agent
        self._fields = dcgm_fields
        self._structs = dcgm_structs
        self._field_ids = [CPU_POWER_FIELD_ID, *(field.field_id for field in CPU_UTILIZATION_FIELDS)]
        self._last_utilization: dict[int, dict[str, float]] = {}
        try:
            self._handle = pydcgm.DcgmHandle(ipAddress=None)
            flags = getattr(dcgm_structs, "DCGM_GEGE_FLAG_ONLY_SUPPORTED", 0)
            self._cpu_ids = list(
                dcgm_agent.dcgmGetEntityGroupEntities(self._handle.handle, dcgm_fields.DCGM_FE_CPU, flags)
            )
        except Exception as exc:
            raise CpuPowerSourceUnavailable(f"cannot enumerate DCGM CPU entities: {exc}") from exc
        if not self._cpu_ids:
            self.close()
            raise CpuPowerSourceUnavailable("DCGM reported no supported CPU entities")
        self._entities = []
        for cpu_id in self._cpu_ids:
            entity = dcgm_structs.c_dcgmGroupEntityPair_t()
            entity.entityGroupId = dcgm_fields.DCGM_FE_CPU
            entity.entityId = cpu_id
            self._entities.append(entity)
        try:
            unique_suffix = f"{os.getpid()}_{time.time_ns()}"
            self._group = pydcgm.DcgmGroup(
                self._handle,
                groupName=f"srtctl_cpu_power_entities_{unique_suffix}",
                groupType=dcgm_structs.DCGM_GROUP_EMPTY,
            )
            for cpu_id in self._cpu_ids:
                self._group.AddEntity(dcgm_fields.DCGM_FE_CPU, cpu_id)
            self._field_group = pydcgm.DcgmFieldGroup(
                self._handle,
                name=f"srtctl_cpu_power_fields_{unique_suffix}",
                fieldIds=list(self._field_ids),
            )
            self._group.samples.WatchFields(
                self._field_group,
                100_000,
                60.0,
                600,
            )
            # A live-data query does not implicitly install a DCGM watch. Force
            # the first watched update so the initial collector sample is real.
            self._agent.dcgmUpdateAllFields(self._handle.handle, True)
        except Exception as exc:
            self.close()
            raise CpuPowerSourceUnavailable(f"cannot watch DCGM CPU power field: {exc}") from exc

    def read_watts(self) -> dict[str, float | None]:
        readings = {sensor_name(DCGM_KIND, cpu_id): None for cpu_id in self._cpu_ids}
        utilization: dict[int, dict[str, float]] = {}
        try:
            values = self._agent.dcgmEntitiesGetLatestValues(
                self._handle.handle,
                self._entities,
                list(self._field_ids),
                0,
            )
        except Exception as exc:
            self._last_utilization = {}
            raise CpuPowerSourceUnavailable(f"DCGM CPU power read failed: {exc}") from exc
        for value in values:
            if value.status != getattr(self._structs, "DCGM_ST_OK", 0):
                continue
            field_id = getattr(value, "fieldId", CPU_POWER_FIELD_ID)
            number = float(value.value.dbl)
            if not math.isfinite(number):
                continue
            if field_id == CPU_POWER_FIELD_ID:
                key = sensor_name(DCGM_KIND, int(value.entityId))
                if key in readings and number > 0:
                    readings[key] = number
                continue
            column = _UTILIZATION_COLUMN_BY_FIELD_ID.get(field_id)
            if column is not None and value.entityId in self._cpu_ids:
                utilization.setdefault(int(value.entityId), {})[column] = number
        self._last_utilization = utilization
        return readings

    def read_utilization(self) -> dict[int, dict[str, float]]:
        return self._last_utilization

    def classify_readings(self, readings: dict[str, float | None]) -> list[RailReading]:
        # One already-aggregated value per socket; no component rails. Whether
        # field 1130 corresponds to the ACPI cpu_rail or the total envelope is
        # unverified, so nothing here claims a rail kind beyond DCGM_KIND.
        return [
            RailReading(int(cpu_id), DCGM_KIND, name, watts)
            for cpu_id in sorted(self._cpu_ids)
            if (watts := readings.get(name := sensor_name(DCGM_KIND, cpu_id))) is not None
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "field_id": CPU_POWER_FIELD_ID,
            "field_name": "DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT",
            "semantics": "instantaneous power usage in watts",
            "sensors": [{"name": sensor_name(DCGM_KIND, cpu_id), "cpu_entity_id": cpu_id} for cpu_id in self._cpu_ids],
            "total_method": "sum of available DCGM CPU entities",
            "aggregate_scope": "cpu_rail_only",
            "utilization_fields": [
                {"column": field.column, "field_id": field.field_id, "field_name": field.field_name}
                for field in CPU_UTILIZATION_FIELDS
            ],
            "utilization_unit": "fraction of socket CPU time (0-1) as reported by DCGM",
        }

    def close(self) -> None:
        group = getattr(self, "_group", None)
        field_group = getattr(self, "_field_group", None)
        if group is not None and field_group is not None:
            with contextlib.suppress(Exception):
                group.samples.UnwatchFields(field_group)
        if field_group is not None:
            with contextlib.suppress(Exception):
                field_group.Delete()
            self._field_group = None
        if group is not None:
            with contextlib.suppress(Exception):
                group.Delete()
            self._group = None
        handle = getattr(self, "_handle", None)
        if handle is not None:
            with contextlib.suppress(Exception):  # DCGM shutdown must not mask completed samples
                handle.Shutdown()
            self._handle = None


def create_reader(source: str) -> CpuPowerReader:
    """Create the requested reader, using BTK's Grace ordering for ``auto``."""
    factories = {"acpi": AcpiPowerMeterReader, "dcgm": DcgmCpuPowerReader}
    if source != "auto":
        try:
            return factories[source]()
        except KeyError as exc:
            raise ValueError(f"unsupported CPU power source: {source}") from exc
    errors: list[str] = []
    for name in ("acpi", "dcgm"):
        try:
            return factories[name]()
        except CpuPowerSourceUnavailable as exc:
            errors.append(f"{name}: {exc}")
    raise CpuPowerSourceUnavailable("; ".join(errors))


def collect(*, output_dir: Path, ready_dir: Path, source: str, interval_seconds: float) -> int:
    """Collect until SIGTERM/SIGINT and leave node-local auditable artifacts."""
    os.umask(0o002)
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_dir.mkdir(parents=True, exist_ok=True)
    hostname = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".", 1)[0]
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        reader = create_reader(source)
    except (CpuPowerSourceUnavailable, ValueError) as exc:
        _atomic_json(ready_dir / f"{hostname}.error.json", {"hostname": hostname, "error": str(exc)})
        return 2

    samples_path = output_dir / f"{hostname}.csv"
    metadata_path = output_dir / f"{hostname}.metadata.json"
    ready_path = ready_dir / f"{hostname}.ready.json"
    started_at = time.time()
    sample_count = 0
    read_failures = 0
    metadata = reader.metadata()
    metadata.update(
        {
            "schema_version": SAMPLES_SCHEMA_VERSION,
            "hostname": hostname,
            "requested_source": source,
            "sample_interval_seconds": interval_seconds,
            "started_at_unix": started_at,
        }
    )
    _atomic_json(metadata_path, metadata)
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLES_HEADER)
        handle.flush()
        _atomic_json(ready_path, {"hostname": hostname, "source": reader.source_name, "ready_at_unix": time.time()})
        next_sample = time.monotonic()
        while not stop:
            timestamp = time.time()
            timestamp_local = format_local_timestamp(timestamp)
            try:
                readings = reader.read_watts()
            except CpuPowerSourceUnavailable:
                read_failures += 1
                readings = {}
            total = reader.aggregate_watts(readings)
            utilization = reader.read_utilization()
            for sample in reader.socket_samples(readings):
                socket_utilization = utilization.get(sample.socket_id, {})
                rails = sample.rails
                writer.writerow(
                    (
                        SAMPLES_SCHEMA_VERSION,
                        repr(timestamp),
                        timestamp_local,
                        hostname,
                        reader.source_name,
                        sample.sensor,
                        sample.socket_id,
                        repr(sample.power_w),
                        *(repr(rails[kind]) if kind in rails else "" for kind in COMPONENT_RAIL_KINDS),
                        "" if total is None else repr(total),
                        *(
                            repr(socket_utilization[column]) if column in socket_utilization else ""
                            for column in UTILIZATION_COLUMNS
                        ),
                    )
                )
                sample_count += 1
            handle.flush()
            next_sample += interval_seconds
            time.sleep(max(0.0, next_sample - time.monotonic()))
    reader.close()
    metadata.update(
        {
            "ended_at_unix": time.time(),
            "sample_row_count": sample_count,
            "read_failure_count": read_failures,
            "status": "complete" if sample_count else "failed",
        }
    )
    _atomic_json(metadata_path, metadata)
    return 0 if sample_count else 3


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-dir", type=Path, required=True)
    parser.add_argument("--source", choices=("auto", "acpi", "dcgm"), default="auto")
    parser.add_argument("--interval-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0:
        parser.error("--interval-seconds must be finite and positive")
    return collect(
        output_dir=args.output_dir,
        ready_dir=args.ready_dir,
        source=args.source,
        interval_seconds=args.interval_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
