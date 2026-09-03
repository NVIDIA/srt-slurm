# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side per-socket CPU power collector.

The implementation follows BTK's CPU-power source ordering for NVIDIA Grace:
Linux ACPI ``power_meter`` CPU rails first, then DCGM CPU entity field 1130.
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
import re
import signal
import socket
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

CPU_POWER_FIELD_ID = 1130
DCGM_PYTHON_BINDING_DIRS = (
    Path("/usr/share/datacenter-gpu-manager-4/bindings/python3"),
    Path("/usr/local/dcgm/bindings/python3"),
)
SAMPLES_HEADER = (
    "schema_version",
    "timestamp_unix",
    "hostname",
    "source",
    "sensor",
    "socket_id",
    "power_w",
    "total_power_w",
)


class CpuPowerSourceUnavailable(RuntimeError):
    """Raised when a CPU power source cannot be used on the current host."""


class CpuPowerReader(ABC):
    """CPU power source interface."""

    source_name: str

    @abstractmethod
    def read_watts(self) -> dict[str, float | None]:
        """Return watts by stable sensor name."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return source and sensor provenance."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""


class AcpiPowerMeterReader(CpuPowerReader):
    """Read Grace CPU-rail ACPI ``power_meter`` channels."""

    source_name = "acpi"
    _CPU_DOMAIN = re.compile(r"\bCPU\s+Power\s+Socket\s+(\d+)\b", re.IGNORECASE)

    def __init__(self, hwmon_root: Path = Path("/sys/class/hwmon")) -> None:
        sensors: list[dict[str, Any]] = []
        available_domains: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        seen_sockets: set[int] = set()
        for hwmon_dir in sorted(hwmon_root.glob("hwmon*")):
            try:
                if (hwmon_dir / "name").read_text().strip() != "power_meter":
                    continue
            except OSError:
                continue
            for attribute_root in (hwmon_dir / "device", hwmon_dir):
                for average_path in sorted(attribute_root.glob("power*_average")):
                    try:
                        identity = str(average_path.resolve())
                    except OSError:
                        identity = str(average_path)
                    if identity in seen_paths:
                        continue
                    seen_paths.add(identity)
                    channel = average_path.stem.removesuffix("_average")
                    input_path = attribute_root / f"{channel}_input"
                    value_path = average_path if average_path.is_file() else input_path
                    if not value_path.is_file():
                        continue
                    domain = _read_optional_text(attribute_root / f"{channel}_oem_info")
                    label = _read_optional_text(attribute_root / f"{channel}_label")
                    display_name = domain or label or channel
                    available_domains.append({"name": display_name, "path": str(value_path)})
                    match = self._CPU_DOMAIN.search(display_name)
                    if match is None:
                        continue
                    socket_id = int(match.group(1))
                    if socket_id in seen_sockets:
                        continue
                    seen_sockets.add(socket_id)
                    sensors.append(
                        {
                            "name": f"CPU{socket_id}:cpuPowerUsageW",
                            "socket_id": socket_id,
                            "domain": display_name,
                            "label": label,
                            "path": value_path,
                            "accuracy_path": attribute_root / f"{channel}_accuracy",
                            "interval_path": attribute_root / f"{channel}_average_interval",
                        }
                    )
        self._sensors = sorted(sensors, key=lambda sensor: sensor["socket_id"])
        self._available_domains = available_domains
        if not self._sensors:
            domains = ", ".join(domain["name"] for domain in available_domains)
            suffix = f"; available domains: {domains}" if domains else ""
            raise CpuPowerSourceUnavailable(
                f"no ACPI 'CPU Power Socket N' power_meter channels under {hwmon_root}{suffix}"
            )

    def read_watts(self) -> dict[str, float | None]:
        readings: dict[str, float | None] = {}
        for sensor in self._sensors:
            try:
                watts = float(sensor["path"].read_text().strip()) / 1_000_000.0
                readings[sensor["name"]] = watts if math.isfinite(watts) and watts >= 0 else None
            except (OSError, ValueError):
                readings[sensor["name"]] = None
        return readings

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
            "semantics": "firmware-reported average CPU rail power in watts",
            "sensors": sensors,
            "available_power_domains": self._available_domains,
            "total_method": "sum of CPU Power Socket N domains",
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
                fieldIds=[CPU_POWER_FIELD_ID],
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
        readings = {f"CPU{cpu_id}:cpuPowerUsageW": None for cpu_id in self._cpu_ids}
        try:
            values = self._agent.dcgmEntitiesGetLatestValues(
                self._handle.handle,
                self._entities,
                [CPU_POWER_FIELD_ID],
                0,
            )
        except Exception as exc:
            raise CpuPowerSourceUnavailable(f"DCGM CPU power read failed: {exc}") from exc
        for value in values:
            if value.status != getattr(self._structs, "DCGM_ST_OK", 0):
                continue
            watts = float(value.value.dbl)
            key = f"CPU{value.entityId}:cpuPowerUsageW"
            if key in readings and math.isfinite(watts) and watts > 0:
                readings[key] = watts
        return readings

    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "field_id": CPU_POWER_FIELD_ID,
            "field_name": "DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT",
            "semantics": "instantaneous power usage in watts",
            "sensors": [{"name": f"CPU{cpu_id}:cpuPowerUsageW", "cpu_entity_id": cpu_id} for cpu_id in self._cpu_ids],
            "total_method": "sum of available DCGM CPU entities",
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
            "schema_version": 1,
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
            try:
                readings = reader.read_watts()
            except CpuPowerSourceUnavailable:
                read_failures += 1
                readings = {}
            valid = {name: watts for name, watts in readings.items() if watts is not None}
            total = sum(valid.values()) if valid else None
            for sensor, watts in sorted(valid.items()):
                socket_match = re.match(r"CPU(\d+):", sensor)
                socket_id = int(socket_match.group(1)) if socket_match else ""
                writer.writerow(
                    (1, repr(timestamp), hostname, reader.source_name, sensor, socket_id, repr(watts), repr(total))
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
