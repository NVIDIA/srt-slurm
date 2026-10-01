# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery shared by the ACPI power reader and Python exporter."""

from pathlib import Path


def is_power_meter(hwmon_dir: Path) -> bool:
    """Accept class attributes and legacy drivers exposing only ``device/name``."""
    for name_path in (hwmon_dir / "name", hwmon_dir / "device" / "name"):
        try:
            if name_path.read_text().strip() == "power_meter":
                return True
        except OSError:
            continue
    return False
