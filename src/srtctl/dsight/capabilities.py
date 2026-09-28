# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Available views are a property of usable evidence, not engine configuration."""

from typing import Any


def capabilities(data: dict[str, Any]) -> dict[str, bool]:
    profiles = data["profiles"]
    metrics = [s for s in data["metrics"] if s["points"]]
    return {
        "requests": bool(data["requests"]),
        "request_breakdown": any(r["lifecycle"]["available"] for r in data["requests"]),
        "server_activity": bool(data.get("server_spans")),
        "metrics": bool(metrics),
        "worker_metrics": any(s.get("group") == "worker" for s in metrics),
        "hardware_metrics": any(s.get("group") == "hardware" for s in metrics),
        "nvtx": any(p["events"] for p in profiles),
        "cpu_samples": any((p.get("cpu") or {}).get("samples") for p in profiles),
        "nsight": any(p["events"] or (p.get("cpu") or {}).get("samples") for p in profiles),
        "iterations": bool(data["iterations"]),
    }
