# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for best-effort live metrics startup."""

import threading

from srtctl.analysis.live_metrics import try_start_snapshotter


def test_try_start_snapshotter_ignores_null_telemetry(monkeypatch, tmp_path):
    """Cluster configs may dump omitted telemetry as null."""
    monkeypatch.setattr("srtctl.core.config.load_cluster_config", lambda: {"telemetry": None})

    assert try_start_snapshotter(tmp_path, threading.Event()) is None


def test_try_start_snapshotter_ignores_null_live_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr("srtctl.core.config.load_cluster_config", lambda: {"telemetry": {"live_metrics": None}})

    assert try_start_snapshotter(tmp_path, threading.Event()) is None
