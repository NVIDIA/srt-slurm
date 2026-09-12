# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin


class _Runtime:
    def __init__(self, log_dir):
        self.log_dir = log_dir


class Harness(PostProcessStageMixin):
    def __init__(self, log_dir):
        self.runtime = _Runtime(log_dir)


def test_start_then_finalize_round_trips(tmp_path):
    harness = Harness(tmp_path)

    harness.start_incremental_power_report()
    harness.finalize_incremental_power_report()  # must not raise


def test_finalize_without_start_is_safe(tmp_path):
    Harness(tmp_path).finalize_incremental_power_report()  # must not raise


@patch("srtctl.analysis.incremental_power.IncrementalPowerWatcher.start", side_effect=RuntimeError("nope"))
def test_a_start_failure_is_absorbed(_mock_start, tmp_path):
    harness = Harness(tmp_path)

    harness.start_incremental_power_report()  # must not raise
    assert not hasattr(harness, "_incremental_power_watcher")
    harness.finalize_incremental_power_report()  # must not raise
