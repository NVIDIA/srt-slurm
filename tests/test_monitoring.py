# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for built-in GPU performance monitoring configuration and benchmark stage integration."""

import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.schema import (
    BenchmarkConfig,
    ModelConfig,
    MonitoringConfig,
    ResourceConfig,
    SrtConfig,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestMonitoringConfig:
    def test_defaults(self):
        m = MonitoringConfig()
        assert m.enabled is True
        assert m.sample_interval == 1.0

    def test_disabled(self):
        m = MonitoringConfig(enabled=False)
        assert m.enabled is False

    def test_custom_interval(self):
        m = MonitoringConfig(sample_interval=2.5)
        assert m.sample_interval == 2.5

    def test_yaml_roundtrip(self):
        schema = MonitoringConfig.Schema()
        result = schema.load({"enabled": True, "sample_interval": 0.5})
        assert result.enabled is True
        assert result.sample_interval == 0.5


class TestSrtConfigMonitoringField:
    def _base_config(self, **kwargs) -> SrtConfig:
        return SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            benchmark=BenchmarkConfig(type="manual"),
            **kwargs,
        )

    def test_monitoring_defaults_to_none(self):
        assert self._base_config().monitoring is None

    def test_monitoring_accepted(self):
        config = self._base_config(monitoring=MonitoringConfig())
        assert config.monitoring is not None
        assert config.monitoring.enabled is True

    def test_monitoring_disabled(self):
        config = self._base_config(monitoring=MonitoringConfig(enabled=False))
        assert config.monitoring.enabled is False


# ---------------------------------------------------------------------------
# BenchmarkStageMixin._start_perf_monitor / _stop_perf_monitor
# ---------------------------------------------------------------------------


def _make_orchestrator(monitoring=None):
    from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

    config = SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image.sqsh", precision="fp8"),
        resources=ResourceConfig(gpu_type="h100", agg_nodes=2, agg_workers=2, gpus_per_node=8),
        benchmark=BenchmarkConfig(type="sa-bench"),
        monitoring=monitoring,
    )

    runtime = MagicMock()
    runtime.nodes.head = "node0"
    runtime.nodes.worker = ("node0", "node1", "node2")
    runtime.container_image = Path("/image.sqsh")
    runtime.container_mounts = {}
    runtime.log_dir = Path("/logs/12345")

    class Orchestrator(BenchmarkStageMixin):
        @property
        def endpoints(self):
            return []

        @property
        def backend_processes(self):
            return []

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = config
    orch.runtime = runtime
    return orch


class TestStartPerfMonitor:
    def test_returns_empty_when_monitoring_is_none(self):
        assert _make_orchestrator(monitoring=None)._start_perf_monitor() == []

    def test_returns_empty_when_disabled(self):
        assert _make_orchestrator(monitoring=MonitoringConfig(enabled=False))._start_perf_monitor() == []

    def test_starts_one_proc_per_worker_node_including_head(self):
        """Starts perfmon on all worker nodes including head (node0)."""
        orch = _make_orchestrator(monitoring=MonitoringConfig())
        mock_proc = MagicMock()

        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc) as mock_srun:
            result = orch._start_perf_monitor()

        assert len(result) == 3
        nodes = [node for node, _ in result]
        assert "node0" in nodes
        assert "node1" in nodes
        assert "node2" in nodes

    def test_output_paths_keyed_by_hostname(self):
        orch = _make_orchestrator(monitoring=MonitoringConfig())
        mock_proc = MagicMock()

        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc) as mock_srun:
            orch._start_perf_monitor()

        all_cmds = [c.kwargs["command"] for c in mock_srun.call_args_list]
        for node in ("node0", "node1", "node2"):
            node_cmds = [cmd for cmd in all_cmds if any(node in arg for arg in cmd)]
            assert any(f"perf_samples_{node}.csv" in arg for arg in node_cmds[0])
            assert any(f"perf_summary_{node}.json" in arg for arg in node_cmds[0])

    def test_perfmon_script_mounted_in_container(self):
        orch = _make_orchestrator(monitoring=MonitoringConfig())
        mock_proc = MagicMock()

        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc) as mock_srun:
            orch._start_perf_monitor()

        for c in mock_srun.call_args_list:
            mounts = c.kwargs["container_mounts"]
            assert Path("/tmp/srt_perfmon.py") in mounts.values()

    def test_failed_node_is_skipped(self):
        orch = _make_orchestrator(monitoring=MonitoringConfig())
        mock_proc = MagicMock()

        def srun_side_effect(**kwargs):
            if kwargs["nodelist"] == ["node1"]:
                raise RuntimeError("srun failed")
            return mock_proc

        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", side_effect=srun_side_effect):
            result = orch._start_perf_monitor()

        assert len(result) == 2
        nodes = [node for node, _ in result]
        assert "node0" in nodes
        assert "node2" in nodes


class TestStopPerfMonitor:
    def test_noop_on_empty_list(self):
        _make_orchestrator()._stop_perf_monitor([])

    def test_sends_sigint_to_running_process(self):
        orch = _make_orchestrator()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0

        orch._stop_perf_monitor([("node1", mock_proc)])

        mock_proc.send_signal.assert_called_once_with(signal.SIGINT)
        mock_proc.wait.assert_called_once_with(timeout=30)

    def test_skips_already_exited_process(self):
        orch = _make_orchestrator()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        orch._stop_perf_monitor([("node1", mock_proc)])

        mock_proc.send_signal.assert_not_called()

    def test_kills_process_on_timeout(self):
        orch = _make_orchestrator()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="perfmon", timeout=30), None]

        orch._stop_perf_monitor([("node1", mock_proc)])

        mock_proc.kill.assert_called_once()

    def test_handles_process_lookup_error(self):
        orch = _make_orchestrator()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.send_signal.side_effect = ProcessLookupError

        orch._stop_perf_monitor([("node1", mock_proc)])  # must not raise

    def test_stops_all_nodes(self):
        orch = _make_orchestrator()
        procs = []
        for node in ("node1", "node2", "node3"):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.wait.return_value = 0
            procs.append((node, mock_proc))

        orch._stop_perf_monitor(procs)

        for _, mock_proc in procs:
            mock_proc.send_signal.assert_called_once_with(signal.SIGINT)
