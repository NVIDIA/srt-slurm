# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for gweperf monitoring configuration and benchmark stage integration."""

import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from srtctl.core.schema import (
    BenchmarkConfig,
    ModelConfig,
    MonitoringConfig,
    MonitoringFeaturesConfig,
    ResourceConfig,
    SrtConfig,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestMonitoringFeaturesConfig:
    def test_defaults(self):
        f = MonitoringFeaturesConfig()
        assert f.dcgm is False
        assert f.rapl is False
        assert f.ipmi is False
        assert f.cpu_pmu is False

    def test_dcgm_only(self):
        f = MonitoringFeaturesConfig(dcgm=True)
        assert f.dcgm is True
        assert f.rapl is False


class TestMonitoringConfig:
    def test_defaults(self):
        m = MonitoringConfig()
        assert m.enabled is True
        assert m.sample_interval == 1.0
        assert isinstance(m.features, MonitoringFeaturesConfig)

    def test_disabled(self):
        m = MonitoringConfig(enabled=False)
        assert m.enabled is False

    def test_custom_interval(self):
        m = MonitoringConfig(sample_interval=2.5)
        assert m.sample_interval == 2.5

    def test_yaml_roundtrip(self):
        """MonitoringConfig round-trips through marshmallow schema."""
        schema = MonitoringConfig.Schema()
        data = {"enabled": True, "sample_interval": 0.5, "features": {"dcgm": True}}
        result = schema.load(data)
        assert result.enabled is True
        assert result.sample_interval == 0.5
        assert result.features.dcgm is True


class TestSrtConfigMonitoringField:
    """SrtConfig accepts monitoring=None (backward compat) and a MonitoringConfig."""

    def _base_config(self, **kwargs) -> SrtConfig:
        return SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            benchmark=BenchmarkConfig(type="manual"),
            **kwargs,
        )

    def test_monitoring_defaults_to_none(self):
        config = self._base_config()
        assert config.monitoring is None

    def test_monitoring_accepted(self):
        config = self._base_config(monitoring=MonitoringConfig())
        assert config.monitoring is not None
        assert config.monitoring.enabled is True

    def test_monitoring_disabled(self):
        config = self._base_config(monitoring=MonitoringConfig(enabled=False))
        assert config.monitoring.enabled is False


# ---------------------------------------------------------------------------
# BenchmarkStageMixin._start_gweperf / _stop_gweperf
# ---------------------------------------------------------------------------


def _make_orchestrator(monitoring=None):
    """Build a minimal BenchmarkStageMixin instance for testing."""
    from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
    from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

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

    # Minimal concrete subclass
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


class TestStartGweperf:
    def test_returns_empty_when_monitoring_is_none(self):
        orch = _make_orchestrator(monitoring=None)
        assert orch._start_gweperf() == []

    def test_returns_empty_when_disabled(self):
        orch = _make_orchestrator(monitoring=MonitoringConfig(enabled=False))
        assert orch._start_gweperf() == []

    def test_returns_empty_when_gweperf_path_not_set(self):
        orch = _make_orchestrator(monitoring=MonitoringConfig())
        with patch("srtctl.cli.mixins.benchmark_stage.get_srtslurm_setting", return_value=None):
            result = orch._start_gweperf()
        assert result == []

    def test_returns_empty_when_gweperf_path_missing(self, tmp_path):
        orch = _make_orchestrator(monitoring=MonitoringConfig())
        missing = str(tmp_path / "nonexistent")
        with patch("srtctl.cli.mixins.benchmark_stage.get_srtslurm_setting", return_value=missing):
            result = orch._start_gweperf()
        assert result == []

    def test_starts_one_proc_per_worker_node_excluding_head(self, tmp_path):
        """Starts gweperf on node1 and node2 (not node0=head)."""
        gweperf_dir = tmp_path / "gweperf"
        gweperf_dir.mkdir()

        orch = _make_orchestrator(monitoring=MonitoringConfig())
        mock_proc = MagicMock()

        with (
            patch("srtctl.cli.mixins.benchmark_stage.get_srtslurm_setting", return_value=str(gweperf_dir)),
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc) as mock_srun,
        ):
            result = orch._start_gweperf()

        assert len(result) == 2
        nodes = [node for node, _ in result]
        assert "node0" not in nodes
        assert "node1" in nodes
        assert "node2" in nodes

    def test_output_paths_keyed_by_hostname(self, tmp_path):
        """perf_samples and perf_summary files are named with the node hostname."""
        gweperf_dir = tmp_path / "gweperf"
        gweperf_dir.mkdir()

        orch = _make_orchestrator(monitoring=MonitoringConfig())
        mock_proc = MagicMock()

        with (
            patch("srtctl.cli.mixins.benchmark_stage.get_srtslurm_setting", return_value=str(gweperf_dir)),
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc) as mock_srun,
        ):
            orch._start_gweperf()

        all_cmds = [c.kwargs["command"] for c in mock_srun.call_args_list]
        for node in ("node1", "node2"):
            node_cmds = [cmd for cmd in all_cmds if any(node in arg for arg in cmd)]
            assert any(f"perf_samples_{node}.csv" in arg for arg in node_cmds[0])
            assert any(f"perf_summary_{node}.json" in arg for arg in node_cmds[0])

    def test_feature_flags_appended_to_command(self, tmp_path):
        """dcgm/rapl/ipmi/cpu_pmu flags are passed to gweperfmon.py."""
        gweperf_dir = tmp_path / "gweperf"
        gweperf_dir.mkdir()

        features = MonitoringFeaturesConfig(dcgm=True, rapl=True)
        orch = _make_orchestrator(monitoring=MonitoringConfig(features=features))
        mock_proc = MagicMock()

        with (
            patch("srtctl.cli.mixins.benchmark_stage.get_srtslurm_setting", return_value=str(gweperf_dir)),
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc) as mock_srun,
        ):
            orch._start_gweperf()

        for c in mock_srun.call_args_list:
            cmd = c.kwargs["command"]
            assert "--dcgm" in cmd
            assert "--rapl" in cmd
            assert "--ipmi" not in cmd
            assert "--cpuPmu" not in cmd

    def test_failed_node_is_skipped(self, tmp_path):
        """If start_srun_process raises on one node, others still start."""
        gweperf_dir = tmp_path / "gweperf"
        gweperf_dir.mkdir()

        orch = _make_orchestrator(monitoring=MonitoringConfig())
        mock_proc = MagicMock()

        def srun_side_effect(**kwargs):
            if kwargs["nodelist"] == ["node1"]:
                raise RuntimeError("srun failed")
            return mock_proc

        with (
            patch("srtctl.cli.mixins.benchmark_stage.get_srtslurm_setting", return_value=str(gweperf_dir)),
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", side_effect=srun_side_effect),
        ):
            result = orch._start_gweperf()

        assert len(result) == 1
        assert result[0][0] == "node2"

    def test_gweperf_mounted_in_container(self, tmp_path):
        """gweperf directory is added to container mounts."""
        gweperf_dir = tmp_path / "gweperf"
        gweperf_dir.mkdir()

        orch = _make_orchestrator(monitoring=MonitoringConfig())
        mock_proc = MagicMock()

        with (
            patch("srtctl.cli.mixins.benchmark_stage.get_srtslurm_setting", return_value=str(gweperf_dir)),
            patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc) as mock_srun,
        ):
            orch._start_gweperf()

        for c in mock_srun.call_args_list:
            mounts = c.kwargs["container_mounts"]
            assert Path("/gweperf") in mounts.values()


class TestStopGweperf:
    def test_noop_on_empty_list(self):
        orch = _make_orchestrator()
        orch._stop_gweperf([])  # should not raise

    def test_sends_sigint_to_running_process(self):
        orch = _make_orchestrator()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_proc.wait.return_value = 0

        orch._stop_gweperf([("node1", mock_proc)])

        mock_proc.send_signal.assert_called_once_with(signal.SIGINT)
        mock_proc.wait.assert_called_once_with(timeout=30)

    def test_skips_already_exited_process(self):
        orch = _make_orchestrator()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already done
        mock_proc.returncode = 0

        orch._stop_gweperf([("node1", mock_proc)])

        mock_proc.send_signal.assert_not_called()

    def test_kills_process_on_timeout(self):
        orch = _make_orchestrator()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # First call (with timeout=) raises; second call (bare wait after kill) returns normally
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="gweperf", timeout=30), None]

        orch._stop_gweperf([("node1", mock_proc)])

        mock_proc.kill.assert_called_once()

    def test_handles_process_lookup_error(self):
        """Process vanished between poll() and send_signal() — should not raise."""
        orch = _make_orchestrator()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.send_signal.side_effect = ProcessLookupError

        orch._stop_gweperf([("node1", mock_proc)])  # must not raise

    def test_stops_all_nodes(self):
        orch = _make_orchestrator()
        procs = []
        for node in ("node1", "node2", "node3"):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.wait.return_value = 0
            procs.append((node, mock_proc))

        orch._stop_gweperf(procs)

        for _, mock_proc in procs:
            mock_proc.send_signal.assert_called_once_with(signal.SIGINT)
