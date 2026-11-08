# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for host_memory_check: each node's MemAvailable against the host pools its workers will pin."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig

GIB_KB = 1024**2  # /proc/meminfo reports kB


def _config(agg_args: dict, **check) -> SrtConfig:
    return SrtConfig.Schema().load(
        {
            "schema": 2,
            "name": "host-memory-test",
            "model": {"path": "/models/test", "container": "test.sqsh", "precision": "bf16"},
            "resources": {"gpu_type": "gb300", "gpus_per_node": 4},
            "engine": "sglang",
            "roles": {"agg": {"nodes": 2, "workers": 2, "gpus": 4, "args": agg_args}},
            "host_memory_check": check,
        }
    )


HICACHE = {"enable-hierarchical-cache": True, "hicache-size": 200}  # GB per rank


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0", "node1")),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/models/test"),
        container_image=Path("/img.sqsh"),
        gpus_per_node=4,
        network_interface=None,
        container_mounts={},
        environment={},
    )


def _meminfo_srun(available_gib: dict[str, float]):
    """Fake start_srun_process that writes each node's MemAvailable line to the step's output."""

    def srun(**kwargs):
        node = kwargs["nodelist"][0]
        Path(kwargs["output"]).write_text(f"MemAvailable:   {int(available_gib[node] * GIB_KB)} kB\n")
        proc = MagicMock()
        proc.wait.return_value = 0
        return proc

    return srun


def _check(config: SrtConfig, tmp_path: Path, available_gib: dict[str, float]) -> MagicMock:
    orch = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    with patch("srtctl.cli.do_sweep.start_srun_process", side_effect=_meminfo_srun(available_gib)) as srun:
        orch._check_host_memory()
    return srun


class TestPlannedBytes:
    def test_hicache_size_times_gpus_per_node(self, tmp_path):
        orch = SweepOrchestrator(config=_config(HICACHE), runtime=_runtime(tmp_path))
        assert orch._planned_host_bytes() == {"node0": 800 * 10**9, "node1": 800 * 10**9}

    @pytest.mark.parametrize(
        "args",
        [{}, {"hicache-size": 200}, {"enable_hierarchical_cache": True}],
        ids=["no-hicache", "size-without-hicache", "ratio-sized"],
    )
    def test_nothing_planned_skips_the_probe(self, tmp_path, args):
        assert not _check(_config(args), tmp_path, {}).called


class TestCheck:
    def test_probes_planned_nodes_on_the_bare_host(self, tmp_path):
        srun = _check(_config(HICACHE), tmp_path, {"node0": 840, "node1": 840})

        assert [c.kwargs["nodelist"] for c in srun.call_args_list] == [["node0"], ["node1"]]
        for call in srun.call_args_list:
            assert call.kwargs["container_image"] is None
            assert call.kwargs["command"] == ["grep", "MemAvailable", "/proc/meminfo"]

    def test_short_node_fails_with_its_numbers(self, tmp_path):
        # 800 GB = 745.1 GiB planned + 32 GiB headroom; node1 has only 760 GiB.
        with pytest.raises(RuntimeError, match=r"node1 has 760\.0 GiB MemAvailable; .* pin 745\.1 GiB") as exc:
            _check(_config(HICACHE), tmp_path, {"node0": 840, "node1": 760})
        assert "node0" not in str(exc.value)

    def test_headroom_is_configurable(self, tmp_path):
        _check(_config(HICACHE, headroom_gib=10), tmp_path, {"node0": 760, "node1": 760})

    def test_warn_continues(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="srtctl.cli.do_sweep"):
            _check(_config(HICACHE, on_shortfall="warn"), tmp_path, {"node0": 700, "node1": 840})
        assert "node0 has 700.0 GiB MemAvailable" in caplog.text
        assert "on_shortfall: warn" in caplog.text

    def test_disabled_skips_the_probe(self, tmp_path):
        assert not _check(_config(HICACHE, enabled=False), tmp_path, {}).called

    def test_unreadable_probe_warns_and_continues(self, tmp_path, caplog):
        proc = MagicMock()
        proc.wait.return_value = 1
        orch = SweepOrchestrator(config=_config(HICACHE), runtime=_runtime(tmp_path))
        with (
            patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc),
            caplog.at_level(logging.WARNING, logger="srtctl.cli.do_sweep"),
        ):
            orch._check_host_memory()
        assert "could not read MemAvailable on node0" in caplog.text
