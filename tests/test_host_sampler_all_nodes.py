# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the all-node host sampler: scheduler-contention fields, the
standalone per-node CLI mode, remote-launch planning, and the multi-file ingest."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from srtctl.analysis import host_sampler
from srtctl.analysis.host_sampler import _proc_sample, plan_remote_sampler_nodes


class TestProcSampleSchedulerFields:
    def test_own_process_has_scheduler_fields(self):
        d = _proc_sample(os.getpid())
        assert d is not None
        # /proc/<pid>/schedstat: cumulative on-CPU ns and run-queue-wait ns.
        assert isinstance(d.get("cpu_ns"), int)
        assert isinstance(d.get("run_delay_ns"), int)
        # se.nr_migrations from /proc/<pid>/sched.
        assert isinstance(d.get("nr_migrations"), int)
        # sched_getaffinity width: the direct pinning-state observable.
        assert isinstance(d.get("affinity_ncpus"), int) and d["affinity_ncpus"] >= 1

    def test_procs_running_blocked(self):
        running, blocked = host_sampler._procs_running_blocked()
        assert isinstance(running, int) and running >= 1
        assert isinstance(blocked, int) and blocked >= 0


class TestPlanRemoteSamplerNodes:
    def test_excludes_local_and_dedups(self):
        assert plan_remote_sampler_nodes(["n1", "n2", "n1", "n3"], "n2") == ["n1", "n3"]

    def test_fqdn_vs_short_hostname(self):
        assert plan_remote_sampler_nodes(["n1.cluster.local", "n2"], "n1") == ["n2"]
        assert plan_remote_sampler_nodes(["n1", "n2"], "n1.cluster.local") == ["n2"]

    def test_all_local(self):
        assert plan_remote_sampler_nodes(["n1", "n1"], "n1") == []


class TestStandaloneMode:
    def test_writes_per_node_file_and_exits(self, tmp_path):
        """python3 host_sampler.py --log-dir D --max-samples 2 writes host_samples_<host>.jsonl."""
        script = Path(host_sampler.__file__).resolve()
        r = subprocess.run(
            [sys.executable, str(script), "--log-dir", str(tmp_path), "--interval", "1", "--max-samples", "2"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r.returncode == 0, r.stderr
        out = tmp_path / f"host_samples_{os.uname().nodename}.jsonl"
        assert out.exists()
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(rows) >= 2
        assert rows[0]["host"] == os.uname().nodename
        assert "t_mono" in rows[0] and "procs_running" in rows[0]


class TestRemoteLaunch:
    def _mixin(self, worker_nodes):
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        m = BenchmarkStageMixin.__new__(BenchmarkStageMixin)
        runtime = MagicMock()
        runtime.nodes.head = worker_nodes[0]
        runtime.nodes.bench = worker_nodes[0]
        runtime.nodes.infra = worker_nodes[0]
        runtime.nodes.worker = tuple(worker_nodes)
        runtime.nodes.het_group_for.return_value = None
        runtime.log_dir = Path("/tmp/logs")
        runtime.srun_options = {}
        m.runtime = runtime
        return m

    def test_launches_on_all_nodes_except_local(self):
        local = os.uname().nodename
        mixin = self._mixin([local, "nodeB", "nodeC"])
        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process") as srun:
            srun.return_value = MagicMock()
            procs = mixin._start_remote_host_samplers()
        assert len(procs) == 1
        kwargs = srun.call_args.kwargs
        assert kwargs["nodelist"] == ["nodeB", "nodeC"]
        assert kwargs["nodes"] == 2 and kwargs["ntasks"] == 2
        assert kwargs["command"][0] == "python3"
        assert kwargs["command"][1].endswith("host_sampler.py")
        assert "--log-dir" in kwargs["command"]

    def test_no_launch_when_only_local(self):
        mixin = self._mixin([os.uname().nodename])
        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process") as srun:
            procs = mixin._start_remote_host_samplers()
        assert procs == [] and srun.call_count == 0


def _rows():
    def proc(pid, cpu_j, invol, delay_ns, migr, aff):
        return {"pid": pid, "name": "dynamo", "cpu_jiffies": cpu_j, "ctx_invol": invol,
                "run_delay_ns": delay_ns, "nr_migrations": migr, "affinity_ncpus": aff,
                "rss_kb": 100, "threads": 4, "open_fds": 10}

    return [
        {"t": 100.0, "host": "nodeB", "cpu_busy_jiffies": 1000, "cpu_total_jiffies": 10000,
         "procs_running": 5, "procs_blocked": 0, "mem": {"MemTotal": 100, "MemAvailable": 50},
         "established_conns": 3, "fd_limit": 1024, "procs": [proc(7, 100, 10, 1_000_000_000, 50, 144)]},
        {"t": 102.0, "host": "nodeB", "cpu_busy_jiffies": 1100, "cpu_total_jiffies": 10200,
         "procs_running": 8, "procs_blocked": 1, "mem": {"MemTotal": 100, "MemAvailable": 40},
         "established_conns": 4, "fd_limit": 1024, "procs": [proc(7, 140, 30, 1_200_000_000, 60, 144)]},
    ]


class TestIngestSchedulerSeries:
    def test_rates_from_rows(self):
        from src.ingest.ingest import _host_series_from_rows

        s = _host_series_from_rows(_rows())
        assert s is not None and s["host"] == "nodeB"
        p = s["procs"]["dynamo:7"]
        # 200ms of run-queue delta over 2s -> 100 ms/s
        assert p["run_delay_ms_per_s"] == [[102.0, 100.0]]
        # 10 migrations over 2s -> 5/s
        assert p["migrations_rate"] == [[102.0, 5.0]]
        assert p["affinity_ncpus"] == [[102.0, 144]]
        assert s["procs_runnable"] == [[102.0, 8]]
        assert s["procs_blocked"] == [[102.0, 1]]

    def test_rates_prefer_monotonic_clock(self):
        """A forward NTP step in wall-clock t must not deflate the rates."""
        from src.ingest.ingest import _host_series_from_rows

        rows = _rows()
        rows[0]["t_mono"], rows[1]["t_mono"] = 500.0, 502.0
        rows[1]["t"] = 112.0  # wall clock stepped +10s mid-window
        s = _host_series_from_rows(rows)
        # dt = 2s from t_mono, not 12s from t: 200ms delta -> 100 ms/s
        assert s["procs"]["dynamo:7"]["run_delay_ms_per_s"] == [[112.0, 100.0]]

    def test_multi_file_hosts_map(self, tmp_path):
        from src.ingest.ingest import run_host_samples

        run_dir, bundle = tmp_path / "run", tmp_path / "bundle"
        run_dir.mkdir()
        bundle.mkdir()
        local = _rows()
        for r in local:
            r["host"] = "orchestrator"
        (run_dir / "host_samples.jsonl").write_text("\n".join(json.dumps(r) for r in local))
        (run_dir / "host_samples_nodeB.jsonl").write_text("\n".join(json.dumps(r) for r in _rows()))
        out = run_host_samples(run_dir, bundle)
        assert out["host"] == "orchestrator"
        assert "nodeB" in out["hosts"]
        assert out["hosts"]["nodeB"]["procs"]["dynamo:7"]["run_delay_ms_per_s"] == [[102.0, 100.0]]
        assert json.loads((bundle / "host_series.json").read_text())["hosts"]["nodeB"]["host"] == "nodeB"

    def test_per_node_files_only(self, tmp_path):
        """Remote-node files alone (no orchestrator file) still produce a bundle."""
        from src.ingest.ingest import run_host_samples

        run_dir, bundle = tmp_path / "run", tmp_path / "bundle"
        run_dir.mkdir()
        bundle.mkdir()
        (run_dir / "host_samples_nodeB.jsonl").write_text("\n".join(json.dumps(r) for r in _rows()))
        out = run_host_samples(run_dir, bundle)
        assert out["hosts"]["nodeB"]["samples"] == 2
