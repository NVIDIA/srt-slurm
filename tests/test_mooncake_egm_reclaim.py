# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise overlapping allocation diagnosis without CUDA, Slurm or host pools."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tools import mooncake_egm_reclaim as diagnostic


def fixture_files(root: Path, files: dict[str, str]) -> None:
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)


def test_inventory_records_actual_cgroup_ancestors_vmstat_driver_and_pool_counts(tmp_path: Path) -> None:
    fixture_files(
        tmp_path,
        {
            "proc/self/cgroup": "0::/slurm/job_1/task_0\n",
            "proc/self/mountinfo": "10 9 0:28 /slurm/job_1 /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            "proc/vmstat": "pgscan_direct 123\ncompact_fail 4\n",
            "proc/pagetypeinfo": "Free pages count per migrate type at order\n",
            "proc/driver/nvidia/version": "test-driver\n",
            "proc/driver/nvidia/params": "EnableSystemMemoryPools: 529\n",
            "sys/devices/system/node/node0/meminfo": "Node 0 MemFree: 1048576 kB\n",
            "sys/devices/system/node/node0/vmstat": "pgscan_direct 44\n",
            "sys/fs/cgroup/task_0/memory.current": "33554432\n",
            "sys/fs/cgroup/task_0/memory.numa_stat": "anon N0=1024 N1=2048\n",
            "sys/fs/cgroup/memory.max": "1073741824\n",
            "sys/kernel/debug/shrinker/nv-sysmem-alloc-node-0-order-9-42/count": "0 80\n",
            "sys/kernel/debug/shrinker/unrelated/count": "unrelated\n",
        },
    )
    result = diagnostic.inventory(tmp_path)
    assert result["files"]["/proc/driver/nvidia/version"]["text"] == "test-driver\n"
    assert result["files"]["/sys/fs/cgroup/task_0/memory.numa_stat"]["text"] == "anon N0=1024 N1=2048\n"
    assert result["files"]["/sys/fs/cgroup/memory.max"]["text"] == "1073741824\n"
    assert result["shrinker"] == {
        "directory": "/sys/kernel/debug/shrinker",
        "status": "available",
        "counts": {"nv-sysmem-alloc-node-0-order-9-42": {"text": "0 80\n"}},
    }
    assert "error" in result["files"]["/proc/zoneinfo"]
    assert "cuda" not in result
    metrics = diagnostic.snapshot_metrics(result)
    assert metrics["nodes"]["0"]["MemFree"] == 1024**3
    assert metrics["vmstat"]["pgscan_direct"] == 123


def test_shrinker_denied_access_is_not_reported_as_an_empty_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.iterdir

    def denied(path: Path) -> Any:
        if path == tmp_path / "sys/kernel/debug/shrinker":
            raise PermissionError("debugfs denied")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    result = diagnostic.shrinker_counts(tmp_path)
    assert result["status"] == "unavailable"
    assert "PermissionError" in result["error"]
    assert "debugfs denied" in result["error"]


def test_shrinker_missing_interface_is_distinct_from_unreadable_count(tmp_path: Path) -> None:
    directory = tmp_path / "sys/kernel/debug/shrinker"
    directory.mkdir(parents=True)
    assert diagnostic.shrinker_counts(tmp_path)["status"] == "no_matching_interface"
    (directory / "nv-sysmem-alloc-node-1-order-0-12").mkdir()
    result = diagnostic.shrinker_counts(tmp_path)
    assert result["status"] == "available"
    assert "FileNotFoundError" in result["counts"]["nv-sysmem-alloc-node-1-order-0-12"]["error"]


def test_cuda_codes_are_preserved_from_python_exceptions_and_native_logs() -> None:
    assert diagnostic.cuda_errors(
        "cuMemCreate failed with CUDA result 2", {"error": "RuntimeError: cuInit failed with CUDA error 100"}
    ) == [{"code": 2, "name": "CUDA_ERROR_OUT_OF_MEMORY"}, {"code": 100, "name": "CUDA_ERROR_NO_DEVICE"}]


def fake_helper(directory: Path) -> Path:
    helper = directory / "fake_donor.py"
    helper.write_text(
        textwrap.dedent(
            """\
            import json
            import os
            import signal
            import time
            from pathlib import Path
            from configs.mooncake_egm_donor import parse_size, provider_config

            def configure_environment(): pass
            def require_cuda_devices(): pass
            def load_store_type(): return Store

            class Store:
                def __init__(self):
                    self.marker = None
                def setup(self, config):
                    directory = Path(os.environ['RECLAIM_TEST_STATE'])
                    self.gpu = os.environ['CUDA_VISIBLE_DEVICES']
                    self.node = config['egm_numa_nodes']
                    self.marker = directory / ('active-' + self.gpu)
                    active = {p.name: json.loads(p.read_text()) for p in directory.glob('active-*')}
                    event = {'action':'setup', 'gpu':self.gpu, 'config':config,
                             'chunk':os.environ.get('MC_MAX_MR_SIZE'), 'active':active, 'pid':os.getpid()}
                    with (directory / 'events.jsonl').open('a') as stream:
                        stream.write(json.dumps(event) + '\\n')
                    if os.environ.get('RECLAIM_TEST_MODE') == 'timeout' and active:
                        time.sleep(2)
                    if os.environ.get('RECLAIM_TEST_MODE') == 'kill-first' and active:
                        os.kill(next(iter(active.values()))['pid'], signal.SIGKILL)
                        time.sleep(1)
                    if os.environ.get('RECLAIM_TEST_MODE') == 'close-fail':
                        self.marker.write_text(json.dumps({'pid':os.getpid(),'node':self.node}))
                        return 0
                    if active and any(value['node'] == self.node for value in active.values()) and not event['chunk']:
                        print('cuMemCreate failed with CUDA result 2', flush=True)
                        return -1
                    self.marker.write_text(json.dumps({'pid':os.getpid(),'node':self.node}))
                    return 0
                def close(self):
                    if self.marker is not None:
                        self.marker.unlink(missing_ok=True)
                    with (Path(os.environ['RECLAIM_TEST_STATE']) / 'events.jsonl').open('a') as stream:
                        stream.write(json.dumps({'action':'close','gpu':getattr(self,'gpu',None)}) + '\\n')
                    return -1 if os.environ.get('RECLAIM_TEST_MODE') == 'close-fail' else 0
            """
        )
    )
    return helper


def run_fake(
    tmp_path: Path, *, cases: str = ",".join(diagnostic.TRIALS), timeout: str = "5"
) -> tuple[subprocess.CompletedProcess[str], Path]:
    config = tmp_path / "master.json"
    config.write_text('{"master_server_address":"192.0.2.1:8700","local_hostname":"192.0.2.2"}')
    output = tmp_path / "run/logs/reclaim"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mooncake_egm_reclaim",
            "run",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--donor-helper",
            str(fake_helper(tmp_path)),
            "--cases",
            cases,
            "--timeout-seconds",
            timeout,
            "--budget-seconds",
            "30",
            "--sample-interval",
            "0.05",
            "--hold-seconds",
            "0.05",
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    return result, output


def test_controlled_trials_keep_first_resident_record_oom_and_preserve_total_pool_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECLAIM_TEST_STATE", str(tmp_path))
    monkeypatch.setenv("MC_MAX_MR_SIZE", "12345")
    result, output = run_fake(tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["parameters"]["inherited_max_mr_size"] == "12345"
    assert [case["allocation_outcome"] for case in summary["cases"]] == [
        "second_allocation_failed",
        "two_pools_allocated",
        "second_allocation_failed",
        "two_pools_allocated",
    ]
    assert all(case["cleanup_confirmed"] for case in summary["cases"])
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    setups = [event for event in events if event["action"] == "setup"]
    assert len(setups) == 8
    assert [event["gpu"] for event in setups] == ["0", "1", "0", "2", "1", "0", "0", "1"]
    for index, event in enumerate(setups):
        assert bool(event["active"]) == bool(index % 2), "Second donor must observe the first native Store resident"
        assert event["config"]["global_segment_size"] == str(250 * 1024**3)
        assert event["config"]["local_buffer_size"] == "0"
        assert event["config"]["protocol"] == "nvlink"
        assert event["config"]["rdma_devices"] == ""
        assert event["config"]["enable_egm_store_pool"] == "true"
        assert event["chunk"] == (str(16 * 1024**3) if index >= 6 else None)
    assert not list(tmp_path.glob("active-*"))
    for case in summary["cases"]:
        assert {"baseline", "before-first", "first-ready", "before-second", "second-result", "after-close"} <= case[
            "snapshots"
        ].keys()
        samples = [json.loads(line) for line in (output / case["samples"]).read_text().splitlines()]
        assert any(sample["phase"] == "allocating-second" for sample in samples)
        assert all(sample.get("time_utc") for sample in samples)
    first = summary["cases"][0]
    assert first["donors"][1]["cuda_errors"] == [{"code": 2, "name": "CUDA_ERROR_OUT_OF_MEMORY"}]
    assert first["donors"][0]["child"]["ready_seen"] is True
    assert first["donors"][0]["stop_time_utc"] >= first["donors"][1]["stop_time_utc"]
    assert diagnostic.report(output.parent.parent, 0) == 0
    text = (output.parent / "perf-analysis.md").read_text()
    assert "**COMPLETE**" in text and "second_allocation_failed" in text
    assert "allocation failures are diagnostic outcomes" in text
    assert "TTFT p50/p95/p99" in text


@pytest.mark.parametrize(
    "mode,reason", [("close-fail", "uncertain_cleanup"), ("timeout", "timeout"), ("kill-first", "first_donor_died")]
)
def test_unsafe_trial_stops_later_cases_and_retains_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, reason: str
) -> None:
    monkeypatch.setenv("RECLAIM_TEST_STATE", str(tmp_path))
    monkeypatch.setenv("RECLAIM_TEST_MODE", mode)
    result, output = run_fake(tmp_path, timeout="0.5" if mode == "timeout" else "5")
    assert result.returncode == 1, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "stopped_incomplete"
    assert len(summary["cases"]) == 1
    assert summary["cases"][0]["failure"] == reason
    assert (output / "same-numa/after-close.json").exists()
    assert diagnostic.report(output.parent.parent, result.returncode) == 1
    assert "**INCOMPLETE**" in (output.parent / "perf-analysis.md").read_text()


def test_report_without_summary_or_cuda_still_explains_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def native_not_allowed(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Reporting must not import native Store or initialize CUDA")

    monkeypatch.setattr(diagnostic.allocation, "load_donor", native_not_allowed)
    monkeypatch.setattr(diagnostic.allocation, "cuda_inventory", native_not_allowed)
    assert diagnostic.report(tmp_path, 17) == 1
    text = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "**INCOMPLETE**" in text and "Launcher exit code: `17`" in text
    assert "Missing diagnostic summary" in text


def test_second_attempt_metrics_keep_node_reclaim_and_cgroup_event_deltas(tmp_path: Path) -> None:
    folder = tmp_path / "logs/reclaim/same-numa"
    folder.mkdir(parents=True)

    def snapshot(free: int, reclaim: int, scans: int, current: int, event: int) -> dict[str, Any]:
        return {
            "files": {
                "/sys/devices/system/node/node0/meminfo": {
                    "text": f"Node 0 MemFree: {free} kB\nNode 0 KReclaimable: {reclaim} kB\n"
                },
                "/proc/vmstat": {"text": f"pgscan_direct {scans}\n"},
                "/sys/fs/cgroup/task/memory.current": {"text": str(current)},
                "/sys/fs/cgroup/task/memory.max": {"text": str(1024**4)},
                "/sys/fs/cgroup/task/memory.events": {"text": f"max {event}\noom 0\n"},
            },
            "cgroups": [{"ancestors": ["/sys/fs/cgroup/task"], "version": 2}],
            "shrinker": {"status": "unavailable", "error": "PermissionError: denied"},
        }

    diagnostic.allocation.write_json(folder / "before-second.json", snapshot(100, 200, 10, 1000, 0))
    diagnostic.allocation.write_json(folder / "second-result.json", snapshot(90, 180, 40, 2000, 2))
    diagnostic.allocation.write_json(
        folder.parent / "summary.json",
        {
            "status": "complete",
            "parameters": {"cases": ["same-numa"]},
            "cases": [
                {
                    "name": "same-numa",
                    "status": "complete",
                    "cleanup_confirmed": True,
                    "snapshots": {"before-second": "before-second.json", "second-result": "second-result.json"},
                }
            ],
        },
    )
    assert diagnostic.report(tmp_path, 0) == 1
    analysis = json.loads((tmp_path / "logs/reclaim-analysis.json").read_text())["trials"][0]
    assert analysis["second_attempt_deltas"]["nodes_bytes"]["0"]["KReclaimable"] == -20 * 1024
    assert analysis["second_attempt_deltas"]["vmstat"]["pgscan_direct"] == 30
    assert analysis["second_attempt_deltas"]["cgroups"]["/sys/fs/cgroup/task"]["events"]["max"] == 2
    text = (tmp_path / "logs/perf-analysis.md").read_text()
    assert "pgscan_direct=+30" in text
    assert "PermissionError: denied" in text


def test_missing_required_snapshot_makes_an_otherwise_complete_report_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECLAIM_TEST_STATE", str(tmp_path))
    result, output = run_fake(tmp_path, cases="same-numa")
    assert result.returncode == 0, result.stderr
    (output / "same-numa/second-result.json").unlink()
    assert diagnostic.report(output.parent.parent, 0) == 1
    text = (output.parent / "perf-analysis.md").read_text()
    assert "**INCOMPLETE**" in text
    assert "second-result: missing or unreadable inventory snapshot" in text
    assert "unavailable: no paired VM counters" in text


def test_sigterm_preserves_partial_summary_and_cleans_first_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECLAIM_TEST_STATE", str(tmp_path))
    monkeypatch.setenv("RECLAIM_TEST_MODE", "timeout")
    config = tmp_path / "master.json"
    config.write_text('{"master_server_address":"192.0.2.1:8700","local_hostname":"192.0.2.2"}')
    output = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "tools.mooncake_egm_reclaim",
        "run",
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--donor-helper",
        str(fake_helper(tmp_path)),
        "--timeout-seconds",
        "30",
        "--budget-seconds",
        "60",
        "--sample-interval",
        "0.02",
        "--hold-seconds",
        "0.05",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import time

    try:
        until = time.monotonic() + 10
        while time.monotonic() < until:
            if (output / "same-numa/donor-1.json").exists():
                break
            time.sleep(0.02)
        else:
            pytest.fail("Second child did not start")
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 1, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "interrupted"
    assert summary["cases"][0]["failure"] == "interrupted"
    assert (output / "same-numa/after-close.json").exists()
    assert not list(tmp_path.glob("active-*"))
