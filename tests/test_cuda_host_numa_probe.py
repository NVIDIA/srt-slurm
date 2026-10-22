# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise allocation lifetimes and evidence handling without CUDA or Slurm."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from tools import cuda_host_numa_probe as diagnostic


def fixture_files(root: Path, files: dict[str, str]) -> None:
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def fake_core(directory: Path) -> Path:
    """Model retained handles with files; never import or call a GPU library."""
    core = directory / "fake-core"
    core.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import argparse
            import json
            import os
            import signal
            import time
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument('--device', required=True)
            parser.add_argument('--numa', required=True)
            parser.add_argument('--size-bytes', type=int, required=True)
            parser.add_argument('--chunk-bytes', type=int, required=True)
            parser.add_argument('--ready-file', type=Path, required=True)
            parser.add_argument('--result-file', type=Path, required=True)
            args = parser.parse_args()
            state = Path(os.environ['CUDA_PROBE_TEST_STATE'])
            mode = os.environ.get('CUDA_PROBE_TEST_MODE', '')
            gpu = os.environ['CUDA_VISIBLE_DEVICES']
            parent_pid = os.getppid()
            stopped = False
            def stop(*_):
                global stopped
                stopped = True
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            def event(action, **extra):
                with (state / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps({'action':action, 'pid':os.getpid(), 'gpu':gpu, **extra}) + '\\n')
            def publish(payload):
                temporary = args.result_file.with_suffix('.tmp')
                temporary.write_text(json.dumps(payload))
                temporary.replace(args.result_file)
            active = {p.name:json.loads(p.read_text()) for p in state.glob('active-*')}
            event('create', visible_ordinal=args.device, numa=args.numa,
                  requested=args.size_bytes, chunk=args.chunk_bytes, active=active)
            if active and mode == 'kill-first':
                os.kill(next(iter(active.values()))['pid'], signal.SIGKILL)
            if active and mode in ('timeout', 'kill-first'):
                while not stopped and os.getppid() == parent_pid:
                    time.sleep(0.01)
                publish({'setup_complete':False, 'allocation_success':False,
                         'held_bytes':0, 'cleanup_confirmed':True})
                event('release', held_bytes=0)
                raise SystemExit(2)
            same_node = any(value['numa'] == args.numa for value in active.values())
            oom = same_node or (not active and mode == 'first-oom')
            other_error = bool(active) and mode == 'other-error'
            held = args.size_bytes if not (oom or other_error) else (args.size_bytes * 13 // 16 if args.chunk_bytes else 0)
            marker = state / ('active-' + str(os.getpid()))
            if held:
                marker.write_text(json.dumps({'pid':os.getpid(), 'numa':args.numa, 'bytes':held, 'gpu':gpu}))
            payload = {'setup_complete':True, 'allocation_success':not (oom or other_error),
                       'held_bytes':held, 'cleanup_confirmed':False,
                       'failed_api':'cuMemCreate' if oom or other_error else None,
                       'failed_result':1 if other_error else (2 if oom else 0)}
            publish(payload)
            args.ready_file.write_text('ready\\n')
            event('ready', held_bytes=held)
            while not stopped and os.getppid() == parent_pid:
                time.sleep(0.01)
            marker.unlink(missing_ok=True)
            payload.update(cleanup_confirmed=mode != 'release-fail', held_bytes=0)
            publish(payload)
            event('release', held_bytes=held)
            raise SystemExit(3 if mode == 'release-fail' else (0 if payload['allocation_success'] else 2))
            """
        )
    )
    core.chmod(0o700)
    return core


def fake_command(
    tmp_path: Path, *, cases: str = ",".join(diagnostic.CASES), timeout: str = "5"
) -> tuple[list[str], Path]:
    output = tmp_path / "run/logs/probe"
    return [
        sys.executable,
        "-m",
        "tools.cuda_host_numa_probe",
        "run",
        "--binary",
        str(fake_core(tmp_path)),
        "--output-dir",
        str(output),
        "--cases",
        cases,
        "--allocation-timeout",
        timeout,
        "--budget",
        "20",
        "--sample-interval",
        "0.05",
        "--hold-seconds",
        "0.05",
    ], output


def run_fake(
    tmp_path: Path, *, cases: str = ",".join(diagnostic.CASES), timeout: str = "5"
) -> tuple[subprocess.CompletedProcess[str], Path]:
    command, output = fake_command(tmp_path, cases=cases, timeout=timeout)
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return result, output


def events(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def test_trials_retain_first_handles_and_record_partial_oom_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_PROBE_TEST_STATE", str(tmp_path))
    result, output = run_fake(tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert [case["outcome"] for case in summary["cases"]] == [
        "second_allocation_oom",
        "two_allocations_succeeded",
        "second_allocation_oom",
        "second_allocation_oom",
    ]
    creates = [event for event in events(tmp_path) if event["action"] == "create"]
    assert len(creates) == 8
    assert [event["gpu"] for event in creates] == ["0", "1", "0", "2", "1", "0", "0", "1"]
    for index, event in enumerate(creates):
        assert event["visible_ordinal"] == "0", "Each child selects ordinal zero inside its own CUDA mask"
        assert event["requested"] == 250 * diagnostic.GIB
        assert event["chunk"] == (16 * diagnostic.GIB if index >= 6 else 0)
        assert len(event["active"]) == index % 2, "The second create must observe the first retained allocation"
    assert not list(tmp_path.glob("active-*"))
    for case in summary["cases"]:
        assert case["status"] == "complete" and case["cleanup_confirmed"] is True
        assert {"baseline", "before-first", "first-result", "before-second", "second-result", "after-release"} <= case[
            "snapshots"
        ].keys()
        ranks = case["ranks"]
        assert ranks[0]["stopped_utc"] >= ranks[1]["stopped_utc"], "Release the second allocation before the first"
        assert all(rank["cleanup_confirmed"] and rank["forced_kill"] is False for rank in ranks)
        samples = [json.loads(line) for line in (output / case["name"] / "samples.jsonl").read_text().splitlines()]
        assert any(sample["phase"] == "allocating-second" for sample in samples)
    chunked = summary["cases"][-1]
    assert chunked["ranks"][1]["observed"]["held_bytes"] == 203.125 * diagnostic.GIB
    assert chunked["ranks"][1]["final"]["held_bytes"] == 0
    assert diagnostic.report(output.parent.parent, 0) == 0
    report = (output.parent / "perf-analysis.md").read_text()
    assert "**COMPLETE**" in report and "second_allocation_oom" in report
    assert "TTFT p50/p95/p99" in report and "N/A" in report


@pytest.mark.parametrize(
    "mode,failure",
    [
        ("timeout", "allocation_timeout"),
        ("kill-first", "first_process_died"),
        ("release-fail", "uncertain_cleanup"),
        ("other-error", "cuda_or_setup_error"),
    ],
)
def test_incomplete_trials_stop_later_cases_and_preserve_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, failure: str
) -> None:
    monkeypatch.setenv("CUDA_PROBE_TEST_STATE", str(tmp_path))
    monkeypatch.setenv("CUDA_PROBE_TEST_MODE", mode)
    result, output = run_fake(tmp_path, timeout="0.6" if mode == "timeout" else "5")
    assert result.returncode == 1, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert len(summary["cases"]) == 1
    case = summary["cases"][0]
    assert case["failure"] == failure
    assert case["status"] == "incomplete"
    assert len([event for event in events(tmp_path) if event["action"] == "create"]) == 2
    assert (output / "same-numa/after-release.json").is_file()
    assert all(rank["returncode"] is not None for rank in case["ranks"])
    assert diagnostic.report(output.parent.parent, result.returncode) == 1
    assert "**INCOMPLETE**" in (output.parent / "perf-analysis.md").read_text()


def test_first_allocation_oom_does_not_start_second_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_PROBE_TEST_STATE", str(tmp_path))
    monkeypatch.setenv("CUDA_PROBE_TEST_MODE", "first-oom")
    result, output = run_fake(tmp_path, cases="same-numa")
    assert result.returncode == 0, result.stderr
    case = json.loads((output / "summary.json").read_text())["cases"][0]
    assert case["outcome"] == "first_allocation_oom"
    assert case["cleanup_confirmed"] is True
    assert len([event for event in events(tmp_path) if event["action"] == "create"]) == 1


def test_sigterm_collects_results_and_releases_only_owned_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_PROBE_TEST_STATE", str(tmp_path))
    monkeypatch.setenv("CUDA_PROBE_TEST_MODE", "timeout")
    command, output = fake_command(tmp_path, timeout="15")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (tmp_path / "events.jsonl").exists() and len(
                [e for e in events(tmp_path) if e["action"] == "create"]
            ) == 2:
                break
            assert process.poll() is None
            time.sleep(0.02)
        else:
            pytest.fail("Second child did not start before signal deadline")
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 1, stdout + stderr
        summary = json.loads((output / "summary.json").read_text())
        case = summary["cases"][0]
        assert case["failure"] == "interrupted"
        assert all(rank["cleanup_confirmed"] for rank in case["ranks"])
        assert not list(tmp_path.glob("active-*"))
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


def test_stale_output_is_refused_without_touching_existing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_PROBE_TEST_STATE", str(tmp_path))
    command, output = fake_command(tmp_path)
    output.mkdir(parents=True)
    sentinel = output / "summary.json"
    sentinel.write_text('"existing evidence"\n')
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert "Refusing existing diagnostic data" in result.stderr
    assert sentinel.read_text() == '"existing evidence"\n'
    assert not (tmp_path / "events.jsonl").exists()


def test_inventory_retains_visible_cgroup_ancestors_and_kernel_pool_counts(tmp_path: Path) -> None:
    fixture_files(
        tmp_path,
        {
            "proc/self/cgroup": "0::/slurm/job_1/task_0\n",
            "proc/self/mountinfo": "10 9 0:28 /slurm/job_1 /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            "proc/vmstat": "pgscan_direct 123\ncompact_fail 4\n",
            "proc/driver/nvidia/version": "test-driver\n",
            "proc/driver/nvidia/params": "EnableSystemMemoryPools: 529\n",
            "sys/devices/system/node/node0/meminfo": "Node 0 MemFree: 1048576 kB\n",
            "sys/fs/cgroup/task_0/memory.numa_stat": "anon N0=1024 N1=2048\n",
            "sys/fs/cgroup/memory.max": "1073741824\n",
            "sys/kernel/debug/shrinker/nv-sysmem-alloc-node-0-order-5-42/count": "0 80\n",
        },
    )
    snapshot = diagnostic.inventory(tmp_path, pids=(123,))
    assert snapshot["files"]["/sys/fs/cgroup/task_0/memory.numa_stat"]["text"] == "anon N0=1024 N1=2048\n"
    assert snapshot["files"]["/sys/fs/cgroup/memory.max"]["text"] == "1073741824\n"
    assert snapshot["shrinker"]["status"] == "available"
    assert snapshot["shrinker"]["counts"]["nv-sysmem-alloc-node-0-order-5-42"]["text"] == "0 80\n"
    assert "FileNotFoundError" in snapshot["files"]["/proc/zoneinfo"]["error"]
    assert "FileNotFoundError" in snapshot["processes"]["123"]["cgroup"]["error"]


def test_missing_and_denied_shrinker_interfaces_are_not_zero_pools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert diagnostic.inventory(tmp_path)["shrinker"]["status"] == "unavailable"
    shrinker = tmp_path / "sys/kernel/debug/shrinker"
    shrinker.mkdir(parents=True)
    assert diagnostic.inventory(tmp_path)["shrinker"]["status"] == "no_matching_interface"
    original = Path.iterdir

    def denied(path: Path) -> Any:
        if path == shrinker:
            raise PermissionError("debugfs denied")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    result = diagnostic.inventory(tmp_path)["shrinker"]
    assert result["status"] == "unavailable"
    assert "PermissionError" in result["error"]


def test_unreadable_shrinker_count_keeps_error_instead_of_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    count = tmp_path / "sys/kernel/debug/shrinker/nv-sysmem-alloc-node-0-order-0-1/count"
    count.parent.mkdir(parents=True)
    original = Path.read_text

    def denied(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == count:
            raise PermissionError("count requires administrator access")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    result = diagnostic.inventory(tmp_path)["shrinker"]["counts"][count.parent.name]
    assert "PermissionError" in result["error"] and "text" not in result


@pytest.mark.parametrize(
    "member,expected",
    [("/", ["/sys/fs/cgroup"]), ("/slurm/job_1/task_0", ["/sys/fs/cgroup/task_0", "/sys/fs/cgroup"])],
)
def test_cgroup_namespace_paths_do_not_escape_visible_mount(member: str, expected: list[str]) -> None:
    groups = diagnostic.cgroups(f"0::{member}\n", "10 9 0:28 /slurm/job_1 /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    assert groups[0]["ancestors"] == expected
    outside = diagnostic.cgroups(
        "0::/slurm/job_2/task_0\n", "10 9 0:28 /slurm/job_1 /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
    )
    assert "outside the visible mount root" in outside[0]["error"]


def test_second_attempt_metrics_keep_counter_deltas_limits_and_missing_evidence() -> None:
    before = {
        "files": {
            "/proc/vmstat": {
                "text": "pgscan_direct 100\npgsteal_direct 90\ncompact_fail 4\ncompact_stall 0\nnr_free_pages 500\n"
            },
            "/sys/fs/cgroup/memory.events": {"text": "max 2\noom 0\noom_kill 0\n"},
        }
    }
    after = {
        "files": {
            "/proc/vmstat": {
                "text": "pgscan_direct 140\npgsteal_direct 115\ncompact_fail 7\ncompact_stall 0\nnr_free_pages 250\npgscan_kswapd 20\n"
            },
            "/sys/fs/cgroup/memory.events": {"text": "max 3\noom 0\noom_kill 0\n"},
            "/sys/fs/cgroup/memory.current": {"text": "12345\n"},
            "/sys/fs/cgroup/memory.max": {"text": "1457265836032\n"},
            "/proc/pagetypeinfo": {"error": "PermissionError: requires administrator access"},
        },
        "cgroups": [{"ancestors": ["/sys/fs/cgroup"]}],
        "shrinker": {"status": "unavailable", "error": "PermissionError: denied"},
    }
    metrics = diagnostic.second_attempt_metrics(before, after)
    assert metrics["vmstat_delta"] == {"pgscan_direct": 40, "pgsteal_direct": 25, "compact_fail": 3, "compact_stall": 0}
    assert metrics["cgroups"]["/sys/fs/cgroup"] == {
        "current": {"text": "12345\n"},
        "limit": {"text": "1457265836032\n"},
        "events_delta": {"max": 1, "oom": 0, "oom_kill": 0},
    }
    assert metrics["missing_files"]["/proc/pagetypeinfo"].startswith("PermissionError")
    assert metrics["shrinker"]["status"] == "unavailable"
    after["files"]["/proc/vmstat"] = {"error": "PermissionError: vmstat unavailable"}
    after["files"]["/sys/fs/cgroup/memory.events"] = {"error": "FileNotFoundError: no visible events"}
    unavailable = diagnostic.second_attempt_metrics(before, after)
    assert unavailable["vmstat_delta"] == {}, "An unavailable counter must not be represented as zero activity"
    assert "/proc/vmstat" in unavailable["missing_files"]
    assert unavailable["cgroups"]["/sys/fs/cgroup"]["events_delta"] == {}


@pytest.mark.parametrize(
    "missing", ["summary.json", "same-numa/first-result.json", "same-numa/rank-1.json", "same-numa/samples.jsonl"]
)
def test_report_rejects_missing_allocation_evidence_even_with_successful_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setenv("CUDA_PROBE_TEST_STATE", str(tmp_path))
    result, output = run_fake(tmp_path, cases="same-numa")
    assert result.returncode == 0, result.stderr
    (output / missing).unlink()
    assert diagnostic.report(output.parent.parent, 0) == 1
    report = (output.parent / "perf-analysis.md").read_text()
    assert "**INCOMPLETE**" in report
    assert "Evidence gaps" in report


def test_report_and_inventory_require_no_cuda_or_framework_imports(tmp_path: Path) -> None:
    source = Path(diagnostic.__file__).resolve()
    guard = textwrap.dedent("""\
        import builtins
        import runpy
        import sys
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'cuda', 'torch', 'cupy', 'mooncake', 'numpy'}:
                raise AssertionError('Unexpected GPU/framework dependency: ' + name)
            return original(name, *args, **kwargs)
        builtins.__import__ = guard
        script, *arguments = sys.argv[1:]
        sys.argv = [script, *arguments]
        runpy.run_path(script, run_name='__main__')
        """)
    inventory_path = tmp_path / "inventory.json"
    result = subprocess.run(
        [sys.executable, "-I", "-c", guard, str(source), "inventory", "--output", str(inventory_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "files" in json.loads(inventory_path.read_text())
    result = subprocess.run(
        [sys.executable, "-I", "-c", guard, str(source), "report", "--run-dir", str(tmp_path), "--exit-code", "1"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, result.stderr
    assert "**INCOMPLETE**" in (tmp_path / "logs/perf-analysis.md").read_text()


@pytest.mark.parametrize("value", ["0", "-1GiB", "250GB", "1.5GiB"])
def test_invalid_pool_size_is_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive bytes or a binary size"):
        diagnostic.parse_size(value)
