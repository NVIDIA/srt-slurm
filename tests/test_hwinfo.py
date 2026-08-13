# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NVLink/MNNVL hardware snapshots taken around a run."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.hwinfo import COLLECT_SCRIPT, record_hwinfo_snapshot

# Diagnostics that must stay in the snapshot: link health, the MNNVL wiring and
# the driver's own account of what went wrong.
REQUIRED_COMMANDS = (
    "nvidia-smi nvlink -s",
    "nvidia-smi nvlink -e",
    "nvidia-smi topo -m",
    "ls -al /dev/nvidia-caps-imex-channels/",
    "cat /etc/nvidia-imex/nodes_config.cfg",
    "nvidia-imex-ctl -c /tmp/imex_hwinfo_config.cfg -N -H",
    "dmesg",
)


def _run_script(parts_dir: Path, phase: str = "before", node: str = "node01", stubs: str = "") -> str:
    """Run the collector with stubbed tools and return the node's part file."""
    harness = "\n".join(
        [
            # Stand-ins are shell functions, not files on PATH, so the test does
            # not need an exec-capable temp dir.
            'nvidia-smi() { echo "nvidia-smi $*"; }',
            'systemctl() { echo "systemctl $*"; }',
            'nvidia-imex-ctl() { echo "nvidia-imex-ctl $*"; }',
            'dmesg() { echo "NVRM: Xid (PCI:0009:01:00): 74, pid=1"; }',
            "export -f nvidia-smi systemctl nvidia-imex-ctl dmesg",
            stubs,
            f"export SLURMD_NODENAME={node}",
            f"bash {shlex.quote(str(COLLECT_SCRIPT))} {shlex.quote(str(parts_dir))} {phase}",
        ]
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return (parts_dir / f"{node}.part").read_text()


class TestCollectScript:
    def test_part_is_named_for_the_slurm_node(self, tmp_path):
        """The orchestrator merges by SLURM node name, so the part must match it."""
        _run_script(tmp_path, node="theia0245")

        assert (tmp_path / "theia0245.part").is_file()

    def test_header_records_phase_and_time(self, tmp_path):
        content = _run_script(tmp_path, phase="after", node="theia0245")

        assert re.match(
            r"^===== theia0245 \| after \| \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z =====$",
            content.splitlines()[0],
        )

    @pytest.mark.parametrize("command", REQUIRED_COMMANDS)
    def test_diagnostic_is_collected(self, tmp_path, command):
        content = _run_script(tmp_path)

        assert command in content

    def test_each_command_is_shown_with_its_output(self, tmp_path):
        """A snapshot is only useful if it says what produced each block."""
        content = _run_script(tmp_path)

        assert "$ nvidia-smi nvlink -e" in content
        assert "nvidia-smi nvlink -e" in content.split("$ nvidia-smi nvlink -e", 1)[1]

    def test_a_missing_tool_is_recorded_and_collection_continues(self, tmp_path):
        """Most clusters lack some of these tools; that must not lose the rest."""
        content = _run_script(tmp_path, stubs="unset -f nvidia-smi")

        assert "[exit 127]" in content
        # Commands after the failing one still ran.
        assert "NVRM: Xid" in content

    def test_a_hung_command_is_bounded(self, tmp_path):
        """A wedged driver call must not hold up job startup or cleanup."""
        content = _run_script(
            tmp_path,
            stubs="nvidia-smi() { sleep 30; }; export -f nvidia-smi; export HWINFO_CMD_TIMEOUT=1",
        )

        assert "[timed out after 1s]" in content

    def test_sections_group_the_output(self, tmp_path):
        content = _run_script(tmp_path)

        assert "# ---------- NVLink ----------" in content
        assert "# ---------- MNNVL / IMEX ----------" in content


def _runtime(tmp_path: Path, worker=("node01", "node02"), het=False, groups=None):
    return SimpleNamespace(
        log_dir=tmp_path / "logs",
        nodes=SimpleNamespace(
            worker=tuple(worker),
            het=het,
            het_group_for=lambda node: (groups or {}).get(node),
        ),
        srun_options={},
    )


def _fake_srun(parts_written: list[str] | None = None):
    """Stand in for srun, writing the parts a real collector run would leave."""

    def start(command, **kwargs):
        parts_dir = Path(command[2])
        parts_dir.mkdir(parents=True, exist_ok=True)
        for node in kwargs["nodelist"]:
            if parts_written is not None and node not in parts_written:
                continue
            (parts_dir / f"{node}.part").write_text(f"===== {node} =====\ncounters\n\n")
        return MagicMock()

    return start


class TestSnapshot:
    def test_parts_are_merged_in_node_order(self, tmp_path):
        runtime = _runtime(tmp_path, worker=("node02", "node01"))

        with patch("srtctl.core.hwinfo.start_srun_process", side_effect=_fake_srun()):
            path = record_hwinfo_snapshot(runtime, "before")

        assert path == tmp_path / "logs" / "hwinfo" / "before.out"
        content = path.read_text()
        assert content.index("node02") < content.index("node01")

    def test_parts_are_cleaned_up(self, tmp_path):
        """Only the merged snapshot is left behind, not the per-node scratch."""
        runtime = _runtime(tmp_path)

        with patch("srtctl.core.hwinfo.start_srun_process", side_effect=_fake_srun()):
            record_hwinfo_snapshot(runtime, "before")

        assert list((tmp_path / "logs" / "hwinfo").iterdir()) == [tmp_path / "logs" / "hwinfo" / "before.out"]

    def test_a_silent_node_is_called_out(self, tmp_path):
        """A node that reported nothing is more interesting than a gap."""
        runtime = _runtime(tmp_path)

        with patch("srtctl.core.hwinfo.start_srun_process", side_effect=_fake_srun(parts_written=["node01"])):
            path = record_hwinfo_snapshot(runtime, "after")

        content = path.read_text()
        assert "===== node02 =====" in content
        assert "[no hwinfo collected from this node]" in content

    def test_stale_parts_from_an_earlier_phase_are_not_reused(self, tmp_path):
        runtime = _runtime(tmp_path)
        stale = tmp_path / "logs" / "hwinfo" / ".before.parts"
        stale.mkdir(parents=True)
        (stale / "node99.part").write_text("stale snapshot\n")

        with patch("srtctl.core.hwinfo.start_srun_process", side_effect=_fake_srun()):
            path = record_hwinfo_snapshot(runtime, "before")

        assert "stale snapshot" not in path.read_text()

    def test_collection_runs_on_the_host(self, tmp_path):
        """The IMEX config and channel devices are not inside the container."""
        runtime = _runtime(tmp_path)

        with patch("srtctl.core.hwinfo.start_srun_process", side_effect=_fake_srun()) as srun:
            record_hwinfo_snapshot(runtime, "before")

        assert srun.call_args.kwargs["container_image"] is None

    def test_het_jobs_get_one_srun_per_component(self, tmp_path):
        """A single srun cannot span two heterogeneous-job components."""
        runtime = _runtime(
            tmp_path,
            worker=("node01", "node02", "node03"),
            het=True,
            groups={"node01": 0, "node02": 1, "node03": 1},
        )

        with patch("srtctl.core.hwinfo.start_srun_process", side_effect=_fake_srun()) as srun:
            record_hwinfo_snapshot(runtime, "before")

        calls = {call.kwargs["het_group"]: call.kwargs["nodelist"] for call in srun.call_args_list}
        assert calls == {0: ["node01"], 1: ["node02", "node03"]}

    def test_a_failure_never_breaks_the_run(self, tmp_path):
        runtime = _runtime(tmp_path)

        with patch("srtctl.core.hwinfo.start_srun_process", side_effect=OSError("srun missing")):
            assert record_hwinfo_snapshot(runtime, "before") is None

    def test_nothing_to_snapshot_without_workers(self, tmp_path):
        runtime = _runtime(tmp_path, worker=())

        with patch("srtctl.core.hwinfo.start_srun_process") as srun:
            assert record_hwinfo_snapshot(runtime, "before") is None

        srun.assert_not_called()
