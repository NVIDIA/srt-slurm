# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the startup gate that reads the before-run hardware snapshot."""

from __future__ import annotations

import textwrap

import pytest

from srtctl.core.preflight import check_snapshot, format_findings

CLEAN_MEMORY = """\
$ nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv
index, memory.total [MiB], memory.used [MiB], memory.free [MiB]
0, 283264 MiB, 3072 MiB, 280192 MiB
1, 283264 MiB, 3072 MiB, 280192 MiB
"""

NO_COMPUTE_APPS = """\
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv
pid, process_name, used_memory [MiB], gpu_uuid
"""

READY_DOMAIN = """\
$ nvidia-imex-ctl -c /tmp/imex_hwinfo_config.cfg -N -H
Node #0   - 10.66.6.8          - READY                - Version: 580.173.02      - Hostname: theia0271
Node #1   * 10.66.6.9 *        - READY                - Version: 580.173.02      - Hostname: theia0272
"""

HEALTHY_FABRIC = """\
$ nvidia-smi -q | grep -A8 -i "^ *Fabric"
    Fabric
        State                                          : Completed
        Status                                         : Success
        CliqueId                                       : 32766
        Health
            Summary                                    : Healthy
"""


def _snapshot(tmp_path, *blocks, node: str = "theia0271"):
    """Write a merged snapshot with the given command blocks for one node."""
    body = f"===== {node} | before | 2026-08-24T07:01:27Z =====\n\n"
    body += "# ---------- GPU inventory ----------\n\n"
    body += "\n".join(textwrap.dedent(block) for block in blocks)
    path = tmp_path / "before.out"
    path.write_text(body)
    return path


class TestCleanSnapshot:
    def test_healthy_hardware_reports_nothing(self, tmp_path):
        path = _snapshot(tmp_path, CLEAN_MEMORY, NO_COMPUTE_APPS, READY_DOMAIN, HEALTHY_FABRIC)

        assert check_snapshot(path, gpu_memory_utilization=0.92) == []

    def test_missing_commands_do_not_raise_false_alarms(self, tmp_path):
        """An older snapshot, or a node where nvidia-smi is absent, must pass."""
        path = _snapshot(
            tmp_path,
            "$ nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv\n[exit 127]\n",
            "$ nvidia-imex-ctl -c /tmp/imex_hwinfo_config.cfg -N -H\n[timed out after 20s]\n",
        )

        assert check_snapshot(path, gpu_memory_utilization=0.92) == []

    def test_unreadable_snapshot_is_skipped(self, tmp_path):
        assert check_snapshot(tmp_path / "missing.out", gpu_memory_utilization=0.9) == []


class TestFindings:
    def test_leftover_process_is_reported_with_its_pid(self, tmp_path):
        path = _snapshot(
            tmp_path,
            """\
            $ nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv
            pid, process_name, used_memory [MiB], gpu_uuid
            4711, python3, 25600 MiB, GPU-d7966220
            """,
        )

        findings = check_snapshot(path)

        assert len(findings) == 1
        assert "pid 4711" in findings[0].detail
        assert "25600 MiB" in findings[0].detail

    def test_card_below_the_requested_share_is_reported(self, tmp_path):
        """The same arithmetic the engine does, three minutes earlier."""
        path = _snapshot(
            tmp_path,
            """\
            $ nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv
            index, memory.total [MiB], memory.used [MiB], memory.free [MiB]
            0, 283264 MiB, 26286 MiB, 256978 MiB
            """,
        )

        findings = check_snapshot(path, gpu_memory_utilization=0.92)

        assert len(findings) == 1
        assert "GPU 0" in findings[0].detail
        assert "250.96 GiB free" in findings[0].detail
        assert "needs 254.50 GiB" in findings[0].detail

    def test_same_card_passes_at_a_lower_share(self, tmp_path):
        path = _snapshot(
            tmp_path,
            """\
            $ nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv
            index, memory.total [MiB], memory.used [MiB], memory.free [MiB]
            0, 283264 MiB, 26286 MiB, 256978 MiB
            """,
        )

        assert check_snapshot(path, gpu_memory_utilization=0.85) == []

    def test_unavailable_domain_node_is_reported_even_outside_the_allocation(self, tmp_path):
        """MNNVL spans the whole domain, so a peer we were not given still breaks it."""
        path = _snapshot(
            tmp_path,
            """\
            $ nvidia-imex-ctl -c /tmp/imex_hwinfo_config.cfg -N -H
            Node #10  - 10.66.6.18         - READY                - Version: 580.173.02      - Hostname: theia0281
            Node #11  - 10.66.6.19         - UNAVAILABLE          - Version:                 - Hostname: theia0282
            """,
        )

        findings = check_snapshot(path)

        assert len(findings) == 1
        assert findings[0].node == "theia0282"
        assert "UNAVAILABLE" in findings[0].detail

    def test_domain_finding_is_reported_once_per_peer(self, tmp_path):
        """Every node sees the same domain; the report must not multiply by observers."""
        broken = textwrap.dedent(
            """\
            $ nvidia-imex-ctl -c /tmp/imex_hwinfo_config.cfg -N -H
            Node #11  - 10.66.6.19         - UNAVAILABLE          - Version:                 - Hostname: theia0282
            """
        )
        path = tmp_path / "before.out"
        path.write_text(
            "".join(
                f"===== theia{index} | before | 2026-08-24T07:01:27Z =====\n\n{broken}"
                for index in (271, 272, 273)
            )
        )

        assert len(check_snapshot(path)) == 1

    @pytest.mark.parametrize(
        ("field", "value"),
        [("State", "Not Started"), ("Status", "Timeout"), ("Summary", "Degraded")],
    )
    def test_gpu_outside_the_fabric_clique_is_reported(self, tmp_path, field, value):
        fabric = {"State": "Completed", "Status": "Success", "Summary": "Healthy"} | {field: value}
        path = _snapshot(
            tmp_path,
            f"""\
            $ nvidia-smi -q | grep -A8 -i "^ *Fabric"
                Fabric
                    State                                          : {fabric["State"]}
                    Status                                         : {fabric["Status"]}
                    Health
                        Summary                                    : {fabric["Summary"]}
            """,
        )

        findings = check_snapshot(path)

        assert len(findings) == 1
        assert value in findings[0].detail


class TestMessage:
    def test_message_names_the_problem_the_node_and_the_file(self, tmp_path):
        path = _snapshot(
            tmp_path,
            """\
            $ nvidia-imex-ctl -c /tmp/imex_hwinfo_config.cfg -N -H
            Node #11  - 10.66.6.19         - UNAVAILABLE          - Version:                 - Hostname: theia0282
            """,
        )
        findings = check_snapshot(path)

        message = format_findings(findings, path)

        assert "1 problem" in message
        assert "theia0282" in message
        assert str(path) in message
        assert "preflight.enabled: false" in message
