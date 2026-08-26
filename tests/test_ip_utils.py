# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for IP address resolution helpers."""

import os

from srtctl.core.ip_utils import get_node_ip


def test_get_node_ip_ignores_srun_step_created_output(tmp_path, monkeypatch):
    """get_node_ip() should ignore SLURM informational lines mixed into output."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/bin/bash\necho 'srun: Step created for StepId=2279904.27' >&2\necho '10.109.25.246'\n",
        encoding="ascii",
    )
    fake_srun.chmod(0o755)

    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    ip = get_node_ip("nvl72156-T15", slurm_job_id="2279904")

    assert ip == "10.109.25.246"


def test_get_node_ip_falls_through_comma_separated_interface_list(tmp_path, monkeypatch):
    """A comma-separated network_interface list tries each candidate in order,
    per node, and uses the first one that actually exists there -- covering a
    fleet where the same physical NIC has different names on different nodes."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    # Fake `ip`: only "eth7" resolves to an address (simulates a node that
    # lacks the first-preference "eth13" but has "eth7").
    fake_ip = fake_bin / "ip"
    fake_ip.write_text(
        "#!/bin/bash\n"
        'if [ "$3" = "eth7" ]; then\n'
        '  echo "    inet 10.1.1.40/16 metric 100 brd 10.1.255.255 scope global dynamic eth7"\n'
        "fi\n",
        encoding="ascii",
    )
    fake_ip.chmod(0o755)

    # Fake `srun`: runs the trailing `bash -c "<script>"` locally instead of
    # dispatching to a real cluster.
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/bin/bash\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "bash" ]; then shift; exec bash "$@"; fi\n'
        "  shift\n"
        "done\n",
        encoding="ascii",
    )
    fake_srun.chmod(0o755)

    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    ip = get_node_ip("compute01", slurm_job_id="1234", network_interface="eth13,eth7,eth4")

    assert ip == "10.1.1.40"
