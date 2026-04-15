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
        "#!/bin/bash\n"
        "echo 'srun: Step created for StepId=2279904.27' >&2\n"
        "echo '10.109.25.246'\n",
        encoding="ascii",
    )
    fake_srun.chmod(0o755)

    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    ip = get_node_ip("nvl72156-T15", slurm_job_id="2279904")

    assert ip == "10.109.25.246"
