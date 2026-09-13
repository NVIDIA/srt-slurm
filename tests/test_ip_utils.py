# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for IP address resolution helpers."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from srtctl.core.ip_utils import get_node_ip

IP_SCRIPT = Path(__file__).resolve().parents[1] / "src/srtctl/core/ip_utils/get_node_ip.sh"


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


def test_shell_resolver_prefers_private_hostname_ip_over_public_default_route(tmp_path):
    """Automatic discovery should select the fabric IP when NIC names vary by node."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "hostname").write_text(
        "#!/bin/bash\n[ \"$1\" = -I ] && echo '203.0.113.8 10.20.30.40'\n",
        encoding="ascii",
    )
    (fake_bin / "ip").write_text(
        "#!/bin/bash\necho '8.8.8.8 via 203.0.113.1 dev eth0 src 203.0.113.8'\n",
        encoding="ascii",
    )
    (fake_bin / "hostname").chmod(0o755)
    (fake_bin / "ip").chmod(0o755)

    result = subprocess.run(
        ["bash", "-c", 'source "$1"; _resolve_ip ""', "bash", str(IP_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.stdout.strip() == "10.20.30.40"


def test_runtime_resolves_control_plane_ips_on_cluster_interface(tmp_path):
    """Runtime endpoints must use the selected private/fabric interface."""
    from srtctl.core.runtime import Nodes, RuntimeContext
    from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

    model = tmp_path / "model"
    model.mkdir()
    container = tmp_path / "image.sqsh"
    container.touch()
    config = SrtConfig(
        name="network-contract",
        model=ModelConfig(path=str(model), container=str(container), precision="fp16"),
        resources=ResourceConfig(gpu_type="mi300x", gpus_per_node=8),
    )
    nodes = Nodes(head="node-a", bench="node-a", infra="node-b", worker=("node-a", "node-b"))

    def setting(name, default=None):
        return "fabric0" if name == "network_interface" else default

    with (
        patch("srtctl.core.runtime.Nodes.from_slurm", return_value=nodes),
        patch("srtctl.core.runtime.get_srtslurm_setting", side_effect=setting),
        patch("srtctl.core.runtime.get_hostname_ip", side_effect=["10.0.0.1", "10.0.0.2"]) as resolve,
    ):
        runtime = RuntimeContext.from_config(config, job_id="42", log_dir_base=Path(tmp_path))

    assert resolve.call_args_list[0].args == ("node-a", "fabric0")
    assert resolve.call_args_list[1].args == ("node-b", "fabric0")
    assert runtime.infra_node_ip == "10.0.0.2"
