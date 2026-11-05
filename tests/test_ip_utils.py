# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for IP address resolution helpers."""

import os
from pathlib import Path
from unittest.mock import patch

from srtctl.core.ip_utils import get_node_ip
from srtctl.core.slurm import get_hostname_ip


def test_get_node_ip_ignores_srun_step_created_output(tmp_path, monkeypatch):
    """get_node_ip() should ignore SLURM informational lines mixed into output."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        f'#!/bin/bash\necho "$*" > {tmp_path}/args\n'
        "echo 'srun: Step created for StepId=2279904.27' >&2\necho '10.109.25.246'\n",
        encoding="ascii",
    )
    fake_srun.chmod(0o755)

    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    ip = get_node_ip("nvl72156-T15", slurm_job_id="2279904")

    assert ip == "10.109.25.246"
    # the workers hold every CPU and GPU of the node: without --overlap the step waits for them until it times out
    assert "--overlap" in (tmp_path / "args").read_text().split()


def test_get_hostname_ip_resolves_each_node_once(monkeypatch):
    """Every worker, scrape target and benchmark setting on a node asks for its address: one lookup serves them all,
    including the DNS fallback after a failed srun lookup."""
    monkeypatch.setenv("SLURM_JOB_ID", "2279904")
    get_hostname_ip.cache_clear()
    with (
        patch(
            "srtctl.core.slurm.get_node_ip", side_effect=lambda node, *_: "10.0.0.1" if node == "a" else None
        ) as srun,
        patch("srtctl.core.slurm.socket.gethostbyname", return_value="10.0.0.2") as dns,
    ):
        assert [get_hostname_ip(node) for node in ("a", "b", "a", "b")] == ["10.0.0.1", "10.0.0.2"] * 2
    get_hostname_ip.cache_clear()
    assert srun.call_count == 2 and dns.call_count == 1


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
    assert (runtime.head_node_ip, runtime.infra_node_ip) == ("10.0.0.1", "10.0.0.2")
