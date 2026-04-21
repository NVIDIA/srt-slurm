# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for frontend topology logic (nginx + multiple frontends)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import FrontendConfig, ResourceConfig, SrtConfig


def make_config(
    *,
    enable_multiple_frontends: bool = True,
    num_additional_frontends: int = 9,
    frontend_type: str = "dynamo",
) -> SrtConfig:
    """Create a minimal SrtConfig for testing."""
    return SrtConfig(
        name="test-config",
        model={"path": "test-model", "container": "test.sqsh", "precision": "fp16"},
        resources=ResourceConfig(
            gpu_type="a100",
            gpus_per_node=8,
            prefill_nodes=1,
            decode_nodes=1,
        ),
        frontend=FrontendConfig(
            type=frontend_type,
            enable_multiple_frontends=enable_multiple_frontends,
            num_additional_frontends=num_additional_frontends,
        ),
    )


def make_runtime(nodes: list[str]) -> RuntimeContext:
    """Create a minimal RuntimeContext for testing."""
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(head=nodes[0], bench=nodes[0], infra=nodes[0], worker=tuple(nodes)),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=Path("/tmp/logs"),
        model_path=Path("/models/test-model"),
        container_image=Path("/path/to/container.sqsh"),
        gpus_per_node=8,
        network_interface=None,
        container_mounts={},
        environment={},
    )


class TestFrontendTopologyDataclass:
    """Tests for the FrontendTopology dataclass."""

    def test_uses_nginx_true(self):
        topology = FrontendTopology(
            nginx_node="node0",
            frontend_nodes=["node1"],
            frontend_port=8180,
            public_port=8000,
        )
        assert topology.uses_nginx is True

    def test_uses_nginx_false(self):
        topology = FrontendTopology(
            nginx_node=None,
            frontend_nodes=["node0"],
            frontend_port=8000,
            public_port=8000,
        )
        assert topology.uses_nginx is False


class TestComputeFrontendTopology:
    """Tests for _compute_frontend_topology method."""

    def test_single_node_no_nginx(self):
        """Single node: no nginx, 1 frontend on head at port 8000."""
        config = make_config(enable_multiple_frontends=True)
        runtime = make_runtime(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = orchestrator._compute_frontend_topology()

        assert topology.nginx_node is None
        assert topology.frontend_nodes == ["node0"]
        assert topology.frontend_port == 8000
        assert topology.public_port == 8000
        assert topology.uses_nginx is False

    def test_multi_node_frontends_disabled(self):
        """Multi-node with enable_multiple_frontends=False: no nginx, 1 frontend on head."""
        config = make_config(enable_multiple_frontends=False)
        runtime = make_runtime(["node0", "node1", "node2"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = orchestrator._compute_frontend_topology()

        assert topology.nginx_node is None
        assert topology.frontend_nodes == ["node0"]
        assert topology.frontend_port == 8000
        assert topology.public_port == 8000
        assert topology.uses_nginx is False

    def test_two_nodes_with_nginx(self):
        """2 nodes + enable_multiple_frontends: nginx on head, 1 frontend on node1."""
        config = make_config(enable_multiple_frontends=True)
        runtime = make_runtime(["node0", "node1"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = orchestrator._compute_frontend_topology()

        assert topology.nginx_node == "node0"
        assert topology.frontend_nodes == ["node1"]
        assert topology.frontend_port == 8180  # Behind nginx
        assert topology.public_port == 8000
        assert topology.uses_nginx is True

    def test_three_nodes_with_nginx(self):
        """3 nodes + enable_multiple_frontends: nginx on head, frontends on node1 and node2."""
        config = make_config(enable_multiple_frontends=True)
        runtime = make_runtime(["node0", "node1", "node2"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = orchestrator._compute_frontend_topology()

        assert topology.nginx_node == "node0"
        assert topology.frontend_nodes == ["node1", "node2"]
        assert topology.frontend_port == 8180
        assert topology.public_port == 8000
        assert topology.uses_nginx is True

    def test_many_nodes_with_nginx(self):
        """Many nodes: nginx on head, frontends on remaining nodes."""
        config = make_config(enable_multiple_frontends=True, num_additional_frontends=9)
        runtime = make_runtime(["node0", "node1", "node2", "node3", "node4", "node5"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = orchestrator._compute_frontend_topology()

        assert topology.nginx_node == "node0"
        assert topology.frontend_nodes == ["node1", "node2", "node3", "node4", "node5"]
        assert len(topology.frontend_nodes) == 5

    def test_frontend_count_limited_by_config(self):
        """Frontend count limited by num_additional_frontends config."""
        config = make_config(enable_multiple_frontends=True, num_additional_frontends=1)
        runtime = make_runtime(["node0", "node1", "node2", "node3", "node4"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = orchestrator._compute_frontend_topology()

        # num_additional_frontends=1 means max 2 frontends total (1 + 1 additional)
        assert topology.nginx_node == "node0"
        assert topology.frontend_nodes == ["node1", "node2"]
        assert len(topology.frontend_nodes) == 2

    def test_frontend_count_limited_by_available_nodes(self):
        """Frontend count limited by available nodes when fewer than config allows."""
        config = make_config(enable_multiple_frontends=True, num_additional_frontends=100)
        runtime = make_runtime(["node0", "node1", "node2"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = orchestrator._compute_frontend_topology()

        # Only 2 nodes available for frontends (node1, node2)
        assert topology.frontend_nodes == ["node1", "node2"]
        assert len(topology.frontend_nodes) == 2


class TestNginxConfigGeneration:
    """Tests for nginx config generation."""

    def test_nginx_config_single_frontend(self):
        """Nginx config with single frontend backend."""
        config = make_config(enable_multiple_frontends=True)
        runtime = make_runtime(["node0", "node1"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = FrontendTopology(
            nginx_node="node0",
            frontend_nodes=["node1"],
            frontend_port=8180,
            public_port=8000,
        )

        with patch.object(orchestrator, "runtime", runtime):
            with patch("srtctl.cli.mixins.frontend_stage.get_hostname_ip", side_effect=lambda x: f"10.0.0.{x[-1]}"):
                nginx_config = orchestrator._generate_nginx_config(topology)

        assert "server 10.0.0.1:8180" in nginx_config
        assert "listen 8000" in nginx_config

    def test_nginx_config_multiple_frontends(self):
        """Nginx config with multiple frontend backends."""
        config = make_config(enable_multiple_frontends=True)
        runtime = make_runtime(["node0", "node1", "node2", "node3"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        topology = FrontendTopology(
            nginx_node="node0",
            frontend_nodes=["node1", "node2", "node3"],
            frontend_port=8180,
            public_port=8000,
        )

        with patch("srtctl.cli.mixins.frontend_stage.get_hostname_ip", side_effect=lambda x: f"10.0.0.{x[-1]}"):
            nginx_config = orchestrator._generate_nginx_config(topology)

        # All three frontends should be in the upstream
        assert "server 10.0.0.1:8180" in nginx_config
        assert "server 10.0.0.2:8180" in nginx_config
        assert "server 10.0.0.3:8180" in nginx_config
        assert "listen 8000" in nginx_config


class TestStartFrontendIntegration:
    """Integration tests for start_frontend method."""

    @patch("srtctl.frontends.dynamo.start_srun_process")
    @patch("srtctl.cli.mixins.frontend_stage.start_srun_process")
    def test_single_node_starts_one_dynamo_frontend(self, mock_mixin_srun, mock_dynamo_srun):
        """Single node starts one dynamo frontend, no nginx."""
        mock_mixin_srun.return_value = MagicMock()
        mock_dynamo_srun.return_value = MagicMock()

        config = make_config(enable_multiple_frontends=True, frontend_type="dynamo")
        runtime = make_runtime(["node0"])
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        registry = MagicMock()
        processes = orchestrator.start_frontend(registry)

        assert len(processes) == 1
        assert processes[0].name == "frontend_0"
        assert processes[0].node == "node0"

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.cli.mixins.frontend_stage.start_srun_process")
    def test_single_node_starts_one_sglang_router(self, mock_mixin_srun, mock_sglang_srun):
        """Single node starts one sglang router, no nginx."""
        mock_mixin_srun.return_value = MagicMock()
        mock_sglang_srun.return_value = MagicMock()

        config = make_config(enable_multiple_frontends=True, frontend_type="sglang")
        runtime = make_runtime(["node0"])
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        orchestrator._backend_processes = []  # No workers for this test

        registry = MagicMock()
        processes = orchestrator.start_frontend(registry)

        assert len(processes) == 1
        assert processes[0].name == "sglang_router_0"
        assert processes[0].node == "node0"

    @patch("srtctl.frontends.dynamo.start_srun_process")
    @patch("srtctl.cli.mixins.frontend_stage.start_srun_process")
    def test_multi_node_starts_nginx_and_frontends(self, mock_mixin_srun, mock_dynamo_srun, tmp_path):
        """Multi-node starts nginx on head + frontends on other nodes."""
        mock_mixin_srun.return_value = MagicMock()
        mock_dynamo_srun.return_value = MagicMock()

        config = make_config(enable_multiple_frontends=True, frontend_type="dynamo")
        runtime = make_runtime(["node0", "node1", "node2"])
        # Use tmp_path for log_dir so nginx config can be written
        runtime = RuntimeContext(
            job_id=runtime.job_id,
            run_name=runtime.run_name,
            nodes=runtime.nodes,
            head_node_ip=runtime.head_node_ip,
            infra_node_ip=runtime.infra_node_ip,
            log_dir=tmp_path,
            model_path=runtime.model_path,
            container_image=runtime.container_image,
            gpus_per_node=runtime.gpus_per_node,
            network_interface=runtime.network_interface,
            container_mounts=runtime.container_mounts,
            environment=runtime.environment,
        )
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        registry = MagicMock()
        processes = orchestrator.start_frontend(registry)

        # Should have: 1 nginx + 2 frontends = 3 processes
        assert len(processes) == 3
        names = [p.name for p in processes]
        assert "nginx" in names
        assert "frontend_0" in names
        assert "frontend_1" in names

        # Verify nginx is on head node
        nginx_proc = next(p for p in processes if p.name == "nginx")
        assert nginx_proc.node == "node0"

        # Verify frontends are on other nodes
        frontend_procs = [p for p in processes if p.name.startswith("frontend_")]
        assert {p.node for p in frontend_procs} == {"node1", "node2"}

        # Verify nginx config was written
        assert (tmp_path / "nginx.conf").exists()

    @patch("srtctl.frontends.sglang.start_srun_process")
    @patch("srtctl.cli.mixins.frontend_stage.start_srun_process")
    def test_multi_node_sglang_with_nginx(self, mock_mixin_srun, mock_sglang_srun, tmp_path):
        """Multi-node with sglang router starts nginx + routers."""
        mock_mixin_srun.return_value = MagicMock()
        mock_sglang_srun.return_value = MagicMock()

        config = make_config(enable_multiple_frontends=True, frontend_type="sglang")
        runtime = make_runtime(["node0", "node1", "node2"])
        # Use tmp_path for log_dir so nginx config can be written
        runtime = RuntimeContext(
            job_id=runtime.job_id,
            run_name=runtime.run_name,
            nodes=runtime.nodes,
            head_node_ip=runtime.head_node_ip,
            infra_node_ip=runtime.infra_node_ip,
            log_dir=tmp_path,
            model_path=runtime.model_path,
            container_image=runtime.container_image,
            gpus_per_node=runtime.gpus_per_node,
            network_interface=runtime.network_interface,
            container_mounts=runtime.container_mounts,
            environment=runtime.environment,
        )
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        orchestrator._backend_processes = []

        registry = MagicMock()
        processes = orchestrator.start_frontend(registry)

        # Should have: 1 nginx + 2 routers = 3 processes
        assert len(processes) == 3
        names = [p.name for p in processes]
        assert "nginx" in names
        assert "sglang_router_0" in names
        assert "sglang_router_1" in names

    @patch("srtctl.frontends.dynamo.start_srun_process")
    @patch("srtctl.cli.mixins.frontend_stage.start_srun_process")
    def test_frontends_disabled_single_frontend_only(self, mock_mixin_srun, mock_dynamo_srun):
        """enable_multiple_frontends=False: only one frontend, no nginx."""
        mock_mixin_srun.return_value = MagicMock()
        mock_dynamo_srun.return_value = MagicMock()

        config = make_config(enable_multiple_frontends=False, frontend_type="dynamo")
        runtime = make_runtime(["node0", "node1", "node2", "node3"])
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        registry = MagicMock()
        processes = orchestrator.start_frontend(registry)

        # Only one frontend, no nginx
        assert len(processes) == 1
        assert processes[0].name == "frontend_0"
        assert processes[0].node == "node0"
