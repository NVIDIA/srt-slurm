# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cluster-style e2e tests for recipe validation."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.config import load_config
from srtctl.core.topology import allocate_endpoints, endpoints_to_processes

RECIPES_DIR = Path(__file__).parent.parent / "recipes"
CI_DIR = Path(__file__).parent.parent / "ci"


# =============================================================================
# Cluster Fixtures
# =============================================================================


class GB200NVLRack:
    """GB200 NVL SLURM rack: 18 nodes × 4 GPUs = 72 total GPUs."""

    NUM_NODES = 18
    GPUS_PER_NODE = 4
    TOTAL_GPUS = NUM_NODES * GPUS_PER_NODE  # 72

    @classmethod
    def nodes(cls) -> list[str]:
        return [f"gb200-{i:02d}" for i in range(1, cls.NUM_NODES + 1)]

    @classmethod
    def slurm_env(cls) -> dict[str, str]:
        return {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOBID": "12345",
            "SLURM_NODELIST": f"gb200-[01-{cls.NUM_NODES:02d}]",
            "SLURM_JOB_NUM_NODES": str(cls.NUM_NODES),
            "SRTCTL_SOURCE_DIR": str(Path(__file__).parent.parent),
        }

    @classmethod
    def mock_scontrol(cls):
        def mock_run(cmd, **kwargs):
            if cmd[0] == "scontrol" and "hostnames" in cmd:
                result = MagicMock()
                result.stdout = "\n".join(cls.nodes())
                result.returncode = 0
                return result
            raise subprocess.CalledProcessError(1, cmd)

        return mock_run


class H100Rack:
    """H100 SLURM rack: 13 nodes × 8 GPUs = 104 total GPUs."""

    NUM_NODES = 13
    GPUS_PER_NODE = 8
    TOTAL_GPUS = NUM_NODES * GPUS_PER_NODE  # 104

    @classmethod
    def nodes(cls) -> list[str]:
        return [f"h100-{i:02d}" for i in range(1, cls.NUM_NODES + 1)]

    @classmethod
    def slurm_env(cls) -> dict[str, str]:
        return {
            "SLURM_JOB_ID": "67890",
            "SLURM_JOBID": "67890",
            "SLURM_NODELIST": f"h100-[01-{cls.NUM_NODES:02d}]",
            "SLURM_JOB_NUM_NODES": str(cls.NUM_NODES),
            "SRTCTL_SOURCE_DIR": str(Path(__file__).parent.parent),
        }

    @classmethod
    def mock_scontrol(cls):
        def mock_run(cmd, **kwargs):
            if cmd[0] == "scontrol" and "hostnames" in cmd:
                result = MagicMock()
                result.stdout = "\n".join(cls.nodes())
                result.returncode = 0
                return result
            raise subprocess.CalledProcessError(1, cmd)

        return mock_run


# =============================================================================
# Tests
# =============================================================================


class TestGB200FP4Cluster:
    """GB200 FP4 1k1k configs on GB200 NVL rack (18 nodes × 4 GPUs)."""

    RACK = GB200NVLRack
    RECIPES = (
        list((RECIPES_DIR / "gb200-fp4" / "1k1k").glob("*.yaml"))
        if (RECIPES_DIR / "gb200-fp4" / "1k1k").exists()
        else []
    )

    @pytest.mark.parametrize("recipe_path", RECIPES, ids=lambda p: p.name)
    def test_gpus_per_node_is_4(self, recipe_path):
        """All GB200 FP4 1k1k configs use 4 GPUs per node."""
        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                assert config.resources.gpus_per_node == self.RACK.GPUS_PER_NODE, (
                    f"{recipe_path.name}: expected gpus_per_node={self.RACK.GPUS_PER_NODE}, "
                    f"got {config.resources.gpus_per_node}"
                )

    @pytest.mark.parametrize("recipe_path", RECIPES, ids=lambda p: p.name)
    def test_fits_in_rack(self, recipe_path):
        """Recipe fits within the GB200 NVL rack (18 nodes)."""
        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources
                total_nodes_needed = (r.prefill_nodes or 0) + (r.decode_nodes or 0) + (r.agg_nodes or 0)
                assert total_nodes_needed <= self.RACK.NUM_NODES, (
                    f"{recipe_path.name}: needs {total_nodes_needed} nodes, rack has {self.RACK.NUM_NODES}"
                )

    @pytest.mark.parametrize("recipe_path", RECIPES, ids=lambda p: p.name)
    def test_endpoint_allocation(self, recipe_path):
        """Endpoints are allocated correctly on GB200 NVL rack."""
        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources

                endpoints = config.backend.allocate_endpoints(
                    num_prefill=r.num_prefill,
                    num_decode=r.num_decode,
                    num_agg=r.num_agg,
                    gpus_per_prefill=r.gpus_per_prefill,
                    gpus_per_decode=r.gpus_per_decode,
                    gpus_per_agg=r.gpus_per_agg,
                    gpus_per_node=r.gpus_per_node,
                    available_nodes=self.RACK.nodes(),
                )

                prefill_eps = [e for e in endpoints if e.mode == "prefill"]
                decode_eps = [e for e in endpoints if e.mode == "decode"]

                assert len(prefill_eps) == r.num_prefill
                assert len(decode_eps) == r.num_decode

                for ep in prefill_eps:
                    assert ep.total_gpus == r.gpus_per_prefill, (
                        f"prefill endpoint {ep.index} has {ep.total_gpus} GPUs, expected {r.gpus_per_prefill}"
                    )

                for ep in decode_eps:
                    assert ep.total_gpus == r.gpus_per_decode, (
                        f"decode endpoint {ep.index} has {ep.total_gpus} GPUs, expected {r.gpus_per_decode}"
                    )


class TestH100Cluster:
    """H100 configs on H100 rack (13 nodes × 8 GPUs = 104 total)."""

    RACK = H100Rack
    RECIPES = list((RECIPES_DIR / "h100").glob("*.yaml")) if (RECIPES_DIR / "h100").exists() else []

    @pytest.mark.parametrize("recipe_path", RECIPES, ids=lambda p: p.name)
    def test_gpus_per_node_is_8(self, recipe_path):
        """All H100 configs use 8 GPUs per node."""
        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                assert config.resources.gpus_per_node == self.RACK.GPUS_PER_NODE, (
                    f"{recipe_path.name}: expected gpus_per_node={self.RACK.GPUS_PER_NODE}, "
                    f"got {config.resources.gpus_per_node}"
                )

    @pytest.mark.parametrize("recipe_path", RECIPES, ids=lambda p: p.name)
    def test_endpoint_allocation(self, recipe_path):
        """Endpoints are allocated correctly on H100 rack."""
        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources

                endpoints = config.backend.allocate_endpoints(
                    num_prefill=r.num_prefill,
                    num_decode=r.num_decode,
                    num_agg=r.num_agg,
                    gpus_per_prefill=r.gpus_per_prefill,
                    gpus_per_decode=r.gpus_per_decode,
                    gpus_per_agg=r.gpus_per_agg,
                    gpus_per_node=r.gpus_per_node,
                    available_nodes=self.RACK.nodes(),
                )

                prefill_eps = [e for e in endpoints if e.mode == "prefill"]
                decode_eps = [e for e in endpoints if e.mode == "decode"]

                assert len(prefill_eps) == r.num_prefill
                assert len(decode_eps) == r.num_decode

                for ep in prefill_eps:
                    assert ep.total_gpus == r.gpus_per_prefill
                for ep in decode_eps:
                    assert ep.total_gpus == r.gpus_per_decode

    @pytest.mark.parametrize("recipe_path", RECIPES, ids=lambda p: p.name)
    def test_multi_node_tp(self, recipe_path):
        """H100 configs with TP > 8 span multiple nodes correctly."""
        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources

                if r.gpus_per_prefill > self.RACK.GPUS_PER_NODE:
                    expected_nodes = r.gpus_per_prefill // self.RACK.GPUS_PER_NODE

                    endpoints = config.backend.allocate_endpoints(
                        num_prefill=r.num_prefill,
                        num_decode=r.num_decode,
                        num_agg=r.num_agg,
                        gpus_per_prefill=r.gpus_per_prefill,
                        gpus_per_decode=r.gpus_per_decode,
                        gpus_per_agg=r.gpus_per_agg,
                        gpus_per_node=r.gpus_per_node,
                        available_nodes=self.RACK.nodes(),
                    )

                    for ep in [e for e in endpoints if e.mode == "prefill"]:
                        assert ep.num_nodes == expected_nodes, (
                            f"prefill endpoint should span {expected_nodes} nodes, got {ep.num_nodes}"
                        )


class TestCIConfigs:
    """CI configs (smaller models) on H100 rack."""

    RACK = H100Rack

    def test_agg_config(self):
        """Aggregated CI config allocates correctly."""
        recipe_path = CI_DIR / "agg.yaml"
        if not recipe_path.exists():
            pytest.skip("agg.yaml not found")

        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources

                endpoints = config.backend.allocate_endpoints(
                    num_prefill=r.num_prefill,
                    num_decode=r.num_decode,
                    num_agg=r.num_agg,
                    gpus_per_prefill=r.gpus_per_prefill,
                    gpus_per_decode=r.gpus_per_decode,
                    gpus_per_agg=r.gpus_per_agg,
                    gpus_per_node=r.gpus_per_node,
                    available_nodes=self.RACK.nodes(),
                )

                agg_eps = [e for e in endpoints if e.mode == "agg"]
                assert len(agg_eps) == r.num_agg
                for ep in agg_eps:
                    assert ep.total_gpus == r.gpus_per_agg

    def test_disagg_config(self):
        """Disaggregated CI config allocates correctly."""
        recipe_path = CI_DIR / "disagg.yaml"
        if not recipe_path.exists():
            pytest.skip("disagg.yaml not found")

        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources

                endpoints = config.backend.allocate_endpoints(
                    num_prefill=r.num_prefill,
                    num_decode=r.num_decode,
                    num_agg=r.num_agg,
                    gpus_per_prefill=r.gpus_per_prefill,
                    gpus_per_decode=r.gpus_per_decode,
                    gpus_per_agg=r.gpus_per_agg,
                    gpus_per_node=r.gpus_per_node,
                    available_nodes=self.RACK.nodes(),
                )

                prefill_eps = [e for e in endpoints if e.mode == "prefill"]
                decode_eps = [e for e in endpoints if e.mode == "decode"]

                assert len(prefill_eps) == r.num_prefill
                assert len(decode_eps) == r.num_decode

                for ep in prefill_eps:
                    assert ep.total_gpus == r.gpus_per_prefill
                for ep in decode_eps:
                    assert ep.total_gpus == r.gpus_per_decode


class TestQwen32BCluster:
    """Qwen3-32B configs with shared node allocation (decode_nodes=0)."""

    RACK = H100Rack
    RECIPES = list((RECIPES_DIR / "qwen3-32b").glob("*.yaml")) if (RECIPES_DIR / "qwen3-32b").exists() else []

    @pytest.mark.parametrize("recipe_path", RECIPES, ids=lambda p: p.name)
    def test_config_loads(self, recipe_path):
        """Qwen3-32B configs load correctly."""
        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                assert config.name is not None
                assert config.resources.gpus_per_node == 8

    def test_disagg_kv_router_shared_node_allocation(self):
        """disagg-kv-sglang.yaml: 6P+2D on 2 nodes with decode_nodes=0."""
        recipe_path = RECIPES_DIR / "qwen3-32b" / "disagg-kv-sglang.yaml"
        if not recipe_path.exists():
            pytest.skip("disagg-kv-sglang.yaml not found")

        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources

                # Verify decode_nodes=0 triggers inheritance from prefill
                assert r.decode_nodes == 0, "decode_nodes should be 0"
                assert r.gpus_per_prefill == 2, "prefill TP should be 2"
                assert r.gpus_per_decode == 2, "decode TP should inherit 2 from prefill"

                # Allocate endpoints
                nodes = self.RACK.nodes()[:2]
                endpoints = allocate_endpoints(
                    num_prefill=r.num_prefill,
                    num_decode=r.num_decode,
                    num_agg=0,
                    gpus_per_prefill=r.gpus_per_prefill,
                    gpus_per_decode=r.gpus_per_decode,
                    gpus_per_agg=8,
                    gpus_per_node=r.gpus_per_node,
                    available_nodes=nodes,
                )

                prefill_eps = [e for e in endpoints if e.mode == "prefill"]
                decode_eps = [e for e in endpoints if e.mode == "decode"]

                assert len(prefill_eps) == 6
                assert len(decode_eps) == 2

                # Check prefill allocation: first 4 on node0, next 2 on node1
                for i, ep in enumerate(prefill_eps[:4]):
                    assert ep.nodes[0] == nodes[0], f"prefill {i} should be on node0"
                for i, ep in enumerate(prefill_eps[4:]):
                    assert ep.nodes[0] == nodes[1], f"prefill {i+4} should be on node1"

                # Check decode allocation: on node1 (GPUs 4-5, 6-7)
                for ep in decode_eps:
                    assert ep.nodes[0] == nodes[1], "decode should be on node1"

                # Verify GPU indices don't overlap on shared node (node1)
                node1_prefill_gpus = set()
                for ep in prefill_eps:
                    if ep.nodes[0] == nodes[1]:
                        node1_prefill_gpus.update(ep.gpu_indices)

                node1_decode_gpus = set()
                for ep in decode_eps:
                    node1_decode_gpus.update(ep.gpu_indices)

                assert node1_prefill_gpus.isdisjoint(node1_decode_gpus), (
                    f"GPU overlap on node1! prefill uses {node1_prefill_gpus}, decode uses {node1_decode_gpus}"
                )

    def test_disagg_kv_router_cuda_visible_devices(self):
        """Processes on shared node have non-overlapping CUDA_VISIBLE_DEVICES."""
        recipe_path = RECIPES_DIR / "qwen3-32b" / "disagg-kv-sglang.yaml"
        if not recipe_path.exists():
            pytest.skip("disagg-kv-sglang.yaml not found")

        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources

                nodes = self.RACK.nodes()[:2]
                endpoints = allocate_endpoints(
                    num_prefill=r.num_prefill,
                    num_decode=r.num_decode,
                    num_agg=0,
                    gpus_per_prefill=r.gpus_per_prefill,
                    gpus_per_decode=r.gpus_per_decode,
                    gpus_per_agg=8,
                    gpus_per_node=r.gpus_per_node,
                    available_nodes=nodes,
                )

                processes = endpoints_to_processes(endpoints)

                # Group processes by node
                node1_processes = [p for p in processes if p.node == nodes[1]]

                # Should have 2 prefill + 2 decode = 4 processes on node1
                assert len(node1_processes) == 4, f"Expected 4 processes on node1, got {len(node1_processes)}"

                # Each process should have unique, non-overlapping GPU indices
                all_gpus_on_node1 = set()
                for proc in node1_processes:
                    for gpu in proc.gpu_indices:
                        assert gpu not in all_gpus_on_node1, (
                            f"GPU {gpu} assigned to multiple processes on {nodes[1]}!"
                        )
                        all_gpus_on_node1.add(gpu)

                # All 8 GPUs on node1 should be used
                assert all_gpus_on_node1 == {0, 1, 2, 3, 4, 5, 6, 7}, (
                    f"Expected all 8 GPUs used on node1, got {all_gpus_on_node1}"
                )

                # Verify CUDA_VISIBLE_DEVICES strings are correct
                for proc in node1_processes:
                    cvd = proc.cuda_visible_devices
                    expected_gpus = sorted(proc.gpu_indices)
                    expected_cvd = ",".join(str(g) for g in expected_gpus)
                    assert cvd == expected_cvd, f"Expected CUDA_VISIBLE_DEVICES={expected_cvd}, got {cvd}"

    def test_disagg_kv_router_total_allocation_fits(self):
        """Total GPU allocation fits within declared nodes."""
        recipe_path = RECIPES_DIR / "qwen3-32b" / "disagg-kv-sglang.yaml"
        if not recipe_path.exists():
            pytest.skip("disagg-kv-sglang.yaml not found")

        with patch.dict(os.environ, self.RACK.slurm_env(), clear=False):
            with patch("subprocess.run", side_effect=self.RACK.mock_scontrol()):
                config = load_config(str(recipe_path))
                r = config.resources

                total_gpus_needed = (
                    r.num_prefill * r.gpus_per_prefill
                    + r.num_decode * r.gpus_per_decode
                )
                total_gpus_available = r.total_nodes * r.gpus_per_node

                assert total_gpus_needed <= total_gpus_available, (
                    f"Need {total_gpus_needed} GPUs but only have {total_gpus_available} "
                    f"({r.total_nodes} nodes × {r.gpus_per_node} GPUs)"
                )
