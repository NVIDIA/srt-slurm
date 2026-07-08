# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for lustre->node-local model staging (model.stage_dir)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from srtctl.backends import TRTLLMProtocol, TRTLLMServerConfig
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig


def _runtime(*, staged=None, hf=False, model="/lustre/DeepSeek-V4-Pro"):
    return RuntimeContext(
        job_id="1",
        run_name="r",
        nodes=Nodes(head="n0", bench="n0", infra="n0", worker=("n1", "n2")),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=Path("/tmp/logs"),
        model_path=Path(model),
        container_image=Path("/img.sqsh"),
        gpus_per_node=4,
        network_interface="eth0",
        is_hf_model=hf,
        staged_model_path=(Path(staged) if staged else None),
    )


class TestWorkerModelArg:
    def test_default_is_model_mount(self):
        assert _runtime().worker_model_arg == "/model"

    def test_staged_path_wins(self):
        rt = _runtime(staged="/raid/scratch/models/DeepSeek-V4-Pro")
        assert rt.worker_model_arg == "/raid/scratch/models/DeepSeek-V4-Pro"

    def test_hf_uses_model_id(self):
        rt = _runtime(hf=True, model="deepseek-ai/DeepSeek-V4-Pro")
        assert rt.worker_model_arg == "deepseek-ai/DeepSeek-V4-Pro"


class TestSchema:
    def test_stage_dir_loads(self):
        data = {
            "name": "stage-test",
            "model": {
                "path": "/lustre/DeepSeek-V4-Pro",
                "container": "trtllm",
                "precision": "fp4",
                "stage_dir": "/raid/scratch/models",
            },
            "resources": {"gpu_type": "gb300", "gpus_per_node": 4, "agg_nodes": 1, "agg_workers": 1},
            "backend": {"type": "trtllm"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = SrtConfig.from_yaml(Path(f.name))
        assert config.model.stage_dir == "/raid/scratch/models"

    def test_stage_dir_defaults_none(self):
        data = {
            "name": "no-stage",
            "model": {"path": "/lustre/m", "container": "trtllm", "precision": "fp4"},
            "resources": {"gpu_type": "gb300", "gpus_per_node": 4, "agg_nodes": 1, "agg_workers": 1},
            "backend": {"type": "trtllm"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = SrtConfig.from_yaml(Path(f.name))
        assert config.model.stage_dir is None


class TestWorkerCommandUsesStagedPath:
    def _proc(self):
        from srtctl.core.topology import Process

        return Process(
            node="n1",
            gpu_indices=frozenset([0]),
            sys_port=8081,
            http_port=6100,
            endpoint_mode="decode",
            endpoint_index=0,
            node_rank=0,
        )

    def _runtime_mock(self, tmp_path, staged_arg):
        rt = MagicMock()
        rt.worker_model_arg = staged_arg
        rt.is_hf_model = False
        rt.model_path = Path("/lustre/DeepSeek-V4-Pro")
        rt.log_dir = Path(tmp_path)
        return rt

    def test_trtllm_serve_worker_uses_staged_path(self, tmp_path):
        backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}))
        cmd = backend.build_worker_command(
            self._proc(), [self._proc()], self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="trtllm_serve",
        )
        assert "/raid/scratch/models/DeepSeek-V4-Pro" in cmd
        assert "/model" not in cmd

    def test_dynamo_worker_uses_staged_path(self, tmp_path):
        backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}))
        cmd = backend.build_worker_command(
            self._proc(), [self._proc()], self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="dynamo",
        )
        # dynamo path passes it as --model-path
        assert "/raid/scratch/models/DeepSeek-V4-Pro" in cmd

    def test_dynamo_worker_publishes_events_by_default(self, tmp_path):
        backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}))
        cmd = backend.build_worker_command(
            self._proc(), [self._proc()], self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="dynamo",
        )
        assert "--publish-events-and-metrics" in cmd

    def test_dynamo_worker_publish_events_disabled(self, tmp_path):
        backend = TRTLLMProtocol(
            trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}),
            publish_events_and_metrics=False,
        )
        cmd = backend.build_worker_command(
            self._proc(), [self._proc()], self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="dynamo",
        )
        assert "--publish-events-and-metrics" not in cmd
