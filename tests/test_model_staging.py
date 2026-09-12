# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for lustre->node-local model staging (model.stage_dir)."""

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from srtctl.backends import TRTLLMProtocol, TRTLLMServerConfig
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import DynamoConfig, SrtConfig


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
    @pytest.mark.parametrize(
        ("publishing", "expected_metrics", "expected_events", "expected_flags"),
        [
            ({}, True, None, ("--publish-metrics",)),
            ({"publish_metrics": False}, False, None, ()),
            ({"publish_metrics": True, "publish_events_and_metrics": None}, True, None, ("--publish-metrics",)),
            ({"publish_metrics": False, "publish_events_and_metrics": None}, False, None, ()),
            ({"publish_metrics": True, "publish_events_and_metrics": False}, True, False, ()),
            ({"publish_metrics": False, "publish_events_and_metrics": False}, False, False, ()),
            (
                {"publish_metrics": False, "publish_events_and_metrics": True},
                False,
                True,
                ("--publish-events-and-metrics",),
            ),
            (
                {"publish_metrics": True, "publish_events_and_metrics": True},
                True,
                True,
                ("--publish-metrics", "--publish-events-and-metrics"),
            ),
        ],
    )
    def test_trtllm_publishing_defaults_and_schema_roundtrip(
        self, publishing, expected_metrics, expected_events, expected_flags
    ):
        data = {
            "name": "publishing-test",
            "model": {"path": "/lustre/m", "container": "trtllm", "precision": "fp4"},
            "resources": {"gpu_type": "gb300", "gpus_per_node": 4, "agg_nodes": 1, "agg_workers": 1},
            "backend": {"type": "trtllm", **publishing},
        }
        schema = SrtConfig.Schema()
        config = schema.load(data)
        dumped = schema.dump(config)
        reloaded = schema.load(dumped)

        assert config.backend.publish_metrics is expected_metrics
        assert config.backend.publish_events_and_metrics is expected_events
        assert dumped["backend"]["publish_metrics"] is expected_metrics
        assert dumped["backend"]["publish_events_and_metrics"] is expected_events
        assert reloaded.backend.publish_metrics is expected_metrics
        assert reloaded.backend.publish_events_and_metrics is expected_events
        assert TRTLLMProtocol(**publishing).dynamo_metrics_flags == expected_flags
        assert config.backend.dynamo_metrics_flags == expected_flags
        assert reloaded.backend.dynamo_metrics_flags == expected_flags

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
            self._proc(),
            [self._proc()],
            self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="trtllm_serve",
        )
        assert "/raid/scratch/models/DeepSeek-V4-Pro" in cmd
        assert "/model" not in cmd

    def test_trtllm_serve_aggregate_worker_binds_public_port(self, tmp_path):
        process = self._proc()
        process = replace(process, endpoint_mode="agg")
        runtime = self._runtime_mock(tmp_path, "/model")
        runtime.frontend_port = 8000
        backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(aggregated={"tensor_parallel_size": 8}))

        cmd = backend.build_worker_command(
            process,
            [process],
            runtime,
            frontend_type="trtllm_serve",
        )

        assert cmd[cmd.index("--port") + 1] == "8000"

    def test_dynamo_worker_uses_staged_path(self, tmp_path):
        backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}))
        cmd = backend.build_worker_command(
            self._proc(),
            [self._proc()],
            self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="dynamo",
        )
        # dynamo path passes it as --model-path
        assert "/raid/scratch/models/DeepSeek-V4-Pro" in cmd

    @pytest.mark.parametrize("mode", ["prefill", "decode", "agg"])
    @pytest.mark.parametrize(
        ("publishing", "expected_flags"),
        [
            ({}, ["--publish-metrics"]),
            ({"publish_metrics": False}, []),
            ({"publish_metrics": True, "publish_events_and_metrics": None}, ["--publish-metrics"]),
            ({"publish_metrics": False, "publish_events_and_metrics": None}, []),
            ({"publish_metrics": True, "publish_events_and_metrics": False}, []),
            ({"publish_metrics": False, "publish_events_and_metrics": False}, []),
            ({"publish_metrics": False, "publish_events_and_metrics": True}, ["--publish-events-and-metrics"]),
            (
                {"publish_metrics": True, "publish_events_and_metrics": True},
                ["--publish-metrics", "--publish-events-and-metrics"],
            ),
        ],
    )
    def test_dynamo_worker_publishing_policy(self, tmp_path, mode, publishing, expected_flags):
        backend = TRTLLMProtocol(**publishing)
        assert backend.publish_metrics is publishing.get("publish_metrics", True)
        assert backend.publish_events_and_metrics is publishing.get("publish_events_and_metrics")
        assert backend.dynamo_metrics_flags == tuple(expected_flags)
        process = replace(self._proc(), endpoint_mode=mode)
        cmd = backend.build_worker_command(
            process,
            [process],
            self._runtime_mock(tmp_path, "/raid/scratch/models/DeepSeek-V4-Pro"),
            frontend_type="dynamo",
        )
        assert sorted(arg for arg in cmd if isinstance(arg, str) and arg.startswith("--publish-")) == sorted(
            expected_flags
        )

    @pytest.mark.parametrize("mode", ["prefill", "decode", "agg"])
    @pytest.mark.parametrize("publish_metrics", [False, True])
    @pytest.mark.parametrize("publish_events_and_metrics", [None, False, True])
    def test_native_worker_ignores_dynamo_publishing_options(
        self, tmp_path, mode, publish_metrics, publish_events_and_metrics
    ):
        process = replace(self._proc(), endpoint_mode=mode)
        runtime = self._runtime_mock(tmp_path, "/model")
        runtime.frontend_port = 8000
        baseline = TRTLLMProtocol(publish_metrics=False, publish_events_and_metrics=False)
        backend = TRTLLMProtocol(publish_metrics=publish_metrics, publish_events_and_metrics=publish_events_and_metrics)

        expected = baseline.build_worker_command(process, [process], runtime, frontend_type="trtllm_serve")
        actual = backend.build_worker_command(process, [process], runtime, frontend_type="trtllm_serve")

        assert actual == expected
        assert "trtllm-serve" in actual
        assert not any(arg.startswith("--publish-") for arg in actual)

    @pytest.mark.parametrize("publish_metrics", [False, True])
    @pytest.mark.parametrize("publish_events_and_metrics", [None, False, True])
    def test_sidecar_worker_ignores_dynamo_publishing_options(
        self, tmp_path, publish_metrics, publish_events_and_metrics
    ):
        process = replace(self._proc(), endpoint_mode="agg")
        runtime = self._runtime_mock(tmp_path, "/model")
        runtime.dynamo = DynamoConfig(sidecar=True)
        baseline = TRTLLMProtocol(publish_metrics=False, publish_events_and_metrics=False)
        backend = TRTLLMProtocol(publish_metrics=publish_metrics, publish_events_and_metrics=publish_events_and_metrics)

        expected = baseline.build_worker_command(process, [process], runtime, frontend_type="dynamo")
        actual = backend.build_worker_command(process, [process], runtime, frontend_type="dynamo")

        assert actual == expected
        assert "dynamo.trtllm.sidecar" in " ".join(actual)
        assert "--publish-" not in " ".join(actual)
