# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TRT-LLM's served model name override."""

from srtctl.backends.trtllm import TRTLLMProtocol, TRTLLMServerConfig


class TestTRTLLMServedModelName:
    """`--served-model-name` is what a client must put in a request's "model" field.

    It defaults to the checkpoint directory name, which is fine when the client
    can be told what to ask for. A client that hardcodes the name — the MLPerf
    harness treats model identity as part of the benchmark definition — can only
    be met by naming the model its way.
    """

    def test_defaults_to_the_checkpoint_directory_name(self):
        assert TRTLLMProtocol().get_served_model_name("deepseek_r1-torch-fp4-v2") == "deepseek_r1-torch-fp4-v2"

    def test_configured_name_wins(self):
        backend = TRTLLMProtocol(served_model_name="deepseek-ai/deepseek-r1")
        assert backend.get_served_model_name("deepseek_r1-torch-fp4-v2") == "deepseek-ai/deepseek-r1"

    def test_empty_string_falls_back_to_the_default(self):
        """An unset-but-present key should not serve the model under an empty name."""
        assert TRTLLMProtocol(served_model_name="").get_served_model_name("ckpt") == "ckpt"

    def test_is_not_written_into_the_engine_yaml(self):
        """It is a dynamo.trtllm CLI flag; leaking it into the engine config would
        inject an option TRT-LLM's LlmArgs does not define."""
        backend = TRTLLMProtocol(
            served_model_name="deepseek-ai/deepseek-r1",
            trtllm_config=TRTLLMServerConfig(aggregated={"tensor_parallel_size": 4}),
        )
        rendered = backend.get_config_for_mode("agg")
        assert "served_model_name" not in rendered
        assert "served-model-name" not in rendered
        assert rendered == {"tensor_parallel_size": 4}

    def test_reaches_the_worker_command(self):
        """The flag the worker is launched with must reflect the override."""
        from pathlib import Path
        from unittest.mock import MagicMock

        backend = TRTLLMProtocol(served_model_name="deepseek-ai/deepseek-r1")
        runtime = MagicMock()
        runtime.model_path = Path("/models/deepseek_r1-torch-fp4-v2")
        runtime.request_plane = "nats"

        process = MagicMock()
        process.endpoint_mode = "agg"
        cmd = backend.build_worker_command(process, [process], runtime)

        assert "--served-model-name" in cmd
        assert cmd[cmd.index("--served-model-name") + 1] == "deepseek-ai/deepseek-r1"
