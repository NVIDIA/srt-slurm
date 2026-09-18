# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``post_eval:`` block: env passthrough and the command override for the eval dispatch."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import PostEvalConfig, SrtConfig

RECIPE = """
name: post-eval-test
model:
  path: /model
  container: /job.sqsh
  precision: bf16
resources:
  gpu_type: h100
  gpus_per_node: 8
  agg_nodes: 1
  agg_workers: 2
  gpus_per_agg: 1
frontend:
  type: dynamo
backend:
  type: sglang
  sglang_config:
    aggregated:
      served-model-name: Qwen/Qwen3-0.6B
benchmark:
  type: sa-bench
  isl: 128
  osl: 128
  concurrencies: "4x8"
"""


def _load(extra: str = "") -> SrtConfig:
    return SrtConfig.Schema().load(yaml.safe_load(RECIPE + extra))


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="1",
        run_name="r",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0",)),
        head_node_ip="10.0.0.10",
        infra_node_ip="10.0.0.10",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/job.sqsh"),
        gpus_per_node=8,
        network_interface=None,
        container_mounts={},
        environment={},
    )


def _run_eval(config: SrtConfig, tmp_path: Path, env: dict[str, str]) -> dict:
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    with (
        patch.dict(os.environ, env, clear=False),
        patch("srtctl.cli.do_sweep.wait_for_port", return_value=True),
        patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc) as srun,
    ):
        assert orchestrator._run_post_eval(threading.Event()) == 0
    return srun.call_args.kwargs


def test_defaults_are_empty() -> None:
    config = _load()
    assert config.post_eval == PostEvalConfig()
    assert config.post_eval.passthrough_env == []
    assert config.post_eval.command is None


def test_passthrough_env_extends_the_builtin_list(tmp_path: Path, monkeypatch) -> None:
    for var in ("EVAL_FRAMEWORK", "EVAL_SUITE", "UNRELATED"):
        monkeypatch.delenv(var, raising=False)
    config = _load("post_eval:\n  passthrough_env:\n    - EVAL_FRAMEWORK\n    - EVAL_SUITE\n")
    kwargs = _run_eval(
        config, tmp_path, {"RUN_EVAL": "true", "EVAL_FRAMEWORK": "lm-eval", "EVAL_SUITE": "gsm8k", "UNRELATED": "x"}
    )
    env = kwargs["env_to_set"]
    assert env["RUN_EVAL"] == "true"  # built-in list still applies
    assert env["EVAL_FRAMEWORK"] == "lm-eval"
    assert env["EVAL_SUITE"] == "gsm8k"
    assert "UNRELATED" not in env
    assert env["MODEL_NAME"] == "Qwen/Qwen3-0.6B"
    assert kwargs["command"][:2] == ["bash", "/srtctl-benchmarks/lm-eval/bench.sh"]


def test_command_override_replaces_the_lm_eval_runner(tmp_path: Path) -> None:
    config = _load('post_eval:\n  command:\n    - bash\n    - /infmax-workspace/evals/run.sh\n    - "{endpoint}"\n')
    kwargs = _run_eval(config, tmp_path, {"RUN_EVAL": "true"})
    assert kwargs["command"] == ["bash", "/infmax-workspace/evals/run.sh", "http://localhost:8000"]
    assert kwargs["env_to_set"]["MODEL_NAME"] == "Qwen/Qwen3-0.6B"


def test_validation() -> None:
    with pytest.raises(ValidationError, match="passthrough_env entries must be environment variable names"):
        _load("post_eval:\n  passthrough_env:\n    - 'not a name'\n")
    with pytest.raises(ValidationError, match="command, if set, must be non-empty"):
        _load("post_eval:\n  command: []\n")
