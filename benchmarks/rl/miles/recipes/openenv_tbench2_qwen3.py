#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEnv Terminal-Bench-2 GRPO for Qwen3-4B against a shared env server.

Miles ships the Terminal-Bench-2 agentic adapter and a launcher for GLM-4.7-Flash
(examples/experimental/openenv/run-openenv-tbench2.py: TP=4, EP=2, GLM parsers).
This recipe keeps that adapter and the shared launch helpers and swaps in a dense
Qwen3-4B profile (TP=2, EP=1, Qwen parsers, TITO model qwen3), so a B200 pool runs
agentic rollouts against one tbench2_env server (--openenv-env-url) that needs no
Docker and no hosted sandbox.

Launched by benchmarks/rl/miles/launch.sh with MILES_RECIPE pointing at this file
(an absolute /benchmarks path). Every field below is overridable through
MILES_SCRIPT_<FIELD>. Extra environment:

  OPENENV_SITE   a site-packages directory holding tbench2_env and openenv, appended
                 to the ray job's PYTHONPATH so the rollout workers can import the
                 env client without shadowing the image's own packages.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import miles.utils.external_utils.command_utils as U
import typer

MILES_ROOT = Path(os.environ.get("MILES_ROOT", "/root/miles"))
OPENENV_EXAMPLE_DIR = MILES_ROOT / "examples" / "experimental" / "openenv"
sys.path.insert(0, str(OPENENV_EXAMPLE_DIR))

import openenv_launch_common as C  # noqa: E402  (lives in Miles's openenv example dir)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "qwen3-4B"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths. ref_load is an existing torch_dist checkpoint, so prepare() is skipped.
    skip_prepare: bool = True
    base_dir: str = "/root"
    model_name: str = "Qwen3-4B"
    hf_checkpoint: str = "/data/home/idhanani/models/Qwen3-4B"
    ref_load: str = "/data/home/idhanani/models/Qwen3-4B_torch_dist"
    save_dir: str = "/data/home/idhanani/miles-runs/qwen3-4b-openenv-tbench2/"
    prompt_data: str = "/data/home/idhanani/openenv/tbench2_train_8.jsonl"

    # Dense profile (from Miles scripts/run_qwen3_dense.py for Qwen3-4B).
    tensor_model_parallel_size: int = 2
    max_tokens_per_gpu: int = 9216
    rollout_num_gpus_per_engine: int = 2
    sglang_mem_fraction_static: float = 0.7

    # Training settings (small; multi-turn so responses run long).
    num_rollout: int = 2
    max_seq_len: int = 16384
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 4
    global_batch_size: int = 32

    # OpenEnv settings, same names and env vars as the upstream launcher.
    openenv_env_url: str = os.environ.get("OPENENV_ENV_URL", "http://localhost:8003")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    openenv_max_turns: int = int(os.environ.get("OPENENV_MAX_TURNS", "12"))
    openenv_max_rollout_time_seconds: int = int(os.environ.get("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "1200"))
    openenv_tb2_tasks_dir: str = os.environ.get("OPENENV_TB2_TASKS_DIR", "")
    openenv_sandbox_backend: str = os.environ.get("OPENENV_SANDBOX_BACKEND", "")
    daytona_api_key_file: str = os.environ.get("DAYTONA_API_KEY_FILE", "")
    e2b_api_key_file: str = os.environ.get("E2B_API_KEY_FILE", "")
    modal_config_file: str = os.environ.get("MODAL_CONFIG_PATH", "")
    dump_details: str = os.environ.get("OPENENV_DUMP_DETAILS", "")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "openenv-tbench2-qwen3")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "openenv-tbench2-qwen3"

    # Prometheus settings (off: nothing scrapes it in this job)
    use_prometheus: bool = False
    prometheus_port: int = 9090
    prometheus_run_name: str = "openenv-tbench2-qwen3"


def prepare(args: ScriptArgs) -> None:
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs) -> None:
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} --save {args.save_dir} --save-interval 100 "
    )

    # The shared helper pins --num-rollout 40; this recipe is sized for a short run.
    rollout_args = C.rollout_args(args).replace("--num-rollout 40 ", f"--num-rollout {args.num_rollout} ")

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-tool-call-parser qwen25 "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-router-port 31000 "
    )

    agent_args = C.agent_args("qwen3", sandbox_backend=C.resolve_sandbox_backend(args))

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_nodes * args.num_gpus_per_node} "
    )

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    dump_args = f"--dump-details {args.dump_details} " if args.dump_details else ""

    train_args = (
        f"{ckpt_args}{rollout_args}{C.optimizer_args()}{C.grpo_args()}{C.wandb_args(args)}"
        f"{C.prometheus_args(args)}{perf_args}{sglang_args}{agent_args}{misc_args}{debug_args}{dump_args}"
    )

    extra_env_vars = C.base_env_vars(args, str(OPENENV_EXAMPLE_DIR), args.megatron_path, U.repo_base_dir)
    C.apply_optional_env_vars(extra_env_vars, args)
    openenv_site = os.environ.get("OPENENV_SITE", "")
    if openenv_site:
        # Appended, not prepended: the workers still see the image's own packages first,
        # and only tbench2_env / openenv come from this directory.
        extra_env_vars["PYTHONPATH"] = f"{extra_env_vars['PYTHONPATH']}:{openenv_site}"

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    C.cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
