# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentPerf harness-run porter (scripts/agentperf/port_harness_run.py)."""

import importlib.util
from pathlib import Path

import yaml

from srtctl.benchmarks.base import SCRIPTS_DIR

_spec = importlib.util.spec_from_file_location(
    "port_harness_run", SCRIPTS_DIR / "agentperf" / "port_harness_run.py"
)
porter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(porter)


CLIENT_BANNER = """0: ============================================================
0: AA-AGENTPERF: Deterministic Load Test
0: ============================================================
0:   Base URL: http://node-01:8333/v1
0:   Model: /lustre/proj/models/dsv4/hf-abc_orig
0:   Server type: trtllm
0:   Concurrencies: [1010]
0:   Max workers: 8
0:   Phase Timeout: 2400.0s
0:   Max ISL offset: 10
0:   User assignments: /tmp/agentperf-1/data/assignments-20k.json
0:   Max tokens: 2000
0:   Reasoning effort: high
0:   Tool-call delays: enabled
0:   Conversation routing headers: enabled
0:   Dynamo conv-aware routing (X-Dynamo-Session-ID): enabled
0:   Seed: 42
0: ============================================================
0: Loading trajectories from /tmp/agentperf-1/data/traj_500.jsonl...
"""

WORKER_ENV = (
    "TLLM_LOG_LEVEL=INFO TRTLLM_ENABLE_PDL=1 DYN_KV_BLOCK_SIZE=128 DYN_UCX_TLS=cuda_ipc,sm,self,tcp "
    "CUDA_VISIBLE_DEVICES=0,1,2,3 ETCD_ENDPOINTS=http://node-09:2379 PATH=/opt/bin "
    "DYN_TOOL_CALL_PARSER=deepseek_v4 PYTHONUNBUFFERED=1"
)

JOB_LOG = f"""[server] Executing: srun -l --export=ALL --nodelist node-09 x.sqsh bash start.sh CTX 0 /model 8102 e2e '1010' true /logs '0-0' /r/ctx_config.yaml '{WORKER_ENV}' none none &> /r/3_output_CTX_0.log &
[server] Executing: srun -l --export=ALL --nodelist node-10 x.sqsh bash start.sh CTX 1 /model 8103 e2e '1010' true /logs '0-0' /r/ctx_config.yaml '{WORKER_ENV}' none none &> /r/3_output_CTX_1.log &
[server] Executing: srun -l --export=ALL --nodelist node-01,node-02,node-03 x.sqsh bash start.sh GEN 0 /model 8101 e2e '1010' true /logs '0-0' /r/gen_config.yaml '{WORKER_ENV}' none none &> /r/3_output_GEN_0.log &
[server] Executing: srun -l --export=ALL,ETCD_ENDPOINTS=http://node-09:2379,ROUTER_MODE=kv,DYN_KV_BLOCK_SIZE=128,DYN_REQUEST_PLANE=tcp,DYN_FRONTEND_ENABLE_KV_EVENTS=0,DYN_ROUTER_TEMPERATURE=0 --nodelist node-09 bash infra.sh frontend node-09 8333 &> /r/4_output_frontend.log &
"""


def make_run_dir(tmp_path: Path, with_taskset: bool = False) -> Path:
    d = tmp_path / "c1010"
    (d / "agentperf").mkdir(parents=True)
    (d / "job_params.env").write_text(
        'MODEL_PATH="/lustre/proj/models/dsv4/hf-abc_orig"\n'
        'CONCURRENCIES="1010"\n'
        'CONTAINER_IMAGE="/lustre/proj/images/dyn.sqsh"\n'
        'SERVER_BACKEND="dynamo"\n'
    )
    (d / "ctx_config.yaml").write_text(
        yaml.safe_dump({"tensor_parallel_size": 4, "kv_cache_config": {"dtype": "fp8"}})
    )
    (d / "gen_config.yaml").write_text(
        yaml.safe_dump({
            "tensor_parallel_size": 12,
            "moe_config": {"load_balancer": "/lustre/proj/eplb/gen.yaml"},
        })
    )
    (d / "client_cmds_base.sh").write_text(
        "srun ... bash -c ' set -e; cd /lustre/proj/agentperf-client-worktrees/abc123; "
        "uv run python agentperf/run.py --config x.yaml'"
    )
    (d / "client.log").write_text(CLIENT_BANNER)
    (d / "job.log").write_text(JOB_LOG)
    (d / "agentperf" / "m__1010u__phase0__dur2400s__settle240s__traj3.json").write_text("{}")
    (d / "3_output_CTX_0.log").write_text("taskset -c 0-35 numactl -m 0,1\n" if with_taskset else "clean\n")
    return d


def run_porter(run_dir: Path, tmp_path: Path, extra: list[str] | None = None):
    out, wl = tmp_path / "recipe.yaml", tmp_path / "workload.yaml"
    rc = porter.main([str(run_dir), "--out", str(out), "--workload-out", str(wl), *(extra or [])])
    assert rc == 0
    return yaml.safe_load(out.read_text()), yaml.safe_load(wl.read_text())


class TestAgentPerfPorter:
    def test_full_port(self, tmp_path):
        recipe, workload = run_porter(make_run_dir(tmp_path), tmp_path)

        # model + default /lustre -> /scratch rewrite
        assert recipe["model"]["path"] == "/scratch/proj/models/dsv4/hf-abc_orig"
        assert recipe["model"]["container"] == "/scratch/proj/images/dyn.sqsh"

        # topology from job.log: 2 CTX single-node workers, 1 GEN over 3 nodes, 4 GPUs/node
        r = recipe["resources"]
        assert (r["prefill_workers"], r["prefill_nodes"]) == (2, 2)
        assert (r["decode_workers"], r["decode_nodes"]) == (1, 3)
        assert r["gpus_per_node"] == 4

        # env translation: rename, drop, passthrough
        penv = recipe["backend"]["prefill_environment"]
        assert penv["DYN_TRTLLM_KV_BLOCK_SIZE"] == "128"
        assert penv["UCX_TLS"] == "cuda_ipc,sm,self,tcp"
        assert "DYN_KV_BLOCK_SIZE" not in penv
        assert "CUDA_VISIBLE_DEVICES" not in penv and "ETCD_ENDPOINTS" not in penv and "PATH" not in penv
        assert penv["DYN_TOOL_CALL_PARSER"] == "deepseek_v4"
        assert penv["DYN_LOG"] == "info"  # pinned to override srt-slurm's noisier default
        assert recipe["frontend"]["env"]["HOME"] == "/tmp"  # image HOME is read-only

        # engine configs verbatim + rewritten paths
        assert recipe["backend"]["trtllm_config"]["prefill"]["tensor_parallel_size"] == 4
        assert recipe["backend"]["trtllm_config"]["decode"]["moe_config"]["load_balancer"] == "/scratch/proj/eplb/gen.yaml"

        # frontend args derived from env; parser flags must NOT be ported
        fargs = recipe["frontend"]["args"]
        assert fargs["router-mode"] == "kv"
        assert fargs["no-router-kv-events"] is True
        assert fargs["kv-cache-block-size"] == 128
        assert "dyn-tool-call-parser" not in fargs

        # baseline: no cpu pinning detected
        assert recipe["backend"]["numa_cpu_bind"] is False

        # benchmark section
        b = recipe["benchmark"]
        assert b["type"] == "agentperf"
        assert b["concurrency"] == 1010
        assert b["agentperf_client_dir"] == "/scratch/proj/agentperf-client-worktrees/abc123"

        # workload knobs from the banner + settle from the filename stem
        assert workload["settling_time_seconds"] == 240.0
        assert workload["phase_timeout_seconds"] == 2400.0
        assert workload["max_workers"] == 8
        assert workload["max_tokens"] == 2000
        assert workload["reasoning_effort"] == "high"
        assert workload["seed"] == 42
        assert workload["send_conversation_routing_headers"] is True
        assert workload["use_dynamo_conv_aware_routing"] is True
        assert workload["base_url"].startswith("http://placeholder")

    def test_taskset_detection(self, tmp_path):
        recipe, _ = run_porter(make_run_dir(tmp_path, with_taskset=True), tmp_path)
        assert recipe["backend"]["numa_cpu_bind"] is True

    def test_dataset_root_resolution(self, tmp_path):
        root = tmp_path / "datasets"
        root.mkdir()
        (root / "traj_500.jsonl").write_text("{}")
        (root / "assignments-20k.json").write_text("{}")
        _, workload = run_porter(make_run_dir(tmp_path), tmp_path, ["--dataset-root", str(root)])
        assert workload["trajectory_path"] == str(root / "traj_500.jsonl")
        assert workload["user_assignments_path"] == str(root / "assignments-20k.json")

    def test_unresolved_datasets_become_placeholders(self, tmp_path):
        _, workload = run_porter(make_run_dir(tmp_path), tmp_path)
        assert workload["trajectory_path"].startswith("/TODO/")
        assert workload["user_assignments_path"].startswith("/TODO/")

    def test_recipe_is_schema_loadable(self, tmp_path):
        """The generated recipe must at least round-trip through SrtConfig's schema."""
        recipe, _ = run_porter(make_run_dir(tmp_path), tmp_path)
        from srtctl.core.schema import SrtConfig

        cfg = SrtConfig.Schema().load(recipe)
        assert cfg.benchmark.type == "agentperf"
        assert cfg.backend.numa_cpu_bind is False
