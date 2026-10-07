# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for configs/hle/run.sh, the HLE eval run as a custom benchmark in the NeMo Skills container."""

import os
import subprocess
from pathlib import Path

from srtctl.core.config import load_config

REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "configs/hle/run.sh"
EXAMPLE = REPO_ROOT / "examples/features/hle.yaml"


def run_script(
    tmp_path: Path, env: dict[str, str], *, write_metrics: bool = True
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run run.sh with a stub `ns` that records its argv, one call per line.

    The stub writes metrics.json on `eval` unless `write_metrics` is False (a failed prepare or judge step).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    stub = bin_dir / "ns"
    metrics = tmp_path / "out/eval-results/hle/metrics.json"
    write = f'mkdir -p "{metrics.parent}" && echo {{}} > "{metrics}"' if write_metrics else ":"
    stub.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{calls}"\n[ "$1" = eval ] && {write}\nexit 0\n')
    stub.chmod(0o755)

    # The script reads its knobs from env; drop any the caller's shell happens to set.
    knobs = {"MODEL", "SPLIT", "NUM_EXAMPLES", "REPEAT", "OPENAI_API_KEY"}
    base = {k: v for k, v in os.environ.items() if k not in knobs and not k.startswith("JUDGE_")}
    base["PATH"] = f"{bin_dir}{os.pathsep}{base['PATH']}"
    base["OUTPUT_DIR"] = str(tmp_path / "out")
    result = subprocess.run(["bash", str(SCRIPT)], env=base | env, capture_output=True, text=True, check=False)
    return result, calls.read_text().splitlines() if calls.exists() else []


def test_defaults_prepare_then_eval_the_text_split(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, {"MODEL": "Qwen/Qwen3-0.6B", "OPENAI_API_KEY": "sk-test"})

    assert result.returncode == 0, result.stderr
    assert calls[0] == "prepare_data hle"
    args = calls[1].split()
    assert args[0] == "eval"
    for expected in (
        "--server_type=openai",
        "--model=Qwen/Qwen3-0.6B",
        "--server_address=http://localhost:8000/v1",
        "--benchmarks=hle:1",
        "--split=text",
        f"--output_dir={tmp_path / 'out'}",
        "--starting_seed=0",
        "++inference.tokens_to_generate=400000",
        "++max_concurrent_requests=512",
        "++inference.temperature=1.0",
        "++inference.top_p=1.0",
        "++inference.timeout=25000000",
    ):
        assert expected in args
    # Unset knobs leave NeMo Skills' own defaults (whole split, default judge).
    assert not [a for a in args if a.startswith(("++max_samples", "--judge_"))]


def test_knobs_and_judge_override_reach_ns_eval(tmp_path: Path) -> None:
    result, calls = run_script(
        tmp_path,
        {
            "MODEL": "m",
            "SPLIT": "math",
            "NUM_EXAMPLES": "50",
            "REPEAT": "4",
            "JUDGE_MODEL": "judge",
            "JUDGE_SERVER_ADDRESS": "http://judge-host:8000/v1",
            "JUDGE_SERVER_TYPE": "openai",
        },
    )

    assert result.returncode == 0, result.stderr
    args = calls[1].split()
    for expected in (
        "--benchmarks=hle:4",
        "--split=math",
        "++max_samples=50",
        "--judge_model=judge",
        "--judge_server_address=http://judge-host:8000/v1",
        "--judge_server_type=openai",
    ):
        assert expected in args


def test_default_judge_without_key_fails_before_generation(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, {"MODEL": "m"})

    assert result.returncode == 1
    assert "OPENAI_API_KEY is required" in result.stderr
    assert calls == []


def test_model_is_required(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, {"OPENAI_API_KEY": "sk-test"})

    assert result.returncode != 0
    assert "MODEL must be set" in result.stderr
    assert calls == []


def test_example_runs_the_script_in_the_nemo_skills_image() -> None:
    config = load_config(EXAMPLE)
    assert config.benchmark.type == "custom"
    assert config.benchmark.command is not None
    assert "/configs/hle/run.sh" in config.benchmark.command
    assert config.benchmark.env["MODEL"] == config.served_model_name


def test_missing_metrics_fails_the_run(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, {"MODEL": "Qwen/Qwen3-0.6B", "OPENAI_API_KEY": "sk-test"}, write_metrics=False)

    assert len(calls) == 2
    assert result.returncode == 1
    assert "metrics.json was not written" in result.stderr
    assert "=== Done ===" not in result.stdout
