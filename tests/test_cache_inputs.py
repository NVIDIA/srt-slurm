# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `srtctl cache-inputs` (SA-Bench dataset prewarming)."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from srtctl.benchmarks.base import SCRIPTS_DIR
from srtctl.benchmarks.sa_bench import DATASET_CACHE_MOUNT
from srtctl.core.cache_inputs import (
    PREWARM_ENV_VAR,
    parse_concurrencies,
    plan_cache_inputs,
    prompt_counts,
)
from srtctl.core.schema import (
    BenchmarkConfig,
    ModelConfig,
    ResourceConfig,
    SlurmConfig,
    SrtConfig,
)

BENCH_SCRIPT = SCRIPTS_DIR / "sa-bench" / "bench.sh"


def _config(tmp_path: Path, model_path: str | None = None, **benchmark_overrides) -> SrtConfig:
    model_dir = tmp_path / "my-model-fp8"
    model_dir.mkdir(exist_ok=True)

    benchmark_fields = {
        "type": "sa-bench",
        "isl": 8192,
        "osl": 1024,
        "concurrencies": "64x128",
        "num_prompts_mult": 8,
        "num_warmup_mult": 1,
        "dataset_cache_dir": str(tmp_path / "cache"),
    }
    benchmark_fields.update(benchmark_overrides)

    return SrtConfig(
        name="prewarm-test",
        model=ModelConfig(path=model_path or str(model_dir), container="my-image:latest", precision="fp4"),
        resources=ResourceConfig(gpu_type="gb200"),
        benchmark=BenchmarkConfig(**benchmark_fields),
        slurm=SlurmConfig(account="my-account", partition="my-partition", time_limit="0:30:00"),
    )


class TestPromptCounts:
    def test_warmup_and_measured_runs_each_get_a_dataset(self):
        """Both runs draw from the seeded RNG, so neither can reuse the other's file."""
        assert prompt_counts((64, 128), num_prompts_mult=8, num_warmup_mult=1) == (64, 512, 128, 1024)

    def test_warmup_is_skipped_when_disabled(self):
        assert prompt_counts((64, 128), num_prompts_mult=8, num_warmup_mult=0) == (512, 1024)

    def test_each_size_is_built_once(self):
        """A warmup count that repeats a measured count must not be built twice."""
        assert prompt_counts((1, 2), num_prompts_mult=2, num_warmup_mult=1) == (1, 2, 4)

    @pytest.mark.parametrize("concurrencies", ["64x128", [64, 128]])
    def test_both_recipe_spellings_are_accepted(self, concurrencies):
        assert parse_concurrencies(concurrencies) == (64, 128)


class TestPlanValidation:
    def test_missing_cache_dir_explains_the_field_to_set(self, tmp_path):
        with pytest.raises(ValueError, match="dataset_cache_dir"):
            plan_cache_inputs(_config(tmp_path, dataset_cache_dir=None))

    def test_other_benchmarks_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="sa-bench"):
            plan_cache_inputs(_config(tmp_path, type="sglang-bench"))

    def test_custom_dataset_is_rejected(self, tmp_path):
        """Only 'random' is generated locally; custom datasets are user files."""
        with pytest.raises(ValueError, match="random"):
            plan_cache_inputs(_config(tmp_path, dataset_name="custom", dataset_path="/data/bench.jsonl"))

    def test_concurrencies_are_required(self, tmp_path):
        with pytest.raises(ValueError, match="concurrencies"):
            plan_cache_inputs(_config(tmp_path, concurrencies=None))

    def test_isl_and_osl_are_required(self, tmp_path):
        with pytest.raises(ValueError, match="isl"):
            plan_cache_inputs(_config(tmp_path, isl=None))


class TestPlan:
    def test_cache_dir_is_created_and_mounted(self, tmp_path):
        """The first prewarm of a new recipe must not fail on a missing dir."""
        config = _config(tmp_path)
        cache_dir = Path(config.benchmark.dataset_cache_dir)
        assert not cache_dir.exists()

        plan = plan_cache_inputs(config)

        assert cache_dir.is_dir()
        assert plan.mounts[cache_dir.resolve()] == Path(DATASET_CACHE_MOUNT)

    def test_tokenizer_and_scripts_are_mounted(self, tmp_path):
        plan = plan_cache_inputs(_config(tmp_path))

        assert plan.mounts[(tmp_path / "my-model-fp8").resolve()] == Path("/model")
        assert plan.mounts[SCRIPTS_DIR.resolve()] == Path("/srtctl-benchmarks")

    def test_hf_models_are_not_mounted(self, tmp_path):
        """An "hf:" id is downloaded in-container, so there is nothing to mount."""
        plan = plan_cache_inputs(_config(tmp_path, model_path="hf:facebook/opt-125m"))

        assert Path("/model") not in plan.mounts.values()

    def test_command_runs_the_benchmark_script_against_the_mounted_cache(self, tmp_path):
        plan = plan_cache_inputs(_config(tmp_path))

        assert plan.command[:2] == ["bash", "/srtctl-benchmarks/sa-bench/bench.sh"]
        assert plan.command[-1] == DATASET_CACHE_MOUNT
        assert "8192" in plan.command

    def test_srun_asks_bench_script_for_prewarm_mode(self, tmp_path):
        plan = plan_cache_inputs(_config(tmp_path))

        srun = plan.srun_command()

        assert f"--export=ALL,{PREWARM_ENV_VAR}=1" in srun
        assert "--container-image=my-image:latest" in srun
        assert srun[-len(plan.command) :] == plan.command

    def test_srun_uses_recipe_slurm_settings(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_JOBID", raising=False)

        srun = plan_cache_inputs(_config(tmp_path)).srun_command()

        assert "--account=my-account" in srun
        assert "--partition=my-partition" in srun
        assert "--time=0:30:00" in srun

    def test_cli_flags_win_over_recipe_slurm_settings(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_JOBID", raising=False)

        plan = plan_cache_inputs(
            _config(tmp_path),
            account="other-account",
            partition="other-partition",
            time_limit="4:00:00",
        )

        srun = plan.srun_command()
        assert "--account=other-account" in srun
        assert "--partition=other-partition" in srun
        assert "--time=4:00:00" in srun

    def test_existing_allocation_is_reused_instead_of_queueing(self, tmp_path, monkeypatch):
        """Run from inside a salloc, prewarming should not wait in the queue again."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")

        plan = plan_cache_inputs(_config(tmp_path))
        srun = plan.srun_command()

        assert plan.attaches_to_current_job
        assert "--jobid" in srun
        assert "--overlap" in srun
        assert not any(arg.startswith(("--time=", "--account=", "--partition=")) for arg in srun)

    def test_worker_count_is_passed_to_the_generator(self, tmp_path):
        srun = plan_cache_inputs(_config(tmp_path), num_workers=32).srun_command()

        assert f"--export=ALL,{PREWARM_ENV_VAR}=1,RANDOM_NUM_WORKERS=32" in srun


class TestBenchScriptPrewarmMode:
    """bench.sh drives the prompt build, so its prewarm path is checked for real."""

    @staticmethod
    def _run(plan_command: list[str], tmp_path: Path, cache_dir: str) -> list[list[str]]:
        """Run bench.sh in prewarm mode with a python3 that only records its args.

        The stand-in is an exported shell function rather than a script on PATH
        so the test does not need an exec-capable temp dir.
        """
        record = tmp_path / "invocations.txt"

        # The container paths in the plan do not exist on the host: point the
        # script and the cache at real ones, leaving every other argument as the
        # container would receive it.
        command = list(plan_command)
        command[1] = str(BENCH_SCRIPT)
        command[-1] = cache_dir

        harness = "\n".join(
            [
                "python3() {",
                # The dependency probe ("python3 -c import ...") must look
                # satisfied so no venv is built for the test.
                '  if [ "$1" = "-c" ]; then return 0; fi',
                f'  echo "$@" >> {shlex.quote(str(record))}',
                "}",
                "export -f python3",
                shlex.join(command),
            ]
        )
        result = subprocess.run(
            ["bash", "-c", harness],
            env={**os.environ, PREWARM_ENV_VAR: "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        if not record.exists():
            return []
        return [line.split() for line in record.read_text().splitlines()]

    def test_every_dataset_the_run_needs_is_built(self, tmp_path):
        plan = plan_cache_inputs(_config(tmp_path))

        invocations = self._run(plan.command, tmp_path, str(plan.cache_dir))

        built = [int(args[args.index("--num-prompts") + 1]) for args in invocations]
        assert built == list(plan.prompt_counts)

    def test_each_build_targets_the_cache_and_never_a_server(self, tmp_path):
        plan = plan_cache_inputs(_config(tmp_path))

        invocations = self._run(plan.command, tmp_path, str(plan.cache_dir))

        for args in invocations:
            assert "--prewarm-dataset-cache" in args
            assert args[args.index("--dataset-cache-dir") + 1] == str(plan.cache_dir)
            assert args[args.index("--random-input-len") + 1] == "8192"
            assert "--max-concurrency" not in args
            assert "--save-result" not in args
