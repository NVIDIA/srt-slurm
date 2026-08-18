# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-generate SA-Bench datasets into ``benchmark.dataset_cache_dir``.

``srtctl cache-inputs`` runs the recipe's container with SA-Bench's prewarm
mode, which builds every dataset the benchmark loop would build and writes it to
the configured cache directory. Only prompt generation runs, so no GPUs, no
frontend and no workers are involved — the expensive tokenize/decode work can
happen while the cluster is busy, and the benchmark job itself starts measuring
almost immediately.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from srtctl.benchmarks.base import SCRIPTS_DIR
from srtctl.benchmarks.sa_bench import (
    DEFAULT_NUM_PROMPTS_MULT,
    DEFAULT_NUM_WARMUP_MULT,
    SABenchRunner,
)
from srtctl.core.config import get_srtslurm_setting
from srtctl.core.runtime import resolve_container_image, resolve_model_path
from srtctl.core.slurm import get_container_mounts_str, get_slurm_job_id
from srtctl.ports import FRONTEND_PUBLIC_PORT

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

# Read by bench.sh to build datasets instead of running the benchmark.
PREWARM_ENV_VAR = "SA_BENCH_PREWARM_ONLY"

JOB_NAME = "srtctl-cache-inputs"

# Used only when the recipe carries no slurm.time_limit. Generous because a
# long-ISL sweep can spend tens of minutes tokenizing.
FALLBACK_TIME_LIMIT = "2:00:00"


@dataclass(frozen=True)
class _PrewarmRuntime:
    """The slice of ``RuntimeContext`` that SA-Bench needs to render its command.

    Prewarming runs without an allocation, so there is no node topology, log dir
    or frontend to describe. Only these fields are meaningful; anything else the
    runner starts reading should fail loudly here rather than silently receive a
    placeholder.
    """

    model_path: Path
    is_hf_model: bool
    container_mounts: dict[Path, Path]
    # bench.sh takes an endpoint as its first argument and ignores it while
    # prewarming. The real default keeps the rendered command honest.
    frontend_port: int = FRONTEND_PUBLIC_PORT


@dataclass(frozen=True)
class CacheInputsPlan:
    """Everything needed to prewarm one recipe, resolved from its config."""

    recipe_name: str
    cache_dir: Path
    container_image: Path
    mounts: dict[Path, Path]
    prompt_counts: tuple[int, ...]
    command: list[str]
    time_limit: str
    account: str | None = None
    partition: str | None = None
    num_workers: int | None = None

    @property
    def attaches_to_current_job(self) -> bool:
        """True when srun joins the caller's allocation instead of queueing one."""
        return bool(get_slurm_job_id())

    def srun_command(self) -> list[str]:
        """Render the srun call that runs the prompt build in the container."""
        srun = ["srun", f"--job-name={JOB_NAME}", "--nodes=1", "--ntasks=1"]

        job_id = get_slurm_job_id()
        if job_id:
            # Inside an allocation already: use it instead of queueing again.
            srun += ["--jobid", job_id, "--overlap"]
        else:
            if self.account:
                srun.append(f"--account={self.account}")
            if self.partition:
                srun.append(f"--partition={self.partition}")
            srun.append(f"--time={self.time_limit}")

        srun += [
            f"--container-image={self.container_image}",
            "--no-container-entrypoint",
            "--no-container-mount-home",
        ]
        if self.mounts:
            srun.append(f"--container-mounts={get_container_mounts_str(self.mounts)}")

        exports = {PREWARM_ENV_VAR: "1"}
        if self.num_workers is not None:
            exports["RANDOM_NUM_WORKERS"] = str(self.num_workers)
        srun.append("--export=ALL," + ",".join(f"{name}={value}" for name, value in exports.items()))

        return srun + self.command


def parse_concurrencies(concurrencies: list[int] | str | None) -> tuple[int, ...]:
    """Accept both recipe spellings: ``[64, 128]`` and ``"64x128"``."""
    if concurrencies is None:
        return ()
    if isinstance(concurrencies, str):
        parts: list[str | int] = [part for part in concurrencies.split("x") if part.strip()]
    else:
        parts = list(concurrencies)
    return tuple(int(part) for part in parts)


def prompt_counts(
    concurrencies: tuple[int, ...],
    num_prompts_mult: int,
    num_warmup_mult: int,
) -> tuple[int, ...]:
    """Datasets a run needs, in the order bench.sh asks for them.

    Warmup and measured runs each draw their own dataset because ``num_prompts``
    changes how many draws come out of the seeded RNG. Identical counts across
    concurrency levels are only built once.
    """
    counts: list[int] = []
    for concurrency in concurrencies:
        candidates = []
        if num_warmup_mult > 0:
            candidates.append(concurrency * num_warmup_mult)
        candidates.append(concurrency * num_prompts_mult)
        for count in candidates:
            if count not in counts:
                counts.append(count)
    return tuple(counts)


def _base_mounts(config: SrtConfig, model_path: Path, is_hf_model: bool) -> dict[Path, Path]:
    """Mounts the prompt build needs: tokenizer, scripts and user extras.

    Deliberately narrower than a benchmark job's mount set — there is no log
    dir to write, no infra binaries to run and no wheelhouse to install from.
    """
    mounts: dict[Path, Path] = {}
    if not is_hf_model:
        mounts[model_path] = Path("/model")
    if SCRIPTS_DIR.exists():
        mounts[SCRIPTS_DIR.resolve()] = Path("/srtctl-benchmarks")

    cluster_mounts = get_srtslurm_setting("default_mounts") or {}
    for host_path, container_path in cluster_mounts.items():
        mounts[Path(os.path.expandvars(host_path)).resolve()] = Path(container_path)

    # Custom tokenizers and other benchmark inputs can live behind extra_mount.
    for mount_spec in config.extra_mount or []:
        host_path, container_path = mount_spec.split(":", 1)
        mounts[Path(os.path.expandvars(host_path)).expanduser().resolve()] = Path(container_path)

    return mounts


def plan_cache_inputs(
    config: SrtConfig,
    *,
    account: str | None = None,
    partition: str | None = None,
    time_limit: str | None = None,
    num_workers: int | None = None,
) -> CacheInputsPlan:
    """Resolve a recipe into a prewarm plan, or explain why it cannot be one."""
    benchmark = config.benchmark

    if benchmark.type != "sa-bench":
        raise ValueError(f"cache-inputs supports benchmark.type 'sa-bench'; this recipe uses '{benchmark.type}'")

    dataset_name = benchmark.dataset_name or "random"
    if dataset_name != "random":
        raise ValueError(
            f"cache-inputs pre-generates the 'random' dataset only; this recipe uses dataset_name '{dataset_name}'"
        )

    if not benchmark.dataset_cache_dir:
        raise ValueError(
            "benchmark.dataset_cache_dir is not set, so there is nowhere to pre-generate into. Add it to the recipe:\n"
            "  benchmark:\n"
            '    dataset_cache_dir: "/lustre/shared/sa-bench-cache"'
        )

    concurrencies = parse_concurrencies(benchmark.concurrencies)
    if not concurrencies:
        raise ValueError("benchmark.concurrencies is required: it determines how many prompts each dataset holds")
    if benchmark.isl is None or benchmark.osl is None:
        raise ValueError("benchmark.isl and benchmark.osl are required to build the random dataset")

    counts = prompt_counts(
        concurrencies,
        benchmark.num_prompts_mult if benchmark.num_prompts_mult is not None else DEFAULT_NUM_PROMPTS_MULT,
        benchmark.num_warmup_mult if benchmark.num_warmup_mult is not None else DEFAULT_NUM_WARMUP_MULT,
    )

    model_path, is_hf_model = resolve_model_path(config.model.path)
    runner = SABenchRunner()
    runtime = cast(
        "RuntimeContext",
        _PrewarmRuntime(
            model_path=model_path,
            is_hf_model=is_hf_model,
            container_mounts=_base_mounts(config, model_path, is_hf_model),
        ),
    )

    return CacheInputsPlan(
        recipe_name=config.name,
        cache_dir=Path(benchmark.dataset_cache_dir).expanduser(),
        container_image=resolve_container_image(config.model.container),
        # Creates the host cache dir, exactly as a benchmark job would.
        mounts=runner.get_container_mounts(config, runtime),
        prompt_counts=counts,
        command=runner.build_command(config, runtime),
        time_limit=time_limit or config.slurm.time_limit or FALLBACK_TIME_LIMIT,
        account=account or config.slurm.account,
        partition=partition or config.slurm.partition,
        num_workers=num_workers,
    )


def run_cache_inputs(plan: CacheInputsPlan) -> int:
    """Run the prewarm step, streaming container output to this terminal."""
    command = plan.srun_command()
    logger.info("srun command: %s", shlex.join(command))
    return subprocess.run(command, check=False).returncode
