#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frozen dataclass schema definitions for job configuration.

Uses marshmallow_dataclass for type-safe configuration with validation.
All config classes are frozen (immutable) after creation.

Backend configs are defined in srtctl.backends.configs/ for modularity.
"""

import builtins
import dataclasses
import hashlib
import itertools
import logging
import math
import os
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import field
from enum import Enum
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    cast,
    get_args,
)

import yaml
from marshmallow import Schema, ValidationError, fields, validate
from marshmallow_dataclass import dataclass

from srtctl.backends import (
    AtomBackend,
    BackendConfig,
    MockerBackend,
    SGLangBackend,
    TileRTBackend,
    TokenSpeedBackend,
    TRTLLMBackend,
    VLLMBackend,
    VLLMMooncakeKVStoreConfig,
)
from srtctl.backends.base import RoleSettings, WorkerMode
from srtctl.core.formatting import (
    FormattablePath,
    FormattablePathField,
)

# Leaf modules (stdlib and prometheus-free imports), so these cannot cycle back into schema.
from srtctl.core.power.contract import (
    CONTAINER_LOG_DIR,
    GPU_UTIL_METRIC,
    SM_ACTIVE_METRIC,
    TEMPERATURE_METRIC,
    UTILIZATION_METRICS,
    UtilizationMetric,
)
from srtctl.core.power.mapping import DCGM_POWER_MAPPING, PowerMetricMapping
from srtctl.core.roles import COLOCATE, PER_ROLE_ENGINE_KEYS, ROLE_NAMES, ROLE_TO_MODE
from srtctl.core.source import DynamoSourceConfig, is_commit_sha
from srtctl.ports import DYNAMO_SIDECAR_GRPC_PORT
from srtctl.services.config import ServiceConfig

if TYPE_CHECKING:
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

logger = logging.getLogger(__name__)


def _dataclass_default(item: dataclasses.Field) -> Any:
    """The default a dataclass field would take when unset (None when it has none)."""
    if item.default is not dataclasses.MISSING:
        return item.default
    if item.default_factory is not dataclasses.MISSING:
        return item.default_factory()
    return None


# Local copies of srtctl.core.power.contract values so that loading a config
# never imports the power package; equality is pinned by tests.
_BENCHMARK_TYPE_SA_BENCH = "sa-bench"
_DCGM_POWER_MAX_SAMPLE_GAP_SECONDS = 3.0
_CPU_POWER_MAX_SAMPLE_GAP_SECONDS = 3.0
_DCGM_POWER_COLLECT_CYCLE_TIMEOUT_GRACE_SECONDS = 1.0


def _is_safe_relative_subpath(value: str) -> bool:
    if not value or value.startswith(("/", "~")):
        return False
    parts = PurePosixPath(value).parts
    return bool(parts) and not any(part in ("..", "") for part in parts)


def _is_finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


# ============================================================================
# Reporting Configuration
# ============================================================================


@dataclass(frozen=True)
class ReportingStatusConfig:
    """Status reporting configuration."""

    # Base URL of one status collector; srtctl POSTs job lifecycle events there (see status-api-spec.md).
    endpoint: str | None = None
    # Several collectors, each sent every event; merged with `endpoint`, deduplicated, trailing slash dropped.
    endpoints: list[str] | None = None
    # Name of the environment variable holding the bearer token the reporter sends as
    # ``Authorization: Bearer`` on every request (default SRTCTL_STATUS_TOKEN). Only the
    # variable name belongs in a recipe: the resolved config is written to the lockfile
    # and the log directory, so a literal token there would leak.
    token_env: str | None = None
    # Seconds between uploads of raw logs and Tachometer captures to every endpoint.
    # Unset disables streaming; lifecycle events are unaffected.
    logging_stream_interval: float | None = field(
        default=None,
        metadata={
            "marshmallow_field": fields.Float(data_key="logging-stream-interval", load_default=None, allow_none=True)
        },
    )

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.logging_stream_interval is not None and not _is_finite_positive(self.logging_stream_interval):
            raise ValidationError(
                f"reporting.status.logging-stream-interval must be positive, got {self.logging_stream_interval!r}"
            )


@dataclass(frozen=True)
class ReportingConfig:
    """Reporting configuration for status updates, AI analysis, and log exports."""

    # Status collector endpoints that receive job lifecycle events. Unset sends nothing.
    status: ReportingStatusConfig | None = None
    # Failure analysis run after a failed job. Unset disables it.
    ai_analysis: "AIAnalysisConfig | None" = None
    # Upload of the log directory to S3-compatible storage after the run. Unset disables it.
    s3: "S3Config | None" = None

    Schema: ClassVar[type[Schema]] = Schema


# ============================================================================
# Cluster Configuration (srtslurm.yaml)
# ============================================================================


# Default prompt template for AI-powered failure analysis
DEFAULT_AI_ANALYSIS_PROMPT = """
You are analyzing benchmark failure logs for an LLM serving system (SGLang/Dynamo).

You have access to:
- Log files in {log_dir}
- The `gh` CLI tool (authenticated) to search GitHub PRs

Your task:
1. Read the log files and identify the root cause of failure
2. Search recent PRs (last {pr_days} days) in {repos} for potentially related changes
3. Write your analysis to ai_analysis.md in {log_dir}

Your analysis should include:
- Summary of the failure
- Root cause identification
- Key error messages found
- Related PRs (if any)
- Suggested next steps

Start by listing and reading the log files, then investigate.
"""


@dataclass(frozen=True)
class AIAnalysisConfig:
    """AI-powered failure analysis configuration.

    This config is typically set in srtslurm.yaml (cluster config) to centralize
    secrets and allow cluster-wide customization. Individual job configs can
    override with `ai_analysis.enabled: false` to disable for specific jobs.

    Uses OpenRouter for Claude Code authentication, which provides a simple API key
    approach that works well in headless/automated environments.
    See: https://openrouter.ai/docs/guides/claude-code-integration

    Attributes:
        enabled: Whether to run AI analysis on benchmark failures
        openrouter_api_key: OpenRouter API key (falls back to OPENROUTER_API_KEY env var)
        gh_token: GitHub token for gh CLI (falls back to GH_TOKEN env var)
        repos_to_search: GitHub repos to search for related PRs
        pr_search_days: Number of days to look back for PRs
        prompt: Custom prompt template (uses DEFAULT_AI_ANALYSIS_PROMPT if None)
            Available variables: {log_dir}, {repos}, {pr_days}
    """

    enabled: bool = False
    openrouter_api_key: str | None = None
    gh_token: str | None = None
    repos_to_search: list[str] = field(default_factory=lambda: ["sgl-project/sglang", "ai-dynamo/dynamo"])
    pr_search_days: int = 14
    prompt: str | None = None

    def get_prompt(self, log_dir: str) -> str:
        """Get the formatted prompt for AI analysis.

        Args:
            log_dir: Path to the log directory

        Returns:
            Formatted prompt string
        """
        template = self.prompt or DEFAULT_AI_ANALYSIS_PROMPT
        repos_str = ", ".join(self.repos_to_search)
        return template.format(
            log_dir=log_dir,
            repos=repos_str,
            pr_days=self.pr_search_days,
        )

    Schema: ClassVar[type[Schema]] = Schema


# What ``aws s3 sync`` skips by default. Patterns follow the AWS CLI rules (relative to the
# log directory, ``*`` matches across directories). The aiperf per-interval scrapes of the
# worker and DCGM ``/metrics`` endpoints are the same time series tachometer stores as
# parquet, at 50 to 100 times the bytes.
#
# The aiperf patterns are scoped to the two directories the aiperf-driven runners write
# to (trace-replay, agentperf and mooncake-router under ``artifacts/<run>/``, sa-bench under
# ``sa-bench_*/conc_*/aiperf_artifacts/``) so a same-named file from another benchmark type
# (a custom runner's own ``inputs.json``, say) is never dropped by accident.
_AIPERF_ARTIFACT_ROOTS = ("artifacts/*", "sa-bench_*/*")
_AIPERF_METRIC_SCRAPES = (
    "server_metrics_export.jsonl",
    "server_metrics_export.json",
    "gpu_telemetry_export.jsonl",
    "inputs.json",
)
DEFAULT_S3_EXCLUDE: tuple[str, ...] = (
    *(f"{root}/{name}" for root in _AIPERF_ARTIFACT_ROOTS for name in _AIPERF_METRIC_SCRAPES),
)
# What goes into the compressed archive uploaded next to the loose files: aiperf's
# per-request records, the raw truth behind every latency number (13 to 40 MB raw, under
# 1 MB compressed). Python ``glob`` rules with ``**``; the same files are excluded from the
# plain sync.
DEFAULT_S3_ARCHIVE: tuple[str, ...] = (
    "artifacts/**/profile_export.jsonl",
    "sa-bench_*/**/profile_export.jsonl",
)


@dataclass(frozen=True)
class S3Config:
    """S3 upload configuration for log artifacts.

    Attributes:
        bucket: S3 bucket name
        prefix: Optional prefix/path within bucket (e.g., "srtslurm/logs")
        region: AWS region (e.g., "us-west-2")
        endpoint_url: Custom S3-compatible endpoint URL (optional)
        access_key_id: AWS access key ID (falls back to AWS_ACCESS_KEY_ID env var)
        secret_access_key: AWS secret access key (falls back to AWS_SECRET_ACCESS_KEY env var)
    """

    bucket: str
    prefix: str | None = None
    region: str | None = None
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    # Patterns `aws s3 sync` skips, relative to the log directory (`*` matches across
    # directories). Omit for the defaults: aiperf's per-interval metrics scrapes and
    # `inputs.json` under `artifacts/*/` and `sa-bench_*/*/` (tachometer already stores that
    # series as parquet). Set to `[]` to ship the whole directory.
    exclude: list[str] | None = None
    # Patterns (Python glob, `**` allowed) packed into one `bundle.tar.zst` uploaded next to the
    # loose files and left out of the plain sync. Omit for the default, aiperf's per-request
    # `profile_export.jsonl`; set to `[]` for no archive.
    archive: list[str] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class PostEvalConfig:
    """How the post-benchmark (or eval-only) accuracy evaluation is dispatched.

    The evaluation runs when the job environment sets ``RUN_EVAL=true`` (after
    the benchmark) or ``EVAL_ONLY=true`` (instead of it). srtctl forwards a
    built-in list of workflow variables into the eval process; downstream runners
    used to patch that list in srtctl's source. This block makes it config.

    Attributes:
        passthrough_env: Extra environment variable names forwarded from the
            orchestrator's environment into the eval process when set (on top
            of the built-in list: RUN_EVAL, EVAL_ONLY, MODEL, ISL, OSL, ...).
        command: Argv that replaces the built-in lm-eval runner command. May use
            the placeholders ``{endpoint}`` (the frontend URL) and
            ``{infmax_workspace}`` (the InferenceMAX workspace mount). Not
            shell-interpreted; wrap in ``bash -lc`` yourself if you need a shell.
    """

    passthrough_env: list[str] = field(default_factory=list)
    command: list[str] | None = None

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        for name in self.passthrough_env:
            if not name.isidentifier():
                raise ValidationError(
                    f"post_eval.passthrough_env entries must be environment variable names, got {name!r}"
                )
        if self.command is not None and not self.command:
            raise ValidationError("post_eval.command, if set, must be non-empty (omit it to use the lm-eval runner)")


@dataclass(frozen=True)
class HostSetupConfig:
    """Commands run on the bare host of each allocated node, outside the container.

    The orchestrator (which itself runs on the host, not in a container) fans these
    out one srun per node before any worker starts, and runs ``teardown`` after the
    workers are torn down. Use this for node state that cannot be set from inside a
    container -- locking GPU clocks with ``nvidia-smi -lmc``, loading a kernel
    module, dropping caches.

    This is the counterpart to ``SrtConfig.setup_script``, which runs *inside* the
    container from /configs.

    Commands run as the submitting user. Anything needing root must go through
    passwordless sudo (``sudo -n ...``); a sudo that prompts will hang until
    ``timeout_seconds`` and fail the job.

    Attributes:
        commands: Shell commands run in order on each node, joined with ``&&``.
        teardown: Shell commands run on each node after workers stop. Runs even
            when the job fails, so state that outlives the allocation (locked
            clocks persist for the next tenant) gets reset.
        nodes: Which nodes to target. "all" covers head, infra, and workers;
            "workers" covers only the nodes running backend workers.
        ignore_failure: When True, a failing node logs a warning instead of
            failing the job.
        timeout_seconds: Per-node wall-clock budget for commands and for teardown.
    """

    commands: list[str] = field(default_factory=list)
    teardown: list[str] = field(default_factory=list)
    nodes: Literal["all", "workers"] = "all"
    ignore_failure: bool = False
    timeout_seconds: int = 300

    Schema: ClassVar[type[Schema]] = Schema

    @property
    def enabled(self) -> bool:
        """True when there is anything to run on the nodes."""
        return bool(self.commands or self.teardown)


@dataclass
class ClusterConfig:
    """Cluster configuration from srtslurm.yaml."""

    cluster: str | None = None  # Cluster name for status reporting
    # Slurm account for recipes that omit `slurm.account`.
    default_account: str | None = None
    # Slurm partition for recipes that omit `slurm.partition`.
    default_partition: str | None = None
    # Job time limit (HH:MM:SS) for recipes that omit `slurm.time_limit`.
    default_time_limit: str | None = None
    # GPUs per node for recipes that omit `resources.gpus_per_node`.
    gpus_per_node: int | None = None
    # Default for ``ResourceConfig.gpu_type`` when the recipe omits it. Lets one
    # recipe move between clusters of different GPU types without an edit.
    default_gpu_type: str | None = None
    # Interface whose IP address frontends use to reach workers (e.g. `ib0`). Unset resolves the hostname.
    network_interface: str | None = None
    # GPU-subset mask passed to workers; ROCm clusters use ROCR_VISIBLE_DEVICES.
    visible_devices_env: str = "CUDA_VISIBLE_DEVICES"
    # Recipe exporter settings win. Explicit null disables the GPU default only.
    default_gpu_exporter: "TelemetryExporterConfig | None" = field(default_factory=lambda: DEFAULT_DCGM_EXPORTER)
    # Emit `#SBATCH --gpus-per-node`. Set false on clusters that reject or ignore it.
    use_gpus_per_node_directive: bool = True
    # Emit `#SBATCH --segment` so the allocation stays inside one topology segment (NVL72 domain).
    use_segment_sbatch_directive: bool = True
    # Emit `#SBATCH --exclusive` to keep other jobs off the allocated nodes.
    use_exclusive_sbatch_directive: bool = False
    # Default for ``ResourceConfig.het_jobs`` when the recipe doesn't set it.
    # When True (and recipe doesn't override), the prefill side and decode side
    # are submitted as two SLURM heterogeneous-job components, each with its
    # own ``--segment``. Lets asymmetric layouts (e.g. prefill 12 + decode 10
    # nodes on GB200/GB300) preserve NVL72 affinity per side.
    use_het_jobs: bool = False
    # Extra `#SBATCH --key=value` lines added to every job; a recipe's `sbatch_directives` wins per key.
    default_sbatch_directives: dict[str, str] | None = None
    # `health_check` block (`max_attempts`, `interval_seconds`) used when a recipe has none.
    default_health_check: dict[str, int] | None = None
    # srtctl checkout on the shared filesystem that compute nodes mount at /srtctl-src. Default: this checkout.
    srtctl_root: str | None = None
    output_dir: str | None = None  # Custom output directory for job logs
    # Cluster-wide default for recording exact realized srun commands. Recipes
    # can opt in independently with output.record_launch_plan.
    record_launch_plan: bool = False
    # Alias -> path map; a recipe's `model.path` may name an alias instead of a path.
    model_paths: dict[str, str] | None = None
    # Alias -> image map, resolved for every container key in a recipe (`model.container`, `roles.<role>.container`, ...).
    containers: dict[str, str] | None = None
    # Free-form cloud settings. Accepted for compatibility; srtctl does not read it.
    cloud: dict[str, str] | None = None
    # Cluster-level container mounts (host_path -> container_path)
    # Applied to all jobs on this cluster, useful for cluster-specific paths
    default_mounts: dict[str, str] | None = None
    # Shell snippet prepended to every container srun (after env exports, before
    # the main command). Useful for cluster-wide ulimits, e.g.
    # ``"ulimit -n 1048576 -s unlimited -u 1048576"``. Silently dropped for
    # sruns that bypass the bash wrapper (distroless containers).
    default_bash_preamble: str | None = None
    # Commands run on every allocated node's bare host, outside the container,
    # before workers start. Recipes override with their own `host_setup:` block.
    default_host_setup: HostSetupConfig | None = None
    # Status collectors, S3 log upload, and failure analysis for every job on this cluster.
    reporting: ReportingConfig | None = None
    telemetry: dict | None = None  # opaque dict, parsed by try_start_snapshotter
    # When set, applied to job configs that omit ``frontend.nginx_raise_ulimit``.
    # Clusters that disallow raising nofile for nginx containers should use false.
    nginx_raise_ulimit: bool | None = None
    # Works around intermittent git smart-HTTP/HTTP2 failures cloning github.com
    # (stalls, or truncated responses git misreports as "could not read
    # Username" auth-prompt failures). See git_clone_command_prefix() in
    # core/config.py -- applied to every git clone/fetch srtctl performs.
    git_http_version: str | None = None
    # Run the pre-submit model.path / model.container / telemetry filesystem
    # checks on ``srtctl apply``. Set false on clusters whose model or image
    # paths exist only on compute nodes (node-local NVMe such as /raid), where
    # the login node cannot stat them; every apply then behaves as if
    # --no-preflight had been passed. The framework still fails loudly at
    # runtime if a path is genuinely missing on the compute node.
    preflight: bool = True

    Schema: ClassVar[type[Schema]] = Schema


# ============================================================================
# Enums
# ============================================================================


class GpuType(str, Enum):
    GB200 = "gb200"
    GB300 = "gb300"
    H100 = "h100"


class Precision(str, Enum):
    FP4 = "fp4"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


class ProfilingType(str, Enum):
    NSYS = "nsys"
    TORCH = "torch"
    NONE = "none"


# ============================================================================
# Marshmallow Custom Fields
# ============================================================================


class BackendConfigField(fields.Field):
    """Marshmallow field for the polymorphic engine: a type string or a mapping with ``type``.

    ``reject_per_role_keys`` is set on the recipe's engine fields (top-level ``engine`` and
    ``roles.<role>.engine``): an engine mapping carries engine-wide knobs only, so the
    pre-2.0 per-mode keys (``sglang_config``, ``prefill_environment``, ``kv_events_config``,
    ...) are refused there with a pointer to ``roles.<role>`` (``env``, ``args``,
    ``extra_args``, ``kv_events``). The engine's ``roles`` is bound by ``SrtConfig`` and
    refused on load by its own field; it is left out of dumps.
    """

    def __init__(self, *, reject_per_role_keys: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reject_per_role_keys = reject_per_role_keys

    def _deserialize(
        self,
        value: Any,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> BackendConfig:
        """Deserialize an engine from its type string or its mapping's ``type`` key."""
        if value is None:
            return SGLangBackend()

        if isinstance(
            value,
            AtomBackend
            | SGLangBackend
            | TileRTBackend
            | TokenSpeedBackend
            | TRTLLMBackend
            | VLLMBackend
            | MockerBackend,
        ):
            return value

        if isinstance(value, str):
            value = {"type": value}
        if not isinstance(value, dict):
            raise ValidationError(f"Expected an engine type or a mapping with 'type', got {type(value).__name__}")

        if self.reject_per_role_keys:
            per_role = sorted(set(value) & PER_ROLE_ENGINE_KEYS)
            if per_role:
                raise ValidationError(
                    "an engine mapping carries engine-wide knobs only; per-role settings ("
                    + ", ".join(per_role)
                    + ") live under roles.<role> (env, args, extra_args, kv_events)"
                )

        backend_type = value.get("type", "sglang")

        if backend_type == "atom":
            return AtomBackend.Schema().load(value)
        elif backend_type == "tilert":
            return TileRTBackend.Schema().load(value)
        elif backend_type == "tokenspeed":
            return TokenSpeedBackend.Schema().load(value)
        elif backend_type == "sglang":
            schema = SGLangBackend.Schema()
            return schema.load(value)
        elif backend_type == "trtllm":
            schema = TRTLLMBackend.Schema()
            return schema.load(value)
        elif backend_type == "vllm":
            schema = VLLMBackend.Schema()
            return schema.load(value)
        elif backend_type == "mocker":
            schema = MockerBackend.Schema()
            return schema.load(value)
        else:
            raise ValidationError(
                f"Unknown engine type: {backend_type!r}. Supported types: atom, sglang, tilert, tokenspeed, trtllm, vllm, mocker"
            )

    def _serialize(self, value: Any | None, attr: str | None, obj: Any, **kwargs) -> Any:
        """Serialize the engine to a dict; the bound ``roles`` are left out (the recipe's ``roles:`` carries them)."""
        if value is None:
            return None
        dumped = self._dump(value)
        dumped.pop("roles", None)
        return dumped

    @staticmethod
    def _dump(value: Any) -> dict[str, Any]:
        if isinstance(value, AtomBackend):
            return AtomBackend.Schema().dump(value)
        if isinstance(value, TileRTBackend):
            return TileRTBackend.Schema().dump(value)
        if isinstance(value, TokenSpeedBackend):
            return TokenSpeedBackend.Schema().dump(value)
        if isinstance(value, SGLangBackend):
            return SGLangBackend.Schema().dump(value)
        if isinstance(value, TRTLLMBackend):
            return TRTLLMBackend.Schema().dump(value)
        if isinstance(value, VLLMBackend):
            return VLLMBackend.Schema().dump(value)
        if isinstance(value, MockerBackend):
            return MockerBackend.Schema().dump(value)
        return value


class SweepConfigField(fields.Field):
    """Marshmallow field for SweepConfig."""

    def _deserialize(self, value: Any, attr: str | None, data: Mapping[str, Any] | None, **kwargs) -> Any:
        if value is None:
            return None
        if isinstance(value, SweepConfig):
            return value
        if not isinstance(value, dict):
            raise ValidationError(f"Expected dict for sweep config, got {type(value).__name__}")

        mode = value.get("mode", "zip")
        parameters: dict[str, list[Any]] = {}

        if "parameters" in value:
            for key, val in value["parameters"].items():
                if not isinstance(val, list):
                    raise ValidationError(f"Sweep parameter '{key}' must be a list")
                parameters[key] = val
        else:
            for key, val in value.items():
                if key == "mode":
                    continue
                if not isinstance(val, list):
                    raise ValidationError(f"Sweep parameter '{key}' must be a list")
                parameters[key] = val

        return SweepConfig(mode=mode, parameters=parameters)

    def _serialize(self, value: Any | None, attr: str | None, obj: Any, **kwargs) -> Any:
        if value is None:
            return None
        if isinstance(value, SweepConfig):
            result: dict[str, Any] = {"mode": value.mode}
            result.update(value.parameters)
            return result
        return value


# ============================================================================
# Sub-Configuration Dataclasses (all frozen)
# ============================================================================


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for benchmark parameter sweeps."""

    # `zip` pairs the i-th value of every list; `grid` takes the Cartesian product.
    mode: Literal["zip", "grid"] = "zip"
    # Parameter name -> list of values to sweep over.
    parameters: dict[str, list[Any]] = field(default_factory=dict)

    def get_combinations(self) -> Iterator[dict[str, Any]]:
        if not self.parameters:
            yield {}
            return

        if self.mode == "zip":
            param_names = list(self.parameters.keys())
            param_lists = [self.parameters[name] for name in param_names]
            for values in zip(*param_lists, strict=False):
                yield dict(zip(param_names, values, strict=False))
        else:
            param_names = list(self.parameters.keys())
            param_lists = [self.parameters[name] for name in param_names]
            for values in itertools.product(*param_lists):
                yield dict(zip(param_names, values, strict=False))

    def __len__(self) -> int:
        if not self.parameters:
            return 1
        if self.mode == "zip":
            return len(next(iter(self.parameters.values())))
        result = 1
        for param_list in self.parameters.values():
            result *= len(param_list)
        return result

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    # Model weights directory, or a `model_paths` alias from srtslurm.yaml. Mounted at /model.
    path: str
    # Container image (`.sqsh` path or registry URI), or a `containers` alias from srtslurm.yaml.
    container: str
    # Weight precision (`fp4`, `fp8`, `fp16`, `bf16`). Recorded with results; engine flags set the actual dtype.
    precision: str
    # Optional: stage the model from shared storage to this node-local dir
    # before workers start (e.g. "/raid/scratch/models"). None = use path directly.
    stage_dir: str | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class IdentityModelConfig:
    """Virtual model identity for runtime verification."""

    repo: str | None = None  # HuggingFace model ID, e.g. "nvidia/Kimi-K2.5-NVFP4"
    revision: str | None = None  # HuggingFace git commit SHA

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class IdentityContainerConfig:
    """Container identity for reproduction (not verified at runtime).

    Recorded so others can pull the same container image to reproduce.
    Cannot be verified at runtime — Pyxis/enroot strips provenance during import.
    """

    image: str | None = None  # Docker URI, e.g. "gitlab-master:5005/.../trtllm-arm64"

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class IdentityConfig:
    """Virtual identity for runtime verification and reproduction.

    These fields declare what *should* be running. They are not used for
    launching — only for verifying the runtime fingerprint matches expectations
    and for helping others reproduce the run.

    - model: HF repo + revision (verified against download metadata at runtime)
    - container: Docker image URI (recorded for reproduction, not verified)
    - frameworks: expected versions for dynamo + one engine (verified via importlib.metadata)
    """

    # Expected HuggingFace repo and revision, checked against the download metadata at runtime.
    model: IdentityModelConfig = field(default_factory=IdentityModelConfig)
    # Container image URI, recorded for reproduction only.
    container: IdentityContainerConfig = field(default_factory=IdentityContainerConfig)
    # Package -> expected version for dynamo and one engine, checked via importlib.metadata at runtime.
    frameworks: dict[str, str] = field(default_factory=dict)

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class HetComponent:
    """One component of a SLURM heterogeneous job.

    A het job is submitted as multiple `#SBATCH` blocks separated by
    `#SBATCH hetjob`. SLURM places each component within a single topology
    segment, so we get per-side NVL72 affinity. At runtime each component
    exposes its own `SLURM_JOB_NODELIST_HET_GROUP_<group>`, and worker srun
    calls target a component with `--het-group=<group>`.
    """

    name: Literal["prefill", "decode"]
    group: int
    nodes: int
    segment: int
    gpus_per_node: int

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class RestartPolicy:
    """How the worker supervisor treats a worker of one role that exits mid-run.

    Modelled on a Kubernetes ``restartPolicy``: the supervisor observes every
    worker step, and when one exits it relaunches the whole endpoint (every
    process of a multi-node worker) on the same nodes and GPUs with the same
    ports, after an exponential backoff. Restarts are counted per endpoint for
    the life of the job; once ``max_restarts`` is spent the role's ``critical``
    flag decides whether the run fails or carries on without that worker.

    Attributes:
        policy: ``never`` leaves a worker exit to ``critical`` (the default,
            today's behavior). ``on-failure`` relaunches after a non-zero exit;
            ``always`` relaunches after any exit, including a clean one.
        max_restarts: Relaunches allowed per endpoint over the whole job.
        backoff_seconds: Delay before the first relaunch. Doubles on every
            further relaunch of the same endpoint (10 s, 20 s, 40 s, ...).
        max_backoff_seconds: Cap on the doubled delay.
    """

    policy: Literal["never", "on-failure", "always"] = "never"
    max_restarts: int = 3
    backoff_seconds: float = 10.0
    max_backoff_seconds: float = 300.0

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.max_restarts < 0:
            raise ValidationError("restart.max_restarts must be 0 or more")
        if self.backoff_seconds < 0:
            raise ValidationError("restart.backoff_seconds must be 0 or more")
        if self.max_backoff_seconds < self.backoff_seconds:
            raise ValidationError("restart.max_backoff_seconds must be at least restart.backoff_seconds")

    @property
    def enabled(self) -> bool:
        """True when the supervisor should relaunch workers of this role."""
        return self.policy != "never"

    def restarts_on(self, exit_code: int | None) -> bool:
        """Whether an exit with ``exit_code`` is one this policy relaunches after."""
        if self.policy == "always":
            return True
        if self.policy == "on-failure":
            return exit_code != 0
        return False

    def backoff(self, restarts: int) -> float:
        """Delay before relaunch number ``restarts`` (1 for the first relaunch)."""
        return min(self.backoff_seconds * (2 ** max(0, restarts - 1)), self.max_backoff_seconds)


class RestartPolicyField(fields.Field):
    def _deserialize(
        self, value: Any, attr: str | None, data: Mapping[str, Any] | None, **kwargs: Any
    ) -> RestartPolicy:
        if isinstance(value, str):
            value = {"policy": value}
        if not isinstance(value, dict):
            raise ValidationError("restart must be a policy name or a mapping")
        return RestartPolicy.Schema().load(value)

    def _serialize(self, value: Any, attr: str | None, obj: Any, **kwargs: Any) -> dict[str, Any]:
        return RestartPolicy.Schema().dump(value)


@dataclass(frozen=True)
class RoleConfig(RoleSettings):
    """One worker role of the recipe: `roles.prefill`, `roles.decode`, or `roles.agg`.

    Everything about a role lives here: the nodes and workers it gets, the GPUs per
    worker, its environment and engine arguments, and optionally its own engine and
    image. ``SrtConfig.topology`` derives the per-role counts the launch path reads;
    ``SrtConfig.backend`` binds the roles onto the engine, which reads ``env`` / ``args`` /
    ``extra_args`` / ``kv_events`` from them.
    """

    # Nodes reserved for this role. `colocate` (decode only) reserves none and packs the
    # decode workers onto the prefill nodes' free GPUs; `gpus` is then required on both
    # roles and the loader rejects a split that does not fit.
    nodes: int | Literal["colocate"] | None = None
    # Number of workers of this role.
    workers: int | None = None
    # GPUs per worker. Defaults to `nodes * gpus_per_node // workers`; required when decode colocates.
    gpus: int | None = None
    # Merged over the recipe srun_options on this role's worker steps only (e.g. a per-step mem cap).
    srun_options: dict[str, str] = field(default_factory=dict)
    # Environment for every worker of this role.
    env: dict[str, str] = field(default_factory=dict)
    # The engine's own CLI flags for this role, as a mapping (`tensor-parallel-size: 4`).
    args: dict[str, Any] = field(default_factory=dict)
    # Raw extra CLI arguments (TRT-LLM only).
    extra_args: list[str] = field(default_factory=list)
    # Engine type or mapping with engine options. Set on every role when no top-level
    # `engine` is declared; the two forms cannot be mixed, and role engines do not inherit
    # options from each other.
    engine: Annotated[BackendConfig | None, BackendConfigField(allow_none=True, reject_per_role_keys=True)] = None
    # Optional role image; accepts cluster container aliases. Defaults to `model.container`.
    container: str | None = None
    # `true` for the default ZMQ publisher, or a mapping with `publisher` / `topic`.
    kv_events: bool | dict[str, Any] | None = None
    # Run the native engine with a Dynamo sidecar (turns on `dynamo.sidecar`); every role must agree.
    sidecar: bool | None = field(
        default=None,
        metadata={"marshmallow_field": fields.Boolean(truthy={True}, falsy={False}, allow_none=True)},
    )
    # A worker of this role exiting fails the run. `false` keeps the run alive for probes that kill workers.
    critical: bool = field(
        default=True,
        metadata={"marshmallow_field": fields.Boolean(truthy={True}, falsy={False})},
    )
    # Relaunch exited workers in place: `never`, `on-failure`, `always`, or a mapping with
    # `policy`, `max_restarts`, `backoff_seconds`, and `max_backoff_seconds`.
    restart: RestartPolicy = field(
        default_factory=RestartPolicy,
        metadata={"marshmallow_field": RestartPolicyField()},
    )

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        nodes = self.nodes
        if isinstance(nodes, bool) or not (nodes is None or isinstance(nodes, int) or nodes == COLOCATE):
            raise ValidationError(f"nodes must be a positive integer or {COLOCATE!r}; got {nodes!r}")
        if self.container is not None and not self.container.strip():
            raise ValidationError("container must be a non-empty string")

    @property
    def colocated(self) -> bool:
        """``nodes: colocate``: the role shares the prefill nodes instead of reserving its own."""
        return self.nodes == COLOCATE

    @property
    def node_count(self) -> int | None:
        """Nodes this role reserves: ``0`` when it colocates, ``None`` when unset."""
        if isinstance(self.nodes, int):
            return self.nodes
        return 0 if self.colocated else None


@dataclass(frozen=True)
class ResourceConfig:
    """Cluster facts and allocation knobs; the worker topology is the `roles:` block."""

    # GPU type (h100, gb200, ...). Cluster fact, not a topology choice. Optional:
    # a recipe that omits it inherits `default_gpu_type` from srtslurm.yaml, and
    # `gpus_per_node` inherits the cluster `gpus_per_node`. Both are still worth
    # setting in a recipe so it is self-describing for result rollups.
    gpu_type: str | None = None
    # GPUs on each node. Inherits the cluster `gpus_per_node` when omitted, else 4.
    gpus_per_node: int = 4

    # If True, place each partial-node worker on its own node instead of
    # packing multiple onto the same node. Caller must reserve enough nodes
    # (e.g. give roles.decode as many nodes as workers when its gpus < gpus_per_node).
    spread_workers: bool = False

    # SLURM heterogeneous-job opt-in. Tri-state: None defers to the cluster
    # default `use_het_jobs` on ClusterConfig; True/False overrides per recipe.
    # When effectively True (and we are in disaggregated mode), the prefill and
    # decode sides are submitted as two het components each with their own
    # `--segment`. See HetComponent above and docs/slurm-faq.md.
    het_jobs: bool | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclasses.dataclass(frozen=True)
class Topology:
    """The worker layout ``roles:`` describes, on nodes of ``gpus_per_node`` GPUs.

    One derivation of every per-role count the launch path, the frontends and the
    validators read (``num_prefill``, ``gpus_per_decode``, ``total_nodes``, ...), built
    once per config as ``SrtConfig.topology``. A role that is not declared has no nodes
    and no workers. ``prefill`` or ``decode`` present means a disaggregated deployment;
    ``agg`` alone is the aggregated one.
    """

    roles: Mapping[str, RoleConfig]
    gpus_per_node: int
    het_jobs: bool | None = None

    def role(self, name: str) -> RoleConfig | None:
        return self.roles.get(name)

    def nodes(self, name: str) -> int | None:
        """Nodes the role reserves (``0`` for a colocated decode), None when the role is absent or unsized."""
        spec = self.roles.get(name)
        return None if spec is None else spec.node_count

    def workers(self, name: str) -> int:
        spec = self.roles.get(name)
        return (spec.workers or 0) if spec is not None else 0

    def gpus_per_worker(self, name: str) -> int:
        """GPUs per worker of the role: its ``gpus``, else its nodes' GPUs split over its workers."""
        spec = self.roles.get(name)
        if spec is None:
            return self.gpus_per_node
        if spec.gpus is not None:
            return spec.gpus
        if spec.node_count and spec.workers:
            return (spec.node_count * self.gpus_per_node) // spec.workers
        if name == "decode" and spec.colocated and spec.workers:
            # A colocated decode shares the prefill nodes and inherits the prefill worker size.
            return self.gpus_per_worker("prefill")
        return self.gpus_per_node

    def worker_critical(self, mode: str) -> bool:
        """Whether a worker of ``mode`` (``prefill``, ``decode``, ``agg``) failing fails the run."""
        spec = self.roles.get(mode)
        return True if spec is None else spec.critical

    def worker_restart(self, mode: str) -> RestartPolicy:
        """Restart policy of a worker role; undeclared roles are never restarted."""
        spec = self.roles.get(mode)
        return RestartPolicy() if spec is None else spec.restart

    @property
    def is_disaggregated(self) -> bool:
        return "prefill" in self.roles or "decode" in self.roles

    @property
    def colocated_decode(self) -> bool:
        """``roles.decode.nodes: colocate``: the decode workers live on the prefill nodes."""
        decode = self.roles.get("decode")
        return decode is not None and decode.colocated

    @property
    def prefill_nodes(self) -> int | None:
        return self.nodes("prefill")

    @property
    def decode_nodes(self) -> int | None:
        return self.nodes("decode")

    @property
    def agg_nodes(self) -> int | None:
        return self.nodes("agg")

    @property
    def prefill_workers(self) -> int | None:
        spec = self.roles.get("prefill")
        return None if spec is None else spec.workers

    @property
    def decode_workers(self) -> int | None:
        spec = self.roles.get("decode")
        return None if spec is None else spec.workers

    @property
    def agg_workers(self) -> int | None:
        spec = self.roles.get("agg")
        return None if spec is None else spec.workers

    @property
    def total_nodes(self) -> int:
        if self.is_disaggregated:
            return (self.prefill_nodes or 0) + (self.decode_nodes or 0)
        return self.agg_nodes or 1

    @property
    def has_engine_workers(self) -> bool:
        """Whether any prefill, decode, or aggregated worker is requested."""
        return (self.num_prefill + self.num_decode + self.num_agg) > 0

    @property
    def num_prefill(self) -> int:
        return self.workers("prefill")

    @property
    def num_decode(self) -> int:
        return self.workers("decode")

    @property
    def num_agg(self) -> int:
        return self.workers("agg")

    @property
    def gpus_per_prefill(self) -> int:
        return self.gpus_per_worker("prefill")

    @property
    def gpus_per_decode(self) -> int:
        return self.gpus_per_worker("decode")

    @property
    def gpus_per_agg(self) -> int:
        return self.gpus_per_worker("agg")

    @property
    def prefill_gpus(self) -> int:
        """Total GPUs used by all prefill workers."""
        return self.num_prefill * self.gpus_per_prefill

    @property
    def decode_gpus(self) -> int:
        """Total GPUs used by all decode workers."""
        return self.num_decode * self.gpus_per_decode

    def het_components(
        self,
        *,
        infra_dedicated: bool,
        cluster_default: bool = False,
    ) -> tuple[HetComponent, HetComponent] | None:
        """Return the (prefill, decode) het components, or None when het is off.

        Het is enabled when ``self.het_jobs`` is True, or when it is None and
        ``cluster_default`` is True. Only valid in disaggregated mode. Group 0
        is prefill (folds in the dedicated infra node when present); group 1 is
        decode. Segment matches each component's node count, so each side lands
        in its own topology segment (NVL72 domain on GB200/GB300).

        Pass ``cluster_default=get_srtslurm_setting("use_het_jobs", False)``
        from callers that have access to the cluster config; schema.py cannot
        import from core.config without a cycle.
        """
        enabled = self.het_jobs if self.het_jobs is not None else cluster_default
        if not enabled or not self.is_disaggregated:
            return None
        prefill_nodes = (self.prefill_nodes or 0) + (1 if infra_dedicated else 0)
        decode_nodes = self.decode_nodes or 0
        return (
            HetComponent(
                name="prefill",
                group=0,
                nodes=prefill_nodes,
                segment=prefill_nodes,
                gpus_per_node=self.gpus_per_node,
            ),
            HetComponent(
                name="decode",
                group=1,
                nodes=decode_nodes,
                segment=decode_nodes,
                gpus_per_node=self.gpus_per_node,
            ),
        )


def _bind_roles(engine: BackendConfig, roles: Mapping[str, RoleConfig]) -> BackendConfig:
    """``engine`` with the recipe's roles bound; the engine reads per-role env, args, extra_args and kv_events from them."""
    if engine.roles:
        raise ValidationError("engine.roles is bound from the recipe's roles block; declare roles at the top level")
    return dataclasses.replace(engine, roles=dict(roles)) if roles else engine


@dataclass(frozen=True)
class SlurmConfig:
    """SLURM job settings."""

    # Slurm account. Unset uses `default_account` from srtslurm.yaml.
    account: str | None = None
    # Slurm partition. Unset uses `default_partition` from srtslurm.yaml.
    partition: str | None = None
    # Job time limit (HH:MM:SS). Unset uses `default_time_limit` from srtslurm.yaml.
    time_limit: str | None = None

    Schema: ClassVar[type[Schema]] = Schema


# ``placement.node`` value that reserves a node for the component.
PLACEMENT_DEDICATED = "dedicated"


@dataclass(frozen=True)
class PlacementConfig:
    """Where a component (the frontend or the benchmark client) runs.

    Attributes:
        node: A location name resolved against the worker topology (``head``, or a
            role-relative name such as ``first_decode`` / ``last_decode``), or
            ``dedicated`` to reserve a node for the component. A dedicated node is
            always the head location, so the two never combine.
    """

    node: str = "head"

    Schema: ClassVar[type[Schema]] = Schema

    @property
    def dedicated(self) -> bool:
        """Whether the component gets a node of its own."""
        return self.node == PLACEMENT_DEDICATED

    @property
    def location(self) -> str:
        """The placement name consumers resolve: ``head`` for a dedicated node."""
        return "head" if self.dedicated else self.node


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark configuration."""

    # Benchmark runner: `manual` (none) or a registered type; see Benchmark types for the keys each accepts.
    type: str = "manual"
    # Mirror benchmark.out to the orchestrator's stdout while the client runs; keep the log file.
    stream_output: bool = False
    # Input sequence length in tokens for synthetic requests.
    isl: int | None = None
    # Output sequence length in tokens for synthetic requests.
    osl: int | None = None
    # Concurrency levels, one benchmark phase each: a list or an `x`-separated string (`"4x8x16"`).
    # Telemetry uses each phase as a measurement window.
    concurrencies: list[int] | str | None = None
    # Request arrival rate in requests/s; `inf` sends as fast as concurrency allows.
    req_rate: str | int | None = "inf"
    # Where the benchmark client runs. placement.node is "head" (default: the
    # orchestrator's node), "last_decode" (the last decode/GEN worker-leader node,
    # isolating the client off the CTX/orchestrator node; use the injected
    # $SRT_FRONTEND_HOST env in the benchmark command's URL), or "dedicated" (a
    # node reserved for the client: needs at least 2 nodes, not supported with
    # resources.het_jobs: true).
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    # Governs how dedicated placements combine when more than one of the
    # benchmark client, the frontend, and the etcd/nats services asks for
    # placement.node: dedicated. If True (default), every requested role
    # shares a single reserved node. If False, each requested role gets its
    # own reserved node (requires enough total nodes: worker count + number
    # of dedicated roles).
    colocate_with_frontend: bool = True
    # Accepted for compatibility; no runner reads it. Sweep a recipe with the top-level `sweep:` block.
    sweep: Annotated[SweepConfig, SweepConfigField(allow_none=True, load_default=None, dump_default=None)] | None = None
    # Accuracy benchmark fields
    num_examples: int | None = None  # Number of evaluation examples; unset uses the runner's default
    max_tokens: int | None = None  # Maximum generated tokens per response
    repeat: int | None = None  # Times each example is evaluated; scores are averaged
    num_threads: int | None = None  # Concurrent evaluation requests
    max_context_length: int | None = None  # LongBench v2: skip examples longer than this many tokens
    categories: list[str] | None = None  # LongBench v2: task categories to run; unset runs all
    num_shots: int | None = None  # GSM8K few-shot examples
    temperature: float | None = None  # Sampling temperature; unset uses the runner's default
    top_p: float | None = None  # Nucleus sampling threshold; unset uses the runner's default
    top_k: int | None = None  # Top-k sampling cutoff; unset uses the runner's default
    # Router benchmark fields
    num_requests: int | None = None
    concurrency: int | None = None  # Single concurrency level (router, agentperf)
    # Router: shared-prefix ratios to test (list or space-separated string).
    prefix_ratios: list[float] | str | None = None
    # Mooncake router benchmark fields (uses aiperf with mooncake_trace)
    mooncake_workload: str | None = None  # "mooncake", "conversation", "synthetic", "toolagent"
    ttft_threshold_ms: int | None = None  # Goodput TTFT threshold in ms (default: 2000)
    itl_threshold_ms: int | None = None  # Goodput ITL threshold in ms (default: 25)
    random_range_ratio: float | None = None  # Random input/output length range ratio (default: 0.8)
    num_prompts_mult: int | None = None  # Multiplier for num_prompts = concurrency * mult (default: 10)
    num_warmup_mult: int | None = None  # Multiplier for warmup prompts = concurrency * mult (default: 2)
    # Custom dataset fields (sa-bench)
    dataset_name: str | None = None  # "random" (default) or "custom"
    dataset_path: str | None = None  # Container path to dataset file (mount via extra_mount)
    # AgentPerf benchmark fields (agentperf-client trajectory replay)
    agentperf_client_dir: str | None = None  # Container path to an agentperf-client checkout (mount via extra_mount)
    # Container path to the client's workload YAML (endpoint/model/concurrency injected)
    agentperf_config: str | None = None
    # Trace replay benchmark fields (uses aiperf with mooncake_trace dataset type)
    trace_file: str | None = None  # Path to trace JSONL file (container path, e.g., /traces/dataset.jsonl)
    custom_tokenizer: str | None = None  # Custom tokenizer class (e.g., "module.path.ClassName")
    use_chat_template: bool = True  # Pass --use-chat-template to benchmark (default: true)
    # SA-Bench Dynamo adapter: reuse a benchmark-scoped HTTP connection pool.
    # Opt-in to preserve the historical per-request ClientSession behavior.
    reuse_http_connections: bool = False
    # Custom benchmark hook.
    # ``command`` is passed to ``bash -lc`` verbatim; srtctl does NOT
    # substitute placeholders like ``{nginx_url}`` or ``{slurm_job_id}``.
    # Render any parameters when generating the recipe. See
    # srtctl.benchmarks.custom.CustomBenchmarkRunner for details.
    command: str | None = None
    # Image the benchmark client runs in (custom, agentperf); unset uses `model.container`.
    container_image: str | None = None
    # Extra environment variables for the benchmark client (custom, agentperf).
    env: dict[str, str] = field(default_factory=dict)
    # aiperf pip install spec (e.g., "aiperf>=0.7.0", "aiperf @ git+https://...@commit")
    # If set, runs pip install <spec> before benchmarking. Upgrades if already installed.
    aiperf_package: str | None = None
    # Extra aiperf CLI flags passed through to bench.sh (e.g., benchmark-duration: 600, workers-max: 200)
    aiperf_args: dict[str, Any] = field(default_factory=dict)
    # SA-Bench: optional SGLang /slow_down on decode workers (sglang frontend only; see benchmark_stage)
    slow_down_sleep_time: float | None = None  # forward_sleep_time (seconds); unset = feature off
    slow_down_wait_time: float | None = None  # seconds until POST clears slow_down; unset = feature off

    def get_concurrency_list(self) -> list[int]:
        if self.concurrencies is None:
            return []
        if isinstance(self.concurrencies, str):
            return [int(x) for x in self.concurrencies.split("x")]
        return list(self.concurrencies)

    Schema: ClassVar[builtins.type[Schema]] = Schema


@dataclass(frozen=True)
class ProfilingPhaseConfig:
    """Profiling config for a single phase (prefill/decode/aggregated)."""

    start_step: int | None = None  # Step to start profiling
    stop_step: int | None = None  # Step to stop profiling
    # `all` profiles every process of the phase; `selected` only `worker_index` / `worker_rank`.
    capture_scope: Literal["selected", "all"] = "all"
    worker_index: int = 0  # Logical worker within the phase
    worker_rank: int = 0  # Physical process rank within that worker

    @property
    def vllm_nsys_delay_iterations(self) -> int:
        """vLLM --profiler-config delay_iterations: engine steps before capture starts."""
        return self.start_step or 0

    @property
    def vllm_nsys_max_iterations(self) -> int:
        """vLLM --profiler-config max_iterations: number of steps to capture (stop - start)."""
        if self.start_step is None or self.stop_step is None:
            return 0
        return max(self.stop_step - self.start_step, 0)

    Schema: ClassVar[builtins.type[Schema]] = Schema


@dataclass(frozen=True)
class ProfilingConfig:
    """Profiling configuration.

    Supports two profiling modes:
    - nsys: NVIDIA Nsight Systems profiling (wraps command with nsys profile)
    - torch: PyTorch profiler (uses SGLANG_TORCH_PROFILER_DIR)

    Per-phase start_step/stop_step are specified in the prefill/decode/aggregated sections.
    """

    type: str = "none"  # "none", "nsys", "nsys-time", or "torch"

    # Extra arguments passed to nsys profile (appended before `-o`; see get_nsys_prefix)
    extra_nsys_args: list[str] | None = None

    # Non-TRT-LLM Nsight activity domains. ``cuda-sw`` can be selected
    # explicitly where software tracing is preferred over hardware tracing.
    nsys_trace: str = "cuda,nvtx"

    # None preserves the existing Dynamo-specific default. Set explicitly for
    # worker launchers that require or cannot tolerate child-process injection.
    trace_fork_before_exec: bool | None = None

    # Non-TRT-LLM behavior when cudaProfilerStop closes a capture range.
    capture_range_end: str = "stop"

    # Optional paths prepended to LD_LIBRARY_PATH for the Nsight wrapper and
    # profiled worker, for containers that do not discover the host libcuda.
    nsys_library_paths: list[str] | None = None

    # Phase-specific profiling step configs (not used for nsys-time)
    prefill: ProfilingPhaseConfig | None = None
    # Step window for the decode role (disaggregated runs).
    decode: ProfilingPhaseConfig | None = None
    # Step window for the `agg` role (aggregated runs).
    aggregated: ProfilingPhaseConfig | None = None

    # nsys-time fields: time-based capture window, same on all workers
    delay_secs: int | None = None  # nsys --delay: seconds from worker launch before capture starts
    duration_secs: int | None = None  # nsys --duration: seconds to capture after delay
    benchmark_duration_secs: int = 300  # total traffic generation duration (must cover delay + duration)

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.type != "none"

    @property
    def is_nsys(self) -> bool:
        """Check if using NVIDIA Nsight Systems profiling (includes nsys-time)."""
        return self.type in ("nsys", "nsys-time")

    @property
    def is_nsys_time(self) -> bool:
        """Check if using time-based nsys capture (--delay/--duration instead of cudaProfilerApi)."""
        return self.type == "nsys-time"

    @property
    def is_torch(self) -> bool:
        """Check if using PyTorch profiler."""
        return self.type == "torch"

    def _get_phase_config(self, mode: str) -> ProfilingPhaseConfig | None:
        """Get the phase config for the given mode."""
        if mode == "prefill":
            return self.prefill
        elif mode == "decode":
            return self.decode
        elif mode in ("agg", "aggregated"):
            return self.aggregated
        return None

    def get_env_vars(self, mode: str, profile_dir: str) -> dict[str, str]:
        """Get profiling-specific environment variables.

        Args:
            mode: Worker mode (prefill/decode/agg)
            profile_dir: Base directory for profiling output.

        Returns:
            Dictionary of environment variables
        """
        if not self.enabled:
            return {}

        env = {"PROFILING_MODE": mode, "PROFILE_TYPE": self.type}

        # Phase-specific start/stop steps
        phase_config = self._get_phase_config(mode)
        if phase_config:
            phase_key = mode.upper() if mode != "agg" else "AGG"
            if phase_config.start_step is not None:
                env[f"PROFILE_{phase_key}_START_STEP"] = str(phase_config.start_step)
            if phase_config.stop_step is not None:
                env[f"PROFILE_{phase_key}_STOP_STEP"] = str(phase_config.stop_step)

        if self.is_torch:
            env["SGLANG_TORCH_PROFILER_DIR"] = f"{profile_dir}/{mode}"

        if self.is_nsys_time:
            env["PROFILE_BENCHMARK_DURATION_SECS"] = str(self.benchmark_duration_secs)
        elif (
            self.is_nsys and phase_config and phase_config.start_step is not None and phase_config.stop_step is not None
        ):
            # TRTLLM iteration-based nsys: PyExecutor triggers cudaProfilerStart/Stop at these boundaries.
            # Harmless on SGLang workers (unknown env vars are ignored).
            env["TLLM_PROFILE_START_STOP"] = f"{phase_config.start_step}-{phase_config.stop_step}"
            env["TLLM_LLMAPI_ENABLE_NVTX"] = "1"

        return env

    def selects_process(self, mode: str, worker_index: int, worker_rank: int) -> bool:
        """Whether an iteration-triggered capture targets this process."""
        phase = self._get_phase_config(mode)
        return bool(
            phase is not None
            and (
                phase.capture_scope == "all"
                or (phase.worker_index == worker_index and phase.worker_rank == worker_rank)
            )
        )

    def captures_all_processes(self, mode: str) -> bool:
        """Whether the phase captures every physical process."""
        phase = self._get_phase_config(mode)
        return bool(phase is not None and phase.capture_scope == "all")

    @property
    def nsys_binary(self) -> str:
        """nsys executable to invoke.

        Defaults to ``nsys`` (resolved on PATH). Override via the
        ``SRTCTL_NSYS_BIN`` environment variable when running inside a
        container that doesn't ship nsys on PATH — e.g. mount the host's
        Nsight Systems install and point this at the absolute path.
        """
        return os.environ.get("SRTCTL_NSYS_BIN", "nsys")

    def _get_nsys_prefix_trtllm(self, output_file: str) -> list[str]:
        """Get nsys command prefix for TRTLLM workers.

        Supports both iteration-based (cudaProfilerApi trigger via TLLM_PROFILE_START_STOP)
        and time-based (--delay/--duration) capture modes.
        """
        if self.is_nsys_time:
            cmd = [
                self.nsys_binary,
                "profile",
                "-t",
                "cuda,nvtx,ucx",
                "--sample=none",
                "--cuda-graph-trace=node",
            ]
            if self.delay_secs is not None:
                cmd += ["--delay", str(self.delay_secs)]
            if self.duration_secs is not None:
                cmd += ["--duration", str(self.duration_secs)]
        else:
            # Iteration-based: TLLM_PROFILE_START_STOP env var triggers cudaProfilerStart/Stop
            cmd = [
                self.nsys_binary,
                "profile",
                "-t",
                "cuda,nvtx,ucx",
                "--sample=none",
                "--cuda-graph-trace=node",
                "-c",
                "cudaProfilerApi",
                "--capture-range-end",
                "stop",
            ]

        if self.extra_nsys_args:
            cmd.extend(self.extra_nsys_args)

        cmd += [
            "--kill",
            "none",
            "--wait",
            "all",
            "--force-overwrite",
            "true",
            "-o",
            output_file,
        ]
        return cmd

    def get_nsys_prefix(
        self, output_file: str, *, frontend_type: str | None = None, backend_type: str | None = None
    ) -> list[str]:
        """Get nsys profiling command prefix.

        Args:
            output_file: Path for nsys output file (without extension)
            frontend_type: Frontend type (e.g., "dynamo", "sglang"). For a frontend whose
                workers are Dynamo processes (``worker_launch == "dynamo"``) with a
                non-trtllm backend, adds --trace-fork-before-exec=true.
            backend_type: Backend type (e.g., "trtllm", "sglang"). When set to "trtllm",
                uses TRTLLM-specific nsys flags (ucx traces, --kill none, --wait all).

        Returns:
            Command prefix list for nsys profiling
        """
        if not self.is_nsys:
            return []

        if backend_type == "trtllm":
            return self._get_nsys_prefix_trtllm(output_file)

        trace_fork_before_exec = self.trace_fork_before_exec
        if trace_fork_before_exec is None:
            # Dynamo workers fork the engine after exec; direct servers do not.
            from srtctl.frontends import get_frontend

            trace_fork_before_exec = frontend_type is not None and get_frontend(frontend_type).worker_launch == "dynamo"

        # Time-based capture for non-TRTLLM backends (vllm, sglang).
        if self.is_nsys_time:
            cmd = [
                self.nsys_binary,
                "profile",
                "-t",
                self.nsys_trace,
                "--cuda-graph-trace=node",
                "--force-overwrite",
                "true",
            ]
            if self.delay_secs is not None:
                cmd += ["--delay", str(self.delay_secs)]
            if self.duration_secs is not None:
                cmd += ["--duration", str(self.duration_secs)]
            if self.extra_nsys_args:
                cmd.extend(self.extra_nsys_args)
            cmd.extend(["-o", output_file])
            if trace_fork_before_exec:
                cmd.insert(-2, "--trace-fork-before-exec=true")
            return cmd

        # SGLang / default path — keep existing behavior
        cmd = [
            self.nsys_binary,
            "profile",
            "-t",
            self.nsys_trace,
            "--cuda-graph-trace=node",
            "-c",
            "cudaProfilerApi",
            "--capture-range-end",
            self.capture_range_end,
            "--force-overwrite",
            "true",
        ]

        if self.extra_nsys_args:
            cmd.extend(self.extra_nsys_args)

        cmd.extend(["-o", output_file])

        if trace_fork_before_exec:
            cmd.insert(-2, "--trace-fork-before-exec=true")

        return cmd

    Schema: ClassVar[builtins.type[Schema]] = Schema


@dataclass(frozen=True)
class GpuLabelsConfig:
    """The labels that identify a GPU in every sample of a GPU exporter's scrape."""

    # Label carrying the node-local GPU index srt-slurm allocates by.
    index: str
    # Label that is stable for one physical GPU across the run; recorded as `gpu_uuid`.
    identity: str
    # Labels marking logical sub-device samples (MIG instances, partitions); such samples are dropped.
    instance: list[str] = field(default_factory=list)

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class GpuMetricConfig:
    """One per-GPU metric in a GPU exporter's scrape."""

    # Prometheus metric name.
    metric: str

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class GpuPowerMetricConfig:
    """The per-GPU power metric in a GPU exporter's scrape, in watts."""

    # Prometheus metric name.
    metric: str
    # What the watts measure, recorded in the power manifest as `power_scope`.
    scope: str

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class GpuMetricsConfig:
    """The per-GPU metrics GPU power telemetry records from a GPU exporter.

    `power` is required; the others are optional and their columns stay empty
    when unset. Units are fixed by the artifact: `gpu_util` is a percent,
    `sm_active` a 0-1 fraction, and `temperature` Celsius.
    """

    # Power draw in watts.
    power: GpuPowerMetricConfig
    # GPU utilization, percent.
    gpu_util: GpuMetricConfig | None = None
    # DCGM's SM-active fraction, 0-1; map only a metric with that meaning and range.
    sm_active: GpuMetricConfig | None = None
    # GPU temperature, Celsius.
    temperature: GpuMetricConfig | None = None

    Schema: ClassVar[type[Schema]] = Schema


# DCGM, the default for exporter blocks that set neither `gpu_labels` nor `gpu_metrics`.
DCGM_GPU_LABELS = GpuLabelsConfig(
    index=DCGM_POWER_MAPPING.gpu_index_label,
    identity=DCGM_POWER_MAPPING.gpu_identity_label,
    instance=list(DCGM_POWER_MAPPING.instance_labels),
)
DCGM_GPU_METRICS = GpuMetricsConfig(
    power=GpuPowerMetricConfig(metric=DCGM_POWER_MAPPING.power_metric, scope=DCGM_POWER_MAPPING.power_scope),
    gpu_util=GpuMetricConfig(metric=GPU_UTIL_METRIC),
    sm_active=GpuMetricConfig(metric=SM_ACTIVE_METRIC),
    temperature=GpuMetricConfig(metric=TEMPERATURE_METRIC),
)


@dataclass(frozen=True)
class TelemetryExporterConfig:
    """Configuration for a metrics exporter deployed on worker nodes.

    Two launch modes. With ``binary`` unset the exporter runs as a pyxis
    container from ``container_image``. With ``binary`` set it runs
    **host-native** -- the executable is started by ``srun`` directly on the
    node with no container; ``container_image`` is ignored (set it to ``""``).
    Relative ``binary`` paths resolve against the srtctl checkout root, which is
    where ``make setup`` installs the host binaries (``configs/nats-server``,
    ``configs/etcd``, ``configs/process-exporter``). Host-native exists because
    some enroot deployments cannot start shell-less ``FROM scratch`` images
    (observed on hecate: ``enroot-switchroot: failed to change directory: /root``,
    then ``/bin/sh: No such file or directory`` with the home mounted), and a
    static Go exporter needs no container at all.
    """

    # Exporter image (registry URI or `containers` alias); ignored, and may be `""`, when `binary` is set.
    container_image: str
    # Port the exporter serves `/metrics` on, on every worker node.
    port: int
    # Command line replacing the image's default entrypoint arguments.
    command: str | None = None
    # Host executable to run without a container; relative paths resolve against the srtctl checkout.
    binary: str | None = None
    # GPU exporter kind: `dcgm` (built-in DCGM command, labels, metrics and tachometer
    # scrape) or `custom` (any other exporter; set `command`, `gpu_labels` and
    # `gpu_metrics`, and tachometer keeps its rows as served).
    kind: Literal["dcgm", "custom"] = "dcgm"
    # GPU power telemetry: labels identifying a GPU in the scrape; unset means DCGM (`gpu`, `UUID`).
    gpu_labels: GpuLabelsConfig | None = None
    # GPU power telemetry: per-GPU metrics to record; unset means DCGM.
    gpu_metrics: GpuMetricsConfig | None = None

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.kind == "dcgm" and self.gpu_labels is not None:
            raise ValidationError("`gpu_labels` needs `kind: custom`; a `kind: dcgm` exporter uses DCGM's labels")
        if self.kind == "custom" and (
            (self.command is None and self.binary is None) or self.gpu_labels is None or self.gpu_metrics is None
        ):
            raise ValidationError("a `kind: custom` GPU exporter needs `command`, `gpu_labels` and `gpu_metrics`")

    @property
    def power_mapping(self) -> PowerMetricMapping:
        """How the power collector reads this exporter's scrape."""
        labels = self.gpu_labels or DCGM_GPU_LABELS
        metrics = self.gpu_metrics or DCGM_GPU_METRICS
        contract = {metric.column: metric for metric in UTILIZATION_METRICS}
        riders = (("gpu_util_pct", metrics.gpu_util), ("sm_active", metrics.sm_active))
        return PowerMetricMapping(
            power_metric=metrics.power.metric,
            power_scope=metrics.power.scope,
            gpu_index_label=labels.index,
            gpu_identity_label=labels.identity,
            utilization_metrics=tuple(
                UtilizationMetric(column, rider.metric, contract[column].unit, contract[column].max_value)
                for column, rider in riders
                if rider is not None
            ),
            instance_labels=tuple(labels.instance),
            temperature_metric=metrics.temperature.metric if metrics.temperature is not None else None,
        )


# Built-in exporter defaults (sweep path only; the --bash lifecycle keys on the
# raw recipe fields and never launches exporter containers). Pinned multi-arch
# registry URIs, so pyxis pulls the node's architecture with zero setup; both
# pins are production-verified on GB300 (all 19 DCGM families; 125 node
# families incl. meminfo, no host /proc mount needed). Ports are deliberately
# offset from the conventional 9400/9100 — managed clusters may already run
# host-level exporters there. Air-gapped or version-pinning clusters override
# the image through the srtslurm.yaml ``containers:`` alias map, which already
# resolves these fields.
DEFAULT_DCGM_EXPORTER = TelemetryExporterConfig(
    container_image="nvcr.io#nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04",
    port=9401,
    # No command: the tachometer launch derives --collect-interval from
    # observability.tachometer.collect_interval_ms so the exporter samples
    # exactly as often as it is scraped. An explicit command still wins.
)
DEFAULT_NODE_EXPORTER = TelemetryExporterConfig(
    container_image="quay.io#prometheus/node-exporter:v1.8.2",
    port=9101,
)
# Per-process and per-thread host telemetry from /proc: CPU seconds by mode,
# thread count and thread CPU by thread name, context switches, RSS, open fds --
# for the frontend, the worker handlers, the engine ranks and the client, grouped
# by command line (see services.exporters.process_exporter_config_yaml). This is the
# signal the Prometheus surface cannot carry: Dynamo publishes no process_* or
# thread metrics, and node_exporter only sees the machine.
#
# Launched HOST-NATIVE from the static Go binary `make setup` installs at
# configs/process-exporter (ncabatoff/process-exporter release tarball for the
# compute arch), like nats-server and etcd. The upstream image is FROM scratch
# (no shell, no /root) and pyxis/enroot on hecate refuses to start it; the binary
# needs neither a container nor privileges and reads the host /proc directly. A
# recipe may still point `process_exporter.container_image` at an image that has
# a shell and leave `binary` unset to get the container launch.
DEFAULT_PROCESS_EXPORTER = TelemetryExporterConfig(
    container_image="",
    port=9256,
    binary="configs/process-exporter",
)


@dataclass(frozen=True)
class TachometerConfig:
    """Native Tachometer collection for an observability-enabled run.

    ``enabled`` is tri-state: ``None`` (the default) means ON — every run
    collects Tachometer data with no ``tachometer:`` block at all; an
    explicit ``false`` opts out. Note that without ``observability.enabled``
    the TRT-LLM worker endpoints may have no engine metrics to serve (the
    observability expansion is what turns their content on); the frontend
    and the exporters are always worth capturing.

    DCGM, node and process exporters default ON via the ``resolved_*``
    properties (sweep path only): an explicit ``dcgm_exporter`` /
    ``node_exporter`` / ``process_exporter`` block always wins,
    ``default_exporters: false`` disables the built-ins, and the raw fields
    stay ``None`` unless the recipe set them — which is what the
    power-telemetry sharing validation and the --bash gate key on.
    """

    # None (default) collects on every run; false opts out.
    enabled: bool | None = None
    # Scraper command or path on the compute nodes.
    binary_path: str = "tachometer-scraper"
    # Milliseconds between scrapes of every endpoint — the same unit and name
    # as dcgm-exporter's --collect-interval. Replaces the retired Hz-based
    # ``default_frequency`` (1000ms == the old 1.0 Hz default).
    collect_interval_ms: int = 1000
    # Seconds between intermediate Parquet compactions; 0 disables them.
    sync_interval_secs: int = 120
    # How long the scraper gets after SIGTERM to flush + compact final.parquet
    # before the SIGKILL escalation. Compaction time scales with the arrow WAL
    # accumulated since the last periodic sync.
    shutdown_grace_secs: float = 120.0
    # Threads for Parquet compaction (POLARS_MAX_THREADS).
    compaction_threads: int = 4
    # Output directory below the run's log directory.
    storage_subdir: str = "tachometer"
    # Static key/value metadata attached to every scraped endpoint.
    extra_metadata: dict[str, str] = field(default_factory=dict)
    # Run the built-in DCGM, node, and process exporters when their blocks are unset; false disables them.
    default_exporters: bool = True
    # Resolved from srtslurm.yaml at load time; never read global config here.
    default_gpu_exporter: TelemetryExporterConfig | None = field(default_factory=lambda: DEFAULT_DCGM_EXPORTER)
    # GPU exporter; unset uses the cluster default (DCGM exporter on port 9401).
    dcgm_exporter: TelemetryExporterConfig | None = None
    # Host metrics exporter; unset uses node-exporter on port 9101.
    node_exporter: TelemetryExporterConfig | None = None
    # Per-process /proc exporter; unset uses the host-native configs/process-exporter on port 9256.
    process_exporter: TelemetryExporterConfig | None = None

    Schema: ClassVar[type[Schema]] = Schema

    @property
    def resolved_dcgm_exporter(self) -> TelemetryExporterConfig | None:
        """Recipe exporter, else the resolved cluster default."""
        if self.dcgm_exporter is not None:
            return self.dcgm_exporter
        return self.default_gpu_exporter if self.default_exporters else None

    @property
    def resolved_node_exporter(self) -> TelemetryExporterConfig | None:
        """User-configured node exporter, else the built-in default."""
        if self.node_exporter is not None:
            return self.node_exporter
        return DEFAULT_NODE_EXPORTER if self.default_exporters else None

    @property
    def resolved_process_exporter(self) -> TelemetryExporterConfig | None:
        """User-configured process exporter, else the built-in default."""
        if self.process_exporter is not None:
            return self.process_exporter
        return DEFAULT_PROCESS_EXPORTER if self.default_exporters else None


@dataclass(frozen=True)
class NsysObservabilityConfig:
    """Automatic NVTX tracing and CPU sampling of workers and Dynamo frontends.

    Enabled by ``observability.enabled`` unless explicitly opted out. An
    explicit top-level ``profiling`` mode takes precedence over this preset.
    By default the benchmark starts capture after warmup and stops it when
    measured work finishes. ``including_startup`` captures from process launch
    through teardown, including initialization and warmup.
    """

    # Set false to keep other observability signals without launching nsys.
    enabled: bool = True
    # measured_workload excludes warmup; including_startup spans process launch through teardown.
    capture_window: Literal["measured_workload", "including_startup"] = "measured_workload"
    # Maximum wait for a control acknowledgment or a step's report finalization.
    report_timeout_secs: int = 1800
    # Optional container path to libToolsInjection64.so for NVTX injection.
    nvtx_injection_path: str | None = None
    # CPU IP sampling and context-switch scope. process-tree fails on engines with
    # many threads ("Not enough resources ... switch to system-wide"); system-wide
    # samples every process on the node, none records NVTX only.
    cpu_sampling: Literal["system-wide", "process-tree", "none"] = "system-wide"

    def __post_init__(self) -> None:
        if self.capture_window not in {"measured_workload", "including_startup"}:
            raise ValidationError("observability.nsys.capture_window must be measured_workload or including_startup")
        if self.cpu_sampling not in {"system-wide", "process-tree", "none"}:
            raise ValidationError("observability.nsys.cpu_sampling must be system-wide, process-tree or none")
        if self.report_timeout_secs <= 0:
            raise ValidationError("observability.nsys.report_timeout_secs must be positive")
        if self.nvtx_injection_path is not None and not self.nvtx_injection_path.startswith("/"):
            raise ValidationError("observability.nsys.nvtx_injection_path must be an absolute container path")

    @property
    def terminate_timeout(self) -> int:
        """Allow report finalization, then the application tree's shutdown grace."""
        return self.report_timeout_secs + 150

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class ObservabilityConfig:
    """Observability configuration for OTEL tracing.

    When enable_otel is True, OTEL environment variables (DYN_LOGGING_JSONL,
    OTEL_EXPORT_ENABLED, OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, OTEL_SERVICE_NAME)
    are automatically injected into all workers and frontends.

    OTEL_SERVICE_NAME defaults to "dynamo-{component}" (e.g. dynamo-prefill,
    dynamo-decode, dynamo-frontend) and can be overridden per-component via
    roles.<role>.env or frontend.env.

    ``enabled`` configures server-side analytics capture. It expands (at config-load time,
    via :func:`srtctl.core.config.expand_observability`) into:

    * ``enable_iter_perf_stats`` + ``return_perf_metrics`` on every engine
      config -- the ``trtllm_kv_cache_*`` occupancy gauges and per-request
      histograms appear on that surface.
    * ``DYN_LOGGING_SPAN_EVENTS`` / ``DYN_LOGGING_JSONL`` / ``DYN_LOG=debug`` on
      prefill, decode and frontend -- per-request ``SPAN_CLOSED`` trace lines.

    and, for the run's server-side capture:

    * Nsight Systems NVTX tracing and CPU sampling on all worker processes/ranks
      and Dynamo frontends (``nsys.enabled: false`` opts out).
      Explicit top-level ``profiling`` takes precedence.
    * native Tachometer collection of every ``/metrics`` endpoint the benchmark
      client does not already poll (see ``TelemetryStageMixin.start_tachometer``
      and ``tachometer`` below).

    Expansion preserves explicit recipe values and leaves publication settings
    unchanged. TRT-LLM engine metrics default on via ``backend.publish_metrics``.
    The legacy combined flag requires explicit ``publish_events_and_metrics: true``.
    KV events require an explicit opt-in; on newer Dynamo builds, set
    ``DYN_TRTLLM_PUBLISH_KV_EVENTS: "true"`` in each worker role's environment.

    Scope is deliberately server-side. The knob configures what the workers and
    frontend *emit*, and captures that surface by scraping the endpoints
    directly. It never asks the benchmark client to re-export what the servers
    already publish. (One indirect exception: on TRT-LLM the client's
    ``AIPERF_SERVER_METRICS_URLS`` worker list exists only when
    the effective publication flags give those endpoints engine metrics — see
    ``BenchmarkStageMixin``.)

    Attributes:
        enabled: Master analytics knob. Default: False.
        enable_otel: If True, inject OTEL environment variables into all workers
            and frontends. Requires otel_endpoint to be set. Default: False.
        otel_endpoint: OTEL collector endpoint (e.g. "http://10.0.0.1:4317").
            Required when enable_otel is True.
        nsys: Automatic Nsight Systems capture, enabled with the master switch.
        tachometer: Native Tachometer capture configuration. Collects on every run,
            independent of ``enabled``, unless ``tachometer.enabled: false`` (see
            :class:`TachometerConfig`).

    The retired ``scrape_metrics`` / ``scrape_interval_seconds`` /
    ``scrape_output`` knobs (the in-job RAW Prometheus scraper) are rejected
    at load like any unknown key.
    """

    enabled: bool = False
    enable_otel: bool = False
    otel_endpoint: str | None = None

    tachometer: TachometerConfig = field(default_factory=TachometerConfig)
    nsys: NsysObservabilityConfig = field(default_factory=NsysObservabilityConfig)

    Schema: ClassVar[type[Schema]] = Schema

    @property
    def tachometer_enabled(self) -> bool:
        """Resolved Tachometer enablement (tri-state ``tachometer.enabled``).

        Tachometer is on by default for every run — server-side capture is
        not an opt-in special occasion, and its cost is bounded (1 Hz,
        best-effort, complement of the client's polling). ``enabled: false``
        opts out; ``observability.enabled`` no longer gates it.
        """
        return True if self.tachometer.enabled is None else self.tachometer.enabled


@dataclass(frozen=True)
class CpuPowerConfig:
    """Host-side CPU power collection on every worker node.

    This is the in-job Python collector (``srtctl.core.cpu_power``): one
    process per backend node reads Linux ACPI ``power_meter`` hwmon channels
    (or DCGM CPU entity field 1130) directly on the bare host and writes its
    own per-node CSV, which the head node aggregates at teardown into
    ``<storage_subdir>/samples.csv`` plus a manifest.

    It is independent of ``cpu_power_exporter`` (the head-node scraper over a
    per-node ``/metrics`` exporter): a recipe may enable either, both, or
    neither. The two legs share no ports and write to different directories.

    Attributes:
        enabled: Master switch for this leg. Default: False.
        source: ``auto`` tries ACPI then DCGM and is best-effort; naming
            ``acpi`` or ``dcgm`` explicitly makes that provider mandatory.
        sample_interval_seconds: Read period on each node, in seconds.
        startup_timeout_seconds: How long to wait for every node's collector
            to publish its ready marker before giving up on readiness.
        required: Fail the job when the leg does not become ready or does not
            produce a valid publication.
        storage_subdir: Directory below the run log directory that holds the
            CPU samples and manifest. Must differ from ``telemetry.storage_subdir``.
    """

    enabled: bool = False
    source: Literal["auto", "acpi", "dcgm"] = "auto"
    sample_interval_seconds: float = 0.1
    startup_timeout_seconds: float = 30.0
    required: bool = False
    storage_subdir: str = "cpu_power"

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class CpuPowerExporterConfig:
    """Best-effort CPU power collection via the cpu-power-exporter binary.

    Presence of this block (not a separate enabled flag) is what turns CPU
    power collection on. Unlike dcgm_exporter/node_exporter, this is not
    containerized -- cpu-power-exporter is a bundled binary installed by
    make setup, launched directly on the bare worker host, with a fallback
    to the Python stdlib exporter when the binary is absent.
    """

    # Port the exporter listens on and the head-node collector scrapes.
    port: int = 9405
    source: Literal["auto", "acpi", "dcgm"] = "auto"
    """Power reading back-end passed through to the bundled Rust binary's own
    ``--source`` flag (``auto`` | ``acpi`` | ``dcgm``). ``auto`` tries DCGM
    first and falls back to ACPI when libdcgm.so is absent or reports no CPU
    entities. Has no effect when the Python stdlib fallback exporter is used
    instead of the binary -- that fallback is ACPI-only.
    """

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class TelemetryConfig:
    """DCGM power telemetry for benchmark measurement windows."""

    # Collect GPU power over each benchmark concurrency window.
    enabled: bool = False
    # GPU power exporter image, port, command, labels and metrics. When `enabled` with
    # no exporter and no CPU leg, the cluster `default_gpu_exporter` is used.
    dcgm_exporter: TelemetryExporterConfig | None = None
    # Milliseconds between collector cycles. Replaces the retired
    # ``default_frequency``, which despite its name was a period in seconds
    # (1000ms == the old 1.0 default).
    collect_interval_ms: int = 1000
    # Output directory below the run's log directory.
    storage_subdir: str = "power"
    # Fail the benchmark when publishable DCGM power artifacts cannot be produced. CPU power stays best-effort.
    required: bool = False
    # Before any server starts, verify every allocation node reports an
    # NTP-synchronised system clock. Sample timestamps (orchestrator host) and
    # window boundaries (benchmark client host) are compared directly, so an
    # unsynchronised node silently misaligns the measurement. Fails the job
    # when ``required`` is true; otherwise the run continues and the manifest
    # records ``clock_sync_unverified`` with ``publication_valid: false``.
    # Set false on clusters where timedatectl/chronyc/ntpq are unavailable to
    # unprivileged users.
    clock_sync_check: bool = True
    # Seconds to wait for the exporters to answer before giving up (DCGM and CPU legs).
    startup_timeout_seconds: float = 30.0
    # Per-request exporter timeout in seconds (DCGM and CPU legs).
    request_timeout_seconds: float = 2.0
    # None derives a safe shutdown budget from request_timeout_seconds.
    collector_join_timeout_seconds: float | None = None
    # Head-node scrape of a per-node CPU power exporter. Setting the block enables it.
    cpu_power_exporter: CpuPowerExporterConfig | None = None
    # In-job CPU power collector on each node, independent of `cpu_power_exporter`.
    cpu_power: CpuPowerConfig = field(default_factory=CpuPowerConfig)

    Schema: ClassVar[type[Schema]] = Schema

    @property
    def resolved_collector_join_timeout_seconds(self) -> float:
        """Return the explicit join timeout or a request-timeout-aware default."""
        if self.collector_join_timeout_seconds is not None:
            return self.collector_join_timeout_seconds
        worst_case = 2 * (2 * self.request_timeout_seconds + _DCGM_POWER_COLLECT_CYCLE_TIMEOUT_GRACE_SECONDS)
        return worst_case + 2.0


def build_otel_env(observability: ObservabilityConfig, component: str) -> dict[str, str]:
    """Build OTEL environment variables for a component.

    Returns an empty dict if OTEL is disabled. Otherwise returns env vars
    with OTEL_SERVICE_NAME set to "dynamo-{component}".
    """
    if not observability.enable_otel or not observability.otel_endpoint:
        return {}
    return {
        "DYN_LOGGING_JSONL": "1",
        "OTEL_EXPORT_ENABLED": "1",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": observability.otel_endpoint,
        "OTEL_SERVICE_NAME": f"dynamo-{component}",
    }


# Env that makes Dynamo emit one JSONL ``SPAN_CLOSED`` line per closed span on
# the component's stdout.
# DYN_LOG=debug is required because the span events are emitted at DEBUG level.
ANALYTICS_SPAN_ENV: dict[str, str] = {
    "DYN_LOGGING_SPAN_EVENTS": "true",
    "DYN_LOGGING_JSONL": "true",
    "DYN_LOG": "debug",
}

# Env that makes the Dynamo *frontend* write one ``dynamo.request.trace.v1``
# ``request_end`` record per request to a JSONL file. This is a different signal
# from the span leg above: spans decompose the router in detail but treat each
# worker as one opaque ``handle_payload``, whereas these records carry the
# frontend's own phase timings -- ``prefill_wait_time_ms`` (receive to dispatch),
# ``prefill_time_ms`` (dispatch to first token) and, on disagg,
# ``kv_transfer_estimated_latency_ms`` -- plus ``x_request_id``, so they join to
# the client and span legs on the key those already use.
#
# Frontend-only: the timings come from the router's RequestTracker
# (``lib/llm/src/protocols/common/timing.rs``) and are emitted from the
# preprocessor. Workers have no tracker and would write empty files.
#
# Only two vars are needed:
#   * DYN_REQUEST_TRACE enables the default request-end records, the file sink,
#     and Dynamo's rotated jsonl_gz format.
#   * DYN_REQUEST_TRACE_FILE_PATH must be overridden. The built-in default is
#     /tmp/dynamo-request-trace, and container /tmp does not survive the job --
#     the capture would be written and then thrown away with the node.
ANALYTICS_REQUEST_TRACE_ENV: dict[str, str] = {
    "DYN_REQUEST_TRACE": "1",
    "DYN_REQUEST_TRACE_FILE_PATH": f"{CONTAINER_LOG_DIR}/dynamo-request-trace",
}

# Engine-config keys that surface per-request and per-iteration statistics on
# the worker's /metrics endpoint. ``enable_iter_perf_stats`` is what produces
# the ``trtllm_kv_cache_{used,free,max}_blocks`` gauges; ``return_perf_metrics``
# adds the per-request latency / KV-transfer histograms.
ANALYTICS_ENGINE_CONFIG: dict[str, bool] = {
    "enable_iter_perf_stats": True,
    "return_perf_metrics": True,
}

# Engine-config default baked in for every ``frontend.type: trtllm_serve`` run,
# independent of ``observability.enabled``. trtllm-serve registers a worker's
# Prometheus route (``/prometheus/metrics``) only when the engine runs with
# ``return_perf_metrics: true`` (TensorRT-LLM ``serve/openai_server.py``,
# ``register_routes``); TensorRT-LLM's own default is ``false``. Tachometer
# scrapes that route on every run, so without this default every trtllm-serve
# worker endpoint answers HTTP 404 and the capture silently has no worker data.
TRTLLM_SERVE_ENGINE_DEFAULTS: dict[str, bool] = {
    "return_perf_metrics": True,
}

# Engine-config default baked in for every TRT-LLM engine section a recipe
# uses, under both the ``dynamo`` and the ``trtllm_serve`` frontend and
# independent of ``observability.enabled``. ``dynamo.trtllm`` derives
# ``enable_iter_perf_stats`` from ``--publish-metrics``, which
# ``backend.publish_metrics`` passes by default, so without this every Dynamo
# worker would run TensorRT-LLM's per-iteration statistics (KV-cache stats and
# CUDA-event step timing on every executor loop) for gauges no benchmark client
# reads. The request-level ``trtllm_*`` Prometheus series (request latency,
# TTFT, TPOT, queue / prefill / decode time, token counters) only need the
# per-request perf metrics, which ``--publish-metrics`` (Dynamo) and
# ``return_perf_metrics: true`` (trtllm-serve) enable on their own. The engine
# YAML is merged over the worker's derived arguments and wins on conflicts
# (TensorRT-LLM ``update_llm_args_with_extra_dict``), so this explicit
# ``false`` keeps that surface and drops the statistics. ``expand_observability``
# runs first and setdefaults ``True`` for analytics runs, which keep their
# iteration-level ``trtllm_kv_cache_*`` gauges; an explicit recipe value
# always wins.
TRTLLM_ENGINE_DEFAULTS: dict[str, bool] = {
    "enable_iter_perf_stats": False,
}


# /configs/dynamo-wheels is the lustre-mounted cache for hash-pinned dynamo
# source builds. The bench/frontend container always mounts srtslurm's
# `configs/` dir at /configs (see RuntimeContext.container_mounts), so this
# path is reachable from every node without any extra recipe wiring.
_DYNAMO_CACHE_ROOT = "/configs/dynamo-wheels"


def dynamo_source_cache_key(dynamo_hash: str, cargo_patches: list[str] | None = None) -> str:
    """Return the cache key shared by Slurm and direct source builds.

    A ref that is not a commit (``refs/pull/14000/head`` when ``srtctl apply``
    could not pin it) is sanitized into a directory name; such a key can go
    stale as the ref moves, which is why apply pins refs to SHAs up front.
    """
    key = dynamo_hash.strip().replace("/", "-")
    if not cargo_patches:
        return key
    # Version the build recipe so a patching change invalidates old artifacts
    # even when the dependency declarations themselves do not change.
    digest = hashlib.sha1(("dep-override-v3\n" + "\n".join(cargo_patches)).encode()).hexdigest()[:8]
    return f"{key}-patch-{digest}"


def dynamo_cargo_patch_commands(cargo_patches: list[str] | None = None) -> tuple[str, ...]:
    """Return shell-safe Cargo.toml replacement commands for a source build."""
    if not cargo_patches:
        return ()
    commands = []
    for entry in cargo_patches:
        crate = entry.split("=", 1)[0].strip()
        if not crate:
            continue
        repl = entry.replace("&", r"\&")  # '&' is the sed replacement metachar
        script = f"s|^{crate}[[:space:]]*=.*|{repl}|"
        commands.append(f"find . -name Cargo.toml -exec sed -i -E {shlex.quote(script)} {{}} +")
    return tuple(commands)


def _git_clone_cmd() -> str:
    """Shell-quoted ``git`` invocation for install-script bash strings; see
    ``srtctl.core.config.git_clone_command_prefix`` for why this exists."""
    from srtctl.core.config import git_clone_command_prefix

    return shlex.join(git_clone_command_prefix())


def _hash_cached_source_install(
    dynamo_hash: str,
    cargo_patches: list[str] | None = None,
    repo_url: str = DynamoSourceConfig.DEFAULT_GIT,
) -> str:
    """Bash for hash-pinned source install with a /configs/dynamo-wheels cache.

    ``dynamo_hash`` is normally a commit SHA (``srtctl apply`` pins
    ``dynamo.source.rev`` before submit). A bare ref such as
    ``refs/pull/14000/head`` still works: it is fetched by name and checked out
    as ``FETCH_HEAD``, since a plain clone does not carry PR refs.

    Cache layout: ``{root}/<key>/`` contains the maturin wheel
    (``ai_dynamo_runtime-*.whl``), a tarball of the dynamo source tree
    (``dynamo-src.tar.gz``), and a ``.complete`` sentinel that's only touched
    on a successful build. flock on the per-key lock file serializes the
    cold-cache build across multiple frontends starting in parallel.

    ``cargo_patches`` (optional) are dependency-declaration overrides — each a full
    ``<crate> = <spec>`` line that replaces that crate's declaration across every
    ``Cargo.toml`` in the tree before ``maturin build`` (typically retargeting a crate
    like ``dynamo-tokenizers`` at a git branch). When set, the cache key is suffixed with
    a digest of the overrides so patched and unpatched builds of the same hash never collide.

    Uses FD 201 (not 200) so it nests cleanly inside the node-local
    ``flock -x 200`` from ``_serialize_node_install``.
    """
    cache_key = dynamo_source_cache_key(dynamo_hash, cargo_patches)
    # Replace each crate's dependency declaration tree-wide. Source replacement
    # (rather than [patch.crates-io]) forces Cargo to resolve the requested source
    # even when its version would not satisfy Dynamo's exact existing pin.
    patch_commands = dynamo_cargo_patch_commands(cargo_patches)
    override_cmd = " && ".join(patch_commands)
    if override_cmd:
        override_cmd += " && "
    cache = f"{_DYNAMO_CACHE_ROOT}/{cache_key}"
    lock = f"{_DYNAMO_CACHE_ROOT}/.{cache_key}.lock"
    checkout_cmd = (
        f"git checkout {dynamo_hash}"
        if is_commit_sha(dynamo_hash)
        else f"git fetch origin {shlex.quote(dynamo_hash)} && git checkout FETCH_HEAD"
    )
    return (
        f"echo 'Installing dynamo from source ({dynamo_hash}, /configs cache)...' && "
        f"mkdir -p {_DYNAMO_CACHE_ROOT} && "
        # Subshell + flock-FD pattern: only the first frontend in a cold-cache
        # job builds; later frontends block on the lock then read .complete.
        f"( "
        f"flock -x 201; "
        f"if [ ! -f {cache}/.complete ]; then "
        # Build tools — install on cold cache only. apt + protoc + cargo + maturin.
        f"apt-get update -qq && apt-get install -y -qq libclang-dev curl git protobuf-compiler > /dev/null 2>&1 && "
        f"if ! command -v cargo &>/dev/null; then "
        f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable -q && "
        f". $HOME/.cargo/env; fi && "
        # Force-reinstall maturin: some images ship the module without the
        # console-script, so `command -v maturin` fails AND a plain pip
        # install reports "already satisfied".
        f"pip install --break-system-packages --force-reinstall --quiet maturin && "
        # Clone + build the runtime wheel.
        f"DYN_BUILD_DIR=$(mktemp -d) && cd $DYN_BUILD_DIR && "
        f"{_git_clone_cmd()} clone {shlex.quote(repo_url)} dynamo && "
        f"cd dynamo && {checkout_cmd} && "
        f"{override_cmd}"
        f"cd lib/bindings/python/ && "
        f'export RUSTFLAGS="${{RUSTFLAGS:-}} -C target-cpu=native --cfg tokio_unstable" && '
        f"rm -f /tmp/ai_dynamo_runtime*.whl && "
        f"maturin build --release -o /tmp && "
        # Populate cache atomically: copy artifacts first, touch .complete last.
        f"mkdir -p {cache} && "
        f"cp /tmp/ai_dynamo_runtime*.whl {cache}/ && "
        f"cd $DYN_BUILD_DIR && "
        # Exclude cargo's target/ (~2 GB of compiled artifacts; not needed at
        # install time) and .git/ (~300 MB of pack files). Drops the tarball
        # from ~3 GB to ~100 MB.
        f"tar --exclude='target' --exclude='.git' -czf {cache}/dynamo-src.tar.gz dynamo && "
        f"touch {cache}/.complete && "
        f"cd / && rm -rf $DYN_BUILD_DIR; "
        f"fi "
        f") 201>{lock} && "
        # Install from the (now warm) cache. Both branches above land here.
        f"pip install --break-system-packages --force-reinstall {cache}/ai_dynamo_runtime-*.whl && "
        f"rm -rf /tmp/dynamo-src && mkdir -p /tmp/dynamo-src && "
        f"tar -xzf {cache}/dynamo-src.tar.gz -C /tmp/dynamo-src && "
        f"pip install --break-system-packages -e /tmp/dynamo-src/dynamo && "
        f"echo 'Dynamo installed from source ({dynamo_hash})'"
    )


def _live_source_install_for_top_of_tree() -> str:
    """Bash for live source install at HEAD — no cache (no stable key).

    Keeps the original SGLang-vs-portable bifurcation: SGLang containers
    already have rust + maturin in the right places at /sgl-workspace; other
    containers (vLLM, etc.) install everything from scratch into /tmp.
    """
    sglang = (
        # protobuf-compiler is required by modelexpress-common's build.rs (prost-build).
        # Some SGLang images ship without /usr/bin/protoc; install it unconditionally.
        "apt-get update -qq && apt-get install -y -qq libclang-dev curl protobuf-compiler > /dev/null 2>&1 && "
        "if ! command -v cargo &>/dev/null; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable -q && source $HOME/.cargo/env; fi && "
        # Force-reinstall maturin: see _hash_cached_source_install.
        "pip install --break-system-packages --force-reinstall --quiet maturin && "
        "cd /sgl-workspace/ && "
        f"{_git_clone_cmd()} clone https://github.com/ai-dynamo/dynamo.git && "
        "cd dynamo && "
        "cd lib/bindings/python/ && "
        'export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native --cfg tokio_unstable" && '
        "maturin build --release -o /tmp && "
        "pip install /tmp/ai_dynamo_runtime*.whl && "
        "cd /sgl-workspace/dynamo/ && "
        "pip install -e . && "
        "cd /sgl-workspace/sglang/ && "
        "echo 'Dynamo installed from source (HEAD)'"
    )

    portable = (
        "if ! command -v cargo &> /dev/null || ! command -v maturin &> /dev/null; then "
        "apt-get update -qq && apt-get install -y -qq git curl libclang-dev protobuf-compiler > /dev/null 2>&1 && "
        "if ! command -v cargo &> /dev/null; then "
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source $HOME/.cargo/env; fi; fi && "
        # Force-reinstall maturin: see _hash_cached_source_install.
        "pip install --break-system-packages --force-reinstall --quiet maturin && "
        "ORIG_DIR=$(pwd) && rm -rf /tmp/dynamo_build && mkdir -p /tmp/dynamo_build && cd /tmp/dynamo_build && "
        f"{_git_clone_cmd()} clone https://github.com/ai-dynamo/dynamo.git && "
        "cd dynamo && "
        "cd lib/bindings/python/ && "
        'export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native --cfg tokio_unstable" && '
        "rm -f /tmp/ai_dynamo_runtime*.whl && "
        "maturin build --release -o /tmp && "
        "pip install --break-system-packages /tmp/ai_dynamo_runtime*.whl --force-reinstall && "
        "cd /tmp/dynamo_build/dynamo/ && "
        "pip install --break-system-packages -e . && "
        "cd $ORIG_DIR && "
        "echo 'Dynamo installed from source (HEAD)'"
    )

    return (
        "echo 'Installing dynamo from source (HEAD)...' && "
        f"if [ -d /sgl-workspace ]; then {sglang}; else {portable}; fi"
    )


def _serialize_node_install(install_cmd: str) -> str:
    """Serialize a node-shared dynamo install across co-located srun tasks.

    With ``--ntasks-per-node > 1`` (e.g. TRTLLM's MPI-style launch, one task
    per GPU), every task on a node runs the worker preamble concurrently
    against the same shared container root. Concurrent pip installs into the
    same site-packages race and corrupt each other. An exclusive ``flock``
    serializes them; a sentinel lets every task after the first short-circuit
    the (idempotent) install entirely.

    The lock/sentinel are anchored in the active Python environment
    (``sys.prefix``) — the exact resource being protected. That location is
    part of the container root filesystem, so it is:
      * shared by every task sharing that site-packages (correct serialization),
      * private to each container instance, so co-located containers with a
        bind-mounted /tmp neither over-serialize nor wrongly skip each other's
        install, and distinct across jobs (no cross-job/version staleness).

    FD 200 (node-local) is kept distinct from the ``flock -x 201`` that the
    hash-pinned source install nests on the /configs cache lock inside a
    subshell. Distinct FDs keep the two node-local and cross-node locks
    independent and refactor-proof even if that inner subshell is removed.
    """
    # Resolve the env dir at runtime; fall back to $HOME (also container-private)
    # if python3 is somehow unavailable before the install runs.
    resolve_dir = 'DYN_LOCK_DIR="$(python3 -c \'import sys; print(sys.prefix)\' 2>/dev/null || echo "${HOME:-/root}")"'
    lock = '"$DYN_LOCK_DIR/.srtctl_dynamo_install.lock"'
    sentinel = '"$DYN_LOCK_DIR/.srtctl_dynamo_install.complete"'
    return (
        f"{resolve_dir} && "
        f"( flock -x 200; "
        f"if [ -f {sentinel} ]; then "
        f"echo 'dynamo install already completed in this environment, skipping'; "
        f"else {{ {install_cmd} ; }} && touch {sentinel}; fi "
        f") 200>{lock}"
    )


RequestPlane = Literal["nats", "tcp", "http"]
EventPlane = Literal["nats", "zmq"]


@dataclass
class DynamoConfig:
    """Dynamo installation configuration.

    ``source`` names the Dynamo to install: a git ref to build, a PyPI release,
    or a staged wheel. ``top_of_tree`` builds HEAD unpinned instead. With
    neither, ``install: true`` pip-installs the PyPI release
    ``DEFAULT_PYPI_VERSION``. ``effective_source`` is what actually gets
    installed; ``pypi_version``, ``git_rev``, ``wheel_version``, and
    ``cargo_patches`` are read-only views of it for the install path.

    Options:
        install: Whether to install dynamo at all (default: True). Set to False
                 if your container already has dynamo pre-installed.
        top_of_tree: Clone repo at HEAD (latest). No immutable equivalent under
                     ``source``; prefer pinning a commit in ``source.rev``.
        source: Which Dynamo to install: ``git`` + ``rev`` (commit, tag, or
               ``refs/pull/<n>/head``; ``srtctl apply`` pins it to ``sha``),
               ``pypi``, or ``wheel``. Cannot be combined with ``top_of_tree``.
        request_plane: Request plane to use (default: "tcp"). Valid values: "nats", "tcp", "http"
        event_plane: Event plane override, sets DYN_EVENT_PLANE (default: None — follow
                     the Dynamo image's own default). Valid values: "nats", "zmq"
        sidecar: Replace the Python workers with native engines and Dynamo sidecars.
    """

    # PyPI release installed when a recipe names no source and does not ask for top_of_tree.
    DEFAULT_PYPI_VERSION: ClassVar[str] = "0.8.0"
    _VALID_REQUEST_PLANES: ClassVar[tuple[str, ...]] = get_args(RequestPlane)
    _VALID_EVENT_PLANES: ClassVar[tuple[str, ...]] = get_args(EventPlane)

    # Install Dynamo into the container before workers start; false when the image already has it.
    install: bool = True
    # Clone and build Dynamo at HEAD (unpinned). No `source` equivalent; prefer a commit in `source.rev`.
    top_of_tree: bool = False
    # Which Dynamo to install: exactly one of git+rev, pypi, or wheel. Unset, and not
    # top_of_tree: the PyPI release DEFAULT_PYPI_VERSION.
    source: DynamoSourceConfig | None = None
    # Transport frontends use to send requests to workers.
    request_plane: RequestPlane = "tcp"
    # Sets DYN_EVENT_PLANE for KV and worker events; unset follows the Dynamo image's default.
    event_plane: EventPlane | None = None
    # Job-wide native sidecar mode; prefer `roles.<role>.sidecar: true`.
    sidecar: bool = False
    # Base loopback gRPC port between engine and sidecar; co-located workers get deterministic offsets.
    sidecar_port: int = DYNAMO_SIDECAR_GRPC_PORT
    # Standalone sidecar executable; unset runs `python3 -m dynamo.<framework>.sidecar`.
    sidecar_binary: str | None = None
    # Seconds to wait for the native engine's gRPC endpoint.
    sidecar_startup_timeout: int = 3600
    # Context length the sidecar advertises (TRT-LLM); unset reads it from the engine.
    sidecar_context_length: int | None = None
    # Extra arguments appended to the sidecar command.
    sidecar_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.top_of_tree and self.source is not None:
            raise ValueError(
                "dynamo.top_of_tree cannot be combined with dynamo.source; pin a commit in source.rev or drop top_of_tree"
            )

        if self.request_plane not in self._VALID_REQUEST_PLANES:
            raise ValueError(
                f"Invalid request_plane '{self.request_plane}', must be one of: {', '.join(self._VALID_REQUEST_PLANES)}"
            )

        if self.event_plane is not None and self.event_plane not in self._VALID_EVENT_PLANES:
            raise ValueError(
                f"Invalid event_plane '{self.event_plane}', must be one of: {', '.join(self._VALID_EVENT_PLANES)}"
            )

        if not 1 <= self.sidecar_port <= 65535:
            raise ValueError(f"dynamo.sidecar_port must be between 1 and 65535, got {self.sidecar_port}")
        if self.sidecar_startup_timeout < 1:
            raise ValueError("dynamo.sidecar_startup_timeout must be at least 1")
        if self.sidecar_binary is not None and not self.sidecar_binary.strip():
            raise ValueError("dynamo.sidecar_binary must be a non-empty executable path")
        if self.sidecar_context_length is not None and self.sidecar_context_length < 1:
            raise ValueError("dynamo.sidecar_context_length must be at least 1")

    @property
    def effective_source(self) -> DynamoSourceConfig | None:
        """What gets installed: ``source`` as written, the PyPI default when nothing was named, None for top_of_tree."""
        if self.source is not None:
            return self.source
        if self.top_of_tree:
            return None
        return DynamoSourceConfig(pypi=self.DEFAULT_PYPI_VERSION)

    @property
    def pypi_version(self) -> str | None:
        """PyPI release to pip-install, when the install is a PyPI release."""
        source = self.effective_source
        return source.pypi if source is not None else None

    @property
    def git_rev(self) -> str | None:
        """Commit (or ref) to build from, when the install is a git source."""
        source = self.effective_source
        return source.checkout if source is not None and source.git is not None else None

    @property
    def cargo_patches(self) -> list[str] | None:
        """Cargo dependency replacements applied before a git source build."""
        source = self.effective_source
        return list(source.patches) if source is not None and source.git is not None and source.patches else None

    @property
    def needs_source_install(self) -> bool:
        """Whether this config requires a source install (git clone + maturin)."""
        return self.top_of_tree or self.git_rev is not None

    @property
    def wheel_version(self) -> str | None:
        """Package version requested for staged wheel installation."""
        source = self.effective_source
        return source.wheel if source is not None else None

    @property
    def wheel_name(self) -> str | None:
        """Return the ai-dynamo wheel filename for the requested package version."""
        version = self.wheel_version
        return f"ai_dynamo-{version}-py3-none-any.whl" if version else None

    def get_wheel_environment(self) -> dict[str, str]:
        """Environment variables consumed by ai-dynamo prefetch/setup scripts."""
        version = self.wheel_version
        if not version:
            return {}
        return {"DYNAMO_WHEEL_NAME": f"ai_dynamo-{version}-py3-none-any.whl", "DYNAMO_VERSION": version}

    def get_install_commands(self) -> str:
        """Get the bash commands to install dynamo.

        The returned command is wrapped in a node-local flock + sentinel so
        that co-located srun tasks (``--ntasks-per-node > 1``, e.g. TRTLLM)
        install once per node instead of racing concurrent pip installs into
        the shared container site-packages. See ``_serialize_node_install``.
        """
        return _serialize_node_install(self._build_install_commands())

    def _build_install_commands(self) -> str:
        """Build the raw (unserialized) dynamo install command."""
        wheel_name = self.wheel_name
        if wheel_name is not None:
            start_message = shlex.quote(f"Installing ai-dynamo-runtime and ai-dynamo from wheel {wheel_name}...")
            done_message = shlex.quote(f"ai-dynamo-runtime and ai-dynamo install path completed for {wheel_name}")
            return (
                f"echo {start_message} && "
                "if [ -f /srtctl-runtime/dynamo_wheels.py ]; then "
                "python3 /srtctl-runtime/dynamo_wheels.py install; "
                "else "
                "echo 'ERROR: /srtctl-runtime/dynamo_wheels.py not found for ai-dynamo wheel install' >&2; "
                "exit 1; "
                "fi && "
                f"echo {done_message}"
            )

        pypi_version = self.pypi_version
        if pypi_version is not None:
            return (
                f"echo 'Installing dynamo {pypi_version}...' && "
                f"pip install --break-system-packages --quiet --extra-index-url https://pypi.nvidia.com ai-dynamo-runtime=={pypi_version} ai-dynamo=={pypi_version} && "
                f"echo 'Dynamo {pypi_version} installed'"
            )

        # Source install. When pinned to an immutable hash, cache the build on
        # /configs (lustre, shared by every job) keyed by hash. First frontend
        # in any job hitting a cold cache builds under flock; everyone else
        # reuses the artifacts. Drops bootstrap from ~5 min + flaky github clone
        # to ~10 sec lustre access for repeat hashes. top_of_tree skips the
        # cache (no stable key) and always live-builds.
        git_rev = self.git_rev
        if git_rev is not None:
            assert self.source is not None and self.source.git is not None
            return _hash_cached_source_install(git_rev, self.cargo_patches, repo_url=self.source.git)

        return _live_source_install_for_top_of_tree()

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class FrontendConfig:
    """Frontend/router configuration.

    Attributes:
        type: Frontend type - "dynamo" (default); "sglang-router" (SGLang Model
            Gateway), "vllm-router", "smg" (Shepherd Model Gateway, any backend), "llm-d"
            (llm-d Endpoint Picker behind Envoy, vLLM), "atomesh", and
            "tilert-router" (static routers); "sglang", "vllm", and
            "trtllm_serve" (direct: the single aggregate worker binds the public
            port, no router process); "none" (services-only job: no router, no
            OpenAI endpoint, no worker-count health gate; requires no engine
            roles). Pre-2.0 recipes spelled the router "sglang"; ``srtctl migrate``
            rewrites that to "sglang-router".
        enable_multiple_frontends: Scale with nginx + multiple routers.
            When ``True`` (default), srtctl stands up nginx and fans out
            to ``num_additional_frontends + 1`` router replicas. When
            ``False``, there is NO nginx proxy — the benchmark must
            target the single master router (or a worker) directly at
            ``http://localhost:<port>``. ``benchmark.command`` has no
            placeholder substitution, so write the URL out literally.
        num_additional_frontends: Additional routers beyond master (default: 9)
        nginx_container: Custom nginx container image (default: nginx:1.27.4)
        nginx_raise_ulimit: Raise nofile before nginx and set ``worker_rlimit_nofile``
            in generated nginx.conf. Off by default; enable on clusters that allow it.
            Override per job or set ``nginx_raise_ulimit`` in srtslurm.yaml for the cluster.
        nginx_session_affinity: Consistently hash ``nginx_session_affinity_header`` to a
            frontend. Requests without that header use a generated request ID and stay distributed.
        nginx_keepalive_timeout: Idle timeout for client and upstream keepalive
            connections in the generated nginx.conf (default "600s"). nginx's own
            default is 75s, which closes a session's connection during the long
            recorded think-time of an agentic replay; the client's next write on
            that pooled socket then fails with "broken pipe" / "server
            disconnected" and nothing is logged server-side.
        nginx_session_affinity_header: Header hashed when affinity is on (default
            ``X-Dynamo-Session-ID``). Set ``X-Correlation-ID`` for clients (e.g. aiperf) that
            carry the session id in that header instead.
        worker_selection: Inline Dynamo worker-selection policy configuration. srtctl
            writes this mapping under the top-level ``worker_selection`` key in a
            generated router policy YAML and passes it to the Dynamo frontend via
            ``--router-policy-config``.
        epp_config: llm-d Endpoint Picker configuration (``EndpointPickerConfig``:
            ``plugins``, ``schedulingProfiles``, ...) for ``frontend.type: llm-d``.
            srtctl writes it to a file with the ``file-discovery`` plugin and
            ``dataLayer.discovery`` added, so the EPP reads the job's workers from
            the endpoints file srtctl renders. Omitted, srtctl runs the scorers of
            llm-d's no-Kubernetes guide (in prefill and decode profiles for a
            prefill/decode job).
        args: CLI arguments passed to the frontend/router process
        env: Environment variables for frontend processes
        container_image: Optional router-specific image. Static routers use the
            model/backend image when omitted.
        numa_bind: Prefix the frontend process command with
            ``numactl --cpunodebind=0 --membind=0``. Off by default. Has no
            effect on direct frontends (``sglang``, ``vllm``, aggregate
            ``trtllm_serve``) that launch no separate frontend process.
    """

    type: str = "dynamo"
    enable_multiple_frontends: bool = True
    num_additional_frontends: int = 9
    nginx_container: str = "nginx:1.27.4"
    nginx_raise_ulimit: bool = False
    nginx_session_affinity: bool = False
    nginx_session_affinity_header: str = "X-Dynamo-Session-ID"
    nginx_keepalive_timeout: str = "600s"
    worker_selection: dict[str, Any] | None = None
    epp_config: dict[str, Any] | None = None
    args: dict[str, Any] | None = None
    env: dict[str, str] | None = None
    container_image: str | None = None
    numa_bind: bool = False
    # trtllm_serve orchestrator (ser.yaml) options; ignored by other frontends.
    ctx_router: dict[str, Any] | None = None  # context_servers.router, e.g. {type: conversation}
    gen_router: dict[str, Any] | None = None  # generation_servers.router
    server_config_extra: dict[str, Any] | None = None  # extra top-level ser.yaml keys
    # Where the frontend (trtllm_serve: the disaggregated orchestrator) runs.
    # placement.node is "head" (default: the first prefill/CTX node), "first_decode"
    # (the first decode/GEN worker-leader node), or "dedicated" (a node reserved for
    # the frontend: needs at least 2 nodes, not supported with resources.het_jobs: true).
    placement: PlacementConfig = field(default_factory=PlacementConfig)

    Schema: ClassVar[builtins.type[Schema]] = Schema


@dataclass(frozen=True)
class OutputConfig:
    """Output paths and optional reproducibility artifacts."""

    # Directory for job logs and results; a FormattablePath, so `{job_id}` and `$VARS` expand.
    log_dir: Annotated[FormattablePath, FormattablePathField()] = field(
        default_factory=lambda: FormattablePath(template="./outputs/{job_id}/logs")
    )
    # Save the realized `srun` scripts and a manifest under `logs/launch-plan/`.
    record_launch_plan: bool = False

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration.

    Attributes:
        max_attempts: Maximum readiness polls of the frontend before the run fails;
            180 x 10 s = 30 minutes by default (large models take time to load).
        interval_seconds: Seconds between readiness polls.
        fatal_log_markers: Fail the run as soon as a worker's log prints a line the
            engine names as fatal (for TRT-LLM, the launcher's ``Rank<N> Task exit
            code: <non-zero>`` and ``Failed to initialize executor``), even while
            its srun step is still running. Without it a worker whose engine died
            behind a live launcher is only noticed when this health window runs out.
        extra_fatal_log_patterns: Additional regular expressions, matched against
            every new worker log line, that fail the run the same way.
    """

    max_attempts: int = 180
    interval_seconds: int = 10
    fatal_log_markers: bool = True
    extra_fatal_log_patterns: list[str] = field(default_factory=list)

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        import re

        for pattern in self.extra_fatal_log_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValidationError(
                    f"health_check.extra_fatal_log_patterns: {pattern!r} is not a valid regular expression: {exc}"
                ) from None


# ============================================================================
# Main Configuration Dataclass
# ============================================================================

# Recipe schema versions. Version 2 is the 2.0 layout and the only one the
# loader accepts; a recipe without a top-level `schema:` key is the pre-2.0
# layout (version 1) and is rejected by `srtctl.core.config.require_current_schema`.
# `srtctl migrate` still reads version 1 and rewrites it to the current one.
CURRENT_SCHEMA_VERSION = 2

# Service kinds that form the discovery plane and share the infra node.
INFRA_SERVICE_TYPES: tuple[str, ...] = ("etcd", "nats")
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (CURRENT_SCHEMA_VERSION,)


@dataclass(frozen=True)
class SrtConfig:
    """Complete srtctl job configuration (frozen, immutable).

    This is the main configuration type returned by load_config(). ``engine`` selects the
    engine (polymorphic on ``type``), ``roles`` carries the worker topology and every
    per-role setting; ``topology`` and ``backend`` are derived from them once.
    """

    # Job name: the Slurm `--job-name` (unless RUNNER_NAME is set) and the run's label in results.
    name: str
    # Model weights, container image, and precision.
    model: ModelConfig
    # GPU type, GPUs per node, and allocation knobs. The worker topology is `roles`.
    resources: ResourceConfig

    # Recipe schema version (YAML key `schema`). A recipe must declare `schema: 2`;
    # `require_current_schema` rejects a missing key as the pre-2.0 layout before
    # the schema ever sees the document. The default only serves documents srtctl
    # itself resolved (a lockfile's recipe, a dumped config).
    schema_version: int = field(
        default=CURRENT_SCHEMA_VERSION,
        metadata={
            "marshmallow_field": fields.Integer(
                data_key="schema",
                load_default=CURRENT_SCHEMA_VERSION,
                validate=validate.OneOf(SUPPORTED_SCHEMA_VERSIONS),
            )
        },
    )

    # Slurm account, partition, and time limit; unset values come from srtslurm.yaml.
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    # The engine every role runs: a type (`sglang`) or a mapping with `type` plus engine-wide
    # knobs (see the engine types). Omit it when every role declares its own `engine`.
    engine: Annotated[BackendConfig | None, BackendConfigField(allow_none=True, reject_per_role_keys=True)] = None
    # One block per worker role (`prefill`, `decode`, `agg`): nodes, workers, GPUs, env, engine args.
    roles: dict[str, RoleConfig] = field(default_factory=dict)
    # The HTTP entry point in front of the workers (Dynamo frontend, router, nginx) and where it runs.
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    # Which Dynamo to install, its request/event planes, and native sidecar mode.
    dynamo: DynamoConfig = field(default_factory=DynamoConfig)
    # The client run once the workers are ready; `type` selects the runner.
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    # Nsight Systems or PyTorch profiling of the workers.
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    # Where the job writes logs and results.
    output: OutputConfig = field(default_factory=OutputConfig)
    # How long to poll the frontend for ready workers before failing the run.
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    # Engine metrics and traces, Tachometer collection, and automatic Nsight tracing.
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    # GPU (DCGM) and CPU power sampling over the benchmark measurement windows.
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # Environment variables for every worker; applied after `roles.<role>.env`, so a key set in both
    # takes this value. Values may use `{node}` and `{node_id}`.
    environment: dict[str, str] = field(default_factory=dict)
    # Host path -> container path mounts for every container; both sides are FormattablePaths.
    container_mounts: dict[
        Annotated[FormattablePath, FormattablePathField()],
        Annotated[FormattablePath, FormattablePathField()],
    ] = field(default_factory=dict)
    # Extra mounts as `host:container[:ro]` strings; `$VARS` expand, `{placeholders}` do not.
    extra_mount: tuple[str, ...] | None = None
    # Extra srun options (`key: value` -> `--key=value`; empty value -> `--key`) for every job step.
    srun_options: dict[str, str] = field(default_factory=dict)
    # Extra `#SBATCH --key=value` lines (empty value -> `--key`); wins over the cluster defaults.
    sbatch_directives: dict[str, str] = field(default_factory=dict)
    # Accepted for compatibility; srtctl does not read it. Workers dump their config where the engine supports it.
    enable_config_dump: bool = True

    # Custom setup script (runs before dynamo install and worker startup)
    # e.g. "custom-setup.sh" -> runs /configs/custom-setup.sh
    setup_script: str | None = None

    # Commands run on each node's bare host, outside the container, before any
    # worker starts. Cluster-wide default lives in srtslurm.yaml as
    # default_host_setup; a recipe that sets this block replaces that default.
    host_setup: HostSetupConfig = field(default_factory=HostSetupConfig)

    # Long-running processes launched next to the job: generic sidecars (an
    # experimental router built from a PR) and typed ones (a standalone Mooncake
    # store per worker node). See docs/services.md.
    services: list[ServiceConfig] = field(default_factory=list)

    # Post-benchmark / eval-only evaluation dispatch: extra env forwarded into the
    # eval process and an optional command override. Replaces the downstream
    # source patch that used to extend the passthrough list in do_sweep.py.
    post_eval: PostEvalConfig = field(default_factory=PostEvalConfig)

    # Virtual identity — declares what *should* be running (verified against fingerprint)
    identity: IdentityConfig = field(default_factory=IdentityConfig)

    # Reporting configuration (status API, future: logs to S3, etc.)
    reporting: ReportingConfig | None = None

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_roles()
        _ = self.backend  # bind the roles onto the engine now, so a bad role setting fails at load
        self._validate_role_backends()
        self._validate_frontend_worker_selection()
        self._validate_profiling()
        self._validate_observability()
        self._validate_telemetry()
        self._validate_mooncake_kv_store()
        self._validate_het_jobs()
        self._validate_colocated_decode()
        self._validate_frontend()
        self._validate_dynamo_sidecar()
        self._validate_vllm_failover()
        self._validate_vllm_discovery_connector()
        self._validate_host_setup()
        self._validate_benchmark_type()
        self._validate_services_only()
        self._validate_services()
        self._warn_dp_launch_mode()

    @cached_property
    def topology(self) -> Topology:
        """The worker layout: every per-role count, derived once from ``roles`` and ``resources``."""
        return Topology(roles=self.roles, gpus_per_node=self.resources.gpus_per_node, het_jobs=self.resources.het_jobs)

    @cached_property
    def role_backends(self) -> dict[str, BackendConfig]:
        """Per-role engines (``roles.<role>.engine``), each bound to its role's settings; empty with one shared engine."""
        if self.engine is not None or all(spec.engine is None for spec in self.roles.values()):
            return {}
        return {
            role: _bind_roles(spec.engine, {role: spec}) for role, spec in self.roles.items() if spec.engine is not None
        }

    @cached_property
    def backend(self) -> BackendConfig:
        """The serving engine with the recipe's roles bound (it reads per-role env, args, extra_args, kv_events).

        With one shared ``engine`` this is that engine (SGLang when none is declared). With
        per-role engines it is the serving role's (decode, else agg, else prefill); workers
        resolve their own through ``backend_for_role``.
        """
        if self.role_backends:
            serving = next(role for role in ("decode", "agg", "prefill") if role in self.role_backends)
            return self.role_backends[serving]
        return _bind_roles(self.engine if self.engine is not None else SGLangBackend(), self.roles)

    @property
    def role_containers(self) -> dict[str, str]:
        """Role images from ``roles.<role>.container``."""
        return {role: spec.container for role, spec in self.roles.items() if spec.container is not None}

    @property
    def has_role_backends(self) -> bool:
        return bool(self.role_backends)

    def backend_for_role(self, mode: str) -> BackendConfig:
        """Resolve the concrete engine for an endpoint's role."""
        from srtctl.core.worker_backends import role_name

        return self.role_backends.get(role_name(mode), self.backend)

    def worker_container_for_role(self, mode: str) -> str:
        """Role-specific worker image, or the shared model image."""
        from srtctl.core.worker_backends import role_name

        image = os.path.expandvars(self.role_containers.get(role_name(mode), self.model.container))
        return str(Path(image).resolve()) if image.startswith(("/", "./")) else image

    def active_role_backends(self) -> list[tuple[str, BackendConfig]]:
        return [
            (role, self.backend_for_role(role))
            for role, count in (
                ("prefill", self.topology.num_prefill),
                ("decode", self.topology.num_decode),
                ("agg", self.topology.num_agg),
            )
            if count
        ]

    def allocate_worker_endpoints(self, nodes: Sequence[str]) -> list["Endpoint"]:
        from srtctl.core.worker_backends import allocate_worker_endpoints

        return allocate_worker_endpoints(self, nodes)

    def worker_processes(
        self, endpoints: list["Endpoint"], port_allocator: "NodePortAllocator | None" = None
    ) -> list["Process"]:
        from srtctl.core.worker_backends import worker_processes

        return worker_processes(self, endpoints, port_allocator)

    def _validate_roles(self) -> None:
        """Rules of ``roles:`` that need the whole recipe.

        Role names, node counts, the engine form (one shared ``engine`` or one on every
        role), the colocated split, and the job-wide sidecar flag.
        """
        for role, spec in self.roles.items():
            if role not in ROLE_TO_MODE:
                raise ValidationError(f"unknown role {role!r}; valid roles are {', '.join(ROLE_NAMES)}")
            if spec.colocated and role != "decode":
                raise ValidationError(f"roles.{role}.nodes: only the decode role can colocate (on the prefill nodes)")
            if isinstance(spec.nodes, int):
                if spec.nodes == 0 and role == "decode":
                    raise ValidationError(
                        "roles.decode.nodes: 0 is not accepted; write nodes: colocate to share the prefill nodes"
                    )
                if spec.nodes < 1:
                    raise ValidationError(f"roles.{role}.nodes must be at least 1; got {spec.nodes}")
            if spec.engine is not None and self.engine is not None:
                raise ValidationError(f"roles.{role}.engine cannot be combined with a top-level engine")
        if self.engine is None and any(spec.engine is not None for spec in self.roles.values()):
            for role, spec in self.roles.items():
                if spec.engine is None:
                    raise ValidationError(f"roles.{role}.engine must name a type when no top-level engine is set")
        for role, spec in self.roles.items():
            engine = spec.engine if spec.engine is not None else self.engine
            engine_type = engine.type if engine is not None else "sglang"
            if spec.extra_args and engine_type != "trtllm":
                raise ValidationError(f"roles.{role}.extra_args is only supported by the trtllm engine")
            if spec.kv_events is not None and engine_type not in ("sglang", "vllm"):
                raise ValidationError(f"roles.{role}.kv_events is not supported by the {engine_type} engine")

        decode = self.roles.get("decode")
        if decode is not None and decode.colocated:
            # A colocated split cannot be derived: the per-node formula would hand prefill every GPU
            # and the decode size would silently inherit it. Both roles must state their worker size.
            missing = [
                role for role in ("prefill", "decode") if self.roles.get(role) is None or self.roles[role].gpus is None
            ]
            if missing:
                raise ValidationError(
                    "roles.decode.nodes: colocate requires an explicit gpus: on both prefill and decode "
                    f"(missing on {', '.join(missing)}); the GPU split is validated against the prefill nodes at load"
                )

        sidecars = {spec.sidecar for spec in self.roles.values() if spec.sidecar is not None}
        if len(sidecars) > 1:
            raise ValidationError("roles.*.sidecar must agree across roles (the Dynamo sidecar mode is job-wide)")
        if sidecars:
            wanted = sidecars.pop()
            if self.dynamo.sidecar and not wanted:
                raise ValidationError("roles.*.sidecar: false disagrees with dynamo.sidecar: true")
            if wanted and not self.dynamo.sidecar:
                # The sidecar mode is job-wide: a role asking for it turns it on for the job.
                object.__setattr__(self, "dynamo", dataclasses.replace(self.dynamo, sidecar=True))

    def _validate_role_backends(self) -> None:
        """Reject job-wide orchestration that is not yet role-aware."""
        if not self.has_role_backends:
            return
        if self.frontend.type == "dynamo" or self.dynamo.sidecar:
            raise ValidationError("role-specific engines do not yet support the Dynamo frontend or sidecars")
        if self.resources.het_jobs is True:
            raise ValidationError("role-specific engines do not yet support resources.het_jobs")
        if self.profiling.enabled or self.observability_nsys_enabled:
            raise ValidationError("role-specific engines do not yet support profiling or observability.nsys")
        gpus_per_node = self.resources.gpus_per_node
        for role, backend in [("default", self.backend), *self.active_role_backends()]:
            if isinstance(backend, VLLMBackend) and backend.discovers_workers():
                raise ValidationError("role-specific engines do not yet support vLLM discovery connectors")
            if backend.mooncake_kv_store is not None or backend.failover is not None:
                raise ValidationError(
                    f"role-specific engines do not yet support implicit Mooncake stores or failover ({role})"
                )
            if role != "default":
                if isinstance(backend, SGLangBackend) and backend.is_grpc_mode(cast("WorkerMode", role)):
                    raise ValidationError("role-specific engines do not yet support SGLang gRPC workers")
                gpus = self.topology.gpus_per_worker(role)
                if gpus > gpus_per_node and (backend.type == "trtllm" or gpus % gpus_per_node):
                    raise ValidationError(
                        f"roles.{role}: role-specific engines require whole-node multi-node workers; "
                        "multi-node TRT-LLM packing is not yet supported"
                    )

    def _validate_services_only(self) -> None:
        """Rules for ``frontend.type: none`` and for services that own nodes (pools).

        ``frontend.type: none`` means no OpenAI endpoint and no worker-count
        health gate: readiness is the services' own probes, and the benchmark
        step is the only thing that runs against them. With engine workers
        present that gate is what keeps a run from benchmarking a half-loaded
        fleet, so ``none`` is refused until per-worker readiness exists.

        A service with ``nodes`` owns a pool that adds to the allocation next to
        the engine roles' nodes. Pools are whole nodes, carved in declaration
        order; a heterogeneous SLURM job cannot carry them.
        """
        owners = self.pool_services
        if owners and self.resources.het_jobs is True:
            raise ValidationError(
                "services[].nodes (pools) are not supported together with resources.het_jobs: true; "
                f"owners: {', '.join(svc.name for svc in owners)}"
            )
        owner_names = {svc.name for svc in owners}
        for svc in self.services:
            pool = svc.placement.pool if svc.placement is not None else None
            if pool is not None and pool not in owner_names:
                raise ValidationError(
                    f"services[{svc.name}].placement.pool {pool!r} names no service that declares nodes "
                    f"(pools: {', '.join(sorted(owner_names)) or 'none'})"
                )
        if self.frontend.type == "none":
            if self.topology.has_engine_workers:
                raise ValidationError(
                    "frontend.type: none is only supported without engine roles (no prefill/decode/agg workers); "
                    "pick a frontend for the workers or drop them"
                )
            if self.frontend.placement.dedicated:
                raise ValidationError(
                    "frontend.type: none has no frontend process; frontend.placement.node: dedicated is invalid"
                )

    def _validate_services(self) -> None:
        """Whole-list checks for ``services:``: unique names, then each kind's recipe-level rules.

        Per-entry checks (empty command, moving-branch source rev, ...) live on
        ``ServiceConfig.__post_init__``; a kind's ``validate`` sees the full
        recipe (a ``mooncake-store`` needs ``engine.mooncake_kv_store``).
        """
        from srtctl.services.registry import get_service_kind

        seen: set[str] = set()
        for service in self.services:
            if service.name in seen:
                raise ValidationError(f"services[].name must be unique; duplicate: {service.name!r}")
            seen.add(service.name)
            get_service_kind(service.type).validate(service, self)

        # etcd and nats share the infra node, so they must agree on whether it is dedicated.
        if len({service.effective_placement == "dedicated" for service in self.infra_services}) > 1:
            raise ValidationError(
                "services etcd and nats must agree on placement.node: dedicated (they share the infra node)"
            )

        # A terminal service is the job's run: the job ends when it exits. It cannot share
        # that role with a benchmark step, and an external service never runs here.
        terminal = self.terminal_services
        if terminal and self.benchmark.type != "manual":
            names = ", ".join(svc.name for svc in terminal)
            raise ValidationError(
                f"services[{names}].terminal ends the job when the service exits, so the job cannot also run "
                f"benchmark.type: {self.benchmark.type}; drop the benchmark block (manual) or the terminal flag"
            )
        for svc in terminal:
            if svc.external:
                raise ValidationError(
                    f"services[{svc.name}].terminal needs a process to wait for; an external service launches nothing"
                )

    def _validate_benchmark_type(self) -> None:
        """Reject a benchmark.type that no runner is registered for.

        An unknown type (a typo like ``gsm8k-bench``, or a removed one) currently
        loads fine and only fails deep in the benchmark stage after a full
        allocation. Catch it at load time against the registry, plus the special
        ``manual`` type (no runner; the server just comes up ready). Import is
        lazy and guarded so a registry import hiccup never blocks a load.
        """
        btype = self.benchmark.type
        try:
            import srtctl.benchmarks  # noqa: F401 - importing the package registers every runner
            from srtctl.benchmarks.base import benchmark_config_fields, list_benchmarks

            allowed = set(list_benchmarks()) | {"manual"}
        except Exception:  # noqa: BLE001 - never block a config load on the registry import
            return
        if btype not in allowed:
            raise ValueError(f"Unknown benchmark.type {btype!r}. Available: {', '.join(sorted(allowed))}")

        # Per-type field split: a field set for a type whose runner never reads it
        # would be a silent no-op (isl on gsm8k, num_shots on sa-bench), so it is
        # rejected. `srtctl migrate` strips such fields from a pre-2.0 recipe.
        accepted = benchmark_config_fields(btype)
        stray = sorted(
            item.name
            for item in dataclasses.fields(BenchmarkConfig)
            if item.name not in accepted and getattr(self.benchmark, item.name) != _dataclass_default(item)
        )
        if stray:
            raise ValueError(
                f"benchmark.type {btype!r} does not use {', '.join(stray)}; fields it accepts: "
                f"{', '.join(sorted(accepted))}"
            )

    def _validate_host_setup(self) -> None:
        """Reject host_setup blocks that would fail or hang mid-job.

        These commands run before any worker starts, so a bad entry costs a full
        allocation to discover. Catch the cheap cases at load time (dry-run).
        """
        setup = self.host_setup
        for attr in ("commands", "teardown"):
            for i, command in enumerate(getattr(setup, attr)):
                if not command.strip():
                    raise ValidationError(f"host_setup.{attr}[{i}] is empty")
        if setup.timeout_seconds < 1:
            raise ValidationError(f"host_setup.timeout_seconds must be at least 1, got {setup.timeout_seconds}")
        if setup.teardown and not setup.commands:
            logger.warning(
                "host_setup.teardown is set without host_setup.commands; "
                "teardown will still run after the job, which is only what you want "
                "if something outside this recipe set the node state"
            )

    def _validate_vllm_discovery_connector(self) -> None:
        """A discovery connector (vLLM MoRI-IO) needs the router that runs its registration endpoint.

        Workers learn each other's transfer addresses from the vLLM Router's ZMQ
        discovery listener, which no other frontend runs. The Router's own rules
        (both roles on the connector, one router on the head node, a P/D
        topology) live in ``VLLMRouterFrontend.validate``.
        """
        if not isinstance(self.backend, VLLMBackend) or not self.backend.discovers_workers():
            return
        if self.frontend.type != "vllm-router":
            raise ValidationError(
                "a discovery connector (engine.connector: moriio) registers workers with the vLLM Router; "
                f"it requires frontend.type: vllm-router (got {self.frontend.type!r})"
            )

    def _validate_vllm_failover(self) -> None:
        """Rules for ``backend.failover`` (vLLM shadow engine recovery).

        The election and the shadow's parked state live in ``dynamo.vllm``
        (``--gms-shadow-mode``), so only the Dynamo frontend can drive it; a static
        router would also list the parked shadows as targets. Data-parallel
        layouts are refused because their per-rank processes would each need a
        GMS session and a lock of their own, which is not modeled.
        """
        failover = self.backend.failover
        if failover is None:
            return
        assert isinstance(self.backend, VLLMBackend)
        if self.frontend.type != "dynamo":
            raise ValidationError(
                f"engine.failover requires frontend.type: dynamo (shadow engines are elected by dynamo.vllm); "
                f"got {self.frontend.type!r}"
            )
        if self.dynamo.sidecar:
            raise ValidationError("engine.failover cannot be combined with dynamo.sidecar: true")
        dp_modes = self.backend.find_dp_modes()
        if dp_modes:
            names = ", ".join(mode for mode, _ in dp_modes)
            raise ValidationError(f"engine.failover does not support data-parallel-size (set on {names})")
        for role, spec in self.roles.items():
            for key, value in spec.args.items():
                if str(key).replace("_", "-") == "load-format" and str(value) != "gms":
                    raise ValidationError(
                        f"engine.failover loads weights through the GPU Memory Service; "
                        f"roles.{role}.args.load-format must be gms or unset, got {value!r}"
                    )
        if installs_dynamo(self):
            logger.warning(
                "engine.failover needs the gpu_memory_service package, which the ai-dynamo PyPI wheel does not "
                "include; the container must ship it (nvcr.io/nvidia/ai-dynamo/vllm-runtime does). "
                "Consider dynamo.install: false."
            )

    def _validate_dynamo_sidecar(self) -> None:
        """Validate native sidecar configuration before job submission."""
        if not self.dynamo.sidecar:
            return
        if self.frontend.type != "dynamo":
            raise ValidationError("dynamo.sidecar: true requires frontend.type: dynamo")
        if not isinstance(self.backend, SGLangBackend | VLLMBackend | TRTLLMBackend):
            raise ValidationError("dynamo.sidecar: true supports sglang, vllm, and trtllm backends only")
        if isinstance(self.backend, VLLMBackend) and self.backend.dp_launch_mode != "per_node":
            raise ValidationError("vLLM sidecar mode requires engine.dp_launch_mode: per_node; per_gpu is unsupported")

    def _warn_dp_launch_mode(self):
        """Warn when a vLLM DP recipe selects the deprecated per-GPU layout.

        Skipped for frontend.type: vllm, where the setting has no effect —
        `vllm serve` owns the local DP ranks, so the layout is one process per
        node whatever dp_launch_mode says.
        """
        if not isinstance(self.backend, VLLMBackend) or self.frontend.type == "vllm" or self.dynamo.sidecar:
            return
        if self.backend.dp_launch_mode != "per_gpu":
            return

        dp_modes = self.backend.find_dp_modes()
        if not dp_modes:
            return

        logger.warning(
            "vLLM DP mode(s) %s use deprecated dp_launch_mode=per_gpu; "
            "use backend.dp_launch_mode: per_node instead. per_gpu will be removed in a future release",
            ", ".join(mode_name for mode_name, _ in dp_modes),
        )

    def _validate_frontend_worker_selection(self):
        """Validate srtctl's inline Dynamo worker-selection shorthand."""
        if self.frontend.worker_selection is None:
            return
        if self.frontend.type != "dynamo":
            raise ValidationError("frontend.worker_selection is only supported with frontend.type: dynamo")

        args = self.frontend.args or {}
        env = self.frontend.env or {}
        if (
            "router-policy-config" in args
            or "DYN_ROUTER_POLICY_CONFIG" in env
            or "DYN_ROUTER_POLICY_CONFIG" in self.environment
        ):
            raise ValidationError(
                "frontend.worker_selection cannot be combined with frontend.args.router-policy-config "
                "or DYN_ROUTER_POLICY_CONFIG in frontend.env/environment"
            )

    def _validate_frontend(self) -> None:
        """``frontend.type`` must be registered, pair with the backend, and pass its own rules.

        The registry in ``srtctl.frontends`` is the only list of frontend types.
        Each implementation carries ``required_backend`` and ``validate``, so this
        schema does not know individual frontends. ``none`` is the services-only
        job and is covered by ``_validate_services_only``.
        """
        if self.frontend.type == "none":
            return
        from srtctl.frontends import get_frontend, list_frontend_types

        try:
            frontend = get_frontend(self.frontend.type)
        except ValueError:
            raise ValidationError(
                f"Unknown frontend.type {self.frontend.type!r}. Available: {', '.join(list_frontend_types())}"
            ) from None
        required = frontend.required_backend
        incompatible = [
            f"{role}={backend.type}" for role, backend in self.active_role_backends() if backend.type != required
        ]
        if required is not None and incompatible:
            raise ValidationError(
                f"frontend.type: {self.frontend.type} requires backend.type: {required}; got {', '.join(incompatible)}"
            )
        try:
            frontend.validate(self)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def _validate_het_jobs(self):
        """When ``resources.het_jobs`` is set to True, enforce supported shape.

        Validation runs only when the per-recipe override is explicitly True;
        a cluster-level default still in effect (recipe None) is permissive at
        load-time and resolved later by callers that pass the cluster default
        into ``het_components()``. This keeps a single recipe that disables het
        via ``het_jobs: false`` from tripping on a cluster default.
        """
        if self.resources.het_jobs is not True:
            return
        topology = self.topology
        if not topology.is_disaggregated:
            raise ValidationError(
                "het_jobs=true requires a disaggregated layout (declare roles.prefill and roles.decode)"
            )
        if (topology.prefill_nodes or 0) < 1 or (topology.decode_nodes or 0) < 1:
            raise ValidationError("het_jobs=true requires roles.prefill.nodes >= 1 and roles.decode.nodes >= 1")
        if self.backend_type != "sglang":
            raise ValidationError(
                f"het_jobs=true is only supported on the sglang backend; got backend.type={self.backend_type!r}"
            )
        if self.frontend.placement.dedicated or self.benchmark.placement.dedicated:
            raise ValidationError(
                "frontend.placement.node: dedicated / benchmark.placement.node: dedicated are not supported "
                "together with het_jobs=true (a dedicated frontend/client node is not carved out of a het allocation)"
            )

    def _validate_colocated_decode(self) -> None:
        """``roles.decode.nodes: colocate`` reserves no nodes of its own, so every decode worker
        has to fit on the GPUs the prefill workers leave free. Run the backend's real packer
        against a placeholder node list of the prefill nodes and turn its failure into a
        load-time error instead of a ``Not enough nodes`` crash inside the SLURM job.
        """
        topology = self.topology
        if not topology.colocated_decode or not topology.num_decode:
            return
        prefill_nodes = topology.prefill_nodes or 0
        if prefill_nodes < 1 or not topology.num_prefill:
            raise ValidationError(
                "roles.decode.nodes: colocate needs at least one prefill node and one prefill worker to share"
            )
        if self.total_nodes != topology.total_nodes:
            return  # the backend packs prefill and decode across extra nodes itself (vLLM)
        capacity = prefill_nodes * topology.gpus_per_node
        demand = topology.prefill_gpus + topology.decode_gpus
        layout = (
            f"{topology.num_prefill} prefill x {topology.gpus_per_prefill} GPU(s) + "
            f"{topology.num_decode} decode x {topology.gpus_per_decode} GPU(s) = {demand} GPU(s) on "
            f"{prefill_nodes} node(s) x {topology.gpus_per_node} GPU(s) = {capacity} GPU(s)"
        )
        if demand > capacity:
            raise ValidationError(f"colocated decode workers do not fit on the prefill nodes: {layout}")
        try:
            self.allocate_worker_endpoints([f"node{i}" for i in range(prefill_nodes)])
        except (ValueError, IndexError) as exc:
            # The packer raises ValueError when it runs out of nodes and IndexError when a
            # partial-node worker overflows the last node; both mean "does not fit".
            detail = str(exc) or "ran out of free GPUs on the prefill nodes"
            raise ValidationError(
                f"colocated decode workers cannot be packed onto the prefill nodes ({layout}): {detail}"
            ) from exc

    def _validate_mooncake_kv_store(self):
        """Catch the common misconfiguration: mooncake_kv_store set but the
        worker-side config doesn't actually wire mooncake into KV transfer.

        Without the per-mode flag (SGLang's ``disaggregation-transfer-backend:
        mooncake`` or vLLM's ``kv-transfer-config`` pointing at
        ``MooncakeConnector``), the master we launch is unused and workers fall
        back to the default transport — almost never what the user intends.
        """
        mooncake_cfg = self.backend.mooncake_kv_store
        if mooncake_cfg is None:
            return
        if isinstance(mooncake_cfg, VLLMMooncakeKVStoreConfig):
            try:
                mooncake_cfg.validate_device_mapping(self.resources.gpus_per_node)
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
        if not self.topology.is_disaggregated:
            return

        if isinstance(self.backend, SGLangBackend):

            def _sglang_has_mooncake(mode_cfg: dict | None) -> bool:
                if not mode_cfg:
                    return False
                # SGLang accepts both "disaggregation-transfer-backend" and the
                # underscore form; _config_to_cli_args normalizes them.
                for key in ("disaggregation-transfer-backend", "disaggregation_transfer_backend"):
                    if mode_cfg.get(key) == "mooncake":
                        return True
                return False

            prefill_ok = _sglang_has_mooncake(self.backend.get_config_for_mode("prefill"))
            decode_ok = _sglang_has_mooncake(self.backend.get_config_for_mode("decode"))

            if not (prefill_ok or decode_ok):
                raise ValidationError(
                    "a mooncake-master service is configured but neither roles.prefill.args nor "
                    "roles.decode.args has 'disaggregation-transfer-backend: mooncake'. "
                    "Add it to both roles (and 'disaggregation-ib-device') so workers "
                    "actually use the mooncake master srtslurm launches for you."
                )
        elif isinstance(self.backend, VLLMBackend):

            def _vllm_has_mooncake(mode_cfg: dict | None) -> bool:
                if not mode_cfg:
                    return False
                # vLLM uses --kv-transfer-config (raw JSON). Mooncake shows up
                # as either ``MooncakeStoreConnector`` / ``MooncakeConnector``
                # at the top level, or nested inside a ``MultiConnector``'s
                # ``kv_connector_extra_config.connectors`` list. A case-insensitive
                # substring match covers all three forms without re-parsing JSON.
                for key in ("kv-transfer-config", "kv_transfer_config"):
                    val = mode_cfg.get(key)
                    if isinstance(val, str) and "mooncake" in val.lower():
                        return True
                return False

            prefill_ok = _vllm_has_mooncake(self.backend.get_config_for_mode("prefill"))
            decode_ok = _vllm_has_mooncake(self.backend.get_config_for_mode("decode"))

            if not (prefill_ok or decode_ok):
                raise ValidationError(
                    "a mooncake-master service is configured but neither roles.prefill.args nor "
                    "roles.decode.args has a kv-transfer-config that references a "
                    "Mooncake connector. Set kv-transfer-config to a JSON value whose "
                    "kv_connector is MooncakeStoreConnector (or MultiConnector wrapping "
                    "one) so workers actually use the mooncake master srtslurm launches "
                    "for you."
                )

    def _profiling_worker_ranks(self, mode: WorkerMode) -> set[int]:
        """Derive selectable physical ranks from the configured worker layout."""
        from srtctl.core.topology import Endpoint

        resources = self.topology
        gpus_per_worker = {
            "prefill": resources.gpus_per_prefill,
            "decode": resources.gpus_per_decode,
            "agg": resources.gpus_per_agg,
        }[mode]
        nodes_per_worker = math.ceil(gpus_per_worker / resources.gpus_per_node)
        # Match allocate_endpoints: multi-node workers use the same full GPU
        # index set on every node (whole-node allocation); partial-node workers
        # use a contiguous subset. Actual placement/ports are resolved later,
        # and _profiling_worker_endpoints checks the selector against that topology.
        local_gpus = resources.gpus_per_node if nodes_per_worker > 1 else gpus_per_worker
        endpoint = Endpoint(
            mode=mode,
            index=0,
            nodes=tuple(f"profiling-node-{rank}" for rank in range(nodes_per_worker)),
            gpu_indices=frozenset(range(local_gpus)),
            gpus_per_node=resources.gpus_per_node,
        )
        # Validation-only expansion uses a fresh port allocator; it neither
        # reserves live ports nor consumes the runtime topology's allocations.
        processes = self.backend.endpoints_to_processes(
            [endpoint],
            frontend_type=self.frontend.type,
            dynamo_sidecar=self.dynamo.sidecar,
        )
        return {process.node_rank for process in processes}

    def _validate_profiling(self):
        """Validate profiling configuration matches serving mode."""
        prof = self.profiling
        if not prof.enabled:
            return

        backend_type = self.backend.type

        # torch profiling is SGLang-only (uses SGLANG_TORCH_PROFILER_DIR)
        if prof.is_torch and backend_type == "trtllm":
            raise ValidationError("torch profiling is not supported for the trtllm backend; use nsys instead")

        # nsys-time (time-based capture via nsys --delay/--duration) is supported
        # for all backends. get_nsys_prefix() emits a time-based command for the
        # non-TRTLLM (vllm/sglang) path too.

        if prof.is_nsys:
            if not prof.nsys_trace.strip():
                raise ValidationError("profiling.nsys_trace must not be empty")
            if not prof.capture_range_end.strip():
                raise ValidationError("profiling.capture_range_end must not be empty")
            if prof.nsys_library_paths is not None and any(not path for path in prof.nsys_library_paths):
                raise ValidationError("profiling.nsys_library_paths must not contain empty paths")

        # nsys-time uses top-level delay/duration — no per-phase step configs needed
        if prof.is_nsys_time:
            if prof.delay_secs is None or prof.duration_secs is None:
                raise ValidationError(
                    "profiling.delay_secs and profiling.duration_secs are required for nsys-time mode"
                )
            return

        r = self.topology
        is_disaggregated = r.is_disaggregated
        has_prefill_prof = prof.prefill is not None
        has_decode_prof = prof.decode is not None
        has_agg_prof = prof.aggregated is not None

        # Validate phase configs match serving mode
        if is_disaggregated:
            if has_agg_prof:
                raise ValidationError(
                    "Disaggregated mode only supports profiling.prefill/decode; profiling.aggregated is not allowed."
                )
            if not has_prefill_prof or not has_decode_prof:
                raise ValidationError(
                    "Disaggregated mode requires both profiling.prefill and profiling.decode "
                    "to be set when profiling is enabled."
                )
            if (r.prefill_workers or 0) <= 0 or (r.decode_workers or 0) <= 0:
                raise ValidationError("Disaggregated mode requires prefill_workers and decode_workers to be > 0.")
        else:
            if has_prefill_prof or has_decode_prof:
                raise ValidationError(
                    "Aggregated mode only supports profiling.aggregated; profiling.prefill/decode are not allowed."
                )
            if not has_agg_prof:
                raise ValidationError(
                    "Aggregated mode requires profiling.aggregated to be set when profiling is enabled."
                )
            if (r.agg_workers or 0) <= 0:
                raise ValidationError("Aggregated mode requires agg_workers to be > 0.")

        if prof.is_nsys:
            phase_workers = (
                (("prefill", prof.prefill, r.prefill_workers), ("decode", prof.decode, r.decode_workers))
                if is_disaggregated
                else (("aggregated", prof.aggregated, r.agg_workers),)
            )
            for phase_name, phase_config, worker_count in phase_workers:
                assert phase_config is not None
                if phase_config.capture_scope not in ("selected", "all"):
                    raise ValidationError(f"profiling.{phase_name}.capture_scope must be 'selected' or 'all'")
                if backend_type == "trtllm":
                    continue
                if phase_config.capture_scope == "all":
                    if phase_config.worker_index != 0 or phase_config.worker_rank != 0:
                        logger.warning(
                            "profiling.%s.capture_scope='all' ignores worker_index=%s and worker_rank=%s; "
                            "all workers remain selected. Set capture_scope='selected' to use these selectors.",
                            phase_name,
                            phase_config.worker_index,
                            phase_config.worker_rank,
                        )
                    continue
                if phase_config.worker_index < 0:
                    raise ValidationError(f"profiling.{phase_name}.worker_index must be non-negative")
                if phase_config.worker_index >= (worker_count or 0):
                    raise ValidationError(
                        f"profiling.{phase_name}.worker_index={phase_config.worker_index} is out of range "
                        f"for {worker_count or 0} configured workers"
                    )
                if phase_config.worker_rank < 0:
                    raise ValidationError(f"profiling.{phase_name}.worker_rank must be non-negative")
                mode = "agg" if phase_name == "aggregated" else phase_name
                try:
                    valid_ranks = self._profiling_worker_ranks(mode)
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                if phase_config.worker_rank not in valid_ranks:
                    ranks = ", ".join(str(rank) for rank in sorted(valid_ranks))
                    raise ValidationError(
                        f"profiling.{phase_name}.worker_rank={phase_config.worker_rank} is not a physical "
                        f"process rank for this worker layout; valid ranks: {ranks}"
                    )
                if phase_config.worker_rank != 0 and self._frontend_profiling_control_is_leader_only():
                    raise ValidationError(
                        f"profiling.{phase_name}.worker_rank={phase_config.worker_rank} has no independent "
                        "control endpoint; direct vLLM and Dynamo sidecar profiling must select rank 0"
                    )

        # Iteration-based nsys (type: nsys) drives the vLLM engine profiler via
        # --profiler-config, derived from the profiling: block. Forbid duplicating
        # it in roles.<role>.args so the two can't diverge silently.
        if prof.type == "nsys" and backend_type == "vllm":
            self._validate_vllm_nsys_profiler_config_not_set()

    def _validate_vllm_nsys_profiler_config_not_set(self):
        """Reject profiler-config.* in a role's args when nsys profiling is enabled.

        srtctl injects --profiler-config from the profiling: block (single source
        of truth), so a user-supplied profiler-config in roles.<role>.args would
        either be overwritten or conflict with a different step window. Fail fast
        at recipe-read time instead.
        """
        if not isinstance(self.backend, VLLMBackend):
            return
        for role, spec in self.roles.items():
            bad = [k for k in spec.args if str(k).replace("_", "-").startswith("profiler-config")]
            if bad:
                raise ValidationError(
                    f"roles.{role}.args sets {bad}, but profiler-config.* is derived automatically "
                    f"from the profiling: block when nsys profiling is enabled. Remove these keys."
                )

    def _validate_dcgm_power(self):
        """Validate DCGM power telemetry.

        It runs its collector in the orchestrator process, so it needs neither
        the scraper image nor node_exporter. Sample timestamps come from the
        orchestrator host and window boundaries from the benchmark client host;
        both are ``time.time()`` and are assumed NTP-synchronised within the
        allocation, so the client may run on any node (``benchmark.placement.node``,
        including ``dedicated``).
        """
        telemetry = self.telemetry
        exporter = telemetry.dcgm_exporter
        if exporter is None:
            raise ValidationError("telemetry.dcgm_exporter is required when telemetry is enabled")
        if not exporter.container_image:
            raise ValidationError("telemetry.dcgm_exporter.container_image must be non-empty")
        if not 1 <= exporter.port <= 65535:
            raise ValidationError("telemetry.dcgm_exporter.port must be in 1..65535")

        for name in ("startup_timeout_seconds",):
            if not _is_finite_positive(getattr(telemetry, name)):
                raise ValidationError(f"telemetry.{name} must be finite and positive")
        if telemetry.collect_interval_ms <= 0:
            raise ValidationError("telemetry.collect_interval_ms must be positive")
        if telemetry.collect_interval_ms > _DCGM_POWER_MAX_SAMPLE_GAP_SECONDS * 1000:
            raise ValidationError(
                f"telemetry.collect_interval_ms={telemetry.collect_interval_ms} exceeds the "
                f"{_DCGM_POWER_MAX_SAMPLE_GAP_SECONDS}s max sample gap the power validator accepts; "
                "every window would fail sample_gap_exceeded. Set it to the intended collector "
                "period (e.g. 1000)."
            )

        if not _is_safe_relative_subpath(telemetry.storage_subdir):
            raise ValidationError("telemetry.storage_subdir must be a safe relative path below the run log directory")

        # `manual` holds the deployment for an external load generator; like serve-only it
        # has no load window, so telemetry captures the whole serve session, best-effort.
        supported_benchmarks = {_BENCHMARK_TYPE_SA_BENCH, "agentic", "agentx", "custom", "manual"}
        if self.benchmark.type not in supported_benchmarks:
            supported = ", ".join(sorted(supported_benchmarks))
            raise ValidationError(f"telemetry requires benchmark.type to be one of: {supported}")

        # NOTE: a dedicated infra node moves nodes.head off the batch host the collector runs on.
        if self.infra_dedicated_node:
            raise ValidationError(
                "telemetry requires the discovery plane on the infra node (no etcd or nats service with "
                "placement.node: dedicated), because a dedicated infra node moves nodes.head off the batch host "
                "and power samples would no longer share the benchmark's clock"
            )

        concurrencies = self.benchmark.get_concurrency_list()
        if not concurrencies or len(set(concurrencies)) != len(concurrencies) or any(c <= 0 for c in concurrencies):
            raise ValidationError("telemetry requires a non-empty list of unique positive benchmark.concurrencies")

    def _frontend_profiling_control_is_leader_only(self) -> bool:
        """Whether the frontend's workers expose one profiler control server per logical endpoint."""
        if self.frontend.type == "none":
            return False
        from srtctl.frontends import get_frontend

        return get_frontend(self.frontend.type).profiling_control_is_leader_only(self)

    def _dynamo_system_ports(self) -> set[int]:
        """System-status ports that backend launches actually bind on worker nodes."""
        from srtctl.frontends import get_frontend

        if self.frontend.type == "none" or get_frontend(self.frontend.type).worker_launch != "dynamo":
            return set()

        resources = self.topology
        nodes = [f"validation-worker-{index}" for index in range(self.total_nodes)]
        endpoints = self.backend.allocate_endpoints(
            num_prefill=resources.num_prefill,
            num_decode=resources.num_decode,
            num_agg=resources.num_agg,
            gpus_per_prefill=resources.gpus_per_prefill,
            gpus_per_decode=resources.gpus_per_decode,
            gpus_per_agg=resources.gpus_per_agg,
            gpus_per_node=resources.gpus_per_node,
            available_nodes=nodes,
            spread_workers=self.resources.spread_workers,
        )
        processes = self.backend.endpoints_to_processes(
            endpoints,
            frontend_type=self.frontend.type,
            dynamo_sidecar=self.dynamo.sidecar,
        )
        if self.backend.get_srun_config().launch_per_endpoint:
            processes = [process for process in processes if process.node_rank == 0]
        return {process.sys_port for process in processes}

    def _validate_collector_budget(self) -> None:
        """Validate the scrape and join budget every enabled leg's collector uses.

        Both the DCGM and CPU legs poll with ``request_timeout_seconds`` and are
        joined with ``collector_join_timeout_seconds``, so a leg running alone
        needs these checked just as much as the pair does.
        """
        telemetry = self.telemetry
        if not _is_finite_positive(telemetry.request_timeout_seconds):
            raise ValidationError("telemetry.request_timeout_seconds must be finite and positive")
        worst_case_join_seconds = 2 * (
            2 * telemetry.request_timeout_seconds + _DCGM_POWER_COLLECT_CYCLE_TIMEOUT_GRACE_SECONDS
        )
        collector_join_timeout_seconds = telemetry.resolved_collector_join_timeout_seconds
        if (
            not _is_finite_positive(collector_join_timeout_seconds)
            or collector_join_timeout_seconds <= worst_case_join_seconds
        ):
            raise ValidationError(
                "telemetry.collector_join_timeout_seconds must be finite, positive, "
                "and greater than two full collector cycles "
                "(2 * (2 * telemetry.request_timeout_seconds + 1 second))"
            )

    def _validate_cpu_power_exporter(self) -> None:
        """Validate the independent, best-effort CPU power exporter, if configured."""
        exporter = self.telemetry.cpu_power_exporter
        if exporter is None:
            return
        if not 1 <= exporter.port <= 65535:
            raise ValidationError("telemetry.cpu_power_exporter.port must be in 1..65535")
        if exporter.source not in ("auto", "acpi", "dcgm"):
            raise ValidationError('telemetry.cpu_power_exporter.source must be one of: "auto", "acpi", "dcgm"')

        neighbours = [("telemetry.dcgm_exporter", self.telemetry.dcgm_exporter)]
        if self.observability.tachometer_enabled:
            tachometer = self.observability.tachometer
            # Compare against the *resolved* exporters: with no explicit block the
            # tachometer still launches its built-in DCGM/node exporters (#358).
            neighbours += [
                ("observability.tachometer.dcgm_exporter", tachometer.resolved_dcgm_exporter),
                ("observability.tachometer.node_exporter", tachometer.resolved_node_exporter),
            ]
        for name, neighbour_exporter in neighbours:
            if neighbour_exporter is not None and neighbour_exporter.port == exporter.port:
                raise ValidationError(
                    f"telemetry.cpu_power_exporter.port={exporter.port} collides with "
                    f"{name}.port; both run on every worker node"
                )

        if exporter.port in self._dynamo_system_ports():
            raise ValidationError(
                f"telemetry.cpu_power_exporter.port={exporter.port} collides with a Dynamo system port "
                "assigned to a backend process on a worker node"
            )

    def _validate_cpu_power(self) -> None:
        """Validate the host-side CPU power collector leg (``telemetry.cpu_power``)."""
        cpu_power = self.telemetry.cpu_power
        for name in ("sample_interval_seconds", "startup_timeout_seconds"):
            if not _is_finite_positive(getattr(cpu_power, name)):
                raise ValidationError(f"telemetry.cpu_power.{name} must be finite and positive")
        if cpu_power.sample_interval_seconds > _CPU_POWER_MAX_SAMPLE_GAP_SECONDS:
            raise ValidationError(
                f"telemetry.cpu_power.sample_interval_seconds={cpu_power.sample_interval_seconds} exceeds the "
                f"{_CPU_POWER_MAX_SAMPLE_GAP_SECONDS}s max sample gap; every window would fail sample_gap_exceeded"
            )

        if not _is_safe_relative_subpath(cpu_power.storage_subdir):
            raise ValidationError(
                "telemetry.cpu_power.storage_subdir must be a safe relative path below the run log directory"
            )
        if cpu_power.storage_subdir == self.telemetry.storage_subdir:
            raise ValidationError(
                "telemetry.cpu_power.storage_subdir must differ from telemetry.storage_subdir; "
                "the two legs write their own samples and manifest"
            )

    def _reject_inert_cpu_power_demand(self) -> None:
        """Reject mandatory CPU power semantics that nothing will act on."""
        cpu_power = self.telemetry.cpu_power
        if cpu_power.required:
            raise ValidationError("telemetry.cpu_power.required has no effect unless telemetry.cpu_power.enabled")
        if cpu_power.source != "auto":
            raise ValidationError(
                f'telemetry.cpu_power.source: "{cpu_power.source}" has no effect unless telemetry.cpu_power.enabled; '
                "it names a mandatory provider for a leg that will not run"
            )

    @property
    def observability_nsys_enabled(self) -> bool:
        """Use the automatic preset only when no explicit profiler owns the run."""
        return self.observability.enabled and self.observability.nsys.enabled and not self.profiling.enabled

    def _validate_observability(self):
        """Validate automatic profiling and Tachometer collection."""
        observability = self.observability
        if self.observability_nsys_enabled and observability.nsys.capture_window == "measured_workload":
            # These scripts own warmup and invoke the acknowledged boundary API.
            # Custom/manual clients receive that API but must invoke it themselves.
            supported = {"sa-bench", "sglang-bench", "trace-replay", "mooncake-router", "custom", "manual"}
            if self.benchmark.type not in supported:
                raise ValidationError(
                    f"observability.nsys.capture_window: measured_workload has no warmup hooks for "
                    f"benchmark.type: {self.benchmark.type}; use capture_window: including_startup, "
                    "disable observability.nsys, or use a custom client with start/stop hooks"
                )
            if self.benchmark.type in {"trace-replay", "mooncake-router"} and any(
                key.replace("_", "-").startswith(("warmup-", "num-warmup-")) and value not in (0, "0", False, None)
                for key, value in self.benchmark.aiperf_args.items()
            ):
                raise ValidationError(
                    "measured_workload nsys capture uses the bundled script's separate warmup; "
                    "additional aiperf_args warmup would occur inside capture. Remove those "
                    "flags or use a custom client with hooks at its actual warmup boundary"
                )
        tachometer = observability.tachometer
        if not observability.tachometer_enabled:
            return
        if self.telemetry.enabled and self.telemetry.dcgm_exporter is not None and tachometer.dcgm_exporter is not None:
            raise ValidationError(
                "configure the shared DCGM exporter under telemetry, not observability.tachometer, "
                "when DCGM power telemetry is enabled"
            )
        if self.telemetry.enabled and tachometer.storage_subdir == self.telemetry.storage_subdir:
            raise ValidationError(
                "observability.tachometer.storage_subdir and telemetry.storage_subdir must be different"
            )

        for name in ("dcgm_exporter", "node_exporter", "process_exporter"):
            exporter = getattr(tachometer, name)
            if exporter is None:
                continue
            if not exporter.container_image and not exporter.binary:
                raise ValidationError(
                    f"observability.tachometer.{name}: set container_image (container launch) or binary (host-native)"
                )
            if not 1 <= exporter.port <= 65535:
                raise ValidationError(f"observability.tachometer.{name}.port must be in 1..65535")
        if not tachometer.binary_path:
            raise ValidationError("observability.tachometer.binary_path must be non-empty")
        if tachometer.collect_interval_ms <= 0:
            raise ValidationError("observability.tachometer.collect_interval_ms must be positive")
        if tachometer.sync_interval_secs < 0:
            raise ValidationError("observability.tachometer.sync_interval_secs must be >= 0")
        if tachometer.compaction_threads < 0:
            raise ValidationError("observability.tachometer.compaction_threads must be >= 0")
        if not _is_safe_relative_subpath(tachometer.storage_subdir):
            raise ValidationError(
                "observability.tachometer.storage_subdir must be a safe relative path below the run log directory"
            )

    def _validate_telemetry(self):
        """Validate telemetry config.

        ``cpu_power_exporter`` (head-node scraper) and ``cpu_power`` (host-side
        collector) are independent legs: each is validated whenever telemetry
        is enabled, regardless of which provider (dcgm-power today) is
        configured. Either is also sufficient on its own -- a recipe may
        enable telemetry for CPU power alone, with no ``dcgm_exporter`` at all.
        """
        telemetry = self.telemetry
        if telemetry is None:
            return
        if not telemetry.enabled:
            if telemetry.cpu_power.enabled:
                raise ValidationError("telemetry.cpu_power.enabled requires telemetry.enabled")
            self._reject_inert_cpu_power_demand()
            return
        if telemetry.cpu_power.enabled:
            if not _is_safe_relative_subpath(telemetry.storage_subdir):
                raise ValidationError(
                    "telemetry.storage_subdir must be a safe relative path below the run log directory"
                )
            self._validate_cpu_power()
        else:
            self._reject_inert_cpu_power_demand()
        if telemetry.dcgm_exporter is not None:
            self._validate_dcgm_power()
        elif telemetry.cpu_power_exporter is None and not telemetry.cpu_power.enabled:
            raise ValidationError(
                "telemetry.enabled requires telemetry.dcgm_exporter, telemetry.cpu_power_exporter, "
                "or telemetry.cpu_power.enabled; otherwise there is nothing to collect"
            )
        self._validate_collector_budget()
        self._validate_cpu_power_exporter()

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "SrtConfig":
        """Load a recipe file without cluster defaults (``load_config`` applies them).

        Runs the same gate and engine-default expansions as ``load_config``: a
        pre-2.0 recipe is rejected before the schema loads the document.
        """
        from srtctl.core.config import expand_engine_config_defaults, resolve_config_with_defaults

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        resolved = resolve_config_with_defaults(data, None)
        expand_engine_config_defaults(resolved)
        schema = cls.Schema()
        return schema.load(resolved)

    @property
    def served_model_name(self) -> str:
        """Get the served model name from backend config or model path."""
        default = Path(self.model.path).name
        role = "decode" if self.topology.num_decode else "agg" if self.topology.num_agg else "prefill"
        if self.frontend.type != "none":
            from srtctl.frontends import get_frontend

            role = get_frontend(self.frontend.type).model_name_role or role
        backend = self.backend_for_role(role)
        if isinstance(backend, AtomBackend):
            # Without served-model-name, ATOM advertises the literal --model
            # argument: the worker's HF ID or container-visible path, including
            # node-local staging.
            model_path = os.path.expandvars(self.model.path)
            if model_path.startswith("hf:"):
                default = model_path[3:]
            elif self.model.stage_dir:
                default = str(Path(os.path.expandvars(self.model.stage_dir)) / Path(model_path).resolve().name)
            else:
                default = "/model"
        return backend.get_served_model_name(default)

    @property
    def pool_services(self) -> list[ServiceConfig]:
        """Services that own nodes (``services[].nodes``), in declaration order: the job's pools."""
        return [svc for svc in self.services if svc.nodes is not None and svc.enabled]

    @property
    def terminal_services(self) -> list[ServiceConfig]:
        """Services marked ``terminal``: the job ends when every one of them has exited."""
        return [svc for svc in self.services if svc.terminal and svc.enabled]

    @property
    def services_node_count(self) -> int | None:
        """Nodes owned by services through ``services[].nodes``, summed; None when no service owns any."""
        counts = [svc.nodes for svc in self.pool_services if svc.nodes is not None]
        return sum(counts) if counts else None

    @property
    def engine_node_count(self) -> int:
        """Nodes the engine roles own; zero when the recipe has no engine workers."""
        if not self.topology.has_engine_workers:
            return 0
        return self._engine_total_nodes()

    @property
    def total_nodes(self) -> int:
        """Node count of the job: the engine roles' nodes plus every service pool, at least one."""
        return (self.engine_node_count + (self.services_node_count or 0)) or 1

    def _engine_total_nodes(self) -> int:
        """Worker node count of the engine roles, adjusted for backend-specific packing."""
        topology = self.topology
        if self.has_role_backends:
            return topology.total_nodes
        if isinstance(self.backend, VLLMBackend) and self.backend.should_colocate_prefill_decode(
            num_prefill=topology.num_prefill,
            num_decode=topology.num_decode,
            num_agg=topology.num_agg,
            gpus_per_prefill=topology.gpus_per_prefill,
            gpus_per_decode=topology.gpus_per_decode,
            gpus_per_agg=topology.gpus_per_agg,
            gpus_per_node=topology.gpus_per_node,
        ):
            total_worker_gpus = topology.prefill_gpus + topology.decode_gpus + topology.num_agg * topology.gpus_per_agg
            return (total_worker_gpus + topology.gpus_per_node - 1) // topology.gpus_per_node
        return topology.total_nodes

    @property
    def backend_type(self) -> str:
        """Get the backend type string."""
        return self.backend.type

    @property
    def infra_services(self) -> list[ServiceConfig]:
        """The declared, enabled discovery-plane entries (``etcd`` / ``nats``)."""
        return [service for service in self.services if service.type in INFRA_SERVICE_TYPES and service.enabled]

    @property
    def infra_dedicated_node(self) -> bool:
        """Whether the discovery plane gets a node of its own: a declared ``etcd`` or ``nats`` placed ``dedicated``."""
        return any(service.effective_placement == "dedicated" for service in self.infra_services)

    @property
    def nats_max_payload_mb(self) -> int | None:
        """The NATS payload limit from a declared ``nats`` entry's ``options.max_payload_mb``; None for the default."""
        for service in self.infra_services:
            if service.type == "nats" and service.options.get("max_payload_mb") is not None:
                return int(service.options["max_payload_mb"])
        return None


def installs_dynamo(config: SrtConfig) -> bool:
    """Whether this config installs dynamo into its containers (needs root inside).

    Single source of truth for the ENROOT_REMAP_ROOT srun injection (workers +
    dynamo frontend) and its dry-run display: dynamo is only installed when the
    dynamo frontend is selected and install isn't disabled.
    """
    return config.frontend.type == "dynamo" and config.dynamo.install
