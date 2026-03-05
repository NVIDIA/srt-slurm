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
import itertools
import logging
from collections.abc import Iterator, Mapping
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
)

import yaml
from marshmallow import Schema, ValidationError, fields
from marshmallow_dataclass import dataclass

from srtctl.backends import (
    BackendConfig,
    SGLangProtocol,
    TRTLLMProtocol,
    VLLMProtocol,
)
from srtctl.core.formatting import (
    FormattablePath,
    FormattablePathField,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Reporting Configuration
# ============================================================================


@dataclass(frozen=True)
class ReportingStatusConfig:
    """Status reporting configuration."""

    endpoint: str | None = None
    endpoints: list[str] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class ReportingConfig:
    """Reporting configuration for status updates, AI analysis, and log exports."""

    status: ReportingStatusConfig | None = None
    ai_analysis: "AIAnalysisConfig | None" = None
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

    Schema: ClassVar[type[Schema]] = Schema


@dataclass
class ClusterConfig:
    """Cluster configuration from srtslurm.yaml."""

    cluster: str | None = None  # Cluster name for status reporting
    default_account: str | None = None
    default_partition: str | None = None
    default_time_limit: str | None = None
    gpus_per_node: int | None = None
    network_interface: str | None = None
    use_gpus_per_node_directive: bool = True
    use_segment_sbatch_directive: bool = True
    use_exclusive_sbatch_directive: bool = False
    srtctl_root: str | None = None
    output_dir: str | None = None  # Custom output directory for job logs
    model_paths: dict[str, str] | None = None
    containers: dict[str, str] | None = None
    cloud: dict[str, str] | None = None
    # Cluster-level container mounts (host_path -> container_path)
    # Applied to all jobs on this cluster, useful for cluster-specific paths
    default_mounts: dict[str, str] | None = None
    reporting: ReportingConfig | None = None

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


class BenchmarkType(str, Enum):
    MANUAL = "manual"
    SA_BENCH = "sa-bench"
    ROUTER = "router"
    MOONCAKE_ROUTER = "mooncake-router"
    MMLU = "mmlu"
    GPQA = "gpqa"
    LONGBENCHV2 = "longbenchv2"


class ProfilingType(str, Enum):
    NSYS = "nsys"
    TORCH = "torch"
    NONE = "none"


# ============================================================================
# Marshmallow Custom Fields
# ============================================================================


class BackendConfigField(fields.Field):
    """Marshmallow field for polymorphic backend deserialization based on type."""

    def _deserialize(
        self,
        value: Any,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> BackendConfig:
        """Deserialize backend config based on 'type' field."""
        if value is None:
            # Default to SGLang
            return SGLangProtocol()

        if isinstance(value, SGLangProtocol | TRTLLMProtocol | VLLMProtocol):
            return value

        if not isinstance(value, dict):
            raise ValidationError(f"Expected dict for backend config, got {type(value).__name__}")

        # Get backend type from the value dict
        backend_type = value.get("type", "sglang")

        if backend_type == "sglang":
            schema = SGLangProtocol.Schema()
            return schema.load(value)
        elif backend_type == "trtllm":
            schema = TRTLLMProtocol.Schema()
            return schema.load(value)
        elif backend_type == "vllm":
            schema = VLLMProtocol.Schema()
            return schema.load(value)
        else:
            raise ValidationError(f"Unknown backend type: {backend_type!r}. Supported types: sglang, trtllm, vllm")

    def _serialize(self, value: Any | None, attr: str | None, obj: Any, **kwargs) -> Any:
        """Serialize backend config to dict."""
        if value is None:
            return None
        if isinstance(value, SGLangProtocol):
            return SGLangProtocol.Schema().dump(value)
        if isinstance(value, TRTLLMProtocol):
            return TRTLLMProtocol.Schema().dump(value)
        if isinstance(value, VLLMProtocol):
            return VLLMProtocol.Schema().dump(value)
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

    mode: Literal["zip", "grid"] = "zip"
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

    path: str
    container: str
    precision: str

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class ResourceConfig:
    """Resource allocation configuration."""

    gpu_type: str
    gpus_per_node: int = 4

    # Disaggregated mode
    prefill_nodes: int | None = None
    decode_nodes: int | None = None
    prefill_workers: int | None = None
    decode_workers: int | None = None

    # Aggregated mode
    agg_nodes: int | None = None
    agg_workers: int | None = None

    # Explicit GPUs per worker (override computed values)
    # Use data_key to map from YAML field names to internal attribute names
    _explicit_gpus_per_prefill: int | None = field(
        default=None,
        metadata={
            "marshmallow_field": fields.Integer(
                data_key="gpus_per_prefill",
                load_default=None,
                allow_none=True,
            )
        },
    )
    _explicit_gpus_per_decode: int | None = field(
        default=None,
        metadata={
            "marshmallow_field": fields.Integer(
                data_key="gpus_per_decode",
                load_default=None,
                allow_none=True,
            )
        },
    )
    _explicit_gpus_per_agg: int | None = field(
        default=None,
        metadata={
            "marshmallow_field": fields.Integer(
                data_key="gpus_per_agg",
                load_default=None,
                allow_none=True,
            )
        },
    )

    @property
    def is_disaggregated(self) -> bool:
        return self.prefill_nodes is not None or self.decode_nodes is not None

    @property
    def total_nodes(self) -> int:
        if self.is_disaggregated:
            return (self.prefill_nodes or 0) + (self.decode_nodes or 0)
        return self.agg_nodes or 1

    @property
    def num_prefill(self) -> int:
        return self.prefill_workers or 0

    @property
    def num_decode(self) -> int:
        return self.decode_workers or 0

    @property
    def num_agg(self) -> int:
        return self.agg_workers or 0

    @property
    def gpus_per_prefill(self) -> int:
        # Use explicit value if set
        if self._explicit_gpus_per_prefill is not None:
            return self._explicit_gpus_per_prefill
        # Fall back to computed value
        if self.prefill_nodes and self.prefill_workers:
            return (self.prefill_nodes * self.gpus_per_node) // self.prefill_workers
        return self.gpus_per_node

    @property
    def gpus_per_decode(self) -> int:
        # Use explicit value if set
        if self._explicit_gpus_per_decode is not None:
            return self._explicit_gpus_per_decode
        # Fall back to computed value
        if self.decode_nodes and self.decode_workers:
            return (self.decode_nodes * self.gpus_per_node) // self.decode_workers
        # decode_nodes=0 with decode_workers means "share nodes with prefill"
        # Inherit TP from prefill in this case
        if self.decode_nodes == 0 and self.decode_workers:
            return self.gpus_per_prefill
        return self.gpus_per_node

    @property
    def gpus_per_agg(self) -> int:
        # Use explicit value if set
        if self._explicit_gpus_per_agg is not None:
            return self._explicit_gpus_per_agg
        # Fall back to computed value
        if self.agg_nodes and self.agg_workers:
            return (self.agg_nodes * self.gpus_per_node) // self.agg_workers
        return self.gpus_per_node

    @property
    def prefill_gpus(self) -> int:
        """Total GPUs used by all prefill workers."""
        return self.num_prefill * self.gpus_per_prefill

    @property
    def decode_gpus(self) -> int:
        """Total GPUs used by all decode workers."""
        return self.num_decode * self.gpus_per_decode

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class SlurmConfig:
    """SLURM job settings."""

    account: str | None = None
    partition: str | None = None
    time_limit: str | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark configuration."""

    type: str = "manual"
    isl: int | None = None
    osl: int | None = None
    concurrencies: list[int] | str | None = None
    req_rate: str | int | None = "inf"
    sweep: Annotated[SweepConfig, SweepConfigField()] | None = None
    # Accuracy benchmark fields
    num_examples: int | None = None
    max_tokens: int | None = None
    repeat: int | None = None
    num_threads: int | None = None
    max_context_length: int | None = None
    categories: list[str] | None = None
    num_shots: int | None = None  # GSM8K few-shot examples
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    # Router benchmark fields
    num_requests: int | None = None
    concurrency: int | None = None
    prefix_ratios: list[float] | str | None = None
    # Mooncake router benchmark fields (uses aiperf with mooncake_trace)
    mooncake_workload: str | None = None  # "mooncake", "conversation", "synthetic", "toolagent"
    ttft_threshold_ms: int | None = None  # Goodput TTFT threshold in ms (default: 2000)
    itl_threshold_ms: int | None = None  # Goodput ITL threshold in ms (default: 25)

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

    Schema: ClassVar[builtins.type[Schema]] = Schema


@dataclass(frozen=True)
class ProfilingConfig:
    """Profiling configuration.

    Supports two profiling modes:
    - nsys: NVIDIA Nsight Systems profiling (wraps command with nsys profile)
    - torch: PyTorch profiler (uses SGLANG_TORCH_PROFILER_DIR)

    Per-phase start_step/stop_step are specified in the prefill/decode/aggregated sections.
    """

    type: str = "none"  # "none", "nsys", or "torch"

    # Phase-specific profiling step configs
    prefill: ProfilingPhaseConfig | None = None
    decode: ProfilingPhaseConfig | None = None
    aggregated: ProfilingPhaseConfig | None = None

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.type != "none"

    @property
    def is_nsys(self) -> bool:
        """Check if using NVIDIA Nsight Systems profiling."""
        return self.type == "nsys"

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
            profile_dir: Base directory for profiling output

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

        return env

    def get_nsys_prefix(self, output_file: str, *, frontend_type: str | None = None) -> list[str]:
        """Get nsys profiling command prefix.

        Args:
            output_file: Path for nsys output file (without extension)
            frontend_type: Frontend type (e.g., "dynamo", "sglangrouter"). When set to "dynamo",
                add flags required for Dynamo's process model.

        Returns:
            Command prefix list for nsys profiling
        """
        if not self.is_nsys:
            return []

        cmd = [
            "nsys",
            "profile",
            "-t",
            "cuda,nvtx",
            "--cuda-graph-trace=node",
            "-c",
            "cudaProfilerApi",
            "--capture-range-end",
            "stop",
            "--force-overwrite",
            "true",
            "-o",
            output_file,
        ]

        if frontend_type == "dynamo":
            cmd.insert(-2, "--trace-fork-before-exec=true")

        return cmd

    Schema: ClassVar[builtins.type[Schema]] = Schema


@dataclass
class DynamoConfig:
    """Dynamo installation configuration.

    Only one of version, hash, or top_of_tree should be specified.
    Defaults to version="0.8.0" (pip install).

    Options:
        install: Whether to install dynamo at all (default: True). Set to False
                 if your container already has dynamo pre-installed.
        version: Install specific version from PyPI (e.g., "0.8.0")
        hash: Clone repo and checkout specific commit hash
        top_of_tree: Clone repo at HEAD (latest)

    If top_of_tree or hash is set, version is automatically cleared.
    """

    install: bool = True
    version: str | None = "0.8.0"
    hash: str | None = None
    top_of_tree: bool = False

    def __post_init__(self) -> None:
        # Auto-clear version if hash or top_of_tree is set
        if self.hash is not None or self.top_of_tree:
            object.__setattr__(self, "version", None)

        # Validate only one source option is set
        if self.hash is not None and self.top_of_tree:
            raise ValueError("Cannot specify both hash and top_of_tree")

    @property
    def needs_source_install(self) -> bool:
        """Whether this config requires a source install (git clone + maturin)."""
        return self.hash is not None or self.top_of_tree

    def get_install_commands(self) -> str:
        """Get the bash commands to install dynamo."""
        if self.version is not None:
            return (
                f"echo 'Installing dynamo {self.version}...' && "
                f"pip install --break-system-packages --quiet ai-dynamo-runtime=={self.version} ai-dynamo=={self.version} && "
                f"echo 'Dynamo {self.version} installed'"
            )

        # Source install (hash or top-of-tree)
        git_ref = self.hash if self.hash else "HEAD"
        checkout_cmd = f"git checkout {self.hash}" if self.hash else ""

        return (
            f"echo 'Installing dynamo from source ({git_ref})...' && "
            "cd /sgl-workspace/ && "
            "git clone https://github.com/ai-dynamo/dynamo.git && "
            "cd dynamo && "
            f"{checkout_cmd + ' && ' if checkout_cmd else ''}"
            "cd lib/bindings/python/ && "
            'export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" && '
            "maturin build -o /tmp && "
            "pip install /tmp/ai_dynamo_runtime*.whl && "
            "cd /sgl-workspace/dynamo/ && "
            "pip install -e . && "
            "cd /sgl-workspace/sglang/ && "
            f"echo 'Dynamo installed from source ({git_ref})'"
        )

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class FrontendConfig:
    """Frontend/router configuration.

    Attributes:
        type: Frontend type - "dynamo" (default) or "sglang"
        enable_multiple_frontends: Scale with nginx + multiple routers
        num_additional_frontends: Additional routers beyond master (default: 9)
        nginx_container: Custom nginx container image (default: nginx:1.27.4)
        args: CLI arguments passed to the frontend/router process
        env: Environment variables for frontend processes
    """

    type: str = "dynamo"
    enable_multiple_frontends: bool = True
    num_additional_frontends: int = 9
    nginx_container: str = "nginx:1.27.4"
    args: dict[str, Any] | None = None
    env: dict[str, str] | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration with formattable paths."""

    log_dir: Annotated[FormattablePath, FormattablePathField()] = field(
        default_factory=lambda: FormattablePath(template="./outputs/{job_id}/logs")
    )

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration."""

    max_attempts: int = 180  # 30 minutes default (large models take time to load)
    interval_seconds: int = 10

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class InfraConfig:
    """Infrastructure configuration for etcd/nats placement.

    Attributes:
        etcd_nats_dedicated_node: If True, run etcd and nats on a dedicated node
            instead of the head node. This reserves the first node exclusively
            for infrastructure services. Default: False.
    """

    etcd_nats_dedicated_node: bool = False

    Schema: ClassVar[type[Schema]] = Schema


# ============================================================================
# Main Configuration Dataclass
# ============================================================================


@dataclass(frozen=True)
class SrtConfig:
    """Complete srtctl job configuration (frozen, immutable).

    This is the main configuration type returned by load_config().

    The backend field supports polymorphic deserialization:
    - type: sglang -> SGLangProtocol
    """

    name: str
    model: ModelConfig
    resources: ResourceConfig

    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    backend: Annotated[BackendConfig, BackendConfigField()] = field(default_factory=SGLangProtocol)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    dynamo: DynamoConfig = field(default_factory=DynamoConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)

    environment: dict[str, str] = field(default_factory=dict)
    container_mounts: dict[
        Annotated[FormattablePath, FormattablePathField()],
        Annotated[FormattablePath, FormattablePathField()],
    ] = field(default_factory=dict)
    extra_mount: tuple[str, ...] | None = None
    srun_options: dict[str, str] = field(default_factory=dict)
    sbatch_directives: dict[str, str] = field(default_factory=dict)
    enable_config_dump: bool = True

    # Custom setup script (runs before dynamo install and worker startup)
    # e.g. "custom-setup.sh" -> runs /configs/custom-setup.sh
    setup_script: str | None = None

    # Reporting configuration (status API, future: logs to S3, etc.)
    reporting: ReportingConfig | None = None

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_profiling()

    def _validate_profiling(self):
        """Validate profiling configuration matches serving mode."""
        prof = self.profiling
        if not prof.enabled:
            return

        r = self.resources
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

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "SrtConfig":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        schema = cls.Schema()
        return schema.load(data)

    @property
    def served_model_name(self) -> str:
        """Get the served model name from backend config or model path."""
        default = Path(self.model.path).name
        return self.backend.get_served_model_name(default)

    @property
    def backend_type(self) -> str:
        """Get the backend type string."""
        return self.backend.type
