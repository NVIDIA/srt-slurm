# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Domain models for benchmark analysis

Centralized location for all data models and type definitions.
Includes both dataclasses (for objects) and TypedDicts (for dict typing).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict


@dataclass
class RunMetadata:
    """Metadata about a benchmark run from {jobid}.json."""

    job_id: str
    path: str
    run_date: str
    container: str
    prefill_nodes: int
    decode_nodes: int
    prefill_workers: int
    decode_workers: int
    mode: str
    # Optional fields
    job_name: str = ""
    partition: str = ""
    model_dir: str = ""
    gpus_per_node: int = 0
    gpu_type: str = ""
    enable_multiple_frontends: bool = False
    num_additional_frontends: int = 0
    # Aggregated mode fields
    agg_nodes: int = 0
    agg_workers: int = 0

    @classmethod
    def from_json(cls, json_data: dict, run_path: str) -> "RunMetadata":
        """Create from {jobid}.json metadata format.

        Supports both old format (with run_metadata key) and new format (flat structure).

        Args:
            json_data: Parsed JSON from {jobid}.json file
            run_path: Path to the run directory

        Returns:
            RunMetadata instance
        """
        # Check if this is the old format (with run_metadata key) or new format (flat)
        if "run_metadata" in json_data:
            # Old format
            run_meta = json_data.get("run_metadata", {})
            mode = run_meta.get("mode", "disaggregated")

            return cls(
                job_id=run_meta.get("slurm_job_id", ""),
                path=run_path,
                run_date=run_meta.get("run_date", ""),
                container=run_meta.get("container", ""),
                prefill_nodes=run_meta.get("prefill_nodes", 0),
                decode_nodes=run_meta.get("decode_nodes", 0),
                prefill_workers=run_meta.get("prefill_workers", 0),
                decode_workers=run_meta.get("decode_workers", 0),
                mode=mode,
                job_name=run_meta.get("job_name", ""),
                partition=run_meta.get("partition", ""),
                model_dir=run_meta.get("model_dir", ""),
                gpus_per_node=run_meta.get("gpus_per_node", 0),
                gpu_type=run_meta.get("gpu_type", ""),
                enable_multiple_frontends=run_meta.get("enable_multiple_frontends", False),
                num_additional_frontends=run_meta.get("num_additional_frontends", 0),
                agg_nodes=run_meta.get("agg_nodes", 0),
                agg_workers=run_meta.get("agg_workers", 0),
            )
        else:
            # New format (flat structure)
            model_data = json_data.get("model", {})
            resources_data = json_data.get("resources", {})
            agg_workers = resources_data.get("agg_workers", 0)

            # Determine mode based on agg_workers
            mode = "aggregated" if agg_workers > 0 else "disaggregated"

            return cls(
                job_id=json_data.get("job_id", ""),
                path=run_path,
                run_date=json_data.get("generated_at", ""),
                container=model_data.get("container", ""),
                prefill_nodes=resources_data.get("prefill_nodes", 0),
                decode_nodes=resources_data.get("decode_nodes", 0),
                prefill_workers=resources_data.get("prefill_workers", 0),
                decode_workers=resources_data.get("decode_workers", 0),
                mode=mode,
                job_name=json_data.get("job_name", ""),
                partition="",  # Not present in new format
                model_dir=model_data.get("path", ""),  # Use model path as model_dir
                gpus_per_node=resources_data.get("gpus_per_node", 0),
                gpu_type=resources_data.get("gpu_type", ""),
                enable_multiple_frontends=False,  # Not present in new format
                num_additional_frontends=0,  # Not present in new format
                agg_nodes=resources_data.get("agg_nodes", 0),  # Not present in new format
                agg_workers=agg_workers,
            )

    @property
    def is_aggregated(self) -> bool:
        """Check if this is an aggregated mode run."""
        return self.mode == "aggregated" or self.agg_nodes > 0

    @property
    def total_gpus(self) -> int:
        """Calculate total GPU count for both modes."""
        if self.is_aggregated:
            return self.agg_nodes * self.gpus_per_node
        return (self.prefill_nodes + self.decode_nodes) * self.gpus_per_node

    @property
    def topology_label(self) -> str:
        """Get topology label appropriate for the mode.

        Returns:
            "XP/YD" for disaggregated, "XA" for aggregated
        """
        if self.is_aggregated:
            return f"{self.agg_workers}A"
        return f"{self.prefill_workers}P/{self.decode_workers}D"

    @property
    def formatted_date(self) -> str:
        """Get human-readable date string (e.g., 'Nov 10').

        Returns:
            Formatted date string like "Nov 10", or raw date if parsing fails
        """
        try:
            # Parse YYYYMMDD_HHMMSS format
            dt = datetime.strptime(self.run_date, "%Y%m%d_%H%M%S")
            return dt.strftime("%b %d").replace(" 0", " ")
        except (ValueError, TypeError):
            return self.run_date


@dataclass
class ProfilerResults:
    """Results from profiler benchmarks.

    Parses 32 out of 39 fields from benchmark JSON output.

    NOT PARSED (7 fields):
    - input_lens, output_lens, ttfts, itls: Per-request arrays (too large for in-memory storage)
    - errors, generated_texts: Per-request data (not needed for aggregate analysis)
    - tokenizer_id, best_of, burstiness: Metadata not critical for dashboards
    """

    profiler_type: str
    isl: str
    osl: str
    concurrencies: str = ""
    req_rate: str = ""

    # Primary throughput metrics (per concurrency level)
    output_tps: list[float] = field(default_factory=list)
    total_tps: list[float] = field(default_factory=list)
    request_throughput: list[float] = field(default_factory=list)
    request_goodput: list[float | None] = field(default_factory=list)
    concurrency_values: list[int] = field(default_factory=list)
    request_rate: list[float] = field(default_factory=list)

    # Latency metrics - mean (per concurrency level)
    mean_ttft_ms: list[float] = field(default_factory=list)
    mean_tpot_ms: list[float] = field(default_factory=list)
    mean_itl_ms: list[float] = field(default_factory=list)
    mean_e2el_ms: list[float] = field(default_factory=list)

    # Latency metrics - median (per concurrency level)
    median_ttft_ms: list[float] = field(default_factory=list)
    median_tpot_ms: list[float] = field(default_factory=list)
    median_itl_ms: list[float] = field(default_factory=list)
    median_e2el_ms: list[float] = field(default_factory=list)

    # Latency metrics - p99 (per concurrency level)
    p99_ttft_ms: list[float] = field(default_factory=list)
    p99_tpot_ms: list[float] = field(default_factory=list)
    p99_itl_ms: list[float] = field(default_factory=list)
    p99_e2el_ms: list[float] = field(default_factory=list)

    # Latency metrics - std dev (per concurrency level)
    std_ttft_ms: list[float] = field(default_factory=list)
    std_tpot_ms: list[float] = field(default_factory=list)
    std_itl_ms: list[float] = field(default_factory=list)
    std_e2el_ms: list[float] = field(default_factory=list)

    # Token counts (per concurrency level)
    total_input_tokens: list[int] = field(default_factory=list)
    total_output_tokens: list[int] = field(default_factory=list)

    # Run metadata (per concurrency level)
    backend: list[str] = field(default_factory=list)
    model_id: list[str] = field(default_factory=list)
    date: list[str] = field(default_factory=list)
    duration: list[float] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    num_prompts: list[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_data: dict) -> "ProfilerResults":
        """Create from {jobid}.json profiler_metadata section.

        Args:
            json_data: Parsed JSON from {jobid}.json file

        Returns:
            ProfilerResults instance (benchmark data added later from result files)
        """
        profiler_meta = json_data.get("benchmark", {})

        return cls(
            profiler_type=profiler_meta.get("type", "unknown"),
            isl=str(profiler_meta.get("isl", "")),
            osl=str(profiler_meta.get("osl", "")),
            concurrencies=profiler_meta.get("concurrencies", ""),
            req_rate=profiler_meta.get("req-rate", ""),
        )

    def add_benchmark_results(self, results: dict) -> None:
        """Add actual benchmark results from profiler output files.

        Args:
            results: Dict with all benchmark metrics from parsed JSON files
        """
        # Primary metrics
        self.concurrency_values = results.get("concurrencies", [])
        self.output_tps = results.get("output_tps", [])
        self.total_tps = results.get("total_tps", [])
        self.request_throughput = results.get("request_throughput", [])
        self.request_goodput = results.get("request_goodput", [])
        self.request_rate = results.get("request_rate", [])

        # Mean latencies
        self.mean_ttft_ms = results.get("mean_ttft_ms", [])
        self.mean_tpot_ms = results.get("mean_tpot_ms", [])
        self.mean_itl_ms = results.get("mean_itl_ms", [])
        self.mean_e2el_ms = results.get("mean_e2el_ms", [])

        # Median latencies
        self.median_ttft_ms = results.get("median_ttft_ms", [])
        self.median_tpot_ms = results.get("median_tpot_ms", [])
        self.median_itl_ms = results.get("median_itl_ms", [])
        self.median_e2el_ms = results.get("median_e2el_ms", [])

        # P99 latencies
        self.p99_ttft_ms = results.get("p99_ttft_ms", [])
        self.p99_tpot_ms = results.get("p99_tpot_ms", [])
        self.p99_itl_ms = results.get("p99_itl_ms", [])
        self.p99_e2el_ms = results.get("p99_e2el_ms", [])

        # Std dev latencies
        self.std_ttft_ms = results.get("std_ttft_ms", [])
        self.std_tpot_ms = results.get("std_tpot_ms", [])
        self.std_itl_ms = results.get("std_itl_ms", [])
        self.std_e2el_ms = results.get("std_e2el_ms", [])

        # Token counts
        self.total_input_tokens = results.get("total_input_tokens", [])
        self.total_output_tokens = results.get("total_output_tokens", [])

        # Metadata
        self.backend = results.get("backend", [])
        self.model_id = results.get("model_id", [])
        self.date = results.get("date", [])
        self.duration = results.get("duration", [])
        self.completed = results.get("completed", [])
        self.num_prompts = results.get("num_prompts", [])


@dataclass
class BenchmarkRun:
    """Complete benchmark run with metadata and profiler results."""

    metadata: RunMetadata
    profiler: ProfilerResults
    is_complete: bool = True
    missing_concurrencies: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_json_file(cls, run_path: str) -> "BenchmarkRun | None":
        """Create from {jobid}.json file in the run directory.

        Args:
            run_path: Path to the run directory containing {jobid}.json

        Returns:
            BenchmarkRun instance or None if file not found/invalid
        """
        import json
        import os

        # Extract job ID from directory name
        dirname = os.path.basename(run_path)
        job_id = dirname.split("_")[0] if "_" in dirname else int(dirname)
        json_path = os.path.join(run_path, f"{job_id}.json")

        if not os.path.exists(json_path):
            return None

        try:
            with open(json_path) as f:
                json_data = json.load(f)

            metadata = RunMetadata.from_json(json_data, run_path)
            profiler = ProfilerResults.from_json(json_data)
            tags = json_data.get("tags", [])

            return cls(metadata=metadata, profiler=profiler, tags=tags)
        except Exception:
            return None

    @property
    def job_id(self) -> str:
        """Convenience property for job ID."""
        return self.metadata.job_id

    @property
    def total_gpus(self) -> int:
        """Calculate total GPU count."""
        return self.metadata.total_gpus

    def check_completeness(self) -> None:
        """Check if all expected benchmark results are present.

        Compares expected concurrencies from profiler metadata with actual results.
        Updates is_complete and missing_concurrencies fields.
        """
        # Parse expected concurrencies from metadata
        if not self.profiler.concurrencies:
            # No expected concurrencies specified, assume manual run
            self.is_complete = True
            self.missing_concurrencies = []
            return

        expected = set()
        for val in self.profiler.concurrencies.split("x"):
            try:
                expected.add(int(val.strip()))
            except ValueError:
                continue

        # Get actual concurrencies from results
        actual = set(self.profiler.concurrency_values)

        # Find missing ones
        missing = expected - actual

        self.is_complete = len(missing) == 0
        self.missing_concurrencies = sorted(missing)


@dataclass
class BatchMetrics:
    """Metrics from a single batch (prefill or decode), parsed from log files."""

    timestamp: str
    dp: int
    tp: int
    ep: int
    batch_type: str  # "prefill" or "decode"
    # Optional metrics
    new_seq: int | None = None
    new_token: int | None = None
    cached_token: int | None = None
    token_usage: float | None = None
    running_req: int | None = None
    queue_req: int | None = None
    prealloc_req: int | None = None
    inflight_req: int | None = None
    input_throughput: float | None = None
    gen_throughput: float | None = None
    transfer_req: int | None = None
    num_tokens: int | None = None
    preallocated_usage: float | None = None

    @property
    def cache_hit_rate(self) -> float | None:
        """Calculate cache hit rate percentage."""
        if self.new_token is not None and self.cached_token is not None:
            total = self.new_token + self.cached_token
            return (self.cached_token / total * 100) if total > 0 else None
        return None


@dataclass
class MemoryMetrics:
    """Memory metrics from log lines."""

    timestamp: str
    dp: int
    tp: int
    ep: int
    metric_type: str  # "memory" or "kv_cache"
    avail_mem_gb: float | None = None
    mem_usage_gb: float | None = None
    kv_cache_gb: float | None = None
    kv_tokens: int | None = None


@dataclass
class NodeMetrics:
    """Metrics from a single node (prefill or decode worker), parsed from log files."""

    node_info: dict  # Has node name, worker type, worker_id
    batches: list[BatchMetrics] = field(default_factory=list)
    memory_snapshots: list[MemoryMetrics] = field(default_factory=list)
    config: dict = field(default_factory=dict)  # TP/DP/EP config
    run_id: str = ""

    @property
    def node_name(self) -> str:
        """Get node name."""
        return self.node_info.get("node", "Unknown")

    @property
    def worker_type(self) -> str:
        """Get worker type (prefill/decode/frontend)."""
        return self.node_info.get("worker_type", "unknown")

    @property
    def is_prefill(self) -> bool:
        """Check if this is a prefill node."""
        return self.worker_type == "prefill"

    @property
    def is_decode(self) -> bool:
        """Check if this is a decode node."""
        return self.worker_type == "decode"


# Config-related TypedDicts (from config_reader.py)
class GPUInfo(TypedDict, total=False):
    """Expected structure of GPU info in node config."""

    count: int
    gpus: list[dict[str, Any]]
    name: str
    memory_total: str
    driver_version: str


class ServerArgs(TypedDict, total=False):
    """Expected structure of server_args in node config.

    Note: This is partial - actual configs may have many more fields.
    Use total=False to allow missing keys.
    """

    tp_size: int
    dp_size: int
    pp_size: int
    ep_size: int
    served_model_name: str
    attention_backend: str
    kv_cache_dtype: str
    max_total_tokens: int
    chunked_prefill_size: int
    disaggregation_mode: str
    context_length: int


class NodeConfig(TypedDict, total=False):
    """Expected structure of a node config JSON file (*_config.json)."""

    filename: str
    gpu_info: GPUInfo
    config: dict[str, Any]  # Contains 'server_args' and other fields
    environment: dict[str, str]


class ParsedCommandInfo(TypedDict):
    """Expected return structure from parse_command_line_from_err."""

    explicit_flags: set
    services: dict[str, list[str]]
