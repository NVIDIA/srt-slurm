# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
RunLoader service for loading and parsing benchmark runs

Clean, JSON-only implementation - requires {jobid}.json in each run directory.
"""

import json
import logging
import os
import re

import pandas as pd

from .cache_manager import CacheManager
from .models import BenchmarkRun

logger = logging.getLogger(__name__)


class RunLoader:
    """Service for loading benchmark run data from directories.

    Requires {jobid}.json metadata file in each run directory.
    """

    def __init__(self, logs_dir: str):
        """Initialize RunLoader with a logs directory.

        Args:
            logs_dir: Path to directory containing benchmark run subdirectories
        """
        self.logs_dir = logs_dir

    def load_all(self) -> list[BenchmarkRun]:
        """Load all benchmark runs from the logs directory.

        Only loads runs that have {jobid}.json metadata files.

        Returns:
            List of BenchmarkRun objects, sorted by job ID (newest first)
        """
        runs, _ = self.load_all_with_skipped()
        return runs

    def load_all_with_skipped(self) -> tuple[list[BenchmarkRun], list[tuple[str, str, str]]]:
        """Load all benchmark runs and track which ones were skipped.

        Returns:
            Tuple of (runs_with_data, skipped_runs)
            - runs_with_data: List of BenchmarkRun objects with benchmark results
            - skipped_runs: List of tuples (job_id, run_dir, reason)
        """
        paths = self._find_run_directories()

        runs = []
        skipped = []

        for path in sorted(paths, key=lambda p: self._extract_job_id(os.path.basename(p)), reverse=True):
            run_dir = os.path.basename(path)

            try:
                run = BenchmarkRun.from_json_file(path)
                if run is not None:
                    # Skip profiling jobs (they don't have benchmark results)
                    if run.profiler.profiler_type == "torch-profiler":
                        reason = "Profiling job (no benchmark results)"
                        logger.debug(f"Skipping profiling job {run.job_id}")
                        skipped.append((run.job_id, run_dir, reason))
                        continue

                    # Load benchmark results from profiler output files
                    self._load_benchmark_results(run)

                    # Check if all expected results are present
                    run.check_completeness()

                    # Warn if job is incomplete
                    if not run.is_complete:
                        logger.warning(
                            f"Job {run.job_id} is incomplete - missing concurrencies: {run.missing_concurrencies}"
                        )

                    # Only include runs with benchmark data
                    if run.profiler.output_tps:
                        runs.append(run)
                    else:
                        reason = "No benchmark results found"
                        logger.debug(f"Skipping run {run.job_id} - {reason}")
                        skipped.append((run.job_id, run_dir, reason))
                else:
                    # Extract job ID from directory name for skipped list
                    job_id = run_dir.split("_")[0] if "_" in run_dir else run_dir
                    reason = "No metadata JSON file"
                    logger.warning(f"No metadata JSON found for {run_dir}")
                    skipped.append((job_id, run_dir, reason))
            except Exception as e:
                # Extract job ID from directory name for skipped list
                job_id = run_dir.split("_")[0] if "_" in run_dir else run_dir
                reason = f"Error loading: {str(e)}"
                logger.error(f"Error loading run from {path}: {e}")
                skipped.append((job_id, run_dir, reason))
                continue

        return runs, skipped

    def load_single(self, run_dir: str) -> BenchmarkRun | None:
        """Load a single benchmark run.

        Args:
            run_dir: Path to run directory (relative to logs_dir or absolute)

        Returns:
            BenchmarkRun object or None if loading failed
        """
        # Handle both relative and absolute paths
        run_path = os.path.join(self.logs_dir, run_dir) if not os.path.isabs(run_dir) else run_dir

        if not os.path.exists(run_path):
            logger.error(f"Run directory does not exist: {run_path}")
            return None

        try:
            run = BenchmarkRun.from_json_file(run_path)
            if run is not None:
                self._load_benchmark_results(run)
                run.check_completeness()
                if not run.is_complete:
                    logger.warning(
                        f"Job {run.job_id} is incomplete - missing concurrencies: {run.missing_concurrencies}"
                    )
            return run
        except Exception as e:
            logger.error(f"Error loading run from {run_path}: {e}")
            return None

    def has_metadata_json(self, run_dir: str) -> bool:
        """Check if a run directory has a metadata JSON file.

        Args:
            run_dir: Path to run directory

        Returns:
            True if {jobid}.json exists, False otherwise
        """
        # Handle both relative and absolute paths
        run_path = os.path.join(self.logs_dir, run_dir) if not os.path.isabs(run_dir) else run_dir

        dirname = os.path.basename(run_path)
        job_id = dirname.split("_")[0]
        json_path = os.path.join(run_path, f"{job_id}.json")
        return os.path.exists(json_path)

    def _find_run_directories(self) -> list[str]:
        """Find all valid benchmark run directories in logs_dir.

        Returns:
            List of absolute paths to run directories
        """
        paths = []

        if not os.path.exists(self.logs_dir):
            logger.error(f"Logs directory does not exist: {self.logs_dir}")
            return paths

        for entry in os.listdir(self.logs_dir):
            # Skip hidden directories and files
            if entry.startswith("."):
                continue
            # Skip common non-job directories
            if entry in ["utils", "__pycache__", "venv", ".venv"]:
                continue
            # Skip Python files
            if ".py" in entry:
                continue

            full_path = os.path.join(self.logs_dir, entry)
            if not os.path.isdir(full_path):
                continue

            # Only include directories that start with a numeric job ID
            first_part = entry.split("_")[0]
            if not first_part.isdigit():
                continue

            paths.append(full_path)

        return paths

    def _extract_job_id(self, dirname: str) -> int:
        """Extract numeric job ID from directory name for sorting.

        Args:
            dirname: Directory name like "3667_1P_1D_20251110_192145"

        Returns:
            Job ID as integer, or -1 if not parseable
        """
        try:
            return int(dirname.split("_")[0])
        except (ValueError, IndexError):
            return -1

    def _load_benchmark_results(self, run: BenchmarkRun) -> None:
        """Load benchmark results from profiler output files.

        Looks for directories like "sa-bench_isl_1024_osl_1024/" or "vllm_isl_1024_osl_1024/" and parses JSON files.
        Uses parquet caching to avoid re-parsing on subsequent loads.

        Checks both the run directory and the logs/ subdirectory for benchmark results.

        Args:
            run: BenchmarkRun object to populate with results
        """
        run_path = run.metadata.path

        # Check both run_path and run_path/logs for benchmark results
        search_paths = [run_path]
        logs_subdir = os.path.join(run_path, "logs")
        if os.path.exists(logs_subdir):
            search_paths.append(logs_subdir)

        # Initialize cache manager
        cache_mgr = CacheManager(run_path)

        # Use profiler_type from metadata to construct directory name
        profiler_type = run.profiler.profiler_type
        pattern_strs = [f"{profiler_type}_isl_{run.profiler.isl}_osl_{run.profiler.osl}"]

        # Define source patterns for cache validation (check all possible patterns)
        source_patterns = [f"{pattern}/*.json" for pattern in pattern_strs]

        # Try to load from cache first
        if cache_mgr.is_cache_valid("benchmark_results", source_patterns):
            cached_df = cache_mgr.load_from_cache("benchmark_results")
            if cached_df is not None and not cached_df.empty:
                # Populate run.profiler from cached DataFrame - load all available columns
                results = {
                    "concurrencies": cached_df["concurrency"].tolist(),
                    "output_tps": cached_df["output_tps"].tolist(),
                    "mean_itl_ms": cached_df["mean_itl_ms"].tolist(),
                    "mean_ttft_ms": cached_df["mean_ttft_ms"].tolist(),
                    "request_rate": cached_df["request_rate"].tolist(),
                }

                # Optional fields (may not exist in old cache files)
                optional_fields = [
                    "mean_tpot_ms",
                    "total_tps",
                    "request_throughput",
                    "request_goodput",
                    "mean_e2el_ms",
                    "median_ttft_ms",
                    "median_tpot_ms",
                    "median_itl_ms",
                    "median_e2el_ms",
                    "p99_ttft_ms",
                    "p99_tpot_ms",
                    "p99_itl_ms",
                    "p99_e2el_ms",
                    "std_ttft_ms",
                    "std_tpot_ms",
                    "std_itl_ms",
                    "std_e2el_ms",
                    "total_input_tokens",
                    "total_output_tokens",
                ]
                for field in optional_fields:
                    if field in cached_df.columns:
                        results[field] = cached_df[field].tolist()
                    else:
                        results[field] = []

                run.profiler.add_benchmark_results(results)
                return

        # Cache miss or invalid - parse from JSON files
        for pattern_str in pattern_strs:
            profiler_pattern = re.compile(pattern_str)
            for search_path in search_paths:
                for entry in os.listdir(search_path):
                    if profiler_pattern.match(entry):
                        result_dir = os.path.join(search_path, entry)
                        if os.path.isdir(result_dir):
                            results = self._parse_profiler_results(result_dir)
                            run.profiler.add_benchmark_results(results)

                            # Save to cache
                            if results["concurrencies"]:
                                # Convert to DataFrame for caching - cache ALL parsed fields
                                cache_data = {
                                    "concurrency": results["concurrencies"],
                                    "output_tps": results["output_tps"],
                                    "mean_itl_ms": results["mean_itl_ms"],
                                    "mean_ttft_ms": results["mean_ttft_ms"],
                                    "request_rate": results["request_rate"],
                                }

                                # Add all optional fields if they have data
                                optional_fields = {
                                    "mean_tpot_ms": "mean_tpot_ms",
                                    "total_tps": "total_tps",
                                    "request_throughput": "request_throughput",
                                    "request_goodput": "request_goodput",
                                    "mean_e2el_ms": "mean_e2el_ms",
                                    "median_ttft_ms": "median_ttft_ms",
                                    "median_tpot_ms": "median_tpot_ms",
                                    "median_itl_ms": "median_itl_ms",
                                    "median_e2el_ms": "median_e2el_ms",
                                    "p99_ttft_ms": "p99_ttft_ms",
                                    "p99_tpot_ms": "p99_tpot_ms",
                                    "p99_itl_ms": "p99_itl_ms",
                                    "p99_e2el_ms": "p99_e2el_ms",
                                    "std_ttft_ms": "std_ttft_ms",
                                    "std_tpot_ms": "std_tpot_ms",
                                    "std_itl_ms": "std_itl_ms",
                                    "std_e2el_ms": "std_e2el_ms",
                                    "total_input_tokens": "total_input_tokens",
                                    "total_output_tokens": "total_output_tokens",
                                }

                                for result_key, cache_key in optional_fields.items():
                                    if results.get(result_key):
                                        cache_data[cache_key] = results[result_key]

                                cache_df = pd.DataFrame(cache_data)
                                cache_mgr.save_to_cache("benchmark_results", cache_df, source_patterns)

                            return  # Found results, stop searching

    def _parse_profiler_results(self, result_dir: str) -> dict:
        """Parse profiler result JSON files.

        Args:
            result_dir: Path to directory containing benchmark result JSON files

        Returns:
            Dict with concurrencies, output_tps, mean_itl_ms, etc.
        """
        result = []

        for file in os.listdir(result_dir):
            if not file.endswith(".json"):
                continue

            filepath = os.path.join(result_dir, file)
            try:
                with open(filepath) as f:
                    content = json.load(f)

                    # Parse all available metrics from benchmark output
                    res = {
                        "max_concurrency": content.get("max_concurrency"),
                        # Throughput metrics
                        "output_throughput": content.get("output_throughput"),
                        "total_token_throughput": content.get("total_token_throughput"),
                        "request_throughput": content.get("request_throughput"),
                        "request_goodput": content.get("request_goodput"),
                        "request_rate": content.get("request_rate"),
                        # Mean latencies
                        "mean_ttft_ms": content.get("mean_ttft_ms"),
                        "mean_tpot_ms": content.get("mean_tpot_ms"),
                        "mean_itl_ms": content.get("mean_itl_ms"),
                        "mean_e2el_ms": content.get("mean_e2el_ms"),
                        # Median latencies
                        "median_ttft_ms": content.get("median_ttft_ms"),
                        "median_tpot_ms": content.get("median_tpot_ms"),
                        "median_itl_ms": content.get("median_itl_ms"),
                        "median_e2el_ms": content.get("median_e2el_ms"),
                        # P99 latencies
                        "p99_ttft_ms": content.get("p99_ttft_ms"),
                        "p99_tpot_ms": content.get("p99_tpot_ms"),
                        "p99_itl_ms": content.get("p99_itl_ms"),
                        "p99_e2el_ms": content.get("p99_e2el_ms"),
                        # Std dev latencies
                        "std_ttft_ms": content.get("std_ttft_ms"),
                        "std_tpot_ms": content.get("std_tpot_ms"),
                        "std_itl_ms": content.get("std_itl_ms"),
                        "std_e2el_ms": content.get("std_e2el_ms"),
                        # Token counts
                        "total_input_tokens": content.get("total_input_tokens"),
                        "total_output_tokens": content.get("total_output_tokens"),
                        # Metadata
                        "backend": content.get("backend"),
                        "model_id": content.get("model_id"),
                        "date": content.get("date"),
                        "duration": content.get("duration"),
                        "completed": content.get("completed"),
                        "num_prompts": content.get("num_prompts"),
                    }

                    result.append(res)
            except Exception as e:
                logger.warning(f"Error parsing {filepath}: {e}")
                continue

        # Organize results - sort by concurrency
        out = {
            # Primary metrics
            "concurrencies": [],
            "output_tps": [],
            "total_tps": [],
            "request_throughput": [],
            "request_goodput": [],
            "request_rate": [],
            # Mean latencies
            "mean_ttft_ms": [],
            "mean_tpot_ms": [],
            "mean_itl_ms": [],
            "mean_e2el_ms": [],
            # Median latencies
            "median_ttft_ms": [],
            "median_tpot_ms": [],
            "median_itl_ms": [],
            "median_e2el_ms": [],
            # P99 latencies
            "p99_ttft_ms": [],
            "p99_tpot_ms": [],
            "p99_itl_ms": [],
            "p99_e2el_ms": [],
            # Std dev latencies
            "std_ttft_ms": [],
            "std_tpot_ms": [],
            "std_itl_ms": [],
            "std_e2el_ms": [],
            # Token counts
            "total_input_tokens": [],
            "total_output_tokens": [],
            # Metadata
            "backend": [],
            "model_id": [],
            "date": [],
            "duration": [],
            "completed": [],
            "num_prompts": [],
        }

        # Sort by concurrency and aggregate
        for data in sorted(result, key=lambda x: x.get("max_concurrency", 0) or 0):
            out["concurrencies"].append(data.get("max_concurrency"))
            # Throughput
            out["output_tps"].append(data.get("output_throughput"))
            out["total_tps"].append(data.get("total_token_throughput"))
            out["request_throughput"].append(data.get("request_throughput"))
            out["request_goodput"].append(data.get("request_goodput"))
            out["request_rate"].append(data.get("request_rate"))
            # Mean latencies
            out["mean_ttft_ms"].append(data.get("mean_ttft_ms"))
            out["mean_tpot_ms"].append(data.get("mean_tpot_ms"))
            out["mean_itl_ms"].append(data.get("mean_itl_ms"))
            out["mean_e2el_ms"].append(data.get("mean_e2el_ms"))
            # Median latencies
            out["median_ttft_ms"].append(data.get("median_ttft_ms"))
            out["median_tpot_ms"].append(data.get("median_tpot_ms"))
            out["median_itl_ms"].append(data.get("median_itl_ms"))
            out["median_e2el_ms"].append(data.get("median_e2el_ms"))
            # P99 latencies
            out["p99_ttft_ms"].append(data.get("p99_ttft_ms"))
            out["p99_tpot_ms"].append(data.get("p99_tpot_ms"))
            out["p99_itl_ms"].append(data.get("p99_itl_ms"))
            out["p99_e2el_ms"].append(data.get("p99_e2el_ms"))
            # Std dev latencies
            out["std_ttft_ms"].append(data.get("std_ttft_ms"))
            out["std_tpot_ms"].append(data.get("std_tpot_ms"))
            out["std_itl_ms"].append(data.get("std_itl_ms"))
            out["std_e2el_ms"].append(data.get("std_e2el_ms"))
            # Token counts
            out["total_input_tokens"].append(data.get("total_input_tokens"))
            out["total_output_tokens"].append(data.get("total_output_tokens"))
            # Metadata
            out["backend"].append(data.get("backend"))
            out["model_id"].append(data.get("model_id"))
            out["date"].append(data.get("date"))
            out["duration"].append(data.get("duration"))
            out["completed"].append(data.get("completed"))
            out["num_prompts"].append(data.get("num_prompts"))

        return out

    def get_run_count(self) -> int:
        """Get count of valid benchmark runs in logs directory.

        Returns:
            Number of valid run directories found
        """
        return len(self._find_run_directories())

    def get_runs_with_metadata(self) -> list[str]:
        """Get list of run directories that have metadata JSON files.

        Returns:
            List of run directory names (not full paths)
        """
        runs_with_metadata = []

        for path in self._find_run_directories():
            if self.has_metadata_json(path):
                runs_with_metadata.append(os.path.basename(path))

        return runs_with_metadata

    def get_runs_without_metadata(self) -> list[str]:
        """Get list of run directories that DON'T have metadata JSON files.

        Useful for identifying which runs need metadata file generation.

        Returns:
            List of run directory names (not full paths)
        """
        runs_without_metadata = []

        for path in self._find_run_directories():
            if not self.has_metadata_json(path):
                runs_without_metadata.append(os.path.basename(path))

        return runs_without_metadata

    def to_dataframe(self, runs: list[BenchmarkRun] | None = None):
        """Convert runs to pandas DataFrame for analysis.

        Args:
            runs: List of BenchmarkRun objects. If None, loads all runs.

        Returns:
            pandas DataFrame with one row per concurrency level
        """
        import pandas as pd

        if runs is None:
            runs = self.load_all()

        rows = []

        for run in runs:
            run_id = (
                f"{run.job_id}_{run.metadata.prefill_workers}P_{run.metadata.decode_workers}D_{run.metadata.run_date}"
            )
            total_gpus = run.total_gpus

            # Create a row for each concurrency level
            for i in range(len(run.profiler.output_tps)):
                # Calculate derived metrics
                tps = run.profiler.output_tps[i]
                tps_per_gpu = tps / total_gpus if total_gpus > 0 else 0

                # Get total TPS (input + output tokens)
                total_token_tps = run.profiler.total_tps[i] if i < len(run.profiler.total_tps) else None
                total_tps_per_gpu = total_token_tps / total_gpus if total_token_tps and total_gpus > 0 else None

                # Output TPS/User = 1000 / TPOT(ms)
                tpot = run.profiler.mean_tpot_ms[i] if i < len(run.profiler.mean_tpot_ms) else None
                tps_per_user = 1000 / tpot if tpot and tpot > 0 else 0

                row = {
                    "Run ID": run_id,
                    "Run Date": run.metadata.run_date,
                    "Profiler": run.profiler.profiler_type,
                    "ISL": run.profiler.isl,
                    "OSL": run.profiler.osl,
                    "Prefill Nodes": run.metadata.prefill_nodes,
                    "Decode Nodes": run.metadata.decode_nodes,
                    "Prefill Workers": run.metadata.prefill_workers,
                    "Decode Workers": run.metadata.decode_workers,
                    "Frontends": run.metadata.num_additional_frontends,
                    "GPUs per Node": run.metadata.gpus_per_node,
                    "Total GPUs": total_gpus,
                    "Request Rate": (run.profiler.request_rate[i] if i < len(run.profiler.request_rate) else "N/A"),
                    "Concurrency": (
                        run.profiler.concurrency_values[i] if i < len(run.profiler.concurrency_values) else "N/A"
                    ),
                    "Output TPS": tps,
                    "Total TPS": total_token_tps if total_token_tps else "N/A",
                    "Output TPS/GPU": tps_per_gpu,
                    "Total TPS/GPU": total_tps_per_gpu if total_tps_per_gpu else "N/A",
                    "Output TPS/User": tps_per_user,
                    "Mean TTFT (ms)": (run.profiler.mean_ttft_ms[i] if i < len(run.profiler.mean_ttft_ms) else "N/A"),
                    "Mean TPOT (ms)": tpot if tpot else "N/A",
                    "Mean ITL (ms)": (run.profiler.mean_itl_ms[i] if i < len(run.profiler.mean_itl_ms) else "N/A"),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def update_tags(self, run_path: str, tags: list[str]) -> bool:
        """Update tags for a run by modifying its {jobid}.json file.

        Args:
            run_path: Path to the run directory
            tags: List of tag strings to set

        Returns:
            True if successful, False otherwise
        """
        # Handle both relative and absolute paths
        if not os.path.isabs(run_path):
            run_path = os.path.join(self.logs_dir, run_path)

        # Extract job ID from directory name
        dirname = os.path.basename(run_path)
        job_id = dirname.split("_")[0]
        json_path = os.path.join(run_path, f"{job_id}.json")

        if not os.path.exists(json_path):
            logger.error(f"Metadata JSON not found: {json_path}")
            return False

        try:
            # Read existing JSON
            with open(json_path) as f:
                json_data = json.load(f)

            # Update tags
            json_data["tags"] = tags

            # Write back to file
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

            logger.info(f"Updated tags for {job_id}: {tags}")
            return True
        except Exception as e:
            logger.error(f"Error updating tags for {json_path}: {e}")
            return False
