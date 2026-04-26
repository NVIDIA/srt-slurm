# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Node analysis service for parsing .err/.out log files

All parsing logic encapsulated in the NodeAnalyzer class.
"""

import logging
import os
import re

import pandas as pd

from .cache_manager import CacheManager

# Configure logging
logger = logging.getLogger(__name__)


class NodeAnalyzer:
    """Service for analyzing node-level metrics from log files.

    Parses .err/.out files to extract batch metrics, memory usage, and configuration.
    All parsing logic is encapsulated as methods.
    """

    def parse_run_logs(self, run_path: str, return_dicts: bool = False) -> list:
        """Parse all node log files in a run directory.

        Uses parquet caching to avoid re-parsing on subsequent loads.

        Args:
            run_path: Path to the run directory containing .err/.out files
            return_dicts: If True, return dicts directly (faster). If False, return NodeMetrics objects.

        Returns:
            List of NodeMetrics objects or dicts, one per node
        """
        # Initialize cache manager
        cache_mgr = CacheManager(run_path)

        # Define source patterns for cache validation (.err and .out files)
        source_patterns = ["*.err", "*.out"]

        # Try to load from cache first
        if cache_mgr.is_cache_valid("node_metrics", source_patterns):
            cached_df = cache_mgr.load_from_cache("node_metrics")
            if cached_df is not None and not cached_df.empty:
                if return_dicts:
                    # Fast path: convert directly to dicts without NodeMetrics objects
                    nodes = self._dataframe_to_dicts(cached_df)
                    logger.info(f"Loaded {len(nodes)} nodes from cache (as dicts)")
                else:
                    # Reconstruct NodeMetrics objects from DataFrame
                    nodes = self._deserialize_node_metrics(cached_df)
                    logger.info(f"Loaded {len(nodes)} nodes from cache")
                return nodes

        # Cache miss or invalid - parse from .err/.out files
        nodes = []

        if not os.path.exists(run_path):
            logger.error(f"Run path does not exist: {run_path}")
            return nodes

        total_err_files = 0
        parsed_successfully = 0

        for file in os.listdir(run_path):
            if (file.endswith(".err") or file.endswith(".out")) and ("prefill" in file or "decode" in file):
                total_err_files += 1
                filepath = os.path.join(run_path, file)
                node = self.parse_single_log(filepath)
                if node:
                    nodes.append(node)
                    parsed_successfully += 1

        logger.info(f"Parsed {parsed_successfully}/{total_err_files} prefill/decode log files from {run_path}")

        if total_err_files == 0:
            logger.warning(f"No prefill/decode log files found in {run_path}")

        # Save to cache if we have data
        if nodes:
            cache_df = self._serialize_node_metrics(nodes)
            cache_mgr.save_to_cache("node_metrics", cache_df, source_patterns)

        return nodes

    def parse_single_log(self, filepath: str):
        """Parse a single node log file.

        Args:
            filepath: Path to the .err/.out log file

        Returns:
            NodeMetrics object or None if parsing failed
        """
        from .models import BatchMetrics, MemoryMetrics, NodeMetrics

        node_info = self._extract_node_info_from_filename(filepath)
        if not node_info:
            logger.warning(
                f"Could not extract node info from filename: {filepath}. "
                f"Expected format: <node>_<service>_<id>.err or .out"
            )
            return None

        batches = []
        memory_snapshots = []
        config = {}

        try:
            with open(filepath) as f:
                for line in f:
                    # Parse prefill batch metrics
                    batch_metrics = self._parse_prefill_batch_line(line)
                    if batch_metrics:
                        batches.append(
                            BatchMetrics(
                                timestamp=batch_metrics["timestamp"],
                                dp=batch_metrics["dp"],
                                tp=batch_metrics["tp"],
                                ep=batch_metrics["ep"],
                                batch_type=batch_metrics["type"],
                                new_seq=batch_metrics.get("new_seq"),
                                new_token=batch_metrics.get("new_token"),
                                cached_token=batch_metrics.get("cached_token"),
                                token_usage=batch_metrics.get("token_usage"),
                                running_req=batch_metrics.get("running_req"),
                                queue_req=batch_metrics.get("queue_req"),
                                prealloc_req=batch_metrics.get("prealloc_req"),
                                inflight_req=batch_metrics.get("inflight_req"),
                                input_throughput=batch_metrics.get("input_throughput"),
                            )
                        )

                    # Parse decode batch metrics
                    decode_metrics = self._parse_decode_batch_line(line)
                    if decode_metrics:
                        batches.append(
                            BatchMetrics(
                                timestamp=decode_metrics["timestamp"],
                                dp=decode_metrics["dp"],
                                tp=decode_metrics["tp"],
                                ep=decode_metrics["ep"],
                                batch_type=decode_metrics["type"],
                                running_req=decode_metrics.get("running_req"),
                                queue_req=decode_metrics.get("queue_req"),
                                prealloc_req=decode_metrics.get("prealloc_req"),
                                transfer_req=decode_metrics.get("transfer_req"),
                                token_usage=decode_metrics.get("token_usage"),
                                preallocated_usage=decode_metrics.get("preallocated_usage"),
                                num_tokens=decode_metrics.get("num_tokens"),
                                gen_throughput=decode_metrics.get("gen_throughput"),
                            )
                        )

                    # Parse memory metrics
                    mem_metrics = self._parse_memory_line(line)
                    if mem_metrics:
                        memory_snapshots.append(
                            MemoryMetrics(
                                timestamp=mem_metrics["timestamp"],
                                dp=mem_metrics["dp"],
                                tp=mem_metrics["tp"],
                                ep=mem_metrics["ep"],
                                metric_type=mem_metrics["type"],
                                avail_mem_gb=mem_metrics.get("avail_mem_gb"),
                                mem_usage_gb=mem_metrics.get("mem_usage_gb"),
                                kv_cache_gb=mem_metrics.get("kv_cache_gb"),
                                kv_tokens=mem_metrics.get("kv_tokens"),
                            )
                        )

                    # Extract TP/DP/EP configuration from command line
                    if "--tp-size" in line:
                        tp_match = re.search(r"--tp-size\s+(\d+)", line)
                        dp_match = re.search(r"--dp-size\s+(\d+)", line)
                        ep_match = re.search(r"--ep-size\s+(\d+)", line)

                        if tp_match:
                            config["tp_size"] = int(tp_match.group(1))
                        if dp_match:
                            config["dp_size"] = int(dp_match.group(1))
                        if ep_match:
                            config["ep_size"] = int(ep_match.group(1))

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None

        # Validation: Log if we found no metrics
        total_metrics = len(batches) + len(memory_snapshots)

        if total_metrics == 0:
            logger.warning(
                f"Parsed {filepath} but found no metrics. "
                f"Expected to find lines with DP/TP/EP tags. "
                f"Log format may have changed."
            )

        logger.debug(f"Parsed {filepath}: {len(batches)} batches, {len(memory_snapshots)} memory snapshots")

        return NodeMetrics(
            node_info=node_info,
            batches=batches,
            memory_snapshots=memory_snapshots,
            config=config,
        )

    def get_prefill_nodes(self, nodes: list):
        """Filter for prefill nodes only.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            Filtered list containing only prefill nodes
        """
        return [n for n in nodes if n.is_prefill]

    def get_decode_nodes(self, nodes: list):
        """Filter for decode nodes only.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            Filtered list containing only decode nodes
        """
        return [n for n in nodes if n.is_decode]

    def get_node_count(self, run_path: str) -> tuple[int, int]:
        """Get count of prefill and decode nodes in a run.

        Args:
            run_path: Path to the run directory

        Returns:
            Tuple of (prefill_count, decode_count)
        """
        nodes = self.parse_run_logs(run_path)

        prefill_count = sum(1 for n in nodes if n.is_prefill)
        decode_count = sum(1 for n in nodes if n.is_decode)

        return (prefill_count, decode_count)

    def has_batch_metrics(self, nodes: list) -> bool:
        """Check if any node has batch-level metrics.

        Useful for detecting if decode nodes are logging batch metrics.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            True if any node has batch data
        """
        return any(len(n.batches) > 0 for n in nodes)

    def _serialize_node_metrics(self, nodes: list) -> pd.DataFrame:
        """Serialize NodeMetrics objects to a DataFrame for caching.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            DataFrame with all batch and memory metrics
        """
        rows = []

        for node in nodes:
            node_info = node.node_info
            config = node.config

            # Serialize batch metrics
            for batch in node.batches:
                row = {
                    # Node identification
                    "node": node_info.get("node", ""),
                    "worker_type": node_info.get("worker_type", ""),
                    "worker_id": node_info.get("worker_id", ""),
                    # Config
                    "tp_size": config.get("tp_size"),
                    "dp_size": config.get("dp_size"),
                    "ep_size": config.get("ep_size"),
                    # Metric type
                    "metric_type": "batch",
                    # Batch data
                    "timestamp": batch.timestamp,
                    "dp": batch.dp,
                    "tp": batch.tp,
                    "ep": batch.ep,
                    "batch_type": batch.batch_type,
                    "new_seq": batch.new_seq,
                    "new_token": batch.new_token,
                    "cached_token": batch.cached_token,
                    "token_usage": batch.token_usage,
                    "running_req": batch.running_req,
                    "queue_req": batch.queue_req,
                    "prealloc_req": batch.prealloc_req,
                    "inflight_req": batch.inflight_req,
                    "transfer_req": batch.transfer_req,
                    "preallocated_usage": batch.preallocated_usage,
                    "num_tokens": batch.num_tokens,
                    "input_throughput": batch.input_throughput,
                    "gen_throughput": batch.gen_throughput,
                }
                rows.append(row)

            # Serialize memory metrics
            for mem in node.memory_snapshots:
                row = {
                    # Node identification
                    "node": node_info.get("node", ""),
                    "worker_type": node_info.get("worker_type", ""),
                    "worker_id": node_info.get("worker_id", ""),
                    # Config
                    "tp_size": config.get("tp_size"),
                    "dp_size": config.get("dp_size"),
                    "ep_size": config.get("ep_size"),
                    # Metric type
                    "metric_type": "memory",
                    # Memory data
                    "timestamp": mem.timestamp,
                    "dp": mem.dp,
                    "tp": mem.tp,
                    "ep": mem.ep,
                    "avail_mem_gb": mem.avail_mem_gb,
                    "mem_usage_gb": mem.mem_usage_gb,
                    "kv_cache_gb": mem.kv_cache_gb,
                    "kv_tokens": mem.kv_tokens,
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def _deserialize_node_metrics(self, df: pd.DataFrame) -> list:
        """Deserialize NodeMetrics objects from a cached DataFrame.

        Args:
            df: DataFrame with cached node metrics

        Returns:
            List of NodeMetrics objects
        """
        import time

        from .models import BatchMetrics, MemoryMetrics, NodeMetrics

        start_time = time.time()
        nodes = []

        # Group by node
        for (node_name, worker_type, worker_id), group_df in df.groupby(
            ["node", "worker_type", "worker_id"], dropna=False
        ):
            node_info = {
                "node": node_name,
                "worker_type": worker_type,
                "worker_id": worker_id,
            }

            # Extract config (same for all rows in this node)
            config = {}
            if not group_df.empty:
                first_row = group_df.iloc[0]
                if pd.notna(first_row.get("tp_size")):
                    config["tp_size"] = int(first_row["tp_size"])
                if pd.notna(first_row.get("dp_size")):
                    config["dp_size"] = int(first_row["dp_size"])
                if pd.notna(first_row.get("ep_size")):
                    config["ep_size"] = int(first_row["ep_size"])

            # Separate batch and memory metrics
            batch_df = group_df[group_df["metric_type"] == "batch"]
            memory_df = group_df[group_df["metric_type"] == "memory"]

            # Reconstruct batch metrics using vectorized operations
            batches = []
            if not batch_df.empty:
                # Convert to dict records in bulk (much faster than iterrows)
                batch_records = batch_df.to_dict("records")
                for row in batch_records:
                    batch = BatchMetrics(
                        timestamp=row["timestamp"],
                        dp=int(row["dp"]) if pd.notna(row["dp"]) else 0,
                        tp=int(row["tp"]) if pd.notna(row["tp"]) else 0,
                        ep=int(row["ep"]) if pd.notna(row["ep"]) else 0,
                        batch_type=row["batch_type"],
                        new_seq=int(row["new_seq"]) if pd.notna(row.get("new_seq")) else None,
                        new_token=int(row["new_token"]) if pd.notna(row.get("new_token")) else None,
                        cached_token=(int(row["cached_token"]) if pd.notna(row.get("cached_token")) else None),
                        token_usage=row.get("token_usage") if pd.notna(row.get("token_usage")) else None,
                        running_req=(int(row["running_req"]) if pd.notna(row.get("running_req")) else None),
                        queue_req=int(row["queue_req"]) if pd.notna(row.get("queue_req")) else None,
                        prealloc_req=(int(row["prealloc_req"]) if pd.notna(row.get("prealloc_req")) else None),
                        inflight_req=(int(row["inflight_req"]) if pd.notna(row.get("inflight_req")) else None),
                        transfer_req=(int(row["transfer_req"]) if pd.notna(row.get("transfer_req")) else None),
                        preallocated_usage=(
                            row.get("preallocated_usage") if pd.notna(row.get("preallocated_usage")) else None
                        ),
                        num_tokens=int(row["num_tokens"]) if pd.notna(row.get("num_tokens")) else None,
                        input_throughput=(
                            row.get("input_throughput") if pd.notna(row.get("input_throughput")) else None
                        ),
                        gen_throughput=(row.get("gen_throughput") if pd.notna(row.get("gen_throughput")) else None),
                    )
                    batches.append(batch)

            # Reconstruct memory metrics using vectorized operations
            memory_snapshots = []
            if not memory_df.empty:
                # Convert to dict records in bulk (much faster than iterrows)
                memory_records = memory_df.to_dict("records")
                for row in memory_records:
                    mem = MemoryMetrics(
                        timestamp=row["timestamp"],
                        dp=int(row["dp"]) if pd.notna(row["dp"]) else 0,
                        tp=int(row["tp"]) if pd.notna(row["tp"]) else 0,
                        ep=int(row["ep"]) if pd.notna(row["ep"]) else 0,
                        metric_type="memory",
                        avail_mem_gb=(row.get("avail_mem_gb") if pd.notna(row.get("avail_mem_gb")) else None),
                        mem_usage_gb=(row.get("mem_usage_gb") if pd.notna(row.get("mem_usage_gb")) else None),
                        kv_cache_gb=(row.get("kv_cache_gb") if pd.notna(row.get("kv_cache_gb")) else None),
                        kv_tokens=int(row["kv_tokens"]) if pd.notna(row.get("kv_tokens")) else None,
                    )
                    memory_snapshots.append(mem)

            # Create NodeMetrics object
            node = NodeMetrics(
                node_info=node_info,
                batches=batches,
                memory_snapshots=memory_snapshots,
                config=config,
            )
            nodes.append(node)

        elapsed = time.time() - start_time
        logger.info(f"Deserialized {len(nodes)} nodes in {elapsed:.2f}s")
        return nodes

    # Private helper methods

    def _parse_dp_tp_ep_tag(self, line: str) -> tuple[int | None, int | None, int | None, str | None]:
        """Extract DP, TP, EP indices and timestamp from log line.

        Supports three formats:
        - Full: [2025-11-04 05:31:43 DP0 TP0 EP0]
        - Simple TP: [2025-11-04 07:05:55 TP0] (defaults DP=0, EP=0)
        - Pipeline: [2025-12-08 14:34:44 PP0] (defaults DP=0, EP=0, TP=PP value)

        Args:
            line: Log line to parse

        Returns:
            (dp, tp, ep, timestamp) or (None, None, None, None) if pattern not found
        """
        # Try full format first: DP0 TP0 EP0
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) DP(\d+) TP(\d+) EP(\d+)\]", line)
        if match:
            timestamp, dp, tp, ep = match.groups()
            return int(dp), int(tp), int(ep), timestamp

        # Try simple format: TP0 only (1P4D style)
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(\d+)\]", line)
        if match:
            timestamp, tp = match.groups()
            return 0, int(tp), 0, timestamp  # Default DP=0, EP=0

        # Try pipeline parallelism format: PP0 (prefill with PP)
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) PP(\d+)\]", line)
        if match:
            timestamp, pp = match.groups()
            return 0, int(pp), 0, timestamp  # Map PP to TP slot, default DP=0, EP=0

        return None, None, None, None

    def _parse_prefill_batch_line(self, line: str) -> dict | None:
        """Parse prefill batch log line for metrics.

        Example line:
        [2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18, #new-token: 16384,
        #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0,
        #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 0.00,
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Prefill batch" not in line:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep, "type": "prefill"}

        # Extract metrics using regex
        patterns = {
            "new_seq": r"#new-seq:\s*(\d+)",
            "new_token": r"#new-token:\s*(\d+)",
            "cached_token": r"#cached-token:\s*(\d+)",
            "token_usage": r"token usage:\s*([\d.]+)",
            "running_req": r"#running-req:\s*(\d+)",
            "queue_req": r"#queue-req:\s*(\d+)",
            "prealloc_req": r"#prealloc-req:\s*(\d+)",
            "inflight_req": r"#inflight-req:\s*(\d+)",
            "input_throughput": r"input throughput \(token/s\):\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                metrics[key] = float(value) if "." in value else int(value)

        return metrics

    def _parse_decode_batch_line(self, line: str) -> dict | None:
        """Parse decode batch log line for metrics.

        Example line:
        [2025-11-04 05:32:32 DP31 TP31 EP31] Decode batch, #running-req: 7, #token: 7040,
        token usage: 0.00, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 0,
        #retracted-req: 0, cuda graph: True, gen throughput (token/s): 6.73, #queue-req: 0,
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Decode batch" not in line:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep, "type": "decode"}

        # Extract metrics using regex
        patterns = {
            "running_req": r"#running-req:\s*(\d+)",
            "num_tokens": r"#token:\s*(\d+)",
            "token_usage": r"token usage:\s*([\d.]+)",
            "preallocated_usage": r"pre-allocated usage:\s*([\d.]+)",
            "prealloc_req": r"#prealloc-req:\s*(\d+)",
            "transfer_req": r"#transfer-req:\s*(\d+)",
            "queue_req": r"#queue-req:\s*(\d+)",
            "gen_throughput": r"gen throughput \(token/s\):\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                metrics[key] = float(value) if "." in value else int(value)

        return metrics

    def _parse_memory_line(self, line: str) -> dict | None:
        """Parse memory-related log lines.

        Examples:
        [2025-11-04 05:27:13 DP0 TP0 EP0] Load weight end. type=DeepseekV3ForCausalLM,
        dtype=torch.bfloat16, avail mem=75.11 GB, mem usage=107.07 GB.

        [2025-11-04 05:27:13 DP0 TP0 EP0] KV Cache is allocated. #tokens: 524288, KV size: 17.16 GB
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None:
            return None

        metrics = {
            "timestamp": timestamp,
            "dp": dp,
            "tp": tp,
            "ep": ep,
        }

        # Parse available memory
        avail_match = re.search(r"avail mem=([\d.]+)\s*GB", line)
        if avail_match:
            metrics["avail_mem_gb"] = float(avail_match.group(1))
            metrics["type"] = "memory"

        # Parse memory usage
        usage_match = re.search(r"mem usage=([\d.]+)\s*GB", line)
        if usage_match:
            metrics["mem_usage_gb"] = float(usage_match.group(1))
            metrics["type"] = "memory"

        # Parse KV cache size
        kv_match = re.search(r"KV size:\s*([\d.]+)\s*GB", line)
        if kv_match:
            metrics["kv_cache_gb"] = float(kv_match.group(1))
            metrics["type"] = "kv_cache"

        # Parse token count for KV cache
        token_match = re.search(r"#tokens:\s*(\d+)", line)
        if token_match:
            metrics["kv_tokens"] = int(token_match.group(1))

        return metrics if "type" in metrics else None

    def _extract_node_info_from_filename(self, filename: str) -> dict | None:
        """Extract node name and worker info from filename.

        Example: watchtower-navy-cn01_prefill_w0.err or r02-p01-dgx-c11_prefill_w0.out
        Returns: {'node': 'watchtower-navy-cn01', 'worker_type': 'prefill', 'worker_id': 'w0'}
        """
        # Use greedy match for node name up to _(prefill|decode|frontend)_
        match = re.match(r"(.+)_(prefill|decode|frontend)_([^.]+)\.(err|out)", os.path.basename(filename))
        if match:
            return {
                "node": match.group(1),
                "worker_type": match.group(2),
                "worker_id": match.group(3),
            }
        return None


# Standalone helper function for visualizations
def get_node_label(node_data: dict) -> str:
    """Generate a display label for a node with its configuration.

    Example: "3320 | 6P1D | 24/32 | cn01-p-w0"
    """
    node_info = node_data["node_info"]
    run_metadata = node_data.get("run_metadata", {})

    # Clean node name
    node_name = (
        node_info["node"].replace("watchtower-navy-", "").replace("watchtower-aqua-", "").replace("inkwell-copper-", "")
    )
    worker_type = node_info["worker_type"][0].lower()  # 'p' for prefill, 'd' for decode
    worker_id = node_info["worker_id"]
    node_short = f"{node_name}-{worker_type}-w{worker_id}"

    # If we have run metadata, use it for context
    if run_metadata:
        job_id = run_metadata.get("job_id", "")
        is_aggregated = run_metadata.get("is_aggregated", False)
        gpus_per_node = run_metadata.get("gpus_per_node", 0)

        if is_aggregated:
            agg_workers = run_metadata.get("agg_workers", 0)
            agg_nodes = run_metadata.get("agg_nodes", 0)
            total_gpus = agg_nodes * gpus_per_node
            # Format: id | xA | total_gpus | node
            return f"{job_id} | {agg_workers}A | {total_gpus} GPUs | {node_short}"
        else:
            prefill_workers = run_metadata.get("prefill_workers", 0)
            decode_workers = run_metadata.get("decode_workers", 0)
            prefill_nodes = run_metadata.get("prefill_nodes", 0)
            decode_nodes = run_metadata.get("decode_nodes", 0)

            prefill_gpus = prefill_nodes * gpus_per_node
            decode_gpus = decode_nodes * gpus_per_node

            # Format: id | xPyD | prefill_gpus/decode_gpus | node
            return f"{job_id} | {prefill_workers}P{decode_workers}D | {prefill_gpus}/{decode_gpus} | {node_short}"
    else:
        # Fallback for old code without metadata
        return node_short
