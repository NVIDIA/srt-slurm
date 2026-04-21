# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Cache manager for storing processed data in parquet format.

Provides efficient disk-based caching to avoid re-parsing log files and JSON data
every time the Streamlit app loads.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages parquet-based caching for benchmark data."""

    def __init__(self, run_dir: str):
        """Initialize cache manager for a specific run directory.

        Args:
            run_dir: Path to the run directory (e.g., "3667_1P_1D_20251110_192145")
        """
        self.run_dir = Path(run_dir)
        self.cache_dir = self.run_dir / "cached_assets"
        self.cache_dir.mkdir(exist_ok=True)

        self.metadata_file = self.cache_dir / "cache_metadata.json"

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file contents for cache validation.

        Args:
            file_path: Path to file

        Returns:
            MD5 hash of file contents
        """
        if not file_path.exists():
            return ""

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_files_hash(self, file_patterns: list[str]) -> dict[str, str]:
        """Get hashes for multiple files matching patterns.

        Args:
            file_patterns: List of glob patterns (e.g., ["*.err", "*.json"])

        Returns:
            Dictionary mapping file paths to their hashes
        """
        hashes = {}
        for pattern in file_patterns:
            for file_path in self.run_dir.glob(pattern):
                if file_path.is_file():
                    hashes[str(file_path.relative_to(self.run_dir))] = self._get_file_hash(file_path)
        return hashes

    def _load_metadata(self) -> dict[str, Any]:
        """Load cache metadata from disk.

        Returns:
            Metadata dictionary, or empty dict if doesn't exist
        """
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save cache metadata to disk.

        Args:
            metadata: Metadata dictionary to save
        """
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def is_cache_valid(self, cache_name: str, source_patterns: list[str]) -> bool:
        """Check if cached data is valid (source files haven't changed).

        Args:
            cache_name: Name of the cache (e.g., "benchmark_run", "node_metrics")
            source_patterns: Glob patterns for source files (e.g., ["*.json", "vllm_*/"])

        Returns:
            True if cache is valid, False otherwise
        """
        cache_file = self.cache_dir / f"{cache_name}.parquet"
        if not cache_file.exists():
            return False

        # Check metadata
        metadata = self._load_metadata()
        if cache_name not in metadata:
            return False

        # Compare file hashes
        current_hashes = self._get_files_hash(source_patterns)
        cached_hashes = metadata[cache_name].get("source_hashes", {})

        return current_hashes == cached_hashes

    def save_to_cache(self, cache_name: str, data: pd.DataFrame | list[dict], source_patterns: list[str]) -> None:
        """Save data to parquet cache.

        Args:
            cache_name: Name of the cache (e.g., "benchmark_run", "node_metrics")
            data: Data to cache (DataFrame or list of dicts)
            source_patterns: Glob patterns for source files
        """
        # Convert to DataFrame if needed
        if isinstance(data, list):
            if not data:
                # Save empty DataFrame
                df = pd.DataFrame()
            else:
                df = pd.DataFrame(data)
        else:
            df = data

        # Save to parquet
        cache_file = self.cache_dir / f"{cache_name}.parquet"
        df.to_parquet(cache_file, index=False, compression="snappy")

        # Update metadata
        metadata = self._load_metadata()
        metadata[cache_name] = {
            "source_hashes": self._get_files_hash(source_patterns),
            "cached_at": pd.Timestamp.now().isoformat(),
            "row_count": len(df),
        }
        self._save_metadata(metadata)

        logger.info(f"Cached {len(df)} rows to {cache_file.name}")

    def load_from_cache(self, cache_name: str) -> pd.DataFrame | None:
        """Load data from parquet cache.

        Args:
            cache_name: Name of the cache (e.g., "benchmark_run", "node_metrics")

        Returns:
            DataFrame if cache exists, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_name}.parquet"
        if not cache_file.exists():
            return None

        try:
            df = pd.read_parquet(cache_file)
            logger.info(f"Loaded {len(df)} rows from {cache_file.name}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file.name}: {e}")
            return None

    def invalidate_cache(self, cache_name: str | None = None) -> None:
        """Invalidate cache (delete cached files).

        Args:
            cache_name: Specific cache to invalidate, or None to invalidate all
        """
        if cache_name:
            # Invalidate specific cache
            cache_file = self.cache_dir / f"{cache_name}.parquet"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Invalidated cache: {cache_name}")

            # Remove from metadata
            metadata = self._load_metadata()
            if cache_name in metadata:
                del metadata[cache_name]
                self._save_metadata(metadata)
        else:
            # Invalidate all caches
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info("Invalidated all caches")
