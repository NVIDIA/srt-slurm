#!/usr/bin/env python3
"""Apply the f946-only Mooncake rank-local RDMA HCA selection patch."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


SITE_PACKAGES = Path(
    os.environ.get("VLLM_SITE_PACKAGES", "/usr/local/lib/python3.12/dist-packages")
)
MOONCAKE_WORKER = SITE_PACKAGES / (
    "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
)
EXPECTED_SHA256 = "a10b3b428c8fd0072a536a0e726ff961d2eb4f455bf88cd91aa9f816fdaee4dc"
MARKER = "DP/TP-rank-local Mooncake HCA selection"
ANCHOR = """        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_config()
"""
REPLACEMENT = """        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_config()
        rank_local_hcas = [
            value.strip()
            for value in os.getenv(
                "VLLM_MOONCAKE_RANK_LOCAL_HCA_MAP", ""
            ).split(",")
            if value.strip()
        ]
        if rank_local_hcas:
            gpu_rank_for_hca = self.dp_rank * self.tp_size + self.tp_rank
            selected_hca = rank_local_hcas[gpu_rank_for_hca % len(rank_local_hcas)]
            store_config.device_name = selected_hca.split(":", 1)[0]
            logger.info(
                "DP/TP-rank-local Mooncake HCA selection: dp_rank=%d tp_rank=%d "
                "gpu_rank=%d device_name=%s",
                self.dp_rank,
                self.tp_rank,
                gpu_rank_for_hca,
                store_config.device_name,
            )
"""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    text = MOONCAKE_WORKER.read_text()
    if MARKER in text:
        print(f"f946 Mooncake rank-local HCA patch already present: {MOONCAKE_WORKER}")
        return

    actual_sha = digest(text)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected source {MOONCAKE_WORKER}: "
            f"sha256={actual_sha}, expected={EXPECTED_SHA256}"
        )
    if text.count(ANCHOR) != 1:
        raise RuntimeError(
            f"expected exactly one patch anchor in {MOONCAKE_WORKER}, "
            f"found {text.count(ANCHOR)}"
        )

    updated = text.replace(ANCHOR, REPLACEMENT, 1)
    compile(updated, str(MOONCAKE_WORKER), "exec")
    MOONCAKE_WORKER.write_text(updated)
    print(f"applied f946 Mooncake rank-local HCA patch: {MOONCAKE_WORKER}")


if __name__ == "__main__":
    main()
