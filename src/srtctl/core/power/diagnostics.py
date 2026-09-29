# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort timing output that cannot hold the sample writer lock."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_MAX_PENDING_RECORDS = 128


class ScrapeDiagnostics:
    """Bounded queue; only the daemon performs potentially blocking file I/O."""

    def __init__(self, path: Path):
        self._path = path
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(_MAX_PENDING_RECORDS)
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, name="PowerScrapeDiagnostics", daemon=True)
        self._thread.start()

    def record(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._stopped.is_set():
                return
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped += 1

    def close(self, deadline: float) -> None:
        with self._lock:
            self._stopped.set()
            # A full queue drains naturally after the stop flag is set.
            with suppress(queue.Full):
                self._queue.put_nowait(None)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _run(self) -> None:
        try:
            with self._path.open("w", encoding="utf-8") as writer:
                while True:
                    record = self._queue.get()
                    if record is None:
                        break
                    writer.write(json.dumps(record, separators=(",", ":")) + "\n")
                    writer.flush()
                    if self._stopped.is_set() and self._queue.empty():
                        break
                writer.write(json.dumps({"event": "diagnostic_summary", "dropped_records": self._dropped}) + "\n")
        except OSError:
            self._stopped.set()
            logger.warning("Power scrape diagnostics unavailable", exc_info=True)
