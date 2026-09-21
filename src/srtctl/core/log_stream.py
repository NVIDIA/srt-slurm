# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort mirroring of a growing log without waiting for a newline."""

from __future__ import annotations

import codecs
import logging
import os
import sys
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


class LogOutputStreamer:
    """Copy available log bytes to stdout when the process monitor polls.

    Slurm remains the log's writer. Reads are bounded by the size observed on
    each poll, so a continuously writing client cannot prevent cancellation.
    """

    def __init__(self, path: Path, output: TextIO | None = None) -> None:
        self.path = path
        self.output = sys.stdout if output is None else output
        self.offset = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.disabled = False

    def poll(self, *, final: bool = False) -> None:
        """Flush new output; tolerate the log not existing until srun starts."""
        if self.disabled:
            return
        try:
            try:
                log = self.path.open("rb")
            except FileNotFoundError:
                log = None
            if log is not None:
                with log:
                    size = os.fstat(log.fileno()).st_size
                    if size < self.offset:
                        self.offset = 0
                        self.decoder.reset()
                    log.seek(self.offset)
                    remaining = size - self.offset
                    while remaining > 0:
                        chunk = log.read(min(remaining, 65536))
                        if not chunk:
                            break
                        self.offset += len(chunk)
                        remaining -= len(chunk)
                        self.output.write(self.decoder.decode(chunk))
            if final:
                self.output.write(self.decoder.decode(b"", final=True))
                self.disabled = True
            self.output.flush()
        except (OSError, ValueError) as exc:
            # An unavailable log or closed stdout must not fail the benchmark.
            self.disabled = True
            logger.warning("Stopped streaming benchmark output from %s: %s", self.path, exc)
