# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured per-submission reporting for ``srtctl apply``."""

import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from srtctl.contract.failures import RECORD_SCHEMA_VERSION, FailureCode, SrtFailure, as_failure


@dataclass
class _Scope:
    identity: dict[str, Any]
    index: int | None = None
    job: dict[str, Any] = field(default_factory=dict)


class ApplyReport:
    """Collects accepted submissions and failures of one apply run and emits them as JSON lines."""

    def __init__(
        self, *, active: bool = False, json_out: TextIO | None = None, failures_file: Path | None = None
    ) -> None:
        self.active = active
        self.json_out = json_out
        self.failures_file = failures_file
        self.submissions: list[dict[str, Any]] = []
        self._scopes: list[_Scope] = []
        self._accepted = 0
        self._failures: list[SrtFailure] = []
        self._submission_count = 0

    @contextmanager
    def item(self, **identity: Any) -> Generator[None]:
        """Scope one submission; a failure leaving it is reported with the scope's identity."""
        if not self.active:
            yield
            return
        scope = _Scope(identity)
        self._scopes.append(scope)
        try:
            yield
        except Exception as exc:
            failure = as_failure(exc)
            if not failure.reported:
                failure.identity = {**self._identity(), **scope.job}
                self.fail(failure)
            if failure is exc:
                raise
            raise failure from exc
        finally:
            self._scopes.pop()

    def accepted(self, slurm_job_id: str, job_name: str, output_dir: Path) -> None:
        """Record and emit that Slurm accepted a job for the current scope."""
        job = {"slurm_job_id": slurm_job_id, "output_dir": str(output_dir)}
        if self._scopes:
            self._scopes[-1].job = job
        self._accepted += 1
        self._emit(
            {
                "schema_version": RECORD_SCHEMA_VERSION,
                "status": "accepted",
                **self._identity(),
                **job,
                "job_name": job_name,
            }
        )

    def submitted(self, record: dict[str, Any]) -> None:
        """Record and emit the final per-submission record."""
        record = {"schema_version": RECORD_SCHEMA_VERSION, **self._identity(), **record}
        self.submissions.append(record)
        self._emit(record)

    def fail(self, exc: BaseException) -> SrtFailure:
        """Report a command-level failure that has no submission identity."""
        failure = as_failure(exc)
        if not failure.reported:
            self._emit(failure.record(), failure=True)
            failure.reported = True
            self._failures.append(failure)
        return failure

    def finish(self) -> None:
        """Emit the no-submissions record when nothing was accepted and nothing failed."""
        if not self._accepted and not self._failures:
            self._emit({"schema_version": RECORD_SCHEMA_VERSION, "status": "no-submissions"})

    @property
    def exit_code(self) -> int:
        """Process exit code: 3 failed after a job was accepted, 2 invalid invocation, 1 failure, 0 success."""
        if self._failures and self._accepted:
            return 3
        if any(failure.code is FailureCode.INVOCATION_INVALID for failure in self._failures):
            return 2
        return 1 if self._failures else 0

    def _identity(self) -> dict[str, Any]:
        if not self._scopes:
            return {}
        scope = self._scopes[-1]
        if scope.index is None:
            self._submission_count += 1
            scope.index = self._submission_count
        identity: dict[str, Any] = {"submission_index": scope.index}
        for outer in self._scopes:
            identity.update(outer.identity)
        return identity

    def _emit(self, record: dict[str, Any], *, failure: bool = False) -> None:
        line = json.dumps(record, default=str)
        if self.json_out is not None:
            try:
                self.json_out.write(line + "\n")
                self.json_out.flush()
            except OSError as e:
                print(f"Could not write record to stdout: {e}", file=sys.stderr)
        if failure and self.failures_file is not None:
            try:
                self.failures_file.parent.mkdir(parents=True, exist_ok=True)
                with self.failures_file.open("a") as handle:
                    handle.write(line + "\n")
            except OSError as e:
                print(f"Could not write failure record to {self.failures_file}: {e}", file=sys.stderr)
