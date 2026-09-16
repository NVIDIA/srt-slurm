# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite store behind the native status collector."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from srtctl.contract import JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_name TEXT NOT NULL,
    cluster TEXT,
    recipe TEXT,
    status TEXT NOT NULL,
    stage TEXT,
    message TEXT,
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    exit_code INTEGER,
    logs_url TEXT,
    benchmark_results TEXT,
    artifacts TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    message TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_cluster ON jobs(cluster);
CREATE INDEX IF NOT EXISTS idx_jobs_submitted_at ON jobs(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id_id ON job_events(job_id, id);
"""

# Columns stored as JSON text and decoded on read.
_JSON_COLUMNS = ("benchmark_results", "artifacts", "metadata")
# PUT fields copied verbatim when present. Fixed allowlist: these names are
# interpolated into SQL below, never anything from the request.
_SCALAR_UPDATE_COLUMNS = ("stage", "message", "started_at", "completed_at", "exit_code", "logs_url")
_EVENT_COLUMNS = "id, job_id, status, stage, message, created_at"


def now_iso() -> str:
    """Current UTC time in the ISO 8601 ``Z`` form the reporter uses."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def placeholder_name(job_id: str) -> str:
    """The job_name a row gets when a PUT arrives for a job the collector never saw a POST for."""
    return f"job-{job_id}"


def _decode_job(row: sqlite3.Row) -> dict[str, Any]:
    job = dict(row)
    for key in _JSON_COLUMNS:
        job[key] = json.loads(job[key]) if job[key] else None
    return job


def _merge_json(existing: str | None, patch: dict) -> str:
    merged = json.loads(existing) if existing else {}
    merged.update(patch)
    return json.dumps(merged)


@dataclass(frozen=True)
class StatusStore:
    """One SQLite file holding every job the collector has seen.

    A connection is opened per operation so the threaded HTTP server can call
    in from any request thread. WAL mode lets pollers of the event feeds read
    while a sweep is writing its updates.
    """

    db_path: Path

    def init(self) -> None:
        """Create the file, its parent directory, and the schema. Safe to repeat."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    # ------------------------------------------------------------------ writes

    def create_job(
        self,
        job_id: str,
        job_name: str,
        *,
        cluster: str | None = None,
        recipe: str | None = None,
        submitted_at: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Insert a job in ``submitted`` state (``POST /api/jobs``).

        Idempotent on status: a repeated POST never rewinds a job that has moved
        on. It does complete the row's identity, though: a placeholder name (the
        row was created by a PUT because the submit-time POST was lost) is
        replaced, a null ``cluster`` or ``recipe`` is filled, ``submitted_at`` is
        moved earlier to the real submit time (never later: a placeholder's value
        is the start time, and a late POST stamped "now" must not reset a running
        job's elapsed time), and ``metadata`` is merged. Existing non-null identity
        is never overwritten.
        Returns ``{"job_id", "status", "created", "backfilled"}``.
        """
        now = now_iso()
        submitted = submitted_at or now
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT status, job_name, cluster, recipe, metadata, submitted_at FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if existing is not None:
                fields: dict[str, Any] = {}
                if existing["job_name"] == placeholder_name(job_id):
                    fields["job_name"] = job_name
                if submitted < existing["submitted_at"]:  # ISO-8601 Z strings order lexically
                    fields["submitted_at"] = submitted
                if existing["cluster"] is None and cluster:
                    fields["cluster"] = cluster
                if existing["recipe"] is None and recipe:
                    fields["recipe"] = recipe
                if metadata:
                    fields["metadata"] = _merge_json(existing["metadata"], metadata)
                if fields:
                    fields["updated_at"] = now
                    assignments = ", ".join(f"{column} = ?" for column in fields)
                    conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", (*fields.values(), job_id))
                return {
                    "job_id": job_id,
                    "status": existing["status"],
                    "created": False,
                    "backfilled": bool(fields),
                }
            conn.execute(
                """
                INSERT INTO jobs (job_id, job_name, cluster, recipe, status, submitted_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_name,
                    cluster,
                    recipe,
                    JobStatus.SUBMITTED.value,
                    submitted,
                    now,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.execute(
                "INSERT INTO job_events (job_id, status, created_at) VALUES (?, ?, ?)",
                (job_id, JobStatus.SUBMITTED.value, submitted),
            )
        return {"job_id": job_id, "status": JobStatus.SUBMITTED.value, "created": True, "backfilled": False}

    def update_job(self, job_id: str, update: dict[str, Any]) -> dict[str, Any]:
        """Apply a validated ``PUT /api/jobs/{job_id}`` body.

        An unknown job gets a placeholder row, so a sweep whose submit-time POST
        was lost (collector down, network blip) still lands every later update.
        The reporter repeats the job's identity in the started report's
        ``metadata`` (``job_name``, ``cluster``); a placeholder name and a null
        cluster are filled from it, so a lost POST costs nothing visible.
        ``artifacts`` and ``metadata`` are merged key-wise into the stored dicts;
        every other field overwrites.

        An event is appended whenever ``(status, stage, message)`` differs from
        the job's last event. That keeps same-status transitions such as
        ``frontend / Starting frontend`` -> ``frontend / Inference endpoint ready``
        while pure artifact or metadata patches stay silent.
        Returns ``{"job_id", "status", "event"}`` where ``event`` says whether one
        was appended.
        """
        status = update["status"]
        stage = update.get("stage")
        message = update.get("message")
        now = update.get("updated_at") or now_iso()

        identity = update.get("metadata") or {}
        identity_name = identity.get("job_name") or None
        identity_cluster = identity.get("cluster") or None

        with self._transaction() as conn:
            row = conn.execute(
                "SELECT artifacts, metadata, job_name, cluster FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

            fields: dict[str, Any] = {"status": status, "updated_at": now}
            for key in _SCALAR_UPDATE_COLUMNS:
                if update.get(key) is not None:
                    fields[key] = update[key]
            if update.get("benchmark_results") is not None:
                fields["benchmark_results"] = json.dumps(update["benchmark_results"])
            for key in ("artifacts", "metadata"):
                if update.get(key) is not None:
                    fields[key] = _merge_json(row[key] if row is not None else None, update[key])
            # Fill identity a lost POST would have supplied; never overwrite a real name or cluster.
            if identity_name and (row is None or row["job_name"] == placeholder_name(job_id)):
                fields["job_name"] = identity_name
            if identity_cluster and (row is None or row["cluster"] is None):
                fields["cluster"] = identity_cluster

            if row is None:
                fields["job_id"] = job_id
                fields.setdefault("job_name", placeholder_name(job_id))
                fields["submitted_at"] = update.get("started_at") or now
                columns = ", ".join(fields)
                marks = ", ".join("?" * len(fields))
                conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({marks})", tuple(fields.values()))
            else:
                assignments = ", ".join(f"{column} = ?" for column in fields)
                conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", (*fields.values(), job_id))

            last = conn.execute(
                "SELECT status, stage, message FROM job_events WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            event = last is None or (last["status"], last["stage"], last["message"]) != (status, stage, message)
            if event:
                conn.execute(
                    "INSERT INTO job_events (job_id, status, stage, message, created_at) VALUES (?, ?, ?, ?, ?)",
                    (job_id, status, stage, message, now),
                )

        return {"job_id": job_id, "status": status, "event": event}

    def delete_job(self, job_id: str) -> bool:
        """Remove a job and its events. Returns False when it did not exist."""
        with self._transaction() as conn:
            deleted = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,)).rowcount
            conn.execute("DELETE FROM job_events WHERE job_id = ?", (job_id,))
        return deleted > 0

    # ------------------------------------------------------------------- reads

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """One job with its full ordered event history, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            events = conn.execute(
                f"SELECT {_EVENT_COLUMNS} FROM job_events WHERE job_id = ? ORDER BY id",
                (job_id,),
            ).fetchall()
        job = _decode_job(row)
        job["events"] = [dict(event) for event in events]
        return job

    def list_jobs(
        self,
        *,
        page: int = 1,
        per_page: int = 50,
        status: str | None = None,
        cluster: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Newest-first page of jobs plus the total matching the filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if cluster:
            clauses.append("cluster = ?")
            params.append(cluster)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM jobs {where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY submitted_at DESC, job_id DESC LIMIT ? OFFSET ?",
                [*params, per_page, (page - 1) * per_page],
            ).fetchall()
        return [_decode_job(row) for row in rows], total

    def list_events(self, *, after: int = 0, limit: int = 100, job_id: str | None = None) -> list[dict[str, Any]]:
        """Events with ``id > after`` in insertion order, optionally for one job.

        ``id`` is the cursor: pass the last id you saw as ``after`` to resume.
        """
        query = f"SELECT {_EVENT_COLUMNS} FROM job_events WHERE id > ?"
        params: list[Any] = [after]
        if job_id is not None:
            query += " AND job_id = ?"
            params.append(job_id)
        query += " ORDER BY id LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
