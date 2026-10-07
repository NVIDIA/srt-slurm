# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite store behind the native status collector."""

from __future__ import annotations

import codecs
import json
import sqlite3
from collections.abc import Generator
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

-- Streamed log and metric output. A chunk is keyed by where it sits in its file,
-- so a resend after a lost response is a no-op.
CREATE TABLE IF NOT EXISTS job_logs (
    job_id TEXT NOT NULL,
    file TEXT NOT NULL,
    offset INTEGER NOT NULL,
    size INTEGER NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, file, offset)
);

-- Immutable Tachometer segments, stored verbatim; readers decode them.
CREATE TABLE IF NOT EXISTS job_captures (
    job_id TEXT NOT NULL, file TEXT NOT NULL,
    total INTEGER NOT NULL, next_offset INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
    PRIMARY KEY (job_id, file)
);
CREATE TABLE IF NOT EXISTS capture_chunks (
    job_id TEXT NOT NULL, file TEXT NOT NULL, offset INTEGER NOT NULL, data BLOB NOT NULL,
    PRIMARY KEY (job_id, file, offset)
);
CREATE TABLE IF NOT EXISTS job_log_ends (
    job_id TEXT NOT NULL, file TEXT NOT NULL, end INTEGER NOT NULL,
    PRIMARY KEY (job_id, file)
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


class LogChunkConflict(ValueError):
    """An incoming chunk disagrees with bytes already stored for the same file."""


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
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
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
            conn.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
            for table in ("job_captures", "capture_chunks", "job_log_ends"):
                conn.execute(f"DELETE FROM {table} WHERE job_id = ?", (job_id,))
        return deleted > 0

    def append_logs(self, job_id: str, chunks: list[dict[str, Any]]) -> int:
        """Store chunks atomically, accepting exact retries and rejecting conflicting ranges."""
        now = now_iso()
        with self._transaction() as conn:
            stored = 0
            for chunk in chunks:
                overlapping = conn.execute(
                    """SELECT offset, size, data FROM job_logs
                    WHERE job_id = ? AND file = ? AND offset < ? ORDER BY offset DESC LIMIT 1""",
                    (job_id, chunk["file"], chunk["offset"] + chunk["size"]),
                ).fetchone()
                if overlapping is not None and overlapping["offset"] + overlapping["size"] > chunk["offset"]:
                    if all(overlapping[key] == chunk[key] for key in ("offset", "size", "data")):
                        continue
                    raise LogChunkConflict(f"Conflicting log chunk for {chunk['file']} at offset {chunk['offset']}")
                conn.execute(
                    "INSERT INTO job_logs (job_id, file, offset, size, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (job_id, chunk["file"], chunk["offset"], chunk["size"], chunk["data"], now),
                )
                stored += 1
            return stored

    def append_raw_log(self, job_id: str, file: str, offset: int, data: bytes, *, final: bool = False) -> int:
        """Keep raw bytes intact; UTF-8 decoding happens when the API is read."""
        stored = (
            self.append_logs(job_id, [{"file": file, "offset": offset, "size": len(data), "data": data}]) if data else 0
        )
        if final:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO job_log_ends VALUES (?, ?, ?) "
                    "ON CONFLICT(job_id, file) DO UPDATE SET end = MAX(end, excluded.end)",
                    (job_id, file, offset + len(data)),
                )
        return stored

    def append_capture(self, job_id: str, file: str, offset: int, total: int, data: bytes) -> dict[str, Any]:
        """Store one chunk of an immutable segment verbatim; nothing is decoded on upload.

        Chunks must arrive in order. A retry of a stored chunk is accepted only if
        its bytes are identical, so a lost response can be resent safely.
        """
        end = offset + len(data)
        with self._transaction() as conn:
            manifest = conn.execute(
                "SELECT total, next_offset FROM job_captures WHERE job_id = ? AND file = ?", (job_id, file)
            ).fetchone()
            if manifest is None:
                if offset != 0:
                    raise LogChunkConflict("Capture upload must start at offset 0")
                conn.execute("INSERT INTO job_captures VALUES (?, ?, ?, 0, ?)", (job_id, file, total, now_iso()))
                next_offset = 0
            elif manifest["total"] != total:
                raise LogChunkConflict("Capture total changed")
            else:
                next_offset = manifest["next_offset"]
            if offset < next_offset:
                stored = conn.execute(
                    "SELECT data FROM capture_chunks WHERE job_id = ? AND file = ? AND offset = ?",
                    (job_id, file, offset),
                ).fetchone()
                if stored is None or stored["data"] != data:
                    raise LogChunkConflict("Capture retry differs from stored bytes")
            elif offset == next_offset:
                conn.execute("INSERT INTO capture_chunks VALUES (?, ?, ?, ?)", (job_id, file, offset, data))
                conn.execute(
                    "UPDATE job_captures SET next_offset = ?, updated_at = ? WHERE job_id = ? AND file = ?",
                    (end, now_iso(), job_id, file),
                )
            else:
                raise LogChunkConflict("Capture chunks must be contiguous")
        return {"job_id": job_id, "next_offset": end, "complete": end == total}

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

    def list_captures(self, job_id: str) -> list[dict[str, Any]]:
        """Every Tachometer segment of a job; only complete ones can be downloaded."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file, total AS size, next_offset = total AS complete, updated_at
                FROM job_captures WHERE job_id = ? ORDER BY file
                """,
                (job_id,),
            ).fetchall()
        return [{**dict(row), "complete": bool(row["complete"])} for row in rows]

    def read_capture(self, job_id: str, file: str) -> bytes | None:
        """The exact bytes of a complete segment, or None while it is missing or partial."""
        with self._connect() as conn:
            manifest = conn.execute(
                "SELECT total, next_offset FROM job_captures WHERE job_id = ? AND file = ?", (job_id, file)
            ).fetchone()
            if manifest is None or manifest["next_offset"] != manifest["total"]:
                return None
            rows = conn.execute(
                "SELECT data FROM capture_chunks WHERE job_id = ? AND file = ? ORDER BY offset", (job_id, file)
            )
            return b"".join(row["data"] for row in rows)

    def list_log_files(self, job_id: str) -> list[dict[str, Any]]:
        """Every streamed file of a job with the bytes received so far."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file, MAX(offset + size) AS size, MAX(created_at) AS updated_at
                FROM job_logs WHERE job_id = ? GROUP BY file ORDER BY file
                """,
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def read_log(self, job_id: str, file: str, *, offset: int = 0, max_bytes: int = 1 << 20) -> tuple[str, int]:
        """Contiguous content of ``file`` from ``offset``, stopping at a gap or after ``max_bytes``.

        Returns ``(data, next_offset)``. ``offset`` is a chunk boundary, i.e. 0 or a
        ``next_offset`` from an earlier read.
        """
        parts: list[str] = []
        cursor = offset
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with self._connect() as conn:
            ending = conn.execute(
                "SELECT end FROM job_log_ends WHERE job_id = ? AND file = ?", (job_id, file)
            ).fetchone()
            rows = conn.execute(
                "SELECT offset, size, data FROM job_logs WHERE job_id = ? AND file = ? AND offset + size > ? ORDER BY offset",
                (job_id, file, offset),
            )
            for row in rows:
                if row["offset"] > cursor or cursor - offset >= max_bytes:
                    break
                if isinstance(row["data"], bytes):
                    raw = row["data"][cursor - row["offset"] : cursor - row["offset"] + max_bytes - (cursor - offset)]
                    parts.append(decoder.decode(raw))
                    cursor += len(raw)
                else:
                    # Legacy JSON chunks use source byte sizes, which can differ
                    # from the UTF-8 size of already-decoded replacement characters.
                    if row["offset"] != cursor:
                        break
                    parts.append(row["data"])
                    cursor += row["size"]
            if ending is not None and cursor == ending["end"]:
                parts.append(decoder.decode(b"", final=True))
            else:
                cursor -= len(decoder.getstate()[0])
        return "".join(parts), cursor
