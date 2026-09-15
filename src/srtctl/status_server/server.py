# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP side of the native status collector (``srtctl status-server``).

Implements the Status API in ``docs/status-api-spec.md`` on the standard
library ``ThreadingHTTPServer`` so it ships with the base install: no ASGI
stack, no extra dependency group. Bodies are validated with the contract
models in ``srtctl.contract``; responses are built from the same models so
the server can never drift from what ``StatusReporter`` sends.

Run it on a login node (or any host the compute nodes can reach) and point
recipes or ``srtslurm.yaml`` at it::

    srtctl status-server --host 0.0.0.0 --port 8080

    reporting:
      status:
        endpoint: "http://login-node:8080"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from srtctl.contract import (
    EventFeedResponse,
    JobCreatePayload,
    JobDetail,
    JobEventListResponse,
    JobListResponse,
    JobResponse,
    JobStage,
    JobStatus,
    JobSummary,
    JobUpdatePayload,
)
from srtctl.status_server.store import StatusStore

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_DB_PATH = Path("~/.local/state/srtctl/status.db")

_JOB_ROUTE = re.compile(r"^/api/jobs/(?P<job_id>[^/]+)$")
_JOB_EVENTS_ROUTE = re.compile(r"^/api/jobs/(?P<job_id>[^/]+)/events$")

Response = tuple[HTTPStatus, dict[str, Any]]


class ApiError(Exception):
    """An HTTP error the handler turns into ``{"detail": ...}``."""

    def __init__(self, status: HTTPStatus, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# --------------------------------------------------------------------- routing


def route(store: StatusStore, method: str, raw_path: str, body: dict[str, Any] | None) -> Response:
    """Dispatch one request. Pure function of (method, path, body) so it is easy to test."""
    url = urlparse(raw_path)
    path = url.path.rstrip("/") or "/"
    query = {key: values[-1] for key, values in parse_qs(url.query).items()}

    if path == "/api/health" and method == "GET":
        return HTTPStatus.OK, {"status": "ok"}
    if path == "/api/jobs":
        if method == "POST":
            return _create_job(store, body)
        if method == "GET":
            return _list_jobs(store, query)
    if path == "/api/events" and method == "GET":
        return _event_feed(store, query)
    if (match := _JOB_EVENTS_ROUTE.match(path)) and method == "GET":
        return _job_events(store, match["job_id"], query)
    if match := _JOB_ROUTE.match(path):
        if method == "GET":
            return _get_job(store, match["job_id"])
        if method == "PUT":
            return _update_job(store, match["job_id"], body)
        if method == "DELETE":
            return _delete_job(store, match["job_id"])
    raise ApiError(HTTPStatus.NOT_FOUND, f"No route for {method} {path}")


def _create_job(store: StatusStore, body: dict[str, Any] | None) -> Response:
    payload = JobCreatePayload.model_validate(body or {})
    result = store.create_job(
        payload.job_id,
        payload.job_name,
        cluster=payload.cluster,
        recipe=payload.recipe,
        submitted_at=payload.submitted_at,
        metadata=payload.metadata,
    )
    if result["created"]:
        where = f" on {payload.cluster}" if payload.cluster else ""
        logger.info("%s submitted: %s%s", payload.job_id, payload.job_name, where)
    return HTTPStatus.CREATED, JobResponse(job_id=result["job_id"], status=result["status"]).model_dump()


def _update_job(store: StatusStore, job_id: str, body: dict[str, Any] | None) -> Response:
    payload = JobUpdatePayload.model_validate(body or {})
    _require_member(JobStatus, payload.status, "status")
    if payload.stage is not None:
        _require_member(JobStage, payload.stage, "stage")
    result = store.update_job(job_id, payload.model_dump(exclude_none=True))
    if result["event"]:
        stage = f"/{payload.stage}" if payload.stage else ""
        logger.info("%s -> %s%s %s", job_id, payload.status, stage, payload.message or "")
    return HTTPStatus.OK, JobResponse(job_id=job_id, status=payload.status).model_dump()


def _get_job(store: StatusStore, job_id: str) -> Response:
    job = store.get_job(job_id)
    if job is None:
        raise ApiError(HTTPStatus.NOT_FOUND, "Job not found")
    return HTTPStatus.OK, JobDetail(**job).model_dump()


def _delete_job(store: StatusStore, job_id: str) -> Response:
    if not store.delete_job(job_id):
        raise ApiError(HTTPStatus.NOT_FOUND, "Job not found")
    return HTTPStatus.OK, {"deleted": True, "job_id": job_id}


def _list_jobs(store: StatusStore, query: dict[str, str]) -> Response:
    page = _int_param(query, "page", 1, minimum=1)
    per_page = _int_param(query, "per_page", 50, minimum=1, maximum=100)
    jobs, total = store.list_jobs(
        page=page, per_page=per_page, status=query.get("status"), cluster=query.get("cluster")
    )
    summaries = [JobSummary(**{name: job[name] for name in JobSummary.model_fields}) for job in jobs]
    return HTTPStatus.OK, JobListResponse(jobs=summaries, total=total, page=page, per_page=per_page).model_dump()


def _job_events(store: StatusStore, job_id: str, query: dict[str, str]) -> Response:
    if store.get_job(job_id) is None:
        raise ApiError(HTTPStatus.NOT_FOUND, "Job not found")
    after = _int_param(query, "after", 0, minimum=0)
    limit = _int_param(query, "limit", 100, minimum=1, maximum=1000)
    events = store.list_events(after=after, limit=limit, job_id=job_id)
    response = JobEventListResponse(job_id=job_id, events=events, next_cursor=_next_cursor(events, after))
    return HTTPStatus.OK, response.model_dump()


def _event_feed(store: StatusStore, query: dict[str, str]) -> Response:
    after = _int_param(query, "after", 0, minimum=0)
    limit = _int_param(query, "limit", 100, minimum=1, maximum=1000)
    events = store.list_events(after=after, limit=limit, job_id=query.get("job_id"))
    return HTTPStatus.OK, EventFeedResponse(events=events, next_cursor=_next_cursor(events, after)).model_dump()


def _next_cursor(events: list[dict[str, Any]], after: int) -> int | None:
    """Cursor for the next poll: the last id seen, or the one the caller passed when nothing new arrived."""
    if events:
        return events[-1]["id"]
    return after or None


def _int_param(query: dict[str, str], name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = query.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, f"{name} must be an integer") from None
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, f"{name} must be {bound}")
    return value


def _require_member(enum: type[JobStatus | JobStage], value: str, field: str) -> None:
    try:
        enum(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum)
        raise ApiError(
            HTTPStatus.UNPROCESSABLE_ENTITY, f"Unknown {field} {value!r}; expected one of: {allowed}"
        ) from None


# ---------------------------------------------------------------------- server


def _handler_class(store: StatusStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "srtctl-status-server"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            # Access log at DEBUG; the INFO lines are the lifecycle transitions in route().
            logger.debug("%s " + format, self.address_string(), *args)

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def do_PUT(self) -> None:
            self._handle("PUT")

        def do_DELETE(self) -> None:
            self._handle("DELETE")

        def _handle(self, method: str) -> None:
            try:
                status, body = route(store, method, self.path, self._read_json())
            except ApiError as exc:
                status, body = exc.status, {"detail": exc.detail}
            except ValidationError as exc:
                status, body = HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": json.loads(exc.json())}
            except Exception:
                logger.exception("Unhandled error serving %s %s", method, self.path)
                status, body = HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "Internal server error"}
            self._send_json(status, body)

        def _read_json(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return None
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"Body is not valid JSON: {exc.msg}") from None
            if not isinstance(parsed, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "Body must be a JSON object")
            return parsed

        def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def make_server(store: StatusStore, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bind a server for ``store``. ``port=0`` picks a free port; read it back from ``server.server_address``."""
    server = ThreadingHTTPServer((host, port), _handler_class(store))
    server.daemon_threads = True
    return server


def serve(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, db_path: Path | None = None) -> None:
    """Run the collector until interrupted."""
    store = StatusStore((db_path or DEFAULT_DB_PATH).expanduser())
    store.init()
    server = make_server(store, host=host, port=port)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    print(f"srtctl status-server listening on http://{bound_host}:{bound_port} (db: {store.db_path})")
    print("Point srtslurm.yaml or a recipe at it with:")
    print("  reporting:")
    print("    status:")
    print(f'      endpoint: "http://<this-host>:{bound_port}"')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``srtctl status-server`` and the standalone ``main``."""
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST}; use 0.0.0.0 so compute nodes can reach it)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Listen port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"SQLite file for jobs and events (default: {DEFAULT_DB_PATH})",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="srtctl status-server",
        description="Run the native status collector that reporting.status.endpoint can point at",
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    serve(host=args.host, port=args.port, db_path=args.db)


if __name__ == "__main__":
    main()
