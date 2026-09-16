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

Authentication is bearer tokens read from the environment. ``$SRTCTL_STATUS_TOKEN``
guards every write and also grants reads; the optional ``$SRTCTL_STATUS_READ_TOKEN``
grants reads only. ``/api/health`` is always open. Binding to anything but
loopback without a token is refused unless ``--allow-unauthenticated`` says the
network is trusted (a cluster's internal network, for example).

``GET /`` serves ``ui/index.html``, a dependency-free single page (jobs table,
per-job facts and event timeline, live global event feed) that talks to the
same ``/api`` routes with the read token the viewer pastes once. The same page
can be hosted elsewhere (a corp-network web server, a file) and pointed at this
API with ``#api=``; the collector then needs ``--cors-origin`` for that origin.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
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
DEFAULT_TOKEN_ENV = "SRTCTL_STATUS_TOKEN"
DEFAULT_READ_TOKEN_ENV = "SRTCTL_STATUS_READ_TOKEN"
# The largest legitimate body is the started-metadata PUT, a few KB. Anything
# beyond this is rejected before it is read so a public endpoint cannot be
# used to fill the disk.
MAX_BODY_BYTES = 1 << 20
HEALTH_PATH = "/api/health"
# The single-page UI. It is static and reveals nothing, so it is served without
# a token; every API call the page makes carries the token the viewer pasted.
UI_DIR = Path(__file__).with_name("ui")
UI_PATHS = frozenset({"/", "/index.html"})

_JOB_ROUTE = re.compile(r"^/api/jobs/(?P<job_id>[^/]+)$")
_JOB_EVENTS_ROUTE = re.compile(r"^/api/jobs/(?P<job_id>[^/]+)/events$")

Response = tuple[HTTPStatus, dict[str, Any]]


class ApiError(Exception):
    """An HTTP error the handler turns into ``{"detail": ...}``."""

    def __init__(self, status: HTTPStatus, detail: str, headers: dict[str, str] | None = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.headers = headers or {}


# ------------------------------------------------------------------------ auth


def _same(presented: str, expected: str | None) -> bool:
    return expected is not None and hmac.compare_digest(presented.encode(), expected.encode())


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


@dataclass(frozen=True)
class AuthPolicy:
    """Bearer-token policy checked before anything else in a request.

    ``write_token`` is required for POST, PUT and DELETE and also grants GET.
    ``read_token`` grants GET only. With no ``write_token`` the server is open;
    ``resolve_auth`` only allows that on loopback or with an explicit flag.
    ``/api/health`` never needs a token, so load balancers and uptime checks
    work, and it reveals nothing but ``{"status": "ok"}``.
    """

    write_token: str | None = None
    read_token: str | None = None

    @property
    def enabled(self) -> bool:
        return self.write_token is not None

    def check(self, method: str, path: str, authorization: str | None) -> None:
        if not self.enabled or path == HEALTH_PATH:
            return
        presented = _bearer(authorization)
        if presented is None:
            raise ApiError(HTTPStatus.UNAUTHORIZED, "Bearer token required", {"WWW-Authenticate": "Bearer"})
        if _same(presented, self.write_token):
            return
        if _same(presented, self.read_token):
            if method == "GET":
                return
            raise ApiError(HTTPStatus.FORBIDDEN, "The read token cannot modify jobs")
        raise ApiError(
            HTTPStatus.UNAUTHORIZED, "Invalid bearer token", {"WWW-Authenticate": 'Bearer error="invalid_token"'}
        )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_auth(
    host: str,
    *,
    token_env: str = DEFAULT_TOKEN_ENV,
    read_token_env: str = DEFAULT_READ_TOKEN_ENV,
    allow_unauthenticated: bool = False,
) -> AuthPolicy:
    """Build the policy from the environment, refusing unsafe combinations.

    Raises ``SystemExit`` with a plain-English reason when the server would be
    reachable beyond loopback with no token, or when only a read token is set.
    """
    write_token = os.environ.get(token_env) or None
    read_token = os.environ.get(read_token_env) or None
    if write_token is None:
        if read_token is not None:
            raise SystemExit(f"${read_token_env} is set but ${token_env} is not; a read token needs a write token.")
        if not _is_loopback(host) and not allow_unauthenticated:
            raise SystemExit(
                f"Refusing to listen on {host} without a bearer token: anyone who can reach this port could read "
                f"and rewrite every job. Set ${token_env} (and optionally ${read_token_env}), or pass "
                "--allow-unauthenticated when the network itself is trusted, such as a cluster login node."
            )
    return AuthPolicy(write_token=write_token, read_token=read_token)


@dataclass(frozen=True)
class CorsPolicy:
    """Opt-in CORS for a UI served from another origin (``--cors-origin``).

    Off by default: no ``Access-Control-*`` header is ever sent and ``OPTIONS``
    is 404. With origins configured, a matching ``Origin`` gets
    ``Access-Control-Allow-Origin`` on every response, errors included, so the
    browser can surface a 401 instead of a generic network error. Preflights are
    answered before auth because browsers send them without the token. Only GET
    and HEAD are offered cross-origin: the UI reads, it never writes. ``"*"``
    allows any origin, including the ``null`` origin of a page opened from a
    file. This is safe because the API uses no cookies and a token stored by one
    origin's localStorage is unreadable to every other origin.
    """

    origins: frozenset[str] = frozenset()

    @classmethod
    def from_origins(cls, origins: Iterable[str]) -> CorsPolicy:
        return cls(frozenset(origin.strip().rstrip("/") for origin in origins if origin.strip()))

    @property
    def enabled(self) -> bool:
        return bool(self.origins)

    def allow_origin(self, origin: str | None) -> str | None:
        if not origin or not self.enabled:
            return None
        if "*" in self.origins:
            return "*"
        return origin if origin.rstrip("/") in self.origins else None

    def headers(self, origin: str | None) -> dict[str, str]:
        allowed = self.allow_origin(origin)
        return {"Access-Control-Allow-Origin": allowed, "Vary": "Origin"} if allowed else {}

    def preflight(self, origin: str | None) -> dict[str, str]:
        headers = self.headers(origin)
        if not headers:
            raise ApiError(HTTPStatus.FORBIDDEN, "CORS is not enabled for this origin")
        return {
            **headers,
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization",
            "Access-Control-Max-Age": "600",
        }


# --------------------------------------------------------------------- routing


def route(store: StatusStore, method: str, raw_path: str, body: dict[str, Any] | None) -> Response:
    """Dispatch one request. Pure function of (method, path, body) so it is easy to test."""
    url = urlparse(raw_path)
    path = url.path.rstrip("/") or "/"
    query = {key: values[-1] for key, values in parse_qs(url.query).items()}

    if path == HEALTH_PATH and method == "GET":
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
    if result.get("backfilled"):
        logger.info("%s identified late by its submit record: %s", payload.job_id, payload.job_name)
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
    logger.info("%s deleted", job_id)
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


def _parse_json(raw: bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Body is not valid JSON: {exc.msg}") from None
    if not isinstance(parsed, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "Body must be a JSON object")
    return parsed


# ---------------------------------------------------------------------- server


def _handler_class(store: StatusStore, auth: AuthPolicy, cors: CorsPolicy) -> type[BaseHTTPRequestHandler]:
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

        def do_HEAD(self) -> None:
            # Uptime checkers and proxies probe with HEAD; answer like GET without a body.
            self._handle("HEAD")

        def do_OPTIONS(self) -> None:
            # CORS preflight. Browsers send it without the token, so it is answered
            # before auth; it discloses only whether this origin may call the API.
            try:
                self._read_body()
                if not cors.enabled:
                    raise ApiError(HTTPStatus.NOT_FOUND, "No route for OPTIONS")
                self._send(HTTPStatus.NO_CONTENT, b"", "text/plain", cors.preflight(self.headers.get("Origin")), True)
            except ApiError as exc:
                self._send(
                    exc.status, json.dumps({"detail": exc.detail}).encode(), "application/json", exc.headers, False
                )

        def _handle(self, method: str) -> None:
            """Order matters: size cap, then the static UI, then auth, then JSON parsing, then routing.

            An unauthenticated caller therefore learns nothing from the response,
            not even whether the body parsed or the job exists.
            """
            head_only = method == "HEAD"
            effective = "GET" if head_only else method
            cors_headers = cors.headers(self.headers.get("Origin"))
            headers: dict[str, str] = {}
            try:
                raw = self._read_body()
                path = urlparse(self.path).path.rstrip("/") or "/"
                if effective == "GET" and path in UI_PATHS:
                    page = (UI_DIR / "index.html").read_bytes()
                    self._send(HTTPStatus.OK, page, "text/html; charset=utf-8", cors_headers, head_only)
                    return
                auth.check(effective, path, self.headers.get("Authorization"))
                status, body = route(store, effective, self.path, _parse_json(raw))
            except ApiError as exc:
                if exc.status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
                    logger.info("%s %s %s from %s", exc.status.value, method, self.path, self.address_string())
                status, body, headers = exc.status, {"detail": exc.detail}, exc.headers
            except ValidationError as exc:
                status, body = HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": json.loads(exc.json())}
            except Exception:
                logger.exception("Unhandled error serving %s %s", method, self.path)
                status, body = HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "Internal server error"}
            # CORS headers ride on every response, errors included, so a browser
            # page can tell a 401 from a blocked request.
            self._send(status, json.dumps(body).encode(), "application/json", {**cors_headers, **headers}, head_only)

        def _read_body(self) -> bytes | None:
            header = self.headers.get("Content-Length")
            try:
                length = int(header) if header else 0
            except ValueError:
                self.close_connection = True
                raise ApiError(HTTPStatus.BAD_REQUEST, "Content-Length must be an integer") from None
            if length > MAX_BODY_BYTES:
                # Not read, so the connection cannot be reused for a keep-alive request.
                self.close_connection = True
                raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Body larger than {MAX_BODY_BYTES} bytes")
            if length == 0:
                return None
            return self.rfile.read(length)

        def _send(
            self, status: HTTPStatus, data: bytes, content_type: str, headers: dict[str, str], head_only: bool
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in headers.items():
                self.send_header(name, value)
            if self.close_connection:
                # Tell keep-alive clients not to reuse a connection whose body was never drained.
                self.send_header("Connection", "close")
            self.end_headers()
            if not head_only:
                self.wfile.write(data)

    return Handler


def make_server(
    store: StatusStore,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    auth: AuthPolicy | None = None,
    cors: CorsPolicy | None = None,
) -> ThreadingHTTPServer:
    """Bind a server for ``store``. ``port=0`` picks a free port; read it back from ``server.server_address``.

    ``auth=None`` means no authentication; callers other than tests should go
    through ``resolve_auth`` so the loopback rule is applied. ``cors=None``
    means CORS off.
    """
    server = ThreadingHTTPServer((host, port), _handler_class(store, auth or AuthPolicy(), cors or CorsPolicy()))
    server.daemon_threads = True
    return server


def serve(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    db_path: Path | None = None,
    token_env: str = DEFAULT_TOKEN_ENV,
    read_token_env: str = DEFAULT_READ_TOKEN_ENV,
    allow_unauthenticated: bool = False,
    cors_origins: Iterable[str] = (),
) -> None:
    """Run the collector until interrupted."""
    auth = resolve_auth(
        host, token_env=token_env, read_token_env=read_token_env, allow_unauthenticated=allow_unauthenticated
    )
    cors = CorsPolicy.from_origins(cors_origins)
    store = StatusStore((db_path or DEFAULT_DB_PATH).expanduser())
    store.init()
    server = make_server(store, host=host, port=port, auth=auth, cors=cors)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    print(f"srtctl status-server listening on http://{bound_host}:{bound_port} (db: {store.db_path})")
    print(f"UI: http://{bound_host}:{bound_port}/  (paste the read token once; the browser keeps it in localStorage)")
    if cors.enabled:
        print(f"cors: GET/HEAD allowed from {', '.join(sorted(cors.origins))}")
    if auth.enabled:
        read = f", read token from ${read_token_env}" if auth.read_token else ""
        print(f"auth: bearer token required on every route except {HEALTH_PATH} (write token from ${token_env}{read})")
    else:
        why = "loopback only" if _is_loopback(host) else "--allow-unauthenticated"
        print(f"auth: none ({why})")
    print("Point srtslurm.yaml or a recipe at it with:")
    print("  reporting:")
    print("    status:")
    print(f'      endpoint: "http://<this-host>:{bound_port}"')
    if auth.enabled:
        print(f"and export ${token_env} in the shell that runs srtctl apply.")
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
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        metavar="VAR",
        help=f"Environment variable holding the write token; grants every route (default: {DEFAULT_TOKEN_ENV})",
    )
    parser.add_argument(
        "--read-token-env",
        default=DEFAULT_READ_TOKEN_ENV,
        metavar="VAR",
        help=f"Environment variable holding a read-only token for GET routes (default: {DEFAULT_READ_TOKEN_ENV})",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Listen beyond loopback with no token set; only for a network that is trusted end to end",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help=(
            "Let the UI served from this origin call the API from a browser (repeatable; "
            "'*' allows any origin, including a page opened from a file). Read-only routes only. Off by default"
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="srtctl status-server",
        description="Run the native status collector that reporting.status.endpoint can point at",
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    serve(
        host=args.host,
        port=args.port,
        db_path=args.db,
        token_env=args.token_env,
        read_token_env=args.read_token_env,
        allow_unauthenticated=args.allow_unauthenticated,
        cors_origins=args.cors_origin,
    )


if __name__ == "__main__":
    main()
