# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single-page UI served at ``/`` by ``srtctl status-server``.

The page is static and must be reachable without a token; everything it
fetches goes through the same authenticated ``/api`` routes as any client.
"""

from __future__ import annotations

import re

import pytest
import requests
from test_status_server import READ, WRITE, _bearer, _create, _put, _running

from srtctl.status_server import StatusStore, make_server
from srtctl.status_server.server import UI_DIR, AuthPolicy


@pytest.fixture
def auth_url(tmp_path):
    store = StatusStore(tmp_path / "status.db")
    store.init()
    auth = AuthPolicy(write_token=WRITE, read_token=READ)
    with _running(make_server(store, host="127.0.0.1", port=0, auth=auth)) as url:
        yield url


def test_ui_is_served_without_a_token(auth_url):
    for path in ("/", "/index.html"):
        response = requests.get(f"{auth_url}{path}", timeout=5)
        assert response.status_code == 200, path
        assert response.headers["Content-Type"] == "text/html; charset=utf-8"
        assert response.headers["Cache-Control"] == "no-store"
        assert "<title>srtctl status</title>" in response.text
    assert response.content == (UI_DIR / "index.html").read_bytes()


def test_ui_has_no_external_resources_and_uses_documented_routes():
    html = (UI_DIR / "index.html").read_text()
    # Works on an air-gapped login node: nothing is loaded from a CDN.
    assert not re.search(r'(src|href)="https?://', html)
    assert "<script src=" not in html and "<link " not in html
    for route in ("/api/jobs?per_page=100", "/api/jobs/${encodeURIComponent(id)}", "/api/events?after="):
        assert route in html, route
    # Every API call carries the bearer token; the token is kept in localStorage only.
    assert 'Authorization: "Bearer " + token' in html
    assert "localStorage" in html
    assert "history.replaceState" in html  # the #token= fragment is removed from the URL


def test_api_next_to_the_ui_still_requires_a_token(auth_url):
    assert requests.get(f"{auth_url}/api/jobs", timeout=5).status_code == 401
    assert requests.get(f"{auth_url}/api/events", timeout=5).status_code == 401
    # Only the two UI paths are static; anything else falls through to auth.
    assert requests.get(f"{auth_url}/app.js", timeout=5).status_code == 401
    assert requests.get(f"{auth_url}/ui", timeout=5).status_code == 401


def test_head_answers_like_get_without_a_body(auth_url):
    page = requests.get(f"{auth_url}/", timeout=5)
    head = requests.head(f"{auth_url}/", timeout=5)
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["Content-Length"] == page.headers["Content-Length"]

    health = requests.head(f"{auth_url}/api/health", timeout=5)
    assert health.status_code == 200
    assert health.content == b""

    denied = requests.head(f"{auth_url}/api/jobs", timeout=5)
    assert denied.status_code == 401
    assert denied.content == b""
    assert denied.headers["WWW-Authenticate"] == "Bearer"

    assert requests.head(f"{auth_url}/api/jobs", headers=_bearer(READ), timeout=5).status_code == 200


def test_job_list_carries_what_the_table_shows(auth_url):
    """The UI's table reads message, completed_at and exit_code straight from the list endpoint."""
    write = _bearer(WRITE)
    requests.post(
        f"{auth_url}/api/jobs",
        json={"job_id": "1", "job_name": "ui", "submitted_at": "2026-01-01T00:00:00Z", "cluster": "c"},
        headers=write,
        timeout=5,
    )
    requests.put(
        f"{auth_url}/api/jobs/1",
        json={
            "status": "completed",
            "stage": "cleanup",
            "message": "Benchmark completed successfully",
            "updated_at": "2026-01-01T00:10:00Z",
            "completed_at": "2026-01-01T00:10:00Z",
            "exit_code": 0,
        },
        headers=write,
        timeout=5,
    )
    row = requests.get(f"{auth_url}/api/jobs", headers=_bearer(READ), timeout=5).json()["jobs"][0]
    assert row["message"] == "Benchmark completed successfully"
    assert row["completed_at"] == "2026-01-01T00:10:00Z"
    assert row["exit_code"] == 0
    assert row["stage"] == "cleanup"


def test_open_server_serves_ui_and_api_without_tokens(tmp_path):
    store = StatusStore(tmp_path / "open.db")
    store.init()
    with _running(make_server(store, host="127.0.0.1", port=0)) as url:
        _create(url, "1")
        _put(url, "1", {"status": "workers"})
        assert requests.get(f"{url}/", timeout=5).status_code == 200
        assert requests.get(f"{url}/api/jobs", timeout=5).json()["total"] == 1
