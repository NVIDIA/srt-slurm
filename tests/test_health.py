# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for health check parsing (Dynamo and SGLang router)."""

import pytest

from srtctl.core.health import (
    WorkerHealthResult,
    check_dynamo_health,
    check_sglang_router_health,
)

# ============================================================================
# Dynamo Health Check Tests
# ============================================================================


class TestDynamoHealthDisaggregated:
    """Test Dynamo /health parsing for disaggregated mode (prefill + decode workers)."""

    def test_all_workers_ready(self):
        """All expected prefill and decode workers are registered."""
        response = {
            "instances": [
                {"endpoint": "generate", "component": "prefill"},
                {"endpoint": "generate", "component": "prefill"},
                {"endpoint": "generate", "component": "decode"},
                {"endpoint": "generate", "component": "decode"},
                {"endpoint": "generate", "component": "decode"},
            ]
        }

        result = check_dynamo_health(response, expected_prefill=2, expected_decode=3)

        assert result.ready is True
        assert result.prefill_ready == 2
        assert result.decode_ready == 3
        assert "Model is ready" in result.message

    def test_missing_prefill_workers(self):
        """Not enough prefill workers registered."""
        response = {
            "instances": [
                {"endpoint": "generate", "component": "prefill"},
                {"endpoint": "generate", "component": "decode"},
                {"endpoint": "generate", "component": "decode"},
            ]
        }

        result = check_dynamo_health(response, expected_prefill=2, expected_decode=2)

        assert result.ready is False
        assert result.prefill_ready == 1
        assert result.prefill_expected == 2
        assert "waiting for 1 prefills" in result.message

    def test_missing_decode_workers(self):
        """Not enough decode workers registered."""
        response = {
            "instances": [
                {"endpoint": "generate", "component": "prefill"},
                {"endpoint": "generate", "component": "prefill"},
                {"endpoint": "generate", "component": "decode"},
            ]
        }

        result = check_dynamo_health(response, expected_prefill=2, expected_decode=4)

        assert result.ready is False
        assert result.decode_ready == 1
        assert result.decode_expected == 4
        assert "waiting for" in result.message
        assert "3 decodes" in result.message

    def test_empty_instances(self):
        """No workers registered yet."""
        response = {"instances": []}

        result = check_dynamo_health(response, expected_prefill=1, expected_decode=1)

        assert result.ready is False
        assert result.prefill_ready == 0
        assert result.decode_ready == 0

    def test_ignores_non_generate_endpoints(self):
        """Only 'generate' endpoint instances count."""
        response = {
            "instances": [
                {"endpoint": "generate", "component": "prefill"},
                {"endpoint": "other", "component": "prefill"},  # Should be ignored
                {"endpoint": "generate", "component": "decode"},
            ]
        }

        result = check_dynamo_health(response, expected_prefill=1, expected_decode=1)

        assert result.ready is True
        assert result.prefill_ready == 1
        assert result.decode_ready == 1


class TestDynamoHealthAggregated:
    """Test Dynamo /health parsing for aggregated mode (backend workers).

    In aggregated mode, workers report as "backend" and count as decode.
    Caller should pass expected_prefill=0, expected_decode=num_agg.
    """

    def test_all_backend_workers_ready(self):
        """All expected backend workers are registered (aggregated mode)."""
        response = {
            "instances": [
                {"endpoint": "generate", "component": "backend"},
                {"endpoint": "generate", "component": "backend"},
            ]
        }

        # Aggregated: expect 0 prefill, N decode (backend counts as decode)
        result = check_dynamo_health(response, expected_prefill=0, expected_decode=2)

        assert result.ready is True
        assert result.decode_ready == 2
        assert result.prefill_ready == 0

    def test_backend_workers_count_as_decode(self):
        """Backend workers always count as decode."""
        response = {
            "instances": [
                {"endpoint": "generate", "component": "backend"},
                {"endpoint": "generate", "component": "backend"},
                {"endpoint": "generate", "component": "backend"},
                {"endpoint": "generate", "component": "backend"},
            ]
        }

        # All 4 backend workers count as decode
        result = check_dynamo_health(response, expected_prefill=0, expected_decode=4)

        assert result.ready is True
        assert result.decode_ready == 4
        assert result.prefill_ready == 0

    def test_not_enough_backend_workers(self):
        """Fewer backend workers than expected."""
        response = {
            "instances": [
                {"endpoint": "generate", "component": "backend"},
            ]
        }

        result = check_dynamo_health(response, expected_prefill=0, expected_decode=2)

        assert result.ready is False
        assert result.decode_ready == 1
        assert result.decode_expected == 2


class TestDynamoHealthErrors:
    """Test Dynamo /health error handling."""

    def test_missing_instances_key(self):
        """Response missing 'instances' key."""
        response = {"status": "ok"}

        result = check_dynamo_health(response, expected_prefill=1, expected_decode=1)

        assert result.ready is False
        assert "instances" in result.message

    def test_empty_response(self):
        """Empty response dict."""
        response = {}

        result = check_dynamo_health(response, expected_prefill=1, expected_decode=1)

        assert result.ready is False


# ============================================================================
# SGLang Router Health Check Tests
# ============================================================================


class TestSGLangRouterHealthDisaggregated:
    """Test SGLang router /workers parsing for disaggregated mode.

    Uses realistic response format from actual SGLang router.
    """

    def test_all_workers_ready_realistic_format(self):
        """All expected workers are registered (realistic response format)."""
        # Realistic format from actual SGLang router
        response = {
            "workers": [
                {"id": "http://10.66.5.20:30000", "worker_type": "decode", "is_healthy": True},
                {"id": "http://10.66.5.15:30000", "worker_type": "decode", "is_healthy": True},
                {"id": "http://10.66.5.14:30000", "worker_type": "prefill", "is_healthy": True},
            ],
            "total": 3,
            "stats": {
                "prefill_count": 1,
                "decode_count": 2,
                "regular_count": 0,
            },
        }

        result = check_sglang_router_health(response, expected_prefill=1, expected_decode=2)

        assert result.ready is True
        assert result.prefill_ready == 1
        assert result.decode_ready == 2
        assert "Model is ready" in result.message

    def test_all_workers_ready(self):
        """All expected workers are registered."""
        response = {
            "workers": [],
            "total": 12,
            "stats": {
                "prefill_count": 4,
                "decode_count": 8,
                "regular_count": 0,
            },
        }

        result = check_sglang_router_health(response, expected_prefill=4, expected_decode=8)

        assert result.ready is True
        assert result.prefill_ready == 4
        assert result.decode_ready == 8
        assert "Model is ready" in result.message

    def test_more_workers_than_expected(self):
        """More workers than expected is still ready."""
        response = {
            "workers": [],
            "total": 16,
            "stats": {
                "prefill_count": 6,
                "decode_count": 10,
                "regular_count": 0,
            },
        }

        result = check_sglang_router_health(response, expected_prefill=4, expected_decode=8)

        assert result.ready is True
        assert result.prefill_ready == 6
        assert result.decode_ready == 10

    def test_missing_prefill_workers(self):
        """Not enough prefill workers."""
        response = {
            "workers": [],
            "total": 10,
            "stats": {
                "prefill_count": 2,
                "decode_count": 8,
                "regular_count": 0,
            },
        }

        result = check_sglang_router_health(response, expected_prefill=4, expected_decode=8)

        assert result.ready is False
        assert result.prefill_ready == 2
        assert result.prefill_expected == 4
        assert "waiting for 2 prefills" in result.message

    def test_missing_decode_workers(self):
        """Not enough decode workers."""
        response = {
            "workers": [],
            "total": 7,
            "stats": {
                "prefill_count": 4,
                "decode_count": 3,
                "regular_count": 0,
            },
        }

        result = check_sglang_router_health(response, expected_prefill=4, expected_decode=8)

        assert result.ready is False
        assert result.decode_ready == 3
        assert "waiting for" in result.message
        assert "5 decodes" in result.message

    def test_zero_workers(self):
        """No workers registered yet."""
        response = {
            "workers": [],
            "total": 0,
            "stats": {
                "prefill_count": 0,
                "decode_count": 0,
                "regular_count": 0,
            },
        }

        result = check_sglang_router_health(response, expected_prefill=2, expected_decode=4)

        assert result.ready is False
        assert result.prefill_ready == 0
        assert result.decode_ready == 0


class TestSGLangRouterHealthAggregated:
    """Test SGLang router /workers parsing for aggregated mode.

    In aggregated mode, workers may report as 'regular' instead of prefill/decode.
    """

    def test_regular_workers_count_as_decode(self):
        """Regular workers (aggregated mode) count towards decode."""
        response = {
            "workers": [],
            "total": 4,
            "stats": {
                "prefill_count": 0,
                "decode_count": 0,
                "regular_count": 4,
            },
        }

        # Aggregated: expect 0 prefill, 4 decode (regular counts as decode)
        result = check_sglang_router_health(response, expected_prefill=0, expected_decode=4)

        assert result.ready is True
        assert result.decode_ready == 4
        assert "regular workers" in result.message

    def test_aggregated_workers_as_decode(self):
        """In aggregated mode, all workers might report as decode."""
        response = {
            "workers": [],
            "total": 4,
            "stats": {
                "prefill_count": 0,
                "decode_count": 4,
                "regular_count": 0,
            },
        }

        # Aggregated: expect 0 prefill, N decode
        result = check_sglang_router_health(response, expected_prefill=0, expected_decode=4)

        assert result.ready is True
        assert result.decode_ready == 4

    def test_mixed_decode_and_regular(self):
        """Mix of decode and regular workers."""
        response = {
            "workers": [],
            "total": 4,
            "stats": {
                "prefill_count": 0,
                "decode_count": 2,
                "regular_count": 2,
            },
        }

        # Both decode and regular should count
        result = check_sglang_router_health(response, expected_prefill=0, expected_decode=4)

        assert result.ready is True
        assert result.decode_ready == 4  # 2 decode + 2 regular


class TestSGLangRouterHealthErrors:
    """Test SGLang router /workers error handling."""

    def test_missing_stats_key(self):
        """Response missing 'stats' key."""
        response = {"workers": []}

        result = check_sglang_router_health(response, expected_prefill=1, expected_decode=1)

        assert result.ready is False
        assert "stats" in result.message

    def test_empty_response(self):
        """Empty response dict."""
        response = {}

        result = check_sglang_router_health(response, expected_prefill=1, expected_decode=1)

        assert result.ready is False

    def test_missing_count_fields_defaults_to_zero(self):
        """Missing count fields default to 0."""
        response = {"stats": {}}

        result = check_sglang_router_health(response, expected_prefill=1, expected_decode=1)

        assert result.ready is False
        assert result.prefill_ready == 0
        assert result.decode_ready == 0


# ============================================================================
# WorkerHealthResult Tests
# ============================================================================


class TestWorkerHealthResult:
    """Test WorkerHealthResult dataclass."""

    def test_default_values(self):
        """Default count values are 0."""
        result = WorkerHealthResult(ready=True, message="OK")

        assert result.prefill_ready == 0
        assert result.prefill_expected == 0
        assert result.decode_ready == 0
        assert result.decode_expected == 0

    def test_with_counts(self):
        """Counts can be set."""
        result = WorkerHealthResult(
            ready=True,
            message="OK",
            prefill_ready=2,
            prefill_expected=2,
            decode_ready=4,
            decode_expected=4,
        )

        assert result.prefill_ready == 2
        assert result.decode_ready == 4


# ============================================================================
# Frontend readiness probes and the wait_for_model loop
# ============================================================================


def _http(status: int, body=None):
    from unittest.mock import Mock

    response = Mock(status_code=status)
    response.json = Mock(return_value=body if body is not None else {})
    return response


class TestFrontendProbes:
    """Each frontend owns one readiness probe; the loop only retries and logs."""

    def test_bare_200_frontend_is_ready_without_a_body(self, monkeypatch):
        from srtctl.core import health
        from srtctl.frontends import get_frontend

        monkeypatch.setattr(health.requests, "get", lambda url, timeout: _http(200))
        result = get_frontend("trtllm_serve").probe_ready("router", 8000, 1, 1)
        assert result.ready is True
        assert "router:8000/health" in result.message

    def test_non_200_is_not_ready_with_the_status_in_the_message(self, monkeypatch):
        from srtctl.core import health
        from srtctl.frontends import get_frontend

        monkeypatch.setattr(health.requests, "get", lambda url, timeout: _http(503))
        for frontend_type in ("dynamo", "sglang-router", "vllm-router", "trtllm_serve", "vllm", "sglang"):
            result = get_frontend(frontend_type).probe_ready("router", 8000, 1, 1)
            assert result.ready is False, frontend_type
            assert "HTTP 503" in result.message, frontend_type

    def test_json_registry_goes_through_parse_health(self, monkeypatch):
        from srtctl.core import health
        from srtctl.frontends import get_frontend

        seen: dict[str, str] = {}

        def fake_get(url, timeout):
            seen["url"] = url
            return _http(200, {"stats": {"prefill_count": 1, "decode_count": 2, "regular_count": 0}, "workers": []})

        monkeypatch.setattr(health.requests, "get", fake_get)
        result = get_frontend("sglang-router").probe_ready("router", 8000, 1, 2)
        assert seen["url"] == "http://router:8000/workers"
        assert result.prefill_ready == 1
        assert result.decode_ready == 2

    def test_direct_server_needs_health_and_a_listed_model(self, monkeypatch):
        from srtctl.core import health
        from srtctl.frontends import get_frontend

        bodies = {
            "http://worker:8000/health": _http(200),
            "http://worker:8000/v1/models": _http(200, {"data": [{"id": "Qwen/Qwen3-0.6B"}]}),
        }
        monkeypatch.setattr(health.requests, "get", lambda url, timeout: bodies[url])
        for frontend_type in ("vllm", "sglang"):
            assert get_frontend(frontend_type).probe_ready("worker", 8000, 0, 1).ready is True

        bodies["http://worker:8000/v1/models"] = _http(200, {"data": []})
        result = get_frontend("vllm").probe_ready("worker", 8000, 0, 1)
        assert result.ready is False
        assert "no models" in result.message

    def test_connection_errors_propagate_for_the_loop_to_retry(self, monkeypatch):
        import requests

        from srtctl.core import health
        from srtctl.frontends import get_frontend

        def refuse(url, timeout):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(health.requests, "get", refuse)
        with pytest.raises(requests.exceptions.RequestException):
            get_frontend("dynamo").probe_ready("router", 8000, 1, 1)


class TestWaitForModel:
    def _stub(self, monkeypatch, probes):
        from types import SimpleNamespace

        import srtctl.frontends

        calls = iter(probes)

        def probe_ready(host, port, expected_prefill, expected_decode):
            outcome = next(calls)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        stub = SimpleNamespace(type="stub", health_endpoint="/probe", probe_ready=probe_ready)
        monkeypatch.setattr(srtctl.frontends, "get_frontend", lambda frontend_type: stub)
        return stub

    def test_returns_true_once_the_probe_reports_ready(self, monkeypatch):
        from srtctl.core.health import wait_for_model

        self._stub(
            monkeypatch,
            [
                WorkerHealthResult(ready=False, message="1/2 workers"),
                WorkerHealthResult(ready=True, message="2/2 workers"),
            ],
        )
        assert wait_for_model("router", 8000, 1, 1, poll_interval=0.001, timeout=5, frontend_type="stub") is True

    def test_returns_false_on_timeout_while_the_endpoint_refuses(self, monkeypatch):
        import itertools

        import requests

        from srtctl.core.health import wait_for_model

        self._stub(monkeypatch, itertools.repeat(requests.exceptions.ConnectionError("refused")))
        assert wait_for_model("router", 8000, 1, 1, poll_interval=0.001, timeout=0.05, frontend_type="stub") is False

    def test_stop_event_aborts(self, monkeypatch):
        import itertools
        import threading

        from srtctl.core.health import wait_for_model

        self._stub(monkeypatch, itertools.repeat(WorkerHealthResult(ready=False, message="waiting")))
        stop = threading.Event()
        stop.set()
        assert (
            wait_for_model("router", 8000, 1, 1, poll_interval=0.001, timeout=5, frontend_type="stub", stop_event=stop)
            is False
        )
