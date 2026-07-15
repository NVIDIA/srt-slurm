# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HTTP health check and port waiting utilities.

This module provides:
- wait_for_port(): Poll TCP port availability
- wait_for_health(): HTTP health check with worker count validation
- wait_for_etcd(): Wait for etcd to be ready
- wait_for_model(): Wait for model with worker count validation (replaces bash version)
- check_dynamo_health(): Parse dynamo /health response for worker counts
- check_sglang_router_health(): Parse sglang /workers response for worker counts
"""

import logging
import socket
import threading
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


# ============================================================================
# Worker Health Check Result
# ============================================================================


@dataclass
class WorkerHealthResult:
    """Result of a worker health check."""

    ready: bool
    message: str
    prefill_ready: int = 0
    prefill_expected: int = 0
    decode_ready: int = 0
    decode_expected: int = 0


# ============================================================================
# Worker Count Parsing (moved from check_server_health.py)
# ============================================================================


def check_sglang_router_health(
    response_json: dict,
    expected_prefill: int,
    expected_decode: int,
) -> WorkerHealthResult:
    """Check health using sglang router /workers endpoint response.

    Expected response format:
    {
        "workers": [
            {"worker_type": "prefill", "is_healthy": true, ...},
            {"worker_type": "decode", "is_healthy": true, ...},
        ],
        "total": 3,
        "stats": {
            "prefill_count": 1,
            "decode_count": 2,
            "regular_count": 0
        }
    }

    For aggregated mode (no prefill/decode split), workers report as "regular"
    and are counted in regular_count. Pass expected_prefill=0 and use
    expected_decode for the total expected workers.

    Args:
        response_json: Parsed JSON from /workers endpoint
        expected_prefill: Expected number of prefill workers
        expected_decode: Expected number of decode workers (or agg workers)

    Returns:
        WorkerHealthResult with ready status and counts
    """
    if "stats" not in response_json:
        return WorkerHealthResult(
            ready=False,
            message=f"Key 'stats' not found in response: {response_json}",
        )

    stats = response_json["stats"]
    actual_prefill = stats.get("prefill_count", 0)
    actual_decode = stats.get("decode_count", 0)
    actual_regular = stats.get("regular_count", 0)

    # For aggregated mode, regular workers count towards decode
    # (caller passes expected_prefill=0, expected_decode=num_agg)
    effective_decode = actual_decode + actual_regular

    ready = actual_prefill >= expected_prefill and effective_decode >= expected_decode

    if ready:
        message = f"Model is ready. Have {actual_prefill} prefills and {effective_decode} decodes."
        if actual_regular > 0:
            message += f" ({actual_regular} regular workers)"
    else:
        message = (
            f"Model is not ready, waiting for "
            f"{max(0, expected_prefill - actual_prefill)} prefills and "
            f"{max(0, expected_decode - effective_decode)} decodes. "
            f"Have {actual_prefill} prefills and {effective_decode} decodes."
        )
        if actual_regular > 0:
            message += f" ({actual_regular} regular workers)"

    return WorkerHealthResult(
        ready=ready,
        message=message,
        prefill_ready=actual_prefill,
        prefill_expected=expected_prefill,
        decode_ready=effective_decode,
        decode_expected=expected_decode,
    )


def check_dynamo_health(
    response_json: dict,
    expected_prefill: int,
    expected_decode: int,
) -> WorkerHealthResult:
    """Check health using dynamo frontend /health endpoint response.

    Args:
        response_json: Parsed JSON from /health endpoint
        expected_prefill: Expected number of prefill workers
        expected_decode: Expected number of decode workers

    Returns:
        WorkerHealthResult with ready status and counts
    """
    if "instances" not in response_json:
        return WorkerHealthResult(
            ready=False,
            message=f"Key 'instances' not found in response: {response_json}",
        )

    prefill_count = 0
    decode_count = 0

    for instance in response_json["instances"]:
        if instance.get("endpoint") == "generate":
            component = instance.get("component")
            if component == "prefill":
                prefill_count += 1
            # TRTLLM decode workers are reported as "tensorrt_llm"
            elif component == "decode" or component == "tensorrt_llm":
                decode_count += 1
            elif component == "backend":
                # In aggregated mode, workers report as "backend"
                # Count them as decode (caller passes expected_prefill=0)
                decode_count += 1

    ready = prefill_count >= expected_prefill and decode_count >= expected_decode

    if ready:
        message = f"Model is ready. Have {prefill_count} prefills and {decode_count} decodes."
    else:
        message = (
            f"Model is not ready, waiting for "
            f"{expected_prefill - prefill_count} prefills and "
            f"{expected_decode - decode_count} decodes. "
            f"Have {prefill_count} prefills and {decode_count} decodes."
        )

    return WorkerHealthResult(
        ready=ready,
        message=message,
        prefill_ready=prefill_count,
        prefill_expected=expected_prefill,
        decode_ready=decode_count,
        decode_expected=expected_decode,
    )


# ============================================================================
# Port and Basic Health Waiting
# ============================================================================


def check_trtllm_serve_health(
    response_json: dict,
    expected_prefill: int,
    expected_decode: int,
) -> WorkerHealthResult:
    """Check trtllm-serve disaggregated health.

    trtllm-serve's /health returns HTTP 200 once the orchestrator is up (the body may
    be empty). The trtllm_serve frontend already gates each worker for readiness before
    starting the orchestrator, so a 200 here means the stack is ready.

    Note: wait_for_model() short-circuits to ready on the 200 for trtllm_serve and does
    not call this, so this exists mainly as the FrontendProtocol.parse_health hook.
    """
    return WorkerHealthResult(
        ready=True,
        message="trtllm-serve orchestrator healthy",
        prefill_ready=expected_prefill,
        prefill_expected=expected_prefill,
        decode_ready=expected_decode,
        decode_expected=expected_decode,
    )


def check_vllm_health(
    host: str,
    port: int,
    health_url: str,
) -> WorkerHealthResult:
    """Check direct vLLM OpenAI server readiness.

    vLLM's /health returns HTTP 200 when the server is up; /v1/models verifies
    that the OpenAI serving stack has exposed at least one model.
    """
    models_url = f"http://{host}:{port}/v1/models"
    try:
        models_response = requests.get(models_url, timeout=5.0)
        if models_response.status_code != 200:
            return WorkerHealthResult(
                ready=False,
                message=f"vLLM /health is up but /v1/models returned {models_response.status_code}",
            )
        data = models_response.json()
        models = data.get("data", [])
        if models:
            return WorkerHealthResult(
                ready=True,
                message=f"vLLM OpenAI server ready at {health_url}; {len(models)} model(s) available",
                decode_ready=1,
                decode_expected=1,
            )
        return WorkerHealthResult(
            ready=False,
            message="vLLM /health is up but /v1/models has no models",
        )
    except (requests.exceptions.RequestException, ValueError) as e:
        return WorkerHealthResult(
            ready=False,
            message=f"vLLM /health is up but /v1/models check failed: {e}",
        )


def wait_for_port(
    host: str,
    port: int,
    timeout: float = 60.0,
    interval: float = 1.0,
) -> bool:
    """Wait for a TCP port to become available.

    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds

    Returns:
        True if port became available, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(interval)

    return False


def wait_for_health(
    host: str,
    port: int,
    max_attempts: int = 60,
    interval: float = 10.0,
    expected_workers: int | None = None,
    stop_event: threading.Event | None = None,
) -> bool:
    """Wait for HTTP health endpoint to return healthy status.

    Checks /health endpoint and optionally /v1/models for worker readiness.

    Args:
        host: Hostname or IP address
        port: HTTP port
        max_attempts: Maximum number of attempts
        interval: Time between attempts in seconds
        expected_workers: Expected number of workers (checks /v1/models)
        stop_event: Optional threading.Event to abort waiting

    Returns:
        True if healthy, False if timeout or aborted
    """
    health_url = f"http://{host}:{port}/health"
    models_url = f"http://{host}:{port}/v1/models"

    for attempt in range(max_attempts):
        if stop_event and stop_event.is_set():
            logger.warning("Wait aborted by stop event")
            return False

        try:
            # Check health endpoint
            response = requests.get(health_url, timeout=5.0)
            if response.status_code != 200:
                logger.debug(
                    "Health check failed (attempt %d/%d): status %d",
                    attempt + 1,
                    max_attempts,
                    response.status_code,
                )
                time.sleep(interval)
                continue

            # If expected_workers specified, check /v1/models
            if expected_workers is not None:
                try:
                    models_response = requests.get(models_url, timeout=5.0)
                    if models_response.status_code == 200:
                        data = models_response.json()
                        # Check if we have the expected number of workers
                        # The response format depends on the backend
                        models = data.get("data", [])
                        if len(models) > 0:
                            logger.info(
                                "Health check passed: %d models available",
                                len(models),
                            )
                            return True
                except Exception as e:
                    logger.debug("Models check failed: %s", e)
                    time.sleep(interval)
                    continue
            else:
                logger.info("Health check passed")
                return True

        except requests.exceptions.RequestException as e:
            logger.debug(
                "Health check failed (attempt %d/%d): %s",
                attempt + 1,
                max_attempts,
                e,
            )

        time.sleep(interval)

    logger.error("Health check failed after %d attempts", max_attempts)
    return False


def wait_for_etcd(
    etcd_url: str,
    max_retries: int = 60,
    interval: float = 2.0,
) -> bool:
    """Wait for etcd to be ready.

    Args:
        etcd_url: Base URL of etcd (e.g., http://node1:2379)
        max_retries: Maximum number of retries
        interval: Time between retries in seconds

    Returns:
        True if etcd is ready, False if timeout
    """
    health_url = f"{etcd_url}/health"

    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5.0)
            if response.status_code == 200:
                logger.info("etcd is ready")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.debug(
            "etcd not ready (attempt %d/%d), retrying...",
            attempt + 1,
            max_retries,
        )
        time.sleep(interval)

    logger.error("etcd not ready after %d attempts", max_retries)
    return False


# ============================================================================
# Wait for Model (replaces bash wait_for_model)
# ============================================================================


def wait_for_model(
    host: str,
    port: int,
    n_prefill: int = 1,
    n_decode: int = 1,
    poll_interval: float = 1.0,
    timeout: float = 600.0,
    report_every: float = 60.0,
    frontend_type: str = "dynamo",
    stop_event: threading.Event | None = None,
) -> bool:
    """Wait for model to be ready with expected worker counts.

    This is the pure Python replacement for the bash wait_for_model function.
    It polls the appropriate health endpoint and validates worker counts.

    Args:
        host: Model server hostname or IP
        port: Model server port
        n_prefill: Expected number of prefill workers
        n_decode: Expected number of decode workers
        poll_interval: Seconds between health checks
        timeout: Maximum wait time in seconds
        report_every: Log progress every N seconds
        frontend_type: Frontend type - "sglang" uses /workers, "dynamo" uses /health
        stop_event: Optional threading.Event to abort waiting

    Returns:
        True if model is ready with expected workers, False if timeout/aborted
    """
    if frontend_type == "sglang":
        health_url = f"http://{host}:{port}/workers"
        logger.info(
            "Polling %s every %.1fs for %d prefills and %d decodes (sglang frontend)",
            health_url,
            poll_interval,
            n_prefill,
            n_decode,
        )
    else:
        health_url = f"http://{host}:{port}/health"
        logger.info(
            "Polling %s every %.1fs for %d prefills and %d decodes",
            health_url,
            poll_interval,
            n_prefill,
            n_decode,
        )

    start_time = time.time()
    last_report_time = start_time

    while True:
        # Check for abort
        if stop_event and stop_event.is_set():
            logger.warning("Wait for model aborted by stop event")
            return False

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            logger.error("Model did not get healthy in %.0f seconds", timeout)
            return False

        # Try to fetch health
        try:
            response = requests.get(health_url, timeout=5.0)
            if response.status_code == 200:
                # trtllm-serve /health may return an empty body; a 200 is sufficient
                # (workers were gated by the frontend before the orchestrator started).
                if frontend_type == "trtllm_serve":
                    logger.info("trtllm-serve frontend healthy at %s", health_url)
                    return True
                if frontend_type == "vllm":
                    result = check_vllm_health(host, port, health_url)
                    if result.ready:
                        logger.info(result.message)
                        return True
                    if time.time() - last_report_time >= report_every:
                        logger.info(result.message)
                        last_report_time = time.time()
                    time.sleep(poll_interval)
                    continue

                response_json = response.json()

                # Check worker counts based on frontend type
                if frontend_type == "sglang":
                    result = check_sglang_router_health(response_json, n_prefill, n_decode)
                else:
                    result = check_dynamo_health(response_json, n_prefill, n_decode)

                if result.ready:
                    logger.info(result.message)
                    return True

                # Report progress periodically
                if time.time() - last_report_time >= report_every:
                    logger.info(result.message)
                    last_report_time = time.time()

        except requests.exceptions.RequestException as e:
            # Report connection errors periodically
            if time.time() - last_report_time >= report_every:
                logger.debug("Health check failed: %s", e)
                last_report_time = time.time()
        except Exception as e:
            logger.debug("Unexpected error during health check: %s", e)

        time.sleep(poll_interval)
