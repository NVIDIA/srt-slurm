# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for trace-replay benchmark.

Runs the actual bench.sh script with aiperf against a mock OpenAI-compatible server.
These tests are slow and require aiperf to be installed, so they are marked with
@pytest.mark.integration and skipped by default (use `uv run pytest -m integration`).
"""

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPTS_DIR = Path(__file__).parent.parent / "src" / "srtctl" / "benchmarks" / "scripts"
BENCH_SCRIPT = SCRIPTS_DIR / "trace-replay" / "bench.sh"


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _create_mock_server(port: int):
    """Create a FastAPI mock server that mimics OpenAI API for aiperf.

    Handles:
    - GET /v1/models - model listing
    - POST /v1/chat/completions - streaming and non-streaming completions
    - GET /health - health check
    - GET /metrics - Prometheus metrics (aiperf probes this)
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse, StreamingResponse

    app = FastAPI()

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": "test-model", "object": "model", "owned_by": "test"}],
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        """Prometheus-style metrics endpoint (aiperf probes this)."""
        return PlainTextResponse(
            "# HELP up Server is up\n# TYPE up gauge\nup 1\n",
            media_type="text/plain",
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        stream = body.get("stream", False)
        # Try to respect max_tokens from request for realistic responses
        max_tokens = body.get("max_tokens", 5)
        # Cap at a reasonable number for test speed
        num_tokens = min(max_tokens, 20)

        if not stream:
            content = " ".join(["word"] * num_tokens)
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": num_tokens,
                    "total_tokens": 10 + num_tokens,
                },
            }

        def generate():
            for _i in range(num_tokens):
                chunk = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "word "},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": num_tokens,
                    "total_tokens": 10 + num_tokens,
                },
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock OpenAI server for the test session."""
    import uvicorn

    port = _find_free_port()
    app = _create_mock_server(port)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        pytest.fail("Mock server did not start in time")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.integration
class TestTraceReplayIntegration:
    """Integration tests that run bench.sh with real aiperf against a mock server."""

    def test_bench_script_exists(self):
        """bench.sh exists."""
        assert BENCH_SCRIPT.exists(), f"bench.sh not found at {BENCH_SCRIPT}"

    def test_fixture_exists(self):
        """Test fixture JSONL exists."""
        fixture = FIXTURES_DIR / "trace_replay_sample.jsonl"
        assert fixture.exists()

    def test_aiperf_installed(self):
        """aiperf is available."""
        result = subprocess.run(["aiperf", "--version"], capture_output=True, text=True)
        assert result.returncode == 0, f"aiperf not installed or broken: {result.stderr}"

    def test_bench_script_rejects_missing_trace(self, tmp_path):
        """bench.sh exits with error when trace file doesn't exist."""
        env = os.environ.copy()
        env["BASE_DIR"] = str(tmp_path)

        result = subprocess.run(
            [
                "bash",
                str(BENCH_SCRIPT),
                "http://localhost:9999",
                "test-model",
                "/nonexistent/trace.jsonl",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode != 0
        assert "not found" in result.stdout.lower()

    def test_full_trace_replay(self, mock_server, tmp_path):
        """Run bench.sh end-to-end against the mock server."""
        fixture = FIXTURES_DIR / "trace_replay_sample.jsonl"
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        env = os.environ.copy()
        env["BASE_DIR"] = str(log_dir)

        result = subprocess.run(
            [
                "bash",
                str(BENCH_SCRIPT),
                mock_server,             # ENDPOINT
                "test-model",            # MODEL_NAME
                str(fixture),            # TRACE_FILE
                "1",                     # CONCURRENCIES
                "5000",                  # TTFT_THRESHOLD
                "100",                   # ITL_THRESHOLD
                "test-model",            # TOKENIZER_PATH (aiperf resolves via HF)
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        print("STDERR:", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)

        assert result.returncode == 0, f"bench.sh failed (exit {result.returncode})"
        assert "Trace Replay Benchmark Complete" in result.stdout

        # Verify artifact directory was created with content
        artifact_dir = log_dir / "artifacts"
        assert artifact_dir.exists(), "artifacts/ directory not created"
        artifact_runs = list(artifact_dir.iterdir())
        assert len(artifact_runs) > 0, "No artifact subdirectories created"

    def test_multi_concurrency_sweep(self, mock_server, tmp_path):
        """bench.sh iterates over multiple concurrency levels."""
        fixture = FIXTURES_DIR / "trace_replay_sample.jsonl"
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        env = os.environ.copy()
        env["BASE_DIR"] = str(log_dir)

        result = subprocess.run(
            [
                "bash",
                str(BENCH_SCRIPT),
                mock_server,
                "test-model",
                str(fixture),
                "1,2",                   # two concurrency levels
                "5000",
                "100",
                "test-model",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        print("STDERR:", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)

        assert result.returncode == 0, f"bench.sh failed (exit {result.returncode})"

        # Should have run both concurrency levels
        assert "Running concurrency=1" in result.stdout
        assert "Running concurrency=2" in result.stdout

        # Should have created artifact dirs for both
        artifact_dir = log_dir / "artifacts"
        artifact_runs = list(artifact_dir.iterdir())
        assert len(artifact_runs) == 2, f"Expected 2 artifact dirs, got {len(artifact_runs)}: {artifact_runs}"
