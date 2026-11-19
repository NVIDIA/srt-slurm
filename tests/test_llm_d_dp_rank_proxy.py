# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

from aiohttp import ClientSession, web

from srtctl.runtime_scripts import llm_d_dp_rank_proxy as proxy

METRICS = """# HELP sglang:num_running_reqs Running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{dp_rank="0",model_name="m"} 3.0
sglang:num_running_reqs{dp_rank="1",model_name="m"} 7.0
sglang:num_running_reqs{dp_rank="10",model_name="m"} 9.0
process_cpu_seconds_total 12.5
"""


def test_rank_metrics_keeps_the_rank_and_unlabelled_samples() -> None:
    assert proxy.rank_metrics(METRICS, 1) == (
        "# HELP sglang:num_running_reqs Running requests\n"
        "# TYPE sglang:num_running_reqs gauge\n"
        'sglang:num_running_reqs{dp_rank="1",model_name="m"} 7.0\n'
        "process_cpu_seconds_total 12.5\n"
    )


def test_rank_metrics_are_unfiltered_for_a_rank_the_node_does_not_export() -> None:
    """A multi-node server only exports its leader node's ranks."""
    assert proxy.rank_metrics(METRICS, 4) == METRICS


def test_listeners_of_one_server_share_a_metrics_fetch_per_scrape_interval() -> None:
    fetches = 0

    class Response:
        async def text(self) -> str:
            return METRICS

    class Session:
        @asynccontextmanager
        async def get(self, url: str, headers: dict[str, str]) -> AsyncIterator[Response]:
            nonlocal fetches
            fetches += 1
            assert headers == {"Accept-Encoding": "identity"}
            yield Response()

    async def scrape() -> None:
        metrics = proxy.shared_metrics(Session(), "http://server/metrics")  # type: ignore[arg-type]
        now = 100.0
        with patch.object(proxy.time, "monotonic", side_effect=lambda: now):
            assert await asyncio.gather(*(metrics() for _ in range(8))) == [METRICS] * 8
            assert fetches == 1
            now += 2 * proxy.METRICS_TTL_S
            await metrics()
            assert fetches == 2

    asyncio.run(scrape())


def test_rank_app_pins_rank_header_preserves_sse_chunks_and_filters_metrics() -> None:
    seen_headers: list[dict[str, str]] = []
    sse_events = [b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"]

    async def upstream_metrics(request: web.Request) -> web.Response:
        return web.Response(text=METRICS, content_type="text/plain")

    async def upstream_chat(request: web.Request) -> web.StreamResponse:
        seen_headers.append(dict(request.headers))
        body = await request.read()
        assert body == b'{"stream":true}'
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "X-Upstream": "ok"})
        await response.prepare(request)
        for event in sse_events:
            await response.write(event)
        await response.write_eof()
        return response

    async def exercise() -> None:
        upstream_app = web.Application()
        upstream_app.router.add_get("/metrics", upstream_metrics)
        upstream_app.router.add_post("/v1/chat/completions", upstream_chat)
        upstream_runner = web.AppRunner(upstream_app)
        await upstream_runner.setup()
        upstream_site = web.TCPSite(upstream_runner, "127.0.0.1", 0)
        await upstream_site.start()
        assert upstream_site._server is not None
        upstream_port = upstream_site._server.sockets[0].getsockname()[1]
        upstream_url = f"http://127.0.0.1:{upstream_port}"

        async with ClientSession(auto_decompress=False) as session:
            metrics = proxy.shared_metrics(session, f"{upstream_url}/metrics")
            app = proxy.rank_app(session, upstream_url, 1, metrics)
            proxy_runner = web.AppRunner(app)
            await proxy_runner.setup()
            proxy_site = web.TCPSite(proxy_runner, "127.0.0.1", 0)
            await proxy_site.start()
            assert proxy_site._server is not None
            proxy_port = proxy_site._server.sockets[0].getsockname()[1]
            proxy_url = f"http://127.0.0.1:{proxy_port}"

            try:
                async with session.get(f"{proxy_url}/metrics") as resp:
                    assert resp.status == 200
                    assert await resp.text() == proxy.rank_metrics(METRICS, 1)

                received_chunks: list[bytes] = []
                async with session.post(
                    f"{proxy_url}/v1/chat/completions?trace=1",
                    headers={"X-Custom": "keep", proxy.RANK_HEADER: "99"},
                    data=b'{"stream":true}',
                ) as resp:
                    assert resp.status == 200
                    assert resp.headers.get("X-Upstream") == "ok"
                    chunk = b""
                    async for data, end_of_chunk in resp.content.iter_chunks():
                        chunk += data
                        if end_of_chunk:
                            received_chunks.append(chunk)
                            chunk = b""
                assert received_chunks == sse_events
                assert len(seen_headers) == 1
                assert seen_headers[0][proxy.RANK_HEADER] == "1"
                assert seen_headers[0]["X-Custom"] == "keep"
            finally:
                await proxy_runner.cleanup()
        await upstream_runner.cleanup()

    asyncio.run(exercise())
