# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outbound subscriber offsets must not hide derived listener ports."""

import ast
import textwrap

import pytest

from tests.test_design_rules import port_arithmetic


def _violations(body: str) -> set[str]:
    tree = ast.parse("def configure(frontend, process, runtime, model, rank):\n" + textwrap.indent(body, "    "))
    return {site for _, site in port_arithmetic("backends/example.py", tree)}


@pytest.mark.parametrize(
    "holder", ["", "target = frontend.kv_events_subscriber(process, runtime, model) if rank else None\n"]
)
@pytest.mark.parametrize("operation", ["+", "-"])
def test_subscriber_destination_offset_is_not_a_listener(holder: str, operation: str) -> None:
    source = "target" if holder else "frontend.kv_events_subscriber(process, runtime, model)"
    assert not _violations(
        holder
        + f"remote_host, remote_port, topic = {source}\n"
        + f'config = {{"endpoint": f"tcp://{{remote_host}}:{{remote_port {operation} rank}}"}}\n'
    )


@pytest.mark.parametrize(
    "body",
    [
        "socket.bind((remote_host, remote_port - rank))",
        "listener_port = remote_port + 1",
        'config = {"endpoint": f"tcp://*:{remote_port - rank}"}',
        'config = {"endpoint": f"tcp://{local_host}:{remote_port - rank}"}',
        'config = {"listen": f"tcp://{remote_host}:{remote_port - rank}"}',
        'remote_host = "0.0.0.0"\nconfig = {"endpoint": f"tcp://{remote_host}:{remote_port - rank}"}',
        'remote_port = 8000\nconfig = {"endpoint": f"tcp://{remote_host}:{remote_port - rank}"}',
    ],
)
def test_subscriber_port_does_not_exempt_listener_or_reassigned_address(body: str) -> None:
    assert _violations(
        "remote_host, remote_port, topic = frontend.kv_events_subscriber(process, runtime, model)\n" + body
    )


def test_unknown_endpoint_source_is_still_checked() -> None:
    assert _violations(
        'host, port, topic = frontend.worker_address(process)\nconfig = {"endpoint": f"tcp://{host}:{port - rank}"}\n'
    ) == {"port - rank"}


def test_subscriber_binding_does_not_leak_into_another_function() -> None:
    tree = ast.parse(
        "def publisher(frontend):\n"
        "    host, port, topic = frontend.kv_events_subscriber()\n"
        '    config = {"endpoint": f"tcp://{host}:{port - rank}"}\n'
        "def listener(host, port, rank):\n"
        '    config = {"endpoint": f"tcp://{host}:{port - rank}"}\n'
    )
    assert list(port_arithmetic("backends/example.py", tree)) == [("backends/example.py", "port - rank")]
