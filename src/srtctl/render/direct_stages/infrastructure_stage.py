# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local NATS, etcd, and Mooncake readiness stage for direct execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common import ManagedProcess


class InfrastructureStageMixin:
    """Start the direct runner's owned infrastructure services."""

    plan: dict[str, Any]
    source_dir: Path
    output_dir: Path

    def _die(self, message: str) -> None:
        raise NotImplementedError

    def _launch(self, label: str, log_name: str, args: list[str], **kwargs: Any) -> ManagedProcess:
        raise NotImplementedError

    def _wait_http_ready(self, url: str, label: str) -> None:
        raise NotImplementedError

    def _wait_tcp_ready(self, host: str, port: int, label: str) -> None:
        raise NotImplementedError

    def _start_infrastructure(self) -> None:
        nats = str(self.source_dir / "configs" / "nats-server")
        etcd = str(self.source_dir / "configs" / "etcd")
        if not os.access(nats, os.X_OK):
            self._die(f"NATS binary is not executable: {nats}")
        if not os.access(etcd, os.X_OK):
            self._die(f"etcd binary is not executable: {etcd}")
        (self.output_dir / "nats").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "etcd").mkdir(parents=True, exist_ok=True)
        self._launch(
            "nats",
            "nats.log",
            [nats, "-js", "-a", "127.0.0.1", "-p", str(self.plan["nats_port"]), "-sd", str(self.output_dir / "nats")],
        )
        client_port = str(self.plan["etcd_client_port"])
        peer_port = str(self.plan["etcd_peer_port"])
        self._launch(
            "etcd",
            "etcd.log",
            [
                etcd,
                "--data-dir",
                str(self.output_dir / "etcd"),
                "--listen-client-urls",
                f"http://127.0.0.1:{client_port}",
                "--advertise-client-urls",
                f"http://127.0.0.1:{client_port}",
                "--listen-peer-urls",
                f"http://127.0.0.1:{peer_port}",
                "--initial-advertise-peer-urls",
                f"http://127.0.0.1:{peer_port}",
                "--initial-cluster",
                f"default=http://127.0.0.1:{peer_port}",
            ],
        )
        self._wait_http_ready(f"http://127.0.0.1:{client_port}/health", "etcd")

    def _start_mooncake(self) -> None:
        command = [str(value) for value in self.plan["mooncake_master_command"]]
        if not command:
            return
        self._wait_tcp_ready("127.0.0.1", int(self.plan["mooncake_master_port"]), "mooncake master")
        self._wait_tcp_ready("127.0.0.1", int(self.plan["mooncake_metadata_port"]), "mooncake metadata")
        self._wait_tcp_ready("127.0.0.1", int(self.plan["mooncake_metrics_port"]), "mooncake metrics")
