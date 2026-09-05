# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang Model Gateway router frontend."""

from typing import Any, ClassVar

from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends.static_router import StaticRouterFrontend


class SGLangFrontend(StaticRouterFrontend):
    """SGLang Model Gateway static router."""

    type: ClassVar[str] = "sglang"
    backend_type: ClassVar[str] = "sglang"
    executable: ClassVar[tuple[str, ...]] = ("python", "-m", "sglang_router.launch_router")
    pd_flag: ClassVar[str] = "--pd-disaggregation"
    process_name: ClassVar[str] = "sglang_router"
    log_label: ClassVar[str] = "router"
    # Preserve the historical launch shape used by dry-run/topology callers
    # that construct the frontend before populating worker processes.
    allow_empty_workers: ClassVar[bool] = True

    def worker_scheme(self, backend: Any, mode: str) -> str:
        return "grpc" if backend.is_grpc_mode(mode) else "http"

    def resolve_worker_host(self, node: str, network_interface: str | None) -> str:
        del network_interface
        return get_hostname_ip(node)

    def start_process(self, **kwargs: Any) -> Any:
        return start_srun_process(**kwargs)
