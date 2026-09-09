# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

    _V1 = True
except ImportError:  # mcp 2.x renamed FastMCP to MCPServer and moved host/port to run()
    from mcp.server.mcpserver import MCPServer as _Server

    _V1 = False

from srtctl.mcp import job_tools
from srtctl.mcp.spec_tools import (
    explain_field as explain_field_impl,
)
from srtctl.mcp.spec_tools import (
    get_config_reference as get_config_reference_impl,
)
from srtctl.mcp.spec_tools import (
    preflight_config as preflight_config_impl,
)
from srtctl.mcp.spec_tools import (
    resolve_config as resolve_config_impl,
)
from srtctl.mcp.spec_tools import (
    schema_summary as schema_summary_impl,
)
from srtctl.mcp.spec_tools import (
    validate_config as validate_config_impl,
)

_HOST = os.getenv("SRTCTL_MCP_HOST", "127.0.0.1")
_PORT = int(os.getenv("SRTCTL_MCP_PORT", "18082"))
mcp = _Server("srtctl-spec", host=_HOST, port=_PORT) if _V1 else _Server("srtctl-spec")


@mcp.tool()
def health() -> dict[str, str]:
    """Return basic liveness for the srtctl spec MCP."""
    return {"status": "ok"}


@mcp.tool()
def schema_summary() -> dict[str, Any]:
    """Return a compact summary of the top-level SrtConfig fields."""
    return schema_summary_impl()


@mcp.tool()
def get_config_reference(query: str | None = None, max_matches: int = 5) -> dict[str, Any]:
    """Search docs/config-reference.md and return relevant snippets."""
    return get_config_reference_impl(query=query, max_matches=max_matches)


@mcp.tool()
def explain_field(path: str) -> dict[str, Any]:
    """Explain a config field path using schema introspection plus config-reference docs."""
    return explain_field_impl(path)


@mcp.tool()
def validate_config(
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = False,
) -> dict[str, Any]:
    """Validate recipe structure only; never read host-side srtslurm.yaml."""
    return validate_config_impl(
        config=config,
        config_yaml=config_yaml,
        apply_cluster_defaults=apply_cluster_defaults,
    )


@mcp.tool()
def preflight_config(
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = False,
) -> dict[str, Any]:
    """Check explicit local paths only; run cluster checks compute-side."""
    return preflight_config_impl(
        config=config,
        config_yaml=config_yaml,
        apply_cluster_defaults=apply_cluster_defaults,
    )


@mcp.tool()
def resolve_config(
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = False,
) -> dict[str, Any]:
    """Resolve schema-only defaults without reading host-side srtslurm.yaml."""
    return resolve_config_impl(
        config=config,
        config_yaml=config_yaml,
        apply_cluster_defaults=apply_cluster_defaults,
    )


# -- job lifecycle -------------------------------------------------------------------
# These call srtctl and Slurm on the machine running the server: start it on a
# login node of the target cluster, inside the checkout that has its srtslurm.yaml.


@mcp.tool()
def submit_job(
    config_path: str,
    set_overrides: list[str] | None = None,
    unset: list[str] | None = None,
    tags: list[str] | None = None,
    serve_only: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Submit a recipe with `srtctl apply -y --json`; returns slurm_job_id and output_dir per submission."""
    return job_tools.submit_job(
        config_path,
        set_overrides=set_overrides,
        unset=unset,
        tags=tags,
        serve_only=serve_only,
        output_dir=output_dir,
    )


@mcp.tool()
def dry_run(
    config_path: str,
    set_overrides: list[str] | None = None,
    unset: list[str] | None = None,
) -> dict[str, Any]:
    """Render a recipe with `srtctl dry-run` (sbatch script, services, mounts, env) without submitting."""
    return job_tools.dry_run(config_path, set_overrides=set_overrides, unset=unset)


@mcp.tool()
def job_status(job_id: str, output_dir: str | None = None, tail: int = 20) -> dict[str, Any]:
    """Slurm accounting plus the job's metadata, current stage, errors, rollup, and sweep-log tail."""
    return job_tools.job_status(job_id, output_dir=output_dir, tail=tail)


@mcp.tool()
def job_logs(job_id: str, name: str | None = None, tail: int = 200, output_dir: str | None = None) -> dict[str, Any]:
    """List a job's log files, or the tail of one (sweep_<id>.log, <node>_<mode>_w<i>.out, service_<name>.out)."""
    return job_tools.job_logs(job_id, name=name, tail=tail, output_dir=output_dir)


@mcp.tool()
def list_jobs(user: str | None = None) -> dict[str, Any]:
    """Pending and running Slurm jobs of a user (default: the current user)."""
    return job_tools.list_jobs(user)


@mcp.tool()
def cancel_job(job_id: str) -> dict[str, Any]:
    """scancel a job; srtctl stops every step it launched on the way out."""
    return job_tools.cancel_job(job_id)


def main() -> None:
    transport = os.getenv("SRTCTL_MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        if _V1:
            mcp.run(transport="streamable-http")
        else:
            mcp.run(transport="streamable-http", host=_HOST, port=_PORT)
    elif transport == "stdio":
        mcp.run()
    else:
        raise ValueError(f"Unsupported MCP transport: {transport}")


if __name__ == "__main__":
    main()
