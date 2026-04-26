# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

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

mcp = FastMCP(
    "srtctl-spec",
    host=os.getenv("SRTCTL_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("SRTCTL_MCP_PORT", "18082")),
)


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


def main() -> None:
    transport = os.getenv("SRTCTL_MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.run(transport="streamable-http")
    elif transport == "stdio":
        mcp.run()
    else:
        raise ValueError(f"Unsupported MCP transport: {transport}")


if __name__ == "__main__":
    main()
