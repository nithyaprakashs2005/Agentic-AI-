"""FastMCP server factory.

Exposes the tool bodies from `app.mcp.tools` over the MCP protocol, so the same
implementations serve both the in-process agent path and an external MCP client.
Registration is driven off `REGISTRY`, which means a new tool becomes available
on both paths at once and the two can never drift.

FastMCP is an optional extra (`uv sync --extra mcp`); this module imports fine
without it and reports the requirement when a server is actually started.
"""

from __future__ import annotations

from typing import Any

from app.mcp.tools import REGISTRY, ToolSpec, call_tool, tools_for
from app.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

SERVERS: tuple[str, ...] = ("crm", "billing", "knowledge", "notifications")


class McpDependencyError(RuntimeError):
    """FastMCP is not installed."""


def build_server(server_name: str) -> Any:
    """Construct a FastMCP server exposing one domain's tools."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise McpDependencyError("fastmcp is not installed. Run: uv sync --extra mcp") from exc

    specs = tools_for(server_name)
    if not specs:
        raise ValueError(f"unknown MCP server {server_name!r}; expected one of {SERVERS}")

    mcp = FastMCP(name=f"support-{server_name}")
    for spec in specs:
        _register(mcp, spec)

    log.info(
        "mcp.server_built",
        extra={"server": server_name, "tools": [s.name for s in specs]},
    )
    return mcp


def _register(mcp: Any, spec: ToolSpec) -> None:
    """Register one tool, routing through `call_tool` for timeout enforcement."""
    # The MCP tool name cannot contain a dot in some clients, so the namespace
    # separator becomes an underscore on the wire.
    wire_name = spec.name.replace(".", "_")

    async def handler(**kwargs: Any) -> dict[str, Any]:
        return await call_tool(spec.name, kwargs)

    handler.__name__ = wire_name
    handler.__doc__ = spec.description

    mcp.tool(
        name=wire_name,
        description=spec.description,
    )(handler)


def describe_all() -> dict[str, Any]:
    """Inventory of every server and tool, for docs and health reporting."""
    return {
        "servers": {
            name: [
                {
                    "name": spec.name,
                    "wire_name": spec.name.replace(".", "_"),
                    "description": spec.description,
                    "input_schema": spec.schema.model_json_schema(),
                }
                for spec in tools_for(name)
            ]
            for name in SERVERS
        },
        "total_tools": len(REGISTRY),
    }


def main(server_name: str) -> None:
    """Run one MCP server over stdio."""
    configure_logging("INFO")
    build_server(server_name).run()
