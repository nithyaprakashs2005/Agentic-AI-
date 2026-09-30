"""Knowledge MCP server entry point (2026 stateless spec).

Run with: uv run python -m app.mcp.knowledge_server
Tool bodies live in app.mcp.tools; see app.mcp.server for the factory.
"""

from __future__ import annotations

from app.mcp.server import main

SERVER_NAME = "knowledge"

if __name__ == "__main__":
    main(SERVER_NAME)
