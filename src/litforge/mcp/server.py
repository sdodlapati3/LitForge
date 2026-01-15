"""
MCP Server for LitForge.

Implements the Model Context Protocol server that exposes LitForge
functionality to AI assistants like Claude, ChemAgent, and OmicsOracle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def create_server(forge: Any = None) -> Any:
    """
    Create an MCP server instance.
    
    Args:
        forge: Optional Forge instance to use for tools.
        
    Returns:
        MCP Server instance.
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import (
            CallToolResult,
            TextContent,
            Tool,
        )
    except ImportError:
        raise ImportError(
            "MCP not installed. Install with: pip install mcp"
        )

    from litforge.mcp.tools import LitForgeTools

    # Create server
    server = Server("litforge")

    # Initialize tools
    tools = LitForgeTools(forge)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available LitForge tools."""
        definitions = tools.get_tool_definitions()
        return [
            Tool(
                name=d["name"],
                description=d["description"],
                inputSchema=d["inputSchema"],
            )
            for d in definitions
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Execute a LitForge tool."""
        try:
            # Route to appropriate tool method
            if name == "smart_search":
                result = await tools.smart_search(**arguments)
            elif name == "search_papers":
                result = await tools.search_papers(**arguments)
            elif name == "lookup_paper":
                result = await tools.lookup_paper(**arguments)
            elif name == "get_citations":
                result = await tools.get_citations(**arguments)
            elif name == "get_references":
                result = await tools.get_references(**arguments)
            elif name == "retrieve_fulltext":
                result = await tools.retrieve_fulltext(**arguments)
            elif name == "index_papers":
                result = await tools.index_papers(**arguments)
            elif name == "ask_papers":
                result = await tools.ask_papers(**arguments)
            elif name == "build_citation_network":
                result = await tools.build_citation_network(**arguments)
            elif name == "summarize_papers":
                result = await tools.summarize_papers(**arguments)
            else:
                result = {"success": False, "error": f"Unknown tool: {name}"}

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                }))]
            )

    return server


async def serve(forge: Any = None) -> None:
    """
    Start the MCP server.
    
    This function starts the MCP server and runs until interrupted.
    It communicates via stdio (stdin/stdout).
    
    Args:
        forge: Optional Forge instance to use for tools.
    """
    try:
        from mcp.server.stdio import stdio_server
    except ImportError:
        raise ImportError(
            "MCP not installed. Install with: pip install mcp"
        )

    server = create_server(forge)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Main entry point for the MCP server."""
    # Configure logging to stderr (stdout is for MCP)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Run the server
    asyncio.run(serve())


if __name__ == "__main__":
    main()
