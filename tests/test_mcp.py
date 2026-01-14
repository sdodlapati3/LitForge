"""
Tests for MCP server and tools.
"""

import pytest


class TestMCPTools:
    """Tests for MCP tools definitions."""

    def test_tools_class_import(self):
        """Test that LitForgeTools class can be imported."""
        from litforge.mcp.tools import LitForgeTools

        assert LitForgeTools is not None

    def test_litforge_tools_creation(self):
        """Test creating LitForgeTools instance."""
        from litforge.mcp.tools import LitForgeTools

        tools = LitForgeTools()
        assert tools is not None

    def test_get_tool_definitions(self):
        """Test that get_tool_definitions returns expected tools."""
        from litforge.mcp.tools import LitForgeTools

        tools = LitForgeTools()
        definitions = tools.get_tool_definitions()

        assert isinstance(definitions, list)
        assert len(definitions) == 9  # We defined 9 tools

    def test_tool_names(self):
        """Test that all expected tools are present."""
        from litforge.mcp.tools import LitForgeTools

        tools = LitForgeTools()
        definitions = tools.get_tool_definitions()
        tool_names = {t["name"] for t in definitions}

        expected = {
            "search_papers",
            "lookup_paper",
            "get_citations",
            "get_references",
            "retrieve_fulltext",
            "index_papers",
            "ask_papers",
            "build_citation_network",
            "summarize_papers",
        }
        assert tool_names == expected

    def test_tool_descriptions(self):
        """Test that all tools have descriptions."""
        from litforge.mcp.tools import LitForgeTools

        tools = LitForgeTools()
        definitions = tools.get_tool_definitions()

        for tool in definitions:
            assert "description" in tool, f"Tool {tool['name']} missing description"
            assert len(tool["description"]) > 10


class TestMCPServer:
    """Tests for MCP server creation."""

    def test_server_module_import(self):
        """Test that server module can be imported."""
        from litforge.mcp import server

        assert hasattr(server, "create_server")
        assert hasattr(server, "serve")

    def test_create_server(self):
        """Test server creation."""
        from litforge.mcp.server import create_server
        from mcp.server import Server

        mcp_server = create_server()

        assert isinstance(mcp_server, Server)
        assert mcp_server.name == "litforge"

    def test_mcp_module_exports(self):
        """Test the MCP module's public API."""
        from litforge import mcp

        assert hasattr(mcp, "create_server")
        assert hasattr(mcp, "serve")
