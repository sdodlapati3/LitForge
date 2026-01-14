"""
Tests for framework integrations.
"""

import pytest


class TestCrewAIIntegration:
    """Tests for CrewAI integration."""

    def test_import(self):
        """Test that CrewAI integration can be imported."""
        from litforge.integrations.crewai import LitForgeTools

        assert LitForgeTools is not None

    def test_tools_creation(self):
        """Test creating LitForgeTools."""
        from litforge.integrations.crewai import LitForgeTools

        tools = LitForgeTools()
        assert tools is not None

    def test_forge_lazy_creation(self):
        """Test that forge is created lazily."""
        from litforge.integrations.crewai import LitForgeTools

        tools = LitForgeTools()
        # forge should be created on access
        assert tools.forge is not None


class TestLangGraphIntegration:
    """Tests for LangGraph integration."""

    def test_import(self):
        """Test that LangGraph integration can be imported."""
        from litforge.integrations.langgraph import LitForgeToolkit

        assert LitForgeToolkit is not None

    def test_toolkit_creation(self):
        """Test creating LitForgeToolkit."""
        from litforge.integrations.langgraph import LitForgeToolkit

        toolkit = LitForgeToolkit()
        assert toolkit is not None

    def test_forge_property(self):
        """Test that forge property works."""
        from litforge.integrations.langgraph import LitForgeToolkit

        toolkit = LitForgeToolkit()
        assert toolkit.forge is not None


class TestLangChainIntegration:
    """Tests for LangChain integration."""

    def test_search_tool_import(self):
        """Test that search tool can be imported."""
        from litforge.integrations.langchain import LitForgeSearchTool

        assert LitForgeSearchTool is not None

    def test_search_tool_creation(self):
        """Test creating search tool."""
        from litforge.integrations.langchain import LitForgeSearchTool

        tool = LitForgeSearchTool()
        assert tool.name == "search_papers"
        assert "search" in tool.description.lower()

    def test_qa_tool_import(self):
        """Test that Q&A tool can be imported."""
        from litforge.integrations.langchain import LitForgeQATool

        assert LitForgeQATool is not None

    def test_citations_tool_import(self):
        """Test that citations tool can be imported."""
        from litforge.integrations.langchain import LitForgeCitationTool

        assert LitForgeCitationTool is not None

    def test_retrieve_tool_import(self):
        """Test that retrieve tool can be imported."""
        from litforge.integrations.langchain import LitForgeRetrieveTool

        assert LitForgeRetrieveTool is not None


class TestIntegrationsModule:
    """Tests for the integrations module exports."""

    def test_module_exports(self):
        """Test the integrations module's public API."""
        from litforge import integrations

        assert hasattr(integrations, "CrewAITools")
        assert hasattr(integrations, "LangGraphToolkit")
        assert hasattr(integrations, "LitForgeSearchTool")
        assert hasattr(integrations, "LitForgeCitationTool")
        assert hasattr(integrations, "LitForgeQATool")
