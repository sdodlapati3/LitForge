"""
Framework integrations for LitForge.

Provides adapters for popular AI agent frameworks:
- CrewAI
- LangGraph
- LangChain
"""

from litforge.integrations.crewai import LitForgeTools as CrewAITools
from litforge.integrations.langchain import (
    LitForgeCitationTool,
    LitForgeQATool,
    LitForgeRetrieveTool,
    LitForgeSearchTool,
)
from litforge.integrations.langgraph import LitForgeToolkit as LangGraphToolkit

__all__ = [
    "CrewAITools",
    "LangGraphToolkit",
    "LitForgeSearchTool",
    "LitForgeRetrieveTool",
    "LitForgeCitationTool",
    "LitForgeQATool",
]
