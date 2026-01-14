"""
CrewAI integration for LitForge.

Provides CrewAI-compatible tools for scientific literature operations.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LitForgeTools:
    """
    LitForge tools for CrewAI.
    
    Provides a set of CrewAI-compatible tools for literature search,
    retrieval, citation analysis, and Q&A.
    
    Usage:
        from crewai import Agent, Task, Crew
        from litforge.integrations.crewai import LitForgeTools
        
        tools = LitForgeTools()
        
        researcher = Agent(
            role="Literature Researcher",
            goal="Find and synthesize relevant papers",
            tools=tools.all(),
        )
    """

    def __init__(self, forge: Any = None):
        """
        Initialize LitForge tools for CrewAI.
        
        Args:
            forge: Optional Forge instance. If not provided, will be created lazily.
        """
        self._forge = forge
        self._tools: list[Any] = []

    @property
    def forge(self) -> Any:
        """Get or create Forge instance."""
        if self._forge is None:
            from litforge.core.forge import Forge
            self._forge = Forge()
        return self._forge

    def all(self) -> list[Any]:
        """Get all available tools."""
        if not self._tools:
            self._tools = [
                self.search_tool(),
                self.lookup_tool(),
                self.retrieve_tool(),
                self.citations_tool(),
                self.references_tool(),
                self.ask_tool(),
            ]
        return self._tools

    def search_tool(self) -> Any:
        """Create paper search tool."""
        try:
            from crewai_tools import BaseTool
        except ImportError:
            raise ImportError(
                "CrewAI tools not installed. Install with: pip install crewai-tools"
            )

        forge = self.forge

        class SearchPapersTool(BaseTool):
            name: str = "search_papers"
            description: str = (
                "Search for scientific papers across OpenAlex, Semantic Scholar, "
                "PubMed, and arXiv. Input should be a search query string. "
                "Returns paper metadata including titles, authors, abstracts, and DOIs."
            )

            def _run(self, query: str) -> str:
                """Execute the search."""
                import asyncio

                async def do_search():
                    papers = await forge.search(query=query, limit=20)
                    results = []
                    for p in papers:
                        results.append({
                            "doi": p.doi,
                            "title": p.title,
                            "authors": [a.name for a in p.authors][:5] if p.authors else [],
                            "year": p.year,
                            "abstract": p.abstract[:500] if p.abstract else None,
                            "citation_count": p.citation_count,
                        })
                    return json.dumps(results, indent=2)

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                        return loop.run_until_complete(do_search())
                    else:
                        return asyncio.run(do_search())
                except Exception as e:
                    return f"Error searching papers: {e}"

        return SearchPapersTool()

    def lookup_tool(self) -> Any:
        """Create paper lookup tool."""
        try:
            from crewai_tools import BaseTool
        except ImportError:
            raise ImportError(
                "CrewAI tools not installed. Install with: pip install crewai-tools"
            )

        forge = self.forge

        class LookupPaperTool(BaseTool):
            name: str = "lookup_paper"
            description: str = (
                "Look up a specific paper by DOI. Input should be a DOI string "
                "(e.g., '10.1038/nature12373'). Returns detailed paper metadata."
            )

            def _run(self, doi: str) -> str:
                """Look up a paper."""
                import asyncio

                async def do_lookup():
                    paper = await forge.lookup(doi=doi.strip())
                    if paper:
                        return json.dumps({
                            "doi": paper.doi,
                            "title": paper.title,
                            "authors": [a.name for a in paper.authors] if paper.authors else [],
                            "year": paper.year,
                            "abstract": paper.abstract,
                            "venue": paper.venue,
                            "citation_count": paper.citation_count,
                            "open_access": paper.open_access,
                        }, indent=2)
                    return "Paper not found"

                try:
                    return asyncio.run(do_lookup())
                except Exception as e:
                    return f"Error looking up paper: {e}"

        return LookupPaperTool()

    def retrieve_tool(self) -> Any:
        """Create full-text retrieval tool."""
        try:
            from crewai_tools import BaseTool
        except ImportError:
            raise ImportError(
                "CrewAI tools not installed. Install with: pip install crewai-tools"
            )

        forge = self.forge

        class RetrieveFulltextTool(BaseTool):
            name: str = "retrieve_fulltext"
            description: str = (
                "Download and extract full text from a paper. Input should be a DOI. "
                "Finds open access versions from Unpaywall, PMC, or arXiv."
            )

            def _run(self, doi: str) -> str:
                """Retrieve full text."""
                import asyncio

                async def do_retrieve():
                    content = await forge.retrieve(doi=doi.strip())
                    if content:
                        # Truncate if too long
                        if len(content) > 10000:
                            return content[:10000] + "\n\n[Content truncated...]"
                        return content
                    return "Full text not available"

                try:
                    return asyncio.run(do_retrieve())
                except Exception as e:
                    return f"Error retrieving full text: {e}"

        return RetrieveFulltextTool()

    def citations_tool(self) -> Any:
        """Create citations tool."""
        try:
            from crewai_tools import BaseTool
        except ImportError:
            raise ImportError(
                "CrewAI tools not installed. Install with: pip install crewai-tools"
            )

        forge = self.forge

        class GetCitationsTool(BaseTool):
            name: str = "get_citations"
            description: str = (
                "Get papers that cite a given paper. Input should be a DOI. "
                "Useful for finding follow-up research and impact assessment."
            )

            def _run(self, doi: str) -> str:
                """Get citing papers."""
                import asyncio

                async def do_get_citations():
                    citations = await forge.get_citations(doi=doi.strip(), limit=20)
                    results = []
                    for p in citations:
                        results.append({
                            "doi": p.doi,
                            "title": p.title,
                            "year": p.year,
                            "citation_count": p.citation_count,
                        })
                    return json.dumps(results, indent=2)

                try:
                    return asyncio.run(do_get_citations())
                except Exception as e:
                    return f"Error getting citations: {e}"

        return GetCitationsTool()

    def references_tool(self) -> Any:
        """Create references tool."""
        try:
            from crewai_tools import BaseTool
        except ImportError:
            raise ImportError(
                "CrewAI tools not installed. Install with: pip install crewai-tools"
            )

        forge = self.forge

        class GetReferencesTool(BaseTool):
            name: str = "get_references"
            description: str = (
                "Get papers referenced by a given paper. Input should be a DOI. "
                "Useful for understanding foundational work and context."
            )

            def _run(self, doi: str) -> str:
                """Get referenced papers."""
                import asyncio

                async def do_get_references():
                    refs = await forge.get_references(doi=doi.strip(), limit=20)
                    results = []
                    for p in refs:
                        results.append({
                            "doi": p.doi,
                            "title": p.title,
                            "year": p.year,
                            "citation_count": p.citation_count,
                        })
                    return json.dumps(results, indent=2)

                try:
                    return asyncio.run(do_get_references())
                except Exception as e:
                    return f"Error getting references: {e}"

        return GetReferencesTool()

    def ask_tool(self) -> Any:
        """Create Q&A tool."""
        try:
            from crewai_tools import BaseTool
        except ImportError:
            raise ImportError(
                "CrewAI tools not installed. Install with: pip install crewai-tools"
            )

        forge = self.forge

        class AskPapersTool(BaseTool):
            name: str = "ask_papers"
            description: str = (
                "Ask a question about indexed papers using RAG. Input should be a question. "
                "Papers must be indexed first using the index_papers tool. "
                "Returns an answer with citations to source papers."
            )

            def _run(self, question: str) -> str:
                """Ask a question."""
                import asyncio

                async def do_ask():
                    answer = await forge.ask(question=question)
                    response = {
                        "answer": answer.text,
                        "citations": [
                            {"doi": c.doi, "title": c.title}
                            for c in answer.citations
                        ],
                    }
                    return json.dumps(response, indent=2)

                try:
                    return asyncio.run(do_ask())
                except Exception as e:
                    return f"Error asking papers: {e}"

        return AskPapersTool()
