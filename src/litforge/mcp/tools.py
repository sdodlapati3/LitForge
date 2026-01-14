"""
MCP Tools for LitForge.

Defines the tools available through the MCP server for literature search,
retrieval, citation analysis, and Q&A.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LitForgeTools:
    """
    LitForge tools for MCP integration.
    
    Provides a set of tools that can be exposed through MCP to AI assistants
    for scientific literature operations.
    """

    def __init__(self, forge: Any = None):
        """
        Initialize LitForge tools.
        
        Args:
            forge: Optional Forge instance. If not provided, will be created lazily.
        """
        self._forge = forge

    @property
    def forge(self) -> Any:
        """Get or create Forge instance."""
        if self._forge is None:
            from litforge.core.forge import Forge
            self._forge = Forge()
        return self._forge

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get MCP tool definitions.
        
        Returns:
            List of tool definitions in MCP format.
        """
        return [
            {
                "name": "search_papers",
                "description": (
                    "Search for scientific papers across multiple databases including "
                    "OpenAlex, Semantic Scholar, PubMed, and arXiv. Returns paper metadata "
                    "including titles, authors, abstracts, DOIs, and citation counts."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keywords, title, author, etc.)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of papers to return (default: 20)",
                            "default": 20,
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Data sources to search: openalex, semantic_scholar, "
                                "pubmed, arxiv. Default searches all."
                            ),
                        },
                        "year_from": {
                            "type": "integer",
                            "description": "Filter papers from this year onwards",
                        },
                        "year_to": {
                            "type": "integer",
                            "description": "Filter papers up to this year",
                        },
                        "open_access": {
                            "type": "boolean",
                            "description": "Only return open access papers",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "lookup_paper",
                "description": (
                    "Look up a specific paper by its identifier (DOI, PubMed ID, arXiv ID, "
                    "or OpenAlex ID). Returns detailed metadata about the paper."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doi": {
                            "type": "string",
                            "description": "Digital Object Identifier (DOI)",
                        },
                        "pmid": {
                            "type": "string",
                            "description": "PubMed ID",
                        },
                        "arxiv_id": {
                            "type": "string",
                            "description": "arXiv ID (e.g., 2301.12345)",
                        },
                        "openalex_id": {
                            "type": "string",
                            "description": "OpenAlex ID (e.g., W1234567890)",
                        },
                    },
                },
            },
            {
                "name": "get_citations",
                "description": (
                    "Get papers that cite a given paper. Useful for finding follow-up "
                    "research and understanding a paper's impact."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doi": {
                            "type": "string",
                            "description": "DOI of the paper to get citations for",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of citing papers to return",
                            "default": 50,
                        },
                    },
                    "required": ["doi"],
                },
            },
            {
                "name": "get_references",
                "description": (
                    "Get papers referenced by a given paper. Useful for understanding "
                    "the foundational work and context of a paper."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doi": {
                            "type": "string",
                            "description": "DOI of the paper to get references for",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of referenced papers to return",
                            "default": 50,
                        },
                    },
                    "required": ["doi"],
                },
            },
            {
                "name": "retrieve_fulltext",
                "description": (
                    "Download and extract full text from a paper. Attempts to find "
                    "open access versions from sources like Unpaywall, PMC, and arXiv."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doi": {
                            "type": "string",
                            "description": "DOI of the paper to retrieve",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["text", "markdown", "chunks"],
                            "description": "Output format (default: text)",
                            "default": "text",
                        },
                    },
                    "required": ["doi"],
                },
            },
            {
                "name": "index_papers",
                "description": (
                    "Index papers into the knowledge base for semantic search and Q&A. "
                    "Papers must be indexed before they can be queried with ask_papers."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dois": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of DOIs to index",
                        },
                        "include_fulltext": {
                            "type": "boolean",
                            "description": "Also retrieve and index full text (slower)",
                            "default": False,
                        },
                    },
                    "required": ["dois"],
                },
            },
            {
                "name": "ask_papers",
                "description": (
                    "Ask a question about indexed papers using RAG (Retrieval Augmented "
                    "Generation). Returns an answer with citations to source papers."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to answer based on indexed papers",
                        },
                        "max_sources": {
                            "type": "integer",
                            "description": "Maximum number of source papers to consider",
                            "default": 5,
                        },
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "build_citation_network",
                "description": (
                    "Build a citation network starting from seed papers. Useful for "
                    "exploring research landscapes and finding influential papers."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "seed_dois": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "DOIs of seed papers to start from",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "How many citation levels to traverse (1-3)",
                            "default": 1,
                        },
                        "max_papers": {
                            "type": "integer",
                            "description": "Maximum total papers in network",
                            "default": 100,
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["citations", "references", "both"],
                            "description": "Direction to traverse",
                            "default": "both",
                        },
                    },
                    "required": ["seed_dois"],
                },
            },
            {
                "name": "summarize_papers",
                "description": (
                    "Generate a summary of multiple papers on a topic. Useful for "
                    "literature reviews and understanding research areas."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dois": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "DOIs of papers to summarize",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Optional focus area for the summary",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["paragraph", "bullets", "table"],
                            "description": "Output format",
                            "default": "paragraph",
                        },
                    },
                    "required": ["dois"],
                },
            },
        ]

    async def search_papers(
        self,
        query: str,
        limit: int = 20,
        sources: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        open_access: bool = False,
    ) -> dict[str, Any]:
        """Search for papers."""
        try:
            papers = await self.forge.search(
                query=query,
                limit=limit,
                sources=sources,
                year_from=year_from,
                year_to=year_to,
                open_access=open_access,
            )

            return {
                "success": True,
                "count": len(papers),
                "papers": [self._paper_to_dict(p) for p in papers],
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"success": False, "error": str(e)}

    async def lookup_paper(
        self,
        doi: str | None = None,
        pmid: str | None = None,
        arxiv_id: str | None = None,
        openalex_id: str | None = None,
    ) -> dict[str, Any]:
        """Look up a specific paper."""
        try:
            paper = await self.forge.lookup(
                doi=doi,
                pmid=pmid,
                arxiv_id=arxiv_id,
                openalex_id=openalex_id,
            )

            if paper:
                return {"success": True, "paper": self._paper_to_dict(paper)}
            else:
                return {"success": False, "error": "Paper not found"}
        except Exception as e:
            logger.error(f"Lookup failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_citations(self, doi: str, limit: int = 50) -> dict[str, Any]:
        """Get papers citing a paper."""
        try:
            citations = await self.forge.get_citations(doi=doi, limit=limit)
            return {
                "success": True,
                "count": len(citations),
                "citations": [self._paper_to_dict(p) for p in citations],
            }
        except Exception as e:
            logger.error(f"Get citations failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_references(self, doi: str, limit: int = 50) -> dict[str, Any]:
        """Get papers referenced by a paper."""
        try:
            references = await self.forge.get_references(doi=doi, limit=limit)
            return {
                "success": True,
                "count": len(references),
                "references": [self._paper_to_dict(p) for p in references],
            }
        except Exception as e:
            logger.error(f"Get references failed: {e}")
            return {"success": False, "error": str(e)}

    async def retrieve_fulltext(
        self,
        doi: str,
        format: str = "text",
    ) -> dict[str, Any]:
        """Retrieve full text of a paper."""
        try:
            content = await self.forge.retrieve(doi=doi, format=format)
            return {"success": True, "content": content}
        except Exception as e:
            logger.error(f"Retrieve fulltext failed: {e}")
            return {"success": False, "error": str(e)}

    async def index_papers(
        self,
        dois: list[str],
        include_fulltext: bool = False,
    ) -> dict[str, Any]:
        """Index papers for Q&A."""
        try:
            indexed = await self.forge.index(dois=dois, include_fulltext=include_fulltext)
            return {"success": True, "indexed_count": indexed}
        except Exception as e:
            logger.error(f"Index papers failed: {e}")
            return {"success": False, "error": str(e)}

    async def ask_papers(
        self,
        question: str,
        max_sources: int = 5,
    ) -> dict[str, Any]:
        """Ask a question about indexed papers."""
        try:
            answer = await self.forge.ask(question=question, max_sources=max_sources)
            return {
                "success": True,
                "answer": answer.text,
                "citations": [
                    {"doi": c.doi, "title": c.title, "relevance": c.relevance}
                    for c in answer.citations
                ],
            }
        except Exception as e:
            logger.error(f"Ask papers failed: {e}")
            return {"success": False, "error": str(e)}

    async def build_citation_network(
        self,
        seed_dois: list[str],
        depth: int = 1,
        max_papers: int = 100,
        direction: str = "both",
    ) -> dict[str, Any]:
        """Build a citation network."""
        try:
            network = await self.forge.build_network(
                seed_dois=seed_dois,
                depth=depth,
                max_papers=max_papers,
                direction=direction,
            )
            return {
                "success": True,
                "node_count": network.node_count,
                "edge_count": network.edge_count,
                "most_cited": [
                    {"doi": p.doi, "title": p.title, "citation_count": p.citation_count}
                    for p in network.most_cited(5)
                ],
            }
        except Exception as e:
            logger.error(f"Build network failed: {e}")
            return {"success": False, "error": str(e)}

    async def summarize_papers(
        self,
        dois: list[str],
        focus: str | None = None,
        format: str = "paragraph",
    ) -> dict[str, Any]:
        """Summarize multiple papers."""
        try:
            summary = await self.forge.summarize(dois=dois, focus=focus, format=format)
            return {"success": True, "summary": summary}
        except Exception as e:
            logger.error(f"Summarize failed: {e}")
            return {"success": False, "error": str(e)}

    def _paper_to_dict(self, paper: Any) -> dict[str, Any]:
        """Convert a Paper object to a dictionary."""
        return {
            "doi": paper.doi,
            "title": paper.title,
            "authors": [a.name for a in paper.authors] if paper.authors else [],
            "abstract": paper.abstract,
            "year": paper.year,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
            "open_access": paper.open_access,
            "url": paper.url,
        }
