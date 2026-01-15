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
                "name": "smart_search",
                "description": (
                    "Intelligent literature search that handles natural language queries. "
                    "The AI assistant should use this tool when the user asks a question-style "
                    "query or uses informal language. The assistant should:\n"
                    "1. Parse the user's intent\n"
                    "2. Expand terminology (e.g., 'liquid foundation models' → 'Liquid Neural Networks', "
                    "'Liquid Time-constant Networks', 'CfC networks')\n"
                    "3. Provide multiple search_queries to cover synonyms and related terms\n"
                    "4. Set appropriate filters\n\n"
                    "This enables semantic understanding that keyword search cannot provide."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "search_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of search queries to execute. Include the main query "
                                "plus synonyms, alternative names, and related terms. "
                                "Example: For 'liquid foundation models', use "
                                "['Liquid Neural Networks', 'Liquid Time-constant Networks', "
                                "'Closed-form continuous-time neural networks', 'CfC networks', "
                                "'Ramin Hasani neural']"
                            ),
                        },
                        "limit_per_query": {
                            "type": "integer",
                            "description": "Max results per query (default: 10)",
                            "default": 10,
                        },
                        "year_from": {
                            "type": "integer",
                            "description": "Filter papers from this year",
                        },
                        "year_to": {
                            "type": "integer",
                            "description": "Filter papers until this year",
                        },
                        "must_contain": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Terms that must appear in the paper title for it to be "
                                "considered relevant. Use lowercase. Example: ['liquid', 'neural']"
                            ),
                        },
                        "exclude_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Terms to exclude from results. Useful for filtering false positives. "
                                "Example: ['ionic liquid', 'liquid crystal', 'liquid biopsy']"
                            ),
                        },
                    },
                    "required": ["search_queries"],
                },
            },
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

    async def smart_search(
        self,
        search_queries: list[str],
        limit_per_query: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        must_contain: list[str] | None = None,
        exclude_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Intelligent multi-query search with filtering.
        
        This tool enables Claude to expand queries semantically and filter
        results for relevance.
        """
        try:
            from litforge.clients import OpenAlexClient, ArxivClient
            
            oa = OpenAlexClient()
            ax = ArxivClient()
            
            all_papers = []
            
            # Execute each query
            for query in search_queries:
                try:
                    # Search OpenAlex
                    results = oa.search(query, limit=limit_per_query)
                    all_papers.extend(results)
                except Exception as e:
                    logger.debug(f"OpenAlex search failed for '{query}': {e}")
                
                try:
                    # Search arXiv
                    results = ax.search(query, limit=limit_per_query // 2)
                    all_papers.extend(results)
                except Exception as e:
                    logger.debug(f"arXiv search failed for '{query}': {e}")
            
            # Deduplicate by title
            seen_titles = set()
            unique_papers = []
            for p in all_papers:
                title_norm = p.title.lower().strip()[:80]
                if title_norm not in seen_titles:
                    seen_titles.add(title_norm)
                    unique_papers.append(p)
            
            # Filter by must_contain terms
            if must_contain:
                must_contain_lower = [t.lower() for t in must_contain]
                unique_papers = [
                    p for p in unique_papers
                    if any(term in p.title.lower() for term in must_contain_lower)
                ]
            
            # Filter out exclude_terms
            if exclude_terms:
                exclude_lower = [t.lower() for t in exclude_terms]
                unique_papers = [
                    p for p in unique_papers
                    if not any(term in p.title.lower() for term in exclude_lower)
                ]
            
            # Filter by year
            if year_from:
                unique_papers = [p for p in unique_papers if p.year and p.year >= year_from]
            if year_to:
                unique_papers = [p for p in unique_papers if p.year and p.year <= year_to]
            
            # Sort by citations
            unique_papers.sort(key=lambda x: x.citation_count, reverse=True)
            
            return {
                "success": True,
                "query_count": len(search_queries),
                "total_found": len(unique_papers),
                "papers": [self._paper_to_dict(p) for p in unique_papers[:50]],
            }
            
        except Exception as e:
            logger.error(f"Smart search failed: {e}")
            return {"success": False, "error": str(e)}

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
            papers = self.forge.search(
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
        """Convert a Paper/Publication object to a dictionary."""
        # Handle both Publication model and simple Paper objects
        authors = []
        if hasattr(paper, 'authors') and paper.authors:
            if hasattr(paper.authors[0], 'name'):
                authors = [a.name for a in paper.authors]
            else:
                authors = list(paper.authors)
        
        return {
            "doi": getattr(paper, 'doi', None),
            "title": getattr(paper, 'title', 'Unknown'),
            "authors": authors,
            "abstract": getattr(paper, 'abstract', None),
            "year": getattr(paper, 'year', None),
            "venue": getattr(paper, 'venue', None),
            "citation_count": getattr(paper, 'citation_count', 0),
            "open_access": getattr(paper, 'is_open_access', getattr(paper, 'open_access', False)),
            "url": getattr(paper, 'url', None),
        }
