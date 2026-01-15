"""
LitForge Simple API - One-liner functions for common tasks.

This module provides the simplest possible interface to LitForge.
Just import and use - no configuration needed.

Example:
    >>> import litforge
    >>> 
    >>> # Search for papers
    >>> papers = litforge.search("CRISPR gene editing")
    >>> 
    >>> # Get citations
    >>> citations = litforge.citations("10.1126/science.1225829")
    >>> 
    >>> # Look up a paper by DOI
    >>> paper = litforge.lookup("10.1038/nature14539")
"""

from __future__ import annotations

import asyncio
import httpx
import uuid
from typing import Any, Optional
from datetime import datetime

# Import the unified Publication model
from litforge.models.publication import Publication, Author

# Paper is an alias for Publication (backwards compatibility)
# Use Paper for simple usage, Publication for full features
Paper = Publication


class LitForgeClient:
    """Simple synchronous client for LitForge operations."""
    
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._llm = None  # Lazy-loaded
    
    def _get_llm(self):
        """Lazy-load LLM for semantic search."""
        if self._llm is None:
            try:
                from litforge.llm import LLMRouter
                self._llm = LLMRouter()
            except Exception:
                pass  # No LLM available - use basic search
        return self._llm
    
    def search(
        self,
        query: str,
        *,
        limit: int = 25,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        sort_by: str = "citations",  # "citations", "date", "relevance"
        use_semantic: bool = True,  # Try semantic search first
    ) -> list[Paper]:
        """
        Search for scientific papers.
        
        Uses semantic search (LLM + embeddings) when available, with
        fast fallback to basic OpenAlex search.
        
        Args:
            query: Search query (natural language or keywords)
            limit: Maximum number of results
            year_from: Filter papers from this year
            year_to: Filter papers until this year
            sort_by: Sort order ("citations", "date", "relevance")
            use_semantic: Try semantic search first (default True)
        
        Returns:
            List of Paper objects
        
        Example:
            >>> papers = litforge.search("CRISPR", limit=10)
            >>> for p in papers:
            ...     print(f"{p.title} ({p.year}) - {p.citations} citations")
        """
        # Try semantic search first (LLM understands natural language)
        if use_semantic:
            llm = self._get_llm()
            if llm:
                try:
                    from litforge.services.semantic_search import semantic_search
                    papers, _ = semantic_search(query, llm, max_results=limit)
                    if papers:
                        # Apply year filter if specified
                        if year_from or year_to:
                            papers = [
                                p for p in papers
                                if (not year_from or (p.year and p.year >= year_from)) and
                                   (not year_to or (p.year and p.year <= year_to))
                            ]
                        return papers[:limit]
                except Exception:
                    pass  # Fall back to basic search
        
        # Basic search fallback (fast, no LLM needed)
        return asyncio.run(self._basic_search_async(query, limit, year_from, year_to, sort_by))
    
    async def _basic_search_async(
        self,
        query: str,
        limit: int,
        year_from: Optional[int],
        year_to: Optional[int],
        sort_by: str,
    ) -> list[Paper]:
        """Basic OpenAlex search (no LLM required)."""
        # Build sort parameter
        sort_map = {
            "citations": "cited_by_count:desc",
            "date": "publication_date:desc",
            "relevance": "relevance_score:desc",
        }
        sort_param = sort_map.get(sort_by, "cited_by_count:desc")
        
        # Build filters
        filters = []
        
        # Use title.search for better precision
        # OpenAlex's fulltext search often returns irrelevant results
        filters.append(f"title.search:{query}")
        
        # Year filters
        if year_from and year_to:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from:
            filters.append(f"publication_year:{year_from}-2030")
        elif year_to:
            filters.append(f"publication_year:1900-{year_to}")
        
        async with httpx.AsyncClient(timeout=30) as client:
            params = {
                "filter": ",".join(filters),
                "per_page": min(limit, 100),
                "sort": sort_param,
            }
            
            resp = await client.get("https://api.openalex.org/works", params=params)
            data = resp.json()
            
            results = [self._parse_openalex_work(work) for work in data.get("results", [])][:limit]
            
            # If no results with title search, fall back to fulltext search
            if not results:
                params = {
                    "search": query,
                    "per_page": min(limit, 100),
                    "sort": sort_param,
                }
                resp = await client.get("https://api.openalex.org/works", params=params)
                data = resp.json()
                results = [self._parse_openalex_work(work) for work in data.get("results", [])][:limit]
            
            return results
    
    def lookup(self, doi: str) -> Optional[Paper]:
        """
        Look up a paper by DOI.
        
        Args:
            doi: Paper DOI (e.g., "10.1038/nature14539")
        
        Returns:
            Paper object or None if not found
        
        Example:
            >>> paper = litforge.lookup("10.1038/nature14539")
            >>> print(paper.title)
            'Deep learning'
        """
        return asyncio.run(self._lookup_async(doi))
    
    async def _lookup_async(self, doi: str) -> Optional[Paper]:
        """Async lookup implementation."""
        # Clean DOI
        doi = doi.replace("https://doi.org/", "").strip()
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Try CrossRef first
            resp = await client.get(f"https://api.crossref.org/works/{doi}")
            if resp.status_code == 200:
                data = resp.json().get("message", {})
                return self._parse_crossref_work(data)
            
            # Fallback to OpenAlex
            resp = await client.get(f"https://api.openalex.org/works/doi:{doi}")
            if resp.status_code == 200:
                return self._parse_openalex_work(resp.json())
        
        return None
    
    def citations(self, doi: str, limit: int = 50) -> list[Paper]:
        """
        Get papers that cite a given paper.
        
        Args:
            doi: DOI of the paper to get citations for
            limit: Maximum number of citing papers
        
        Returns:
            List of citing Paper objects
        
        Example:
            >>> cites = litforge.citations("10.1126/science.1225829", limit=10)
            >>> print(f"Found {len(cites)} citing papers")
        """
        return asyncio.run(self._citations_async(doi, limit))
    
    async def _citations_async(self, doi: str, limit: int) -> list[Paper]:
        """Async citations implementation."""
        doi = doi.replace("https://doi.org/", "").strip()
        papers = []
        
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://api.openalex.org/works",
                params={
                    "filter": f"cites:doi:{doi}",
                    "per_page": min(limit, 100),
                    "sort": "cited_by_count:desc",
                }
            )
            data = resp.json()
            
            for work in data.get("results", []):
                papers.append(self._parse_openalex_work(work))
        
        return papers[:limit]
    
    def references(self, doi: str, limit: int = 50) -> list[Paper]:
        """
        Get papers referenced by a given paper.
        
        Args:
            doi: DOI of the paper to get references for
            limit: Maximum number of referenced papers
        
        Returns:
            List of referenced Paper objects
        """
        return asyncio.run(self._references_async(doi, limit))
    
    async def _references_async(self, doi: str, limit: int) -> list[Paper]:
        """Async references implementation."""
        doi = doi.replace("https://doi.org/", "").strip()
        papers = []
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Get the paper's referenced works
            resp = await client.get(f"https://api.openalex.org/works/doi:{doi}")
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            ref_ids = data.get("referenced_works", [])[:limit]
            
            # Fetch each reference
            for ref_id in ref_ids:
                try:
                    resp = await client.get(ref_id)
                    if resp.status_code == 200:
                        papers.append(self._parse_openalex_work(resp.json()))
                except:
                    continue
        
        return papers
    
    def _parse_openalex_work(self, work: dict) -> Publication:
        """Parse OpenAlex work to Publication."""
        # Parse authors as Author objects
        authors = []
        for i, a in enumerate(work.get("authorships", [])):
            author_data = a.get("author", {})
            name = author_data.get("display_name", "")
            if name:
                authors.append(Author(
                    name=name,
                    openalex_id=author_data.get("id"),
                    position=i + 1,
                ))
        
        venue = ""
        loc = work.get("primary_location", {})
        if loc and isinstance(loc, dict):
            source = loc.get("source", {})
            if source:
                venue = source.get("display_name", "")
        
        # Reconstruct abstract
        abstract = ""
        if work.get("abstract_inverted_index"):
            idx = work["abstract_inverted_index"]
            try:
                words = [""] * (max(max(p) for p in idx.values()) + 1)
                for word, positions in idx.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words)
            except:
                pass
        
        doi = work.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")
        
        # Generate ID from OpenAlex ID or DOI
        openalex_id = work.get("id", "")
        paper_id = openalex_id.split("/")[-1] if openalex_id else (doi or str(uuid.uuid4())[:8])
        
        return Publication(
            id=paper_id,
            title=work.get("title") or "Unknown",
            authors=authors,
            year=work.get("publication_year"),
            doi=doi or None,
            citation_count=work.get("cited_by_count", 0),
            venue=venue or None,
            abstract=abstract or None,
            url=openalex_id or None,
            openalex_id=openalex_id or None,
            sources=["openalex"],
        )
    
    def _parse_crossref_work(self, work: dict) -> Publication:
        """Parse CrossRef work to Publication."""
        # Parse authors as Author objects
        authors = []
        for i, a in enumerate(work.get("author", [])):
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(Author(
                    name=name,
                    given_name=given or None,
                    family_name=family or None,
                    orcid=a.get("ORCID"),
                    position=i + 1,
                ))
        
        title = work.get("title", [""])[0] if work.get("title") else "Unknown"
        
        year = None
        if work.get("published-print", {}).get("date-parts"):
            year = work["published-print"]["date-parts"][0][0]
        elif work.get("published-online", {}).get("date-parts"):
            year = work["published-online"]["date-parts"][0][0]
        
        venue = work.get("container-title", [""])[0] if work.get("container-title") else ""
        doi = work.get("DOI", "")
        
        return Publication(
            id=doi or str(uuid.uuid4())[:8],
            title=title,
            authors=authors,
            year=year,
            doi=doi or None,
            citation_count=work.get("is-referenced-by-count", 0),
            venue=venue or None,
            abstract=work.get("abstract") or None,
            url=work.get("URL") or None,
            sources=["crossref"],
        )


# Global client instance
_client = LitForgeClient()

# Module-level functions for simplest usage
def search(query: str, **kwargs) -> list[Paper]:
    """Search for papers. See LitForgeClient.search for details."""
    return _client.search(query, **kwargs)

def lookup(doi: str) -> Optional[Paper]:
    """Look up a paper by DOI. See LitForgeClient.lookup for details."""
    return _client.lookup(doi)

def citations(doi: str, limit: int = 50) -> list[Paper]:
    """Get citing papers. See LitForgeClient.citations for details."""
    return _client.citations(doi, limit)

def references(doi: str, limit: int = 50) -> list[Paper]:
    """Get referenced papers. See LitForgeClient.references for details."""
    return _client.references(doi, limit)


# For agent systems - expose tools directly
def get_tools() -> dict:
    """
    Get LitForge tools for agent integration.
    
    Returns a dict of callable tools that can be registered with any agent system.
    
    Example:
        >>> from litforge import get_tools
        >>> tools = get_tools()
        >>> # Register with your agent
        >>> agent.register_tools(tools)
    """
    return {
        "search_papers": search,
        "lookup_paper": lookup,
        "get_citations": citations,
        "get_references": references,
    }
