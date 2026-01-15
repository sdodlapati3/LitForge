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
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """Simple paper representation."""
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    citations: int = 0
    venue: str = ""
    abstract: str = ""
    url: str = ""
    
    def __repr__(self):
        return f"Paper('{self.title[:50]}...' by {', '.join(self.authors[:2])} ({self.year}))"


class LitForgeClient:
    """Simple synchronous client for LitForge operations."""
    
    def __init__(self):
        self._cache: dict[str, Any] = {}
    
    def search(
        self,
        query: str,
        *,
        limit: int = 25,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        sort_by: str = "citations",  # "citations", "date", "relevance"
    ) -> list[Paper]:
        """
        Search for scientific papers.
        
        Args:
            query: Search query (e.g., "linear attention transformers")
            limit: Maximum number of results
            year_from: Filter papers from this year
            year_to: Filter papers until this year
            sort_by: Sort order ("citations", "date", "relevance")
        
        Returns:
            List of Paper objects
        
        Example:
            >>> papers = litforge.search("CRISPR", limit=10)
            >>> for p in papers:
            ...     print(f"{p.title} ({p.year}) - {p.citations} citations")
        """
        return asyncio.run(self._search_async(query, limit, year_from, year_to, sort_by))
    
    def _clean_query(self, query: str) -> str:
        """Clean natural language query to extract key search terms."""
        import re
        
        # Remove common natural language phrases (multi-word)
        noise_phrases = [
            "find me", "list of", "papers on", "papers about", "research on",
            "show me", "get me", "i want", "i need", "looking for",
            "articles on", "articles about", "publications on", "publications about",
            "can you find", "please find", "search for", "give me",
            "what are", "what is", "tell me about", "explain", "describe",
            "related to them", "related to it", "related to this",
            "me all the", "all the papers", "all papers",
            "can you", "could you", "would you", "please",
        ]
        
        cleaned = query.lower()
        for phrase in noise_phrases:
            cleaned = cleaned.replace(phrase, " ")
        
        # Remove question marks and other punctuation
        cleaned = re.sub(r'[?!.,;:]', ' ', cleaned)
        
        # Remove common stopwords that hurt search quality
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'me', 'my', 'i', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
            'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
        }
        
        words = cleaned.split()
        filtered_words = [w for w in words if w not in stopwords and len(w) > 1]
        
        # Join back
        cleaned = " ".join(filtered_words)
        
        return cleaned.strip() or query
    
    async def _search_async(
        self,
        query: str,
        limit: int,
        year_from: Optional[int],
        year_to: Optional[int],
        sort_by: str,
    ) -> list[Paper]:
        """Async search implementation."""
        papers = []
        
        # Clean query to extract key terms
        clean_query = self._clean_query(query)
        
        # Build sort parameter
        sort_map = {
            "citations": "cited_by_count:desc",
            "date": "publication_date:desc",
            "relevance": "relevance_score:desc",
        }
        sort_param = sort_map.get(sort_by, "cited_by_count:desc")
        
        # Build filter - use OpenAlex range format
        filters = []
        if year_from and year_to:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from:
            filters.append(f"publication_year:{year_from}-2030")
        elif year_to:
            filters.append(f"publication_year:1900-{year_to}")
        
        # Extract key terms for relevance filtering
        key_terms = [t.lower() for t in clean_query.split() if len(t) > 2]
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Request more to allow for filtering
            params = {
                "search": clean_query,
                "per_page": min(limit * 3, 100),  # Get extra for filtering
                "sort": sort_param,
            }
            if filters:
                params["filter"] = ",".join(filters)
            
            resp = await client.get("https://api.openalex.org/works", params=params)
            data = resp.json()
            
            for work in data.get("results", []):
                paper = self._parse_openalex_work(work)
                
                # Score relevance based on title match
                title_lower = paper.title.lower()
                relevance_score = sum(1 for term in key_terms if term in title_lower)
                paper._relevance = relevance_score
                
                papers.append(paper)
            
            # Sort by relevance first, then citations
            papers.sort(key=lambda p: (getattr(p, '_relevance', 0), p.citations), reverse=True)
        
        return papers[:limit]
    
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
    
    def _parse_openalex_work(self, work: dict) -> Paper:
        """Parse OpenAlex work to Paper."""
        authors = []
        for a in work.get("authorships", []):
            name = a.get("author", {}).get("display_name", "")
            if name:
                authors.append(name)
        
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
        
        return Paper(
            title=work.get("title") or "Unknown",
            authors=authors,
            year=work.get("publication_year"),
            doi=doi,
            citations=work.get("cited_by_count", 0),
            venue=venue,
            abstract=abstract,
            url=work.get("id", ""),
        )
    
    def _parse_crossref_work(self, work: dict) -> Paper:
        """Parse CrossRef work to Paper."""
        authors = []
        for a in work.get("author", []):
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)
        
        title = work.get("title", [""])[0] if work.get("title") else "Unknown"
        
        year = None
        if work.get("published-print", {}).get("date-parts"):
            year = work["published-print"]["date-parts"][0][0]
        elif work.get("published-online", {}).get("date-parts"):
            year = work["published-online"]["date-parts"][0][0]
        
        venue = work.get("container-title", [""])[0] if work.get("container-title") else ""
        
        return Paper(
            title=title,
            authors=authors,
            year=year,
            doi=work.get("DOI", ""),
            citations=work.get("is-referenced-by-count", 0),
            venue=venue,
            abstract=work.get("abstract", ""),
            url=work.get("URL", ""),
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
