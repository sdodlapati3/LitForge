"""
Discovery Service - Paper search and discovery.

Provides unified search across multiple data sources with deduplication
and result merging.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from litforge.config import LitForgeConfig, get_config
from litforge.models import Publication, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class DiscoveryService:
    """
    Service for discovering and searching scientific publications.
    
    Aggregates results from multiple data sources (OpenAlex, Semantic Scholar,
    PubMed, etc.) and provides deduplication and result merging.
    """
    
    def __init__(self, config: LitForgeConfig | None = None):
        """
        Initialize the discovery service.
        
        Args:
            config: LitForge configuration
        """
        self.config = config or get_config()
        self._clients: dict[str, Any] = {}
    
    def _get_client(self, source: str) -> Any:
        """Get or create a client for the given source."""
        if source not in self._clients:
            self._clients[source] = self._create_client(source)
        return self._clients[source]
    
    def _create_client(self, source: str) -> Any:
        """Create a client for the given source."""
        if source == "openalex":
            from litforge.clients.openalex import OpenAlexClient
            return OpenAlexClient(
                email=self.config.sources.openalex_email
            )
        elif source == "semantic_scholar":
            from litforge.clients.semantic_scholar import SemanticScholarClient
            return SemanticScholarClient(
                api_key=self.config.sources.semantic_scholar_api_key
            )
        elif source == "pubmed":
            from litforge.clients.pubmed import PubMedClient
            return PubMedClient(
                email=self.config.sources.pubmed_email,
                api_key=self.config.sources.pubmed_api_key,
            )
        elif source == "arxiv":
            from litforge.clients.arxiv import ArxivClient
            return ArxivClient()
        elif source == "crossref":
            from litforge.clients.crossref import CrossrefClient
            return CrossrefClient()
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def search(self, query: SearchQuery) -> SearchResult:
        """
        Search for publications.
        
        Args:
            query: Search query with filters and options
            
        Returns:
            SearchResult with publications
        """
        import time
        start_time = time.time()
        
        all_papers: list[Publication] = []
        source_counts: dict[str, int] = {}
        
        # Search each source
        for source in query.sources:
            try:
                client = self._get_client(source)
                papers = client.search(
                    query=query.query,
                    limit=query.limit,
                    filters=query.filters,
                )
                
                # Tag papers with source
                for paper in papers:
                    if source not in paper.sources:
                        paper.sources.append(source)
                
                all_papers.extend(papers)
                source_counts[source] = len(papers)
                
                logger.info(f"Found {len(papers)} papers from {source}")
                
            except Exception as e:
                logger.warning(f"Error searching {source}: {e}")
                source_counts[source] = 0
        
        # Deduplicate
        if query.deduplicate:
            all_papers = self._deduplicate(all_papers)
        
        # Sort
        all_papers = self._sort_papers(all_papers, query.sort_by, query.sort_order)
        
        # Apply limit
        all_papers = all_papers[:query.limit]
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            query=query,
            publications=all_papers,
            total_count=len(all_papers),
            returned_count=len(all_papers),
            source_counts=source_counts,
            search_time_ms=elapsed_ms,
        )
    
    def lookup(
        self,
        *,
        doi: str | None = None,
        pmid: str | None = None,
        arxiv_id: str | None = None,
        openalex_id: str | None = None,
        semantic_scholar_id: str | None = None,
    ) -> Publication | None:
        """
        Look up a specific publication by identifier.
        
        Tries multiple sources to find the most complete information.
        """
        paper: Publication | None = None
        
        # Try OpenAlex first (usually most complete)
        if openalex_id or doi:
            try:
                client = self._get_client("openalex")
                paper = client.get_paper(
                    openalex_id=openalex_id,
                    doi=doi,
                )
            except Exception as e:
                logger.debug(f"OpenAlex lookup failed: {e}")
        
        # Try Semantic Scholar
        if (not paper) and (semantic_scholar_id or doi):
            try:
                client = self._get_client("semantic_scholar")
                paper = client.get_paper(
                    paper_id=semantic_scholar_id,
                    doi=doi,
                )
            except Exception as e:
                logger.debug(f"Semantic Scholar lookup failed: {e}")
        
        # Try PubMed
        if (not paper) and (pmid or doi):
            try:
                client = self._get_client("pubmed")
                paper = client.get_paper(pmid=pmid, doi=doi)
            except Exception as e:
                logger.debug(f"PubMed lookup failed: {e}")
        
        # Try arXiv
        if (not paper) and arxiv_id:
            try:
                client = self._get_client("arxiv")
                paper = client.get_paper(arxiv_id=arxiv_id)
            except Exception as e:
                logger.debug(f"arXiv lookup failed: {e}")
        
        return paper
    
    def recommend(
        self,
        papers: Sequence[Publication] | Sequence[str],
        limit: int = 20,
    ) -> list[Publication]:
        """
        Get paper recommendations based on seed papers.
        
        Uses Semantic Scholar's recommendation API.
        """
        # Get paper IDs
        paper_ids = []
        for p in papers:
            if isinstance(p, str):
                paper_ids.append(p)
            elif p.semantic_scholar_id:
                paper_ids.append(p.semantic_scholar_id)
            elif p.doi:
                paper_ids.append(p.doi)
        
        if not paper_ids:
            return []
        
        try:
            client = self._get_client("semantic_scholar")
            return client.get_recommendations(paper_ids, limit=limit)
        except Exception as e:
            logger.warning(f"Recommendation failed: {e}")
            return []
    
    def _deduplicate(self, papers: list[Publication]) -> list[Publication]:
        """Deduplicate papers by DOI and title similarity."""
        from litforge.services.utils import deduplicate_papers
        return deduplicate_papers(papers)
    
    def _sort_papers(
        self,
        papers: list[Publication],
        sort_by: str,
        sort_order: str,
    ) -> list[Publication]:
        """Sort papers by the specified field."""
        reverse = sort_order == "desc"
        
        if sort_by == "citations":
            return sorted(papers, key=lambda p: p.citation_count, reverse=reverse)
        elif sort_by == "date":
            return sorted(
                papers,
                key=lambda p: p.publication_date or p.year or 0,
                reverse=reverse,
            )
        elif sort_by == "title":
            return sorted(papers, key=lambda p: p.title.lower(), reverse=reverse)
        else:
            # Relevance - keep original order
            return papers
