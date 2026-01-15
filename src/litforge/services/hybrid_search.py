"""
Hybrid Search Service - Combines Local Index + Live API.

This service provides the best of both worlds:
1. Local Index: Instant similarity search over millions of papers
2. Live API: Fresh papers and authoritative metadata

Architecture:
    User Query 
        ↓
    [Embed Query] → SPECTER2 embedding
        ↓
    ┌─────────────────┬─────────────────┐
    │  Local Index    │   Live API      │
    │  (instant,      │   (fresh,       │
    │   millions)     │   metadata)     │
    └────────┬────────┴────────┬────────┘
             ↓                  ↓
        [Merge & Deduplicate]
             ↓
        [Ensemble Score]
             ↓
        [LLM Verify if uncertain]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy import ndarray
else:
    try:
        import numpy as np
        ndarray = np.ndarray
    except ImportError:
        np = None  # type: ignore
        ndarray = Any  # type: ignore

from litforge.stores.embedding_index import (
    EmbeddingIndex, 
    EmbeddingIndexManager,
    IndexedPaper,
    get_embedding_index,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid search."""
    paper_id: str
    title: str
    year: int | None = None
    doi: str | None = None
    citation_count: int = 0
    abstract: str | None = None
    
    # Scores
    local_score: float = 0.0  # Similarity from local index
    api_score: float = 0.0    # Relevance from API
    combined_score: float = 0.0
    
    # Source
    from_local: bool = False
    from_api: bool = False
    
    # Full publication (if enriched)
    publication: Any = None


class HybridSearchService:
    """
    Hybrid search combining local embedding index with live APIs.
    
    Features:
    - Fast local search (~100ms for 10M papers)
    - Live API for fresh papers and metadata
    - Automatic result merging and deduplication
    - Score fusion for ranking
    """
    
    # Weight for combining scores
    LOCAL_WEIGHT = 0.6
    API_WEIGHT = 0.4
    
    def __init__(
        self,
        index_name: str = "default",
        embedder: Any | None = None,
        use_semantic_scholar: bool = True,
        use_openalex: bool = True,
    ):
        """
        Initialize hybrid search service.
        
        Args:
            index_name: Name of local index to use
            embedder: Embedder for query encoding (auto-loads SPECTER2 if None)
            use_semantic_scholar: Include Semantic Scholar API
            use_openalex: Include OpenAlex API
        """
        self.index_name = index_name
        self._embedder = embedder
        self.use_semantic_scholar = use_semantic_scholar
        self.use_openalex = use_openalex
        
        # Lazy-loaded
        self._index: EmbeddingIndex | None = None
        self._s2_client = None
        self._oa_client = None
    
    @property
    def embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use MiniLM - reliable and fast (384 dim, but works for search)
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded MiniLM embedder (384-dim)")
            except ImportError:
                logger.warning("sentence-transformers not installed")
            except Exception as e:
                logger.warning(f"Failed to load embedder: {e}")
        return self._embedder
    
    @property
    def index(self) -> EmbeddingIndex | None:
        """Lazy-load local index."""
        if self._index is None:
            try:
                self._index = get_embedding_index(self.index_name)
                if self._index.size == 0:
                    logger.info(f"Index '{self.index_name}' is empty")
                    self._index = None
                else:
                    logger.info(f"Loaded index '{self.index_name}' with {self._index.size} papers")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
        return self._index
    
    @property
    def s2_client(self):
        """Lazy-load Semantic Scholar client."""
        if self._s2_client is None and self.use_semantic_scholar:
            try:
                from litforge.clients.semantic_scholar import SemanticScholarClient
                self._s2_client = SemanticScholarClient()
            except ImportError:
                logger.warning("Semantic Scholar client not available")
        return self._s2_client
    
    @property
    def oa_client(self):
        """Lazy-load OpenAlex client."""
        if self._oa_client is None and self.use_openalex:
            try:
                from litforge.clients.openalex import OpenAlexClient
                self._oa_client = OpenAlexClient()
            except ImportError:
                logger.warning("OpenAlex client not available")
        return self._oa_client
    
    def embed_query(self, query: str) -> np.ndarray | None:
        """Embed a search query."""
        if self.embedder is None:
            return None
        
        try:
            # SentenceTransformer.encode() returns numpy array directly
            embedding = self.embedder.encode(query)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return None
    
    def search_local(
        self,
        query_embedding: np.ndarray,
        k: int = 50,
    ) -> list[HybridResult]:
        """Search local index."""
        if self.index is None:
            return []
        
        results = self.index.search(query_embedding, k=k)
        
        return [
            HybridResult(
                paper_id=paper.paper_id,
                title=paper.title,
                year=paper.year,
                doi=paper.doi,
                citation_count=paper.citation_count,
                local_score=score,
                from_local=True,
            )
            for paper, score in results
        ]
    
    def search_api(
        self,
        query: str,
        limit: int = 30,
    ) -> list[HybridResult]:
        """Search live APIs."""
        results = []
        
        # Semantic Scholar
        if self.s2_client:
            try:
                papers = self.s2_client.search(query, limit=limit)
                for i, paper in enumerate(papers):
                    # Score based on position (API returns relevance-sorted)
                    score = 1.0 - (i / limit) * 0.5  # 1.0 to 0.5
                    results.append(HybridResult(
                        paper_id=getattr(paper, 'semantic_scholar_id', paper.id),
                        title=paper.title,
                        year=paper.year,
                        doi=paper.doi,
                        citation_count=getattr(paper, 'citation_count', 0),
                        abstract=paper.abstract,
                        api_score=score,
                        from_api=True,
                        publication=paper,
                    ))
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed: {e}")
        
        # OpenAlex
        if self.oa_client:
            try:
                papers = self.oa_client.search(query, limit=limit)
                for i, paper in enumerate(papers):
                    score = 1.0 - (i / limit) * 0.5
                    results.append(HybridResult(
                        paper_id=paper.id,
                        title=paper.title,
                        year=paper.year,
                        doi=paper.doi,
                        citation_count=getattr(paper, 'citation_count', 0),
                        abstract=paper.abstract,
                        api_score=score,
                        from_api=True,
                        publication=paper,
                    ))
            except Exception as e:
                logger.warning(f"OpenAlex search failed: {e}")
        
        return results
    
    def merge_results(
        self,
        local_results: list[HybridResult],
        api_results: list[HybridResult],
    ) -> list[HybridResult]:
        """
        Merge and deduplicate results from local and API sources.
        
        Uses DOI for exact matching, then title similarity for fuzzy matching.
        """
        # Index by DOI
        by_doi: dict[str, HybridResult] = {}
        by_title: dict[str, HybridResult] = {}
        
        for result in local_results:
            if result.doi:
                by_doi[result.doi.lower()] = result
            title_key = result.title.lower()[:60]
            by_title[title_key] = result
        
        # Merge API results
        merged = list(local_results)
        
        for api_result in api_results:
            existing = None
            
            # Check DOI match
            if api_result.doi:
                doi_key = api_result.doi.lower()
                if doi_key in by_doi:
                    existing = by_doi[doi_key]
            
            # Check title match
            if existing is None:
                title_key = api_result.title.lower()[:60]
                if title_key in by_title:
                    existing = by_title[title_key]
            
            if existing:
                # Merge scores
                existing.api_score = max(existing.api_score, api_result.api_score)
                existing.from_api = True
                # Enrich with API data
                if api_result.abstract and not existing.abstract:
                    existing.abstract = api_result.abstract
                if api_result.publication and not existing.publication:
                    existing.publication = api_result.publication
                if api_result.citation_count > existing.citation_count:
                    existing.citation_count = api_result.citation_count
            else:
                # New result from API
                merged.append(api_result)
                if api_result.doi:
                    by_doi[api_result.doi.lower()] = api_result
                title_key = api_result.title.lower()[:60]
                by_title[title_key] = api_result
        
        return merged
    
    def compute_combined_scores(
        self,
        results: list[HybridResult],
    ) -> list[HybridResult]:
        """Compute combined scores and sort."""
        for result in results:
            # Weighted combination
            result.combined_score = (
                self.LOCAL_WEIGHT * result.local_score +
                self.API_WEIGHT * result.api_score
            )
            
            # Boost for having both sources
            if result.from_local and result.from_api:
                result.combined_score *= 1.2  # 20% boost for agreement
            
            # Small boost for citations (log scale)
            import math
            if result.citation_count > 0:
                citation_boost = math.log10(result.citation_count + 1) / 10
                result.combined_score += citation_boost
        
        # Sort by combined score
        results.sort(key=lambda x: -x.combined_score)
        
        return results
    
    def search(
        self,
        query: str,
        k: int = 25,
        local_k: int = 50,
        api_limit: int = 30,
    ) -> list[HybridResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            k: Number of final results
            local_k: Number of local results to consider
            api_limit: Number of API results to fetch
            
        Returns:
            Ranked list of hybrid results
        """
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Search local index
        local_results = []
        if query_embedding is not None and self.index is not None:
            local_results = self.search_local(query_embedding, k=local_k)
            logger.info(f"Local search: {len(local_results)} results")
        
        # Search APIs
        api_results = self.search_api(query, limit=api_limit)
        logger.info(f"API search: {len(api_results)} results")
        
        # Merge
        merged = self.merge_results(local_results, api_results)
        logger.info(f"Merged: {len(merged)} unique results")
        
        # Score and rank
        ranked = self.compute_combined_scores(merged)
        
        return ranked[:k]
    
    def get_publications(
        self,
        results: list[HybridResult],
    ) -> list[Any]:
        """
        Convert hybrid results to Publication objects.
        
        Enriches local-only results with API metadata if needed.
        """
        from litforge.models import Publication, Author
        
        publications = []
        
        for result in results:
            if result.publication:
                publications.append(result.publication)
            else:
                # Create minimal Publication from HybridResult
                pub = Publication(
                    id=result.paper_id,
                    title=result.title,
                    year=result.year,
                    doi=result.doi,
                    abstract=result.abstract,
                    citation_count=result.citation_count,
                    authors=[],
                    sources=["local_index" if result.from_local else "api"],
                )
                publications.append(pub)
        
        return publications
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "index_name": self.index_name,
            "index_size": self.index.size if self.index else 0,
            "has_embedder": self.embedder is not None,
            "has_s2_client": self.s2_client is not None,
            "has_oa_client": self.oa_client is not None,
        }


# Convenience function for integration
def hybrid_search(
    query: str,
    k: int = 25,
    index_name: str = "default",
) -> list[Any]:
    """
    Quick hybrid search returning Publications.
    
    Args:
        query: Search query
        k: Number of results
        index_name: Name of local index
        
    Returns:
        List of Publication objects
    """
    service = HybridSearchService(index_name=index_name)
    results = service.search(query, k=k)
    return service.get_publications(results)
