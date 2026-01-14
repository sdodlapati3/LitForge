"""
Forge - The main entry point for LitForge.

The Forge class provides a unified, high-level API for all LitForge functionality.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from litforge.config import LitForgeConfig, get_config
from litforge.models import (
    Publication,
    SearchQuery,
    SearchResult,
    SearchFilter,
    CitationNetwork,
)

logger = logging.getLogger(__name__)


class Forge:
    """
    🔥 LitForge - Forging Knowledge from Literature
    
    The main entry point for all LitForge functionality.
    
    Example:
        >>> from litforge import Forge
        >>> forge = Forge()
        
        >>> # Search for papers
        >>> papers = forge.search("CRISPR gene editing", limit=50)
        
        >>> # Build knowledge base
        >>> forge.index(papers)
        
        >>> # Ask questions
        >>> answer = forge.ask("What are the main CRISPR mechanisms?")
        >>> print(answer.text)
    """
    
    def __init__(
        self,
        config: LitForgeConfig | None = None,
        *,
        # Quick config overrides
        sources: list[str] | None = None,
        embeddings: str | None = None,
        vector_store: str | None = None,
        llm: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        """
        Initialize LitForge.
        
        Args:
            config: Full configuration object (optional)
            sources: Override default sources (e.g., ["openalex", "pubmed"])
            embeddings: Override embedding provider ("openai", "local")
            vector_store: Override vector store ("chroma", "qdrant", "faiss")
            llm: Override LLM provider ("openai", "anthropic", "ollama")
            cache_dir: Override cache directory
        """
        self.config = config or get_config()
        
        # Apply quick overrides
        if sources:
            self.config.sources.default_sources = sources
        if embeddings:
            self.config.embeddings.provider = embeddings  # type: ignore
        if vector_store:
            self.config.vector_store.provider = vector_store  # type: ignore
        if llm:
            self.config.llm.provider = llm  # type: ignore
        if cache_dir:
            self.config.cache.directory = Path(cache_dir)
        
        # Lazy-initialized services
        self._discovery: Any = None
        self._retrieval: Any = None
        self._citation: Any = None
        self._knowledge: Any = None
        self._qa: Any = None
        
        logger.info(
            f"LitForge initialized with sources={self.config.sources.default_sources}, "
            f"embeddings={self.config.embeddings.provider}, "
            f"vector_store={self.config.vector_store.provider}"
        )
    
    # =========================================================================
    # Service Properties (Lazy Loading)
    # =========================================================================
    
    @property
    def discovery(self):
        """Get discovery service."""
        if self._discovery is None:
            from litforge.services.discovery import DiscoveryService
            self._discovery = DiscoveryService(config=self.config)
        return self._discovery
    
    @property
    def retrieval(self):
        """Get retrieval service."""
        if self._retrieval is None:
            from litforge.services.retrieval import RetrievalService
            self._retrieval = RetrievalService(config=self.config)
        return self._retrieval
    
    @property
    def citation(self):
        """Get citation service."""
        if self._citation is None:
            from litforge.services.citation import CitationService
            self._citation = CitationService(
                config=self.config,
                discovery=self.discovery,
            )
        return self._citation
    
    @property
    def knowledge(self):
        """Get knowledge base service."""
        if self._knowledge is None:
            from litforge.services.knowledge import KnowledgeService
            self._knowledge = KnowledgeService(config=self.config)
        return self._knowledge
    
    @property
    def qa(self):
        """Get Q&A service."""
        if self._qa is None:
            from litforge.services.qa import QAService
            self._qa = QAService(
                config=self.config,
                knowledge=self.knowledge,
            )
        return self._qa
    
    # =========================================================================
    # Discovery Methods
    # =========================================================================
    
    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        sources: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        open_access_only: bool = False,
        fields_of_study: list[str] | None = None,
        **kwargs,
    ) -> list[Publication]:
        """
        Search for publications across data sources.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            sources: Data sources to search (default from config)
            year_from: Minimum publication year
            year_to: Maximum publication year
            open_access_only: Only return open access papers
            fields_of_study: Filter by field (e.g., ["Biology", "Medicine"])
            **kwargs: Additional filter options
            
        Returns:
            List of matching publications
            
        Example:
            >>> papers = forge.search("CRISPR", limit=100, year_from=2020)
            >>> print(f"Found {len(papers)} papers")
        """
        search_filter = SearchFilter(
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
            fields_of_study=fields_of_study,
            **kwargs,
        )
        
        search_query = SearchQuery(
            query=query,
            sources=sources or self.config.sources.default_sources,
            filters=search_filter,
            limit=limit,
        )
        
        result = self.discovery.search(search_query)
        return result.publications
    
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
        
        Args:
            doi: Digital Object Identifier
            pmid: PubMed ID
            arxiv_id: arXiv ID
            openalex_id: OpenAlex work ID
            semantic_scholar_id: Semantic Scholar paper ID
            
        Returns:
            Publication if found, None otherwise
            
        Example:
            >>> paper = forge.lookup(doi="10.1038/nature12373")
            >>> print(paper.title)
        """
        return self.discovery.lookup(
            doi=doi,
            pmid=pmid,
            arxiv_id=arxiv_id,
            openalex_id=openalex_id,
            semantic_scholar_id=semantic_scholar_id,
        )
    
    def recommend(
        self,
        papers: Sequence[Publication] | Sequence[str],
        limit: int = 20,
    ) -> list[Publication]:
        """
        Get paper recommendations based on seed papers.
        
        Args:
            papers: Seed papers (Publication objects or IDs/DOIs)
            limit: Number of recommendations
            
        Returns:
            List of recommended publications
        """
        return self.discovery.recommend(papers, limit=limit)
    
    # =========================================================================
    # Retrieval Methods
    # =========================================================================
    
    def retrieve(
        self,
        paper: Publication | str,
        *,
        download_pdf: bool = True,
        extract_text: bool = True,
    ) -> Publication:
        """
        Retrieve full content for a publication.
        
        Attempts to download PDF from open access sources and extract full text.
        
        Args:
            paper: Publication object or DOI/ID
            download_pdf: Whether to download PDF
            extract_text: Whether to extract text from PDF
            
        Returns:
            Publication with full_text populated (if available)
        """
        return self.retrieval.retrieve(
            paper,
            download_pdf=download_pdf,
            extract_text=extract_text,
        )
    
    def retrieve_batch(
        self,
        papers: Sequence[Publication],
        *,
        max_concurrent: int = 5,
        progress: bool = True,
    ) -> list[Publication]:
        """
        Retrieve full content for multiple publications.
        
        Args:
            papers: Publications to retrieve
            max_concurrent: Maximum concurrent downloads
            progress: Show progress bar
            
        Returns:
            Publications with full_text populated where available
        """
        return self.retrieval.retrieve_batch(
            papers,
            max_concurrent=max_concurrent,
            progress=progress,
        )
    
    # =========================================================================
    # Citation Network Methods
    # =========================================================================
    
    def build_network(
        self,
        seeds: Sequence[Publication] | Sequence[str],
        *,
        depth: int = 2,
        max_papers: int = 500,
        direction: str = "both",
    ) -> CitationNetwork:
        """
        Build a citation network from seed papers.
        
        Args:
            seeds: Seed papers (Publication objects or DOIs/IDs)
            depth: How many citation levels to traverse
            max_papers: Maximum total papers in network
            direction: "citing", "cited", or "both"
            
        Returns:
            CitationNetwork with papers and citation relationships
            
        Example:
            >>> paper = forge.lookup(doi="10.1126/science.aad5227")
            >>> network = forge.build_network([paper], depth=2)
            >>> influential = network.most_cited(n=10)
        """
        return self.citation.build_network(
            seeds,
            depth=depth,
            max_papers=max_papers,
            direction=direction,
        )
    
    def find_clusters(
        self,
        network: CitationNetwork,
        algorithm: str = "louvain",
    ) -> list[Any]:
        """
        Find clusters in a citation network.
        
        Args:
            network: Citation network to analyze
            algorithm: Clustering algorithm ("louvain", "leiden", "label_propagation")
            
        Returns:
            List of Cluster objects
        """
        return self.citation.find_clusters(network, algorithm=algorithm)
    
    # =========================================================================
    # Knowledge Base Methods
    # =========================================================================
    
    def index(
        self,
        papers: Sequence[Publication],
        *,
        include_full_text: bool = True,
        batch_size: int = 100,
        progress: bool = True,
    ) -> int:
        """
        Index publications into the knowledge base.
        
        Args:
            papers: Publications to index
            include_full_text: Index full text (if available)
            batch_size: Batch size for embedding generation
            progress: Show progress bar
            
        Returns:
            Number of papers indexed
            
        Example:
            >>> papers = forge.search("CRISPR", limit=100)
            >>> forge.index(papers)
            >>> answer = forge.ask("What is CRISPR?")
        """
        return self.knowledge.index(
            papers,
            include_full_text=include_full_text,
            batch_size=batch_size,
            progress=progress,
        )
    
    def search_knowledge(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[Publication, float]]:
        """
        Search the knowledge base for relevant papers.
        
        Args:
            query: Search query
            limit: Maximum results
            threshold: Minimum similarity score
            
        Returns:
            List of (Publication, score) tuples
        """
        return self.knowledge.search(query, limit=limit, threshold=threshold)
    
    def clear_knowledge(self) -> None:
        """Clear the knowledge base."""
        self.knowledge.clear()
    
    # =========================================================================
    # Q&A Methods
    # =========================================================================
    
    def ask(
        self,
        question: str,
        *,
        max_sources: int = 5,
        include_citations: bool = True,
    ) -> Any:  # Returns QAResponse
        """
        Ask a question about the indexed literature.
        
        Args:
            question: Natural language question
            max_sources: Maximum papers to use as context
            include_citations: Include citation references
            
        Returns:
            QAResponse with answer text and source citations
            
        Example:
            >>> forge.index(papers)
            >>> answer = forge.ask("What are the mechanisms of CRISPR?")
            >>> print(answer.text)
            >>> print(answer.citations)
        """
        return self.qa.ask(
            question,
            max_sources=max_sources,
            include_citations=include_citations,
        )
    
    def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
    ) -> Any:  # Returns ChatResponse
        """
        Multi-turn chat about the indexed literature.
        
        Args:
            message: User message
            session_id: Session ID for conversation continuity
            
        Returns:
            ChatResponse with answer and session info
        """
        return self.qa.chat(message, session_id=session_id)
    
    def summarize(
        self,
        papers: Sequence[Publication],
        *,
        focus: str | None = None,
        max_length: int = 500,
    ) -> str:
        """
        Summarize a collection of papers.
        
        Args:
            papers: Papers to summarize
            focus: Optional focus topic
            max_length: Maximum summary length (words)
            
        Returns:
            Summary text
        """
        return self.qa.summarize(papers, focus=focus, max_length=max_length)
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def export_bibtex(self, papers: Sequence[Publication], path: str | Path) -> None:
        """Export papers to BibTeX file."""
        with open(path, "w") as f:
            for paper in papers:
                f.write(paper.to_bibtex())
                f.write("\n\n")
    
    def export_json(self, papers: Sequence[Publication], path: str | Path) -> None:
        """Export papers to JSON file."""
        import json
        
        data = [p.model_dump(exclude={"embedding", "full_text", "raw_data"}) for p in papers]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def export_network(
        self,
        network: CitationNetwork,
        path: str | Path,
        format: str = "graphml",
    ) -> None:
        """
        Export citation network to file.
        
        Args:
            network: Network to export
            path: Output file path
            format: Export format (graphml, gexf, json)
        """
        network.export(str(path), format=format)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def stats(self) -> dict[str, Any]:
        """Get statistics about the current state."""
        return {
            "config": {
                "sources": self.config.sources.default_sources,
                "embeddings": self.config.embeddings.provider,
                "vector_store": self.config.vector_store.provider,
                "llm": self.config.llm.provider,
            },
            "knowledge_base": self.knowledge.stats() if self._knowledge else None,
            "cache": {
                "enabled": self.config.cache.enabled,
                "directory": str(self.config.cache.directory),
            },
        }
    
    def __repr__(self) -> str:
        return (
            f"Forge(sources={self.config.sources.default_sources}, "
            f"embeddings={self.config.embeddings.provider})"
        )
