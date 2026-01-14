"""
Knowledge Service - Vector store and semantic search.

Manages document indexing and similarity search with RAG support.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Sequence

from litforge.config import LitForgeConfig, get_config
from litforge.models import Publication

logger = logging.getLogger(__name__)


class KnowledgeService:
    """
    Service for managing a knowledge base of publications.
    
    Provides:
    - Document indexing with embeddings
    - Semantic similarity search
    - Chunk-level retrieval for RAG
    """
    
    def __init__(self, config: LitForgeConfig | None = None):
        """
        Initialize the knowledge service.
        
        Args:
            config: LitForge configuration
        """
        self.config = config or get_config()
        self._vector_store: Any | None = None
        self._embedder: Any | None = None
        self._retrieval: Any | None = None
    
    @property
    def retrieval(self) -> Any:
        """Lazy-load retrieval service."""
        if self._retrieval is None:
            from litforge.services.retrieval import RetrievalService
            self._retrieval = RetrievalService(self.config)
        return self._retrieval
    
    @property
    def embedder(self) -> Any:
        """Lazy-load embedding provider."""
        if self._embedder is None:
            self._embedder = self._create_embedder()
        return self._embedder
    
    @property
    def vector_store(self) -> Any:
        """Lazy-load vector store."""
        if self._vector_store is None:
            self._vector_store = self._create_vector_store()
        return self._vector_store
    
    def _create_embedder(self) -> Any:
        """Create embedding provider based on config."""
        provider = self.config.embedding.provider
        
        if provider == "openai":
            from litforge.embedding.openai import OpenAIEmbedder
            return OpenAIEmbedder(
                api_key=self.config.embedding.openai_api_key,
                model=self.config.embedding.model,
            )
        elif provider == "sentence_transformers":
            from litforge.embedding.sentence_transformers import LocalEmbedder
            return LocalEmbedder(model=self.config.embedding.model)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    def _create_vector_store(self) -> Any:
        """Create vector store based on config."""
        backend = self.config.vector_store.backend
        persist_dir = Path(self.config.vector_store.path)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        if backend == "chromadb":
            from litforge.stores.chromadb import ChromaDBStore
            return ChromaDBStore(
                persist_dir=persist_dir,
                collection_name=self.config.vector_store.collection_name,
            )
        elif backend == "qdrant":
            from litforge.stores.qdrant import QdrantStore
            return QdrantStore(
                path=persist_dir,
                collection_name=self.config.vector_store.collection_name,
            )
        elif backend == "faiss":
            from litforge.stores.faiss import FAISSStore
            return FAISSStore(
                path=persist_dir,
                dimension=self.config.embedding.dimensions,
            )
        else:
            raise ValueError(f"Unknown vector store backend: {backend}")
    
    def index(
        self,
        publications: Sequence[Publication],
        include_fulltext: bool = True,
        section_aware: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> int:
        """
        Index publications into the knowledge base.
        
        Args:
            publications: Publications to index
            include_fulltext: Whether to retrieve and index full text
            section_aware: Use section-aware chunking (requires full text)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks indexed
        """
        total_chunks = 0
        
        for pub in publications:
            try:
                chunks = self._process_publication(
                    pub,
                    include_fulltext=include_fulltext,
                    section_aware=section_aware,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                
                if not chunks:
                    continue
                
                # Generate embeddings in batches
                texts = [c["text"] for c in chunks]
                embeddings = self.embedder.embed(texts)
                
                # Store in vector store
                for chunk, embedding in zip(chunks, embeddings):
                    self.vector_store.add(
                        id=chunk["id"],
                        embedding=embedding,
                        metadata=chunk["metadata"],
                        text=chunk["text"],
                    )
                
                total_chunks += len(chunks)
                logger.info(f"Indexed {len(chunks)} chunks from {pub.doi or pub.title}")
                
            except Exception as e:
                logger.error(f"Error indexing {pub.doi or pub.title}: {e}")
        
        return total_chunks
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        section_filter: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            limit: Maximum results
            filter_metadata: Optional metadata filters
            section_filter: Filter by section name(s) (e.g., "methods", ["abstract", "results"])
            
        Returns:
            List of matching chunks with metadata and scores
        """
        # Build metadata filter
        if section_filter:
            if filter_metadata is None:
                filter_metadata = {}
            if isinstance(section_filter, str):
                filter_metadata["section"] = section_filter
            else:
                # Multiple sections - will need to search each and combine
                all_results = []
                for section in section_filter:
                    section_filter_meta = {**filter_metadata, "section": section}
                    query_embedding = self.embedder.embed([query])[0]
                    results = self.vector_store.search(
                        embedding=query_embedding,
                        limit=limit,
                        filter_metadata=section_filter_meta,
                    )
                    all_results.extend(results)
                # Sort by score and limit
                all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                return all_results[:limit]
        
        # Generate query embedding
        query_embedding = self.embedder.embed([query])[0]
        
        # Search vector store
        results = self.vector_store.search(
            embedding=query_embedding,
            limit=limit,
            filter_metadata=filter_metadata,
        )
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            limit: Maximum results
            semantic_weight: Weight for semantic search scores (0-1)
            keyword_weight: Weight for keyword search scores (0-1)
            
        Returns:
            List of matching chunks with combined scores
        """
        # Get semantic results (more than needed for re-ranking)
        semantic_results = self.search(query, limit=limit * 2)
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Score results by keyword matches
        scored_results = []
        for result in semantic_results:
            text = result.get("text", "").lower()
            
            # Calculate keyword score
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
            keyword_score = keyword_matches / max(len(keywords), 1)
            
            # Combine scores
            semantic_score = result.get("score", 0)
            combined_score = (
                semantic_weight * semantic_score + keyword_weight * keyword_score
            )
            
            result["semantic_score"] = semantic_score
            result["keyword_score"] = keyword_score
            result["score"] = combined_score
            scored_results.append(result)
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_results[:limit]
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from a query."""
        # Remove common stopwords and keep meaningful terms
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "and",
            "but", "if", "or", "because", "until", "while", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "about",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        keywords = [w for w in words if w not in stopwords]
        
        return keywords

    def get_context(
        self,
        query: str,
        limit: int = 5,
        max_tokens: int = 4000,
    ) -> str:
        """
        Get context for a query (for RAG).
        
        Args:
            query: Query to find context for
            limit: Maximum chunks to retrieve
            max_tokens: Maximum tokens in context
            
        Returns:
            Concatenated context string
        """
        results = self.search(query, limit=limit)
        
        context_parts = []
        total_tokens = 0
        
        for result in results:
            text = result.get("text", "")
            # Rough token estimate (1 token ≈ 4 chars)
            tokens = len(text) // 4
            
            if total_tokens + tokens > max_tokens:
                break
            
            # Format context with source info
            metadata = result.get("metadata", {})
            source_info = f"[Source: {metadata.get('title', 'Unknown')}]"
            context_parts.append(f"{source_info}\n{text}")
            total_tokens += tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear the knowledge base."""
        self.vector_store.clear()
        logger.info("Knowledge base cleared")
    
    def _process_publication(
        self,
        publication: Publication,
        include_fulltext: bool,
        section_aware: bool,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict[str, Any]]:
        """Process a publication into chunks."""
        chunks = []
        base_metadata = {
            "doi": publication.doi,
            "title": publication.title,
            "authors": ", ".join(a.name for a in publication.authors[:5]),
            "year": publication.year,
            "journal": publication.venue,
            "openalex_id": publication.openalex_id,
        }
        
        # Always index abstract
        if publication.abstract:
            chunks.append({
                "id": self._generate_chunk_id(publication, "abstract"),
                "text": f"Title: {publication.title}\n\nAbstract: {publication.abstract}",
                "metadata": {**base_metadata, "chunk_type": "abstract", "section": "abstract"},
            })
        
        # Get full text if requested
        if include_fulltext:
            if section_aware:
                # Use section-aware retrieval and chunking
                pub_with_sections = self.retrieval.retrieve_with_sections(publication)
                if pub_with_sections.sections:
                    # Index each section separately
                    for section_name, section_content in pub_with_sections.sections.items():
                        if section_name == "references":
                            continue  # Skip references section
                        
                        # Chunk each section
                        section_chunks = self._chunk_text(
                            section_content, chunk_size, chunk_overlap
                        )
                        for i, text in enumerate(section_chunks):
                            chunks.append({
                                "id": self._generate_chunk_id(
                                    publication, f"{section_name}_{i}"
                                ),
                                "text": f"[{section_name.upper()}]\n{text}",
                                "metadata": {
                                    **base_metadata,
                                    "chunk_type": "section",
                                    "section": section_name,
                                    "chunk_index": i,
                                },
                            })
                elif pub_with_sections.full_text:
                    # Fall back to regular chunking
                    text_chunks = self._chunk_text(
                        pub_with_sections.full_text, chunk_size, chunk_overlap
                    )
                    for i, text in enumerate(text_chunks):
                        chunks.append({
                            "id": self._generate_chunk_id(publication, f"fulltext_{i}"),
                            "text": text,
                            "metadata": {
                                **base_metadata,
                                "chunk_type": "fulltext",
                                "chunk_index": i,
                            },
                        })
            else:
                # Regular full text retrieval
                fulltext = self.retrieval.retrieve(publication)
                if fulltext:
                    text_chunks = self._chunk_text(fulltext, chunk_size, chunk_overlap)
                    for i, text in enumerate(text_chunks):
                        chunks.append({
                            "id": self._generate_chunk_id(publication, f"fulltext_{i}"),
                            "text": text,
                            "metadata": {
                                **base_metadata,
                                "chunk_type": "fulltext",
                                "chunk_index": i,
                            },
                        })
        
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < text_len:
                # Look for sentence end
                for sep in [". ", ".\n", "? ", "! "]:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
        
        return chunks
    
    def _generate_chunk_id(self, publication: Publication, suffix: str) -> str:
        """Generate a unique chunk ID."""
        base = publication.doi or publication.openalex_id or publication.title
        return hashlib.md5(f"{base}:{suffix}".encode()).hexdigest()
