"""
Knowledge Service - Vector store and semantic search.

Manages document indexing and similarity search.
"""

from __future__ import annotations

import hashlib
import logging
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
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> int:
        """
        Index publications into the knowledge base.
        
        Args:
            publications: Publications to index
            include_fulltext: Whether to retrieve and index full text
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
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                
                if not chunks:
                    continue
                
                # Generate embeddings
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
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            limit: Maximum results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching chunks with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embedder.embed([query])[0]
        
        # Search vector store
        results = self.vector_store.search(
            embedding=query_embedding,
            limit=limit,
            filter_metadata=filter_metadata,
        )
        
        return results
    
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
                "metadata": {**base_metadata, "chunk_type": "abstract"},
            })
        
        # Get full text if requested
        if include_fulltext:
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
