"""
ChromaDB vector store implementation.

ChromaDB is the default store - simple, embedded, works everywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from litforge.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaDBStore(BaseVectorStore):
    """
    ChromaDB vector store.
    
    ChromaDB provides a simple, embedded vector database that works
    out of the box without any external dependencies.
    """
    
    def __init__(
        self,
        persist_dir: Path | str,
        collection_name: str = "litforge",
    ):
        """
        Initialize ChromaDB store.
        
        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the collection
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None
    
    @property
    def client(self) -> Any:
        """Lazy-load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(
                        anonymized_telemetry=False,
                    ),
                )
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. Install with: pip install chromadb"
                )
        return self._client
    
    @property
    def collection(self) -> Any:
        """Get or create collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection
    
    def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        text: str,
    ) -> None:
        """Add a document to the store."""
        # ChromaDB requires string values in metadata
        clean_metadata = self._clean_metadata(metadata)
        
        self.collection.upsert(
            ids=[id],
            embeddings=[embedding],
            metadatas=[clean_metadata],
            documents=[text],
        )
    
    def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        """Add multiple documents in batch."""
        clean_metadatas = [self._clean_metadata(m) for m in metadatas]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=clean_metadatas,
            documents=texts,
        )
    
    def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        where = None
        if filter_metadata:
            where = self._build_where_clause(filter_metadata)
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted = []
        
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            for i, doc_id in enumerate(ids):
                formatted.append({
                    "id": doc_id,
                    "text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": 1 - (distances[i] if i < len(distances) else 0),  # Convert distance to similarity
                })
        
        return formatted
    
    def delete(self, id: str) -> None:
        """Delete a document by ID."""
        self.collection.delete(ids=[id])
    
    def clear(self) -> None:
        """Clear all documents."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self._collection = None  # Force recreation
    
    def count(self) -> int:
        """Get document count."""
        return self.collection.count()
    
    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata for ChromaDB (requires string/int/float/bool values)."""
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                # Convert list to string
                clean[key] = ", ".join(str(v) for v in value)
            else:
                clean[key] = str(value)
        return clean
    
    def _build_where_clause(
        self,
        filter_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build ChromaDB where clause from filter."""
        conditions = []
        
        for key, value in filter_metadata.items():
            if isinstance(value, list):
                # OR condition for list values
                conditions.append({
                    "$or": [{key: {"$eq": v}} for v in value]
                })
            else:
                conditions.append({key: {"$eq": value}})
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        else:
            return {}
