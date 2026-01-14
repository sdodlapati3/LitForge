"""
Qdrant vector store implementation.

Qdrant is a high-performance vector database suitable for production deployments,
supporting both cloud-hosted and self-hosted options.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from litforge.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    """
    Qdrant vector store.
    
    Qdrant provides a production-ready vector database with support for
    filtering, payloads, and horizontal scaling.
    
    Can connect to:
    - Local in-memory storage (for testing)
    - Local on-disk storage (for development)
    - Qdrant Cloud (for production)
    - Self-hosted Qdrant server (for enterprise)
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "litforge",
        persist_dir: Path | str | None = None,
        embedding_dim: int = 1536,
        in_memory: bool = False,
    ):
        """
        Initialize Qdrant store.
        
        Args:
            url: Qdrant server URL (e.g., "http://localhost:6333" or cloud URL)
            api_key: API key for Qdrant Cloud
            collection_name: Name of the collection
            persist_dir: Directory for local on-disk storage (if no URL provided)
            embedding_dim: Dimension of embeddings (default: 1536 for OpenAI)
            in_memory: Use in-memory storage (for testing)
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.embedding_dim = embedding_dim
        self.in_memory = in_memory
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-load Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                # Determine connection mode
                if self.url:
                    # Remote or local server
                    self._client = QdrantClient(
                        url=self.url,
                        api_key=self.api_key,
                    )
                elif self.in_memory:
                    # In-memory (for testing)
                    self._client = QdrantClient(":memory:")
                elif self.persist_dir:
                    # Local on-disk storage
                    self.persist_dir.mkdir(parents=True, exist_ok=True)
                    self._client = QdrantClient(path=str(self.persist_dir))
                else:
                    # Default to in-memory
                    self._client = QdrantClient(":memory:")

                # Ensure collection exists
                self._ensure_collection()

            except ImportError:
                raise ImportError(
                    "Qdrant client not installed. Install with: pip install qdrant-client"
                )
        return self._client

    def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        from qdrant_client.models import Distance, VectorParams

        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        text: str,
    ) -> None:
        """Add a document to the store."""
        from qdrant_client.models import PointStruct

        # Include text in payload
        payload = {**metadata, "_text": text}

        # Convert string ID to UUID if needed
        point_id = self._to_point_id(id)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        """Add multiple documents in batch."""
        from qdrant_client.models import PointStruct

        points = []
        for i, (doc_id, embedding, metadata, text) in enumerate(
            zip(ids, embeddings, metadatas, texts)
        ):
            payload = {**metadata, "_text": text}
            points.append(
                PointStruct(
                    id=self._to_point_id(doc_id),
                    vector=embedding,
                    payload=payload,
                )
            )

        # Batch in chunks of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

    def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        query_filter = None
        if filter_metadata:
            query_filter = self._build_filter(filter_metadata)

        # Use query_points (new API) instead of search (deprecated)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        # Format results
        formatted = []
        for point in results.points:
            payload = dict(point.payload) if point.payload else {}
            text = payload.pop("_text", "")
            formatted.append({
                "id": str(point.id),
                "text": text,
                "metadata": payload,
                "score": point.score,
            })

        return formatted

    def delete(self, id: str) -> None:
        """Delete a document by ID."""
        from qdrant_client.models import PointIdsList

        point_id = self._to_point_id(id)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[point_id]),
        )

    def clear(self) -> None:
        """Clear all documents."""
        from qdrant_client.models import Distance, VectorParams

        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def count(self) -> int:
        """Get document count."""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

    def _to_point_id(self, id: str) -> str | int:
        """Convert string ID to Qdrant-compatible point ID."""
        # Try to convert to int, otherwise use UUID
        try:
            return int(id)
        except ValueError:
            # Use hash to create deterministic UUID from string
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, id))

    def _build_filter(self, filter_metadata: dict[str, Any]) -> Any:
        """Build Qdrant filter from metadata filter."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        conditions = []

        for key, value in filter_metadata.items():
            if isinstance(value, list):
                # Match any of the values
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=conditions) if conditions else None
