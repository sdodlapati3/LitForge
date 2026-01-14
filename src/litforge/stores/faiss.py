"""
FAISS vector store implementation.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity
search and clustering of dense vectors. Ideal for high-performance local search.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from litforge.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSStore(BaseVectorStore):
    """
    FAISS vector store.
    
    FAISS provides extremely fast similarity search for large-scale vectors.
    Best for scenarios where:
    - High query throughput is needed
    - Memory efficiency is critical
    - No external database is desired
    """

    def __init__(
        self,
        persist_dir: Path | str,
        embedding_dim: int = 1536,
        index_type: str = "Flat",
        use_gpu: bool = False,
    ):
        """
        Initialize FAISS store.
        
        Args:
            persist_dir: Directory for persistent storage
            embedding_dim: Dimension of embeddings (default: 1536 for OpenAI)
            index_type: FAISS index type:
                - "Flat": Exact search (brute force)
                - "IVF": Inverted file index (faster, approximate)
                - "HNSW": Hierarchical Navigable Small World (good quality/speed)
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu

        # File paths for persistence
        self.index_path = self.persist_dir / "index.faiss"
        self.metadata_path = self.persist_dir / "metadata.pkl"

        # In-memory storage
        self._index: Any = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._texts: dict[str, str] = {}
        self._next_idx: int = 0

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load index and metadata from disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu "
                "or pip install faiss-gpu for GPU support"
            )

        if self.index_path.exists() and self.metadata_path.exists():
            # Load FAISS index
            self._index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
                self._id_to_idx = data.get("id_to_idx", {})
                self._idx_to_id = data.get("idx_to_id", {})
                self._metadata = data.get("metadata", {})
                self._texts = data.get("texts", {})
                self._next_idx = data.get("next_idx", 0)

            logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
        else:
            # Create new index
            self._create_index()

    def _create_index(self) -> None:
        """Create a new FAISS index."""
        import faiss

        if self.index_type == "Flat":
            # Exact search using L2 distance
            self._index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine
        elif self.index_type == "IVF":
            # Inverted file index for approximate search
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = 100  # Number of clusters
            self._index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            # IVF needs training, but we'll train on first batch add
        elif self.index_type == "HNSW":
            # HNSW index
            self._index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 neighbors
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except Exception as e:
                logger.warning(f"GPU not available, using CPU: {e}")

        logger.info(f"Created FAISS {self.index_type} index")

    def _save(self) -> None:
        """Save index and metadata to disk."""
        import faiss

        # Move to CPU for saving if on GPU
        index_to_save = self._index
        if self.use_gpu:
            try:
                index_to_save = faiss.index_gpu_to_cpu(self._index)
            except Exception:
                pass  # Already on CPU

        # Save FAISS index
        faiss.write_index(index_to_save, str(self.index_path))

        # Save metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "id_to_idx": self._id_to_idx,
                "idx_to_id": self._idx_to_id,
                "metadata": self._metadata,
                "texts": self._texts,
                "next_idx": self._next_idx,
            }, f)

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """Normalize vector for cosine similarity."""
        import numpy as np
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tolist()

    def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        text: str,
    ) -> None:
        """Add a document to the store."""
        import numpy as np

        # Normalize for cosine similarity
        normalized = self._normalize_vector(embedding)
        vector = np.array([normalized], dtype=np.float32)

        # Check if ID already exists (update case)
        if id in self._id_to_idx:
            # FAISS doesn't support in-place updates easily
            # We'll just update metadata and text, leaving the vector
            # For a proper update, call delete then add
            self._metadata[id] = metadata
            self._texts[id] = text
        else:
            # Add new vector
            idx = self._next_idx
            self._index.add(vector)

            # Update mappings
            self._id_to_idx[id] = idx
            self._idx_to_id[idx] = id
            self._metadata[id] = metadata
            self._texts[id] = text
            self._next_idx += 1

        self._save()

    def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        """Add multiple documents in batch."""
        import numpy as np

        # Filter out existing IDs (update metadata only for those)
        new_ids = []
        new_embeddings = []

        for doc_id, embedding, metadata, text in zip(ids, embeddings, metadatas, texts):
            if doc_id in self._id_to_idx:
                # Update metadata only
                self._metadata[doc_id] = metadata
                self._texts[doc_id] = text
            else:
                new_ids.append(doc_id)
                new_embeddings.append(embedding)
                self._metadata[doc_id] = metadata
                self._texts[doc_id] = text

        if new_embeddings:
            # Normalize vectors
            normalized = [self._normalize_vector(e) for e in new_embeddings]
            vectors = np.array(normalized, dtype=np.float32)

            # Train IVF index if needed
            if self.index_type == "IVF" and not self._index.is_trained:
                self._index.train(vectors)

            # Add vectors
            self._index.add(vectors)

            # Update mappings
            for doc_id in new_ids:
                idx = self._next_idx
                self._id_to_idx[doc_id] = idx
                self._idx_to_id[idx] = doc_id
                self._next_idx += 1

        self._save()

    def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        import numpy as np

        if self._index.ntotal == 0:
            return []

        # Normalize query vector
        normalized = self._normalize_vector(embedding)
        query = np.array([normalized], dtype=np.float32)

        # Search with a larger limit if filtering
        search_limit = limit * 10 if filter_metadata else limit
        search_limit = min(search_limit, self._index.ntotal)

        # Perform search
        scores, indices = self._index.search(query, search_limit)

        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue

            doc_id = self._idx_to_id.get(idx)
            if doc_id is None:
                continue

            metadata = self._metadata.get(doc_id, {})

            # Apply metadata filter
            if filter_metadata:
                if not self._matches_filter(metadata, filter_metadata):
                    continue

            results.append({
                "id": doc_id,
                "text": self._texts.get(doc_id, ""),
                "metadata": metadata,
                "score": float(score),
            })

            if len(results) >= limit:
                break

        return results

    def _matches_filter(
        self,
        metadata: dict[str, Any],
        filter_metadata: dict[str, Any],
    ) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False

            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False

        return True

    def delete(self, id: str) -> None:
        """
        Delete a document by ID.
        
        Note: FAISS doesn't support efficient single-item deletion.
        This implementation marks the item as deleted in metadata
        but doesn't remove from the index. For a clean index,
        rebuild periodically using clear() + add_batch().
        """
        if id in self._id_to_idx:
            # Remove from metadata (vector remains in index)
            del self._id_to_idx[id]
            if id in self._metadata:
                del self._metadata[id]
            if id in self._texts:
                del self._texts[id]
            self._save()

    def clear(self) -> None:
        """Clear all documents."""
        # Reset everything
        self._id_to_idx = {}
        self._idx_to_id = {}
        self._metadata = {}
        self._texts = {}
        self._next_idx = 0

        # Recreate index
        self._create_index()

        # Remove persisted files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

    def count(self) -> int:
        """Get document count."""
        # Return count of valid documents (not index total due to deletions)
        return len(self._id_to_idx)

    def rebuild_index(self) -> None:
        """
        Rebuild the FAISS index from scratch.
        
        Call this periodically to clean up deleted vectors and
        optimize index structure.
        """

        if not self._id_to_idx:
            self._create_index()
            return

        # Collect all valid vectors
        # This requires having stored the original vectors, which we don't
        # So this is a no-op warning for now
        logger.warning(
            "Index rebuild requires original vectors. "
            "Consider re-indexing your documents."
        )
