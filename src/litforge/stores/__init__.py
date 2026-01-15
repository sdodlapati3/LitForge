"""Vector and document stores."""

from litforge.stores.base import BaseVectorStore
from litforge.stores.chromadb import ChromaDBStore

# Lazy imports for optional dependencies
def get_qdrant_store():
    """Get QdrantStore (requires qdrant-client)."""
    from litforge.stores.qdrant import QdrantStore
    return QdrantStore

def get_faiss_store():
    """Get FAISSStore (requires faiss-cpu or faiss-gpu)."""
    from litforge.stores.faiss import FAISSStore
    return FAISSStore

def get_embedding_index(name: str = "default"):
    """Get EmbeddingIndex for local paper similarity search."""
    from litforge.stores.embedding_index import get_embedding_index as _get
    return _get(name)

def get_embedding_index_manager():
    """Get EmbeddingIndexManager for managing multiple indices."""
    from litforge.stores.embedding_index import EmbeddingIndexManager
    return EmbeddingIndexManager()

__all__ = [
    "BaseVectorStore",
    "ChromaDBStore",
    "get_qdrant_store",
    "get_faiss_store",
    "get_embedding_index",
    "get_embedding_index_manager",
]
