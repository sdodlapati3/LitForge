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

__all__ = [
    "BaseVectorStore",
    "ChromaDBStore",
    "get_qdrant_store",
    "get_faiss_store",
]
