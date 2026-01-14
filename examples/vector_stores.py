"""
Vector store comparison example.

This example demonstrates the different vector store options in LitForge
and helps you choose the right one for your use case.
"""

import tempfile
import time
from pathlib import Path


def test_chromadb():
    """Test ChromaDB (default, embedded)."""
    print("\n📦 ChromaDB (Default)")
    print("-" * 40)
    
    from litforge.stores.chromadb import ChromaDBStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaDBStore(persist_dir=tmpdir)
        
        # Add documents
        start = time.time()
        for i in range(100):
            store.add(
                id=f"doc_{i}",
                embedding=[0.1 * (i % 10)] * 1536,
                metadata={"index": i, "category": f"cat_{i % 5}"},
                text=f"Document {i} content",
            )
        add_time = time.time() - start
        
        # Search
        start = time.time()
        results = store.search([0.5] * 1536, limit=10)
        search_time = time.time() - start
        
        print(f"  Add 100 docs: {add_time:.3f}s")
        print(f"  Search: {search_time:.3f}s")
        print(f"  Results: {len(results)}")
        print(f"  Best match: {results[0]['id']} (score: {results[0]['score']:.4f})")


def test_qdrant():
    """Test Qdrant (production-ready)."""
    print("\n📦 Qdrant (Production)")
    print("-" * 40)
    
    from litforge.stores.qdrant import QdrantStore
    
    store = QdrantStore(in_memory=True)
    
    # Add documents
    start = time.time()
    for i in range(100):
        store.add(
            id=f"doc_{i}",
            embedding=[0.1 * (i % 10)] * 1536,
            metadata={"index": i, "category": f"cat_{i % 5}"},
            text=f"Document {i} content",
        )
    add_time = time.time() - start
    
    # Search
    start = time.time()
    results = store.search([0.5] * 1536, limit=10)
    search_time = time.time() - start
    
    print(f"  Add 100 docs: {add_time:.3f}s")
    print(f"  Search: {search_time:.3f}s")
    print(f"  Results: {len(results)}")
    print(f"  Best match: {results[0]['id']} (score: {results[0]['score']:.4f})")


def test_faiss():
    """Test FAISS (high-performance local)."""
    print("\n📦 FAISS (High-Performance)")
    print("-" * 40)
    
    from litforge.stores.faiss import FAISSStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FAISSStore(persist_dir=tmpdir)
        
        # Add documents
        start = time.time()
        for i in range(100):
            store.add(
                id=f"doc_{i}",
                embedding=[0.1 * (i % 10)] * 1536,
                metadata={"index": i, "category": f"cat_{i % 5}"},
                text=f"Document {i} content",
            )
        add_time = time.time() - start
        
        # Search
        start = time.time()
        results = store.search([0.5] * 1536, limit=10)
        search_time = time.time() - start
        
        print(f"  Add 100 docs: {add_time:.3f}s")
        print(f"  Search: {search_time:.3f}s")
        print(f"  Results: {len(results)}")
        print(f"  Best match: {results[0]['id']} (score: {results[0]['score']:.4f})")


def main():
    print("🔬 Vector Store Comparison")
    print("=" * 40)
    
    print("""
Choose the right vector store:

• ChromaDB: Best for development and small projects
  - Easy setup, no external dependencies
  - Good for up to ~100K documents
  
• Qdrant: Best for production deployments
  - Cloud-hosted or self-hosted options
  - Advanced filtering, horizontal scaling
  - Good for millions of documents
  
• FAISS: Best for high-performance local search
  - Extremely fast search
  - GPU acceleration available
  - Good for read-heavy workloads
""")
    
    print("\nRunning benchmarks...\n")
    
    test_chromadb()
    test_qdrant()
    test_faiss()
    
    print("\n" + "=" * 40)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
