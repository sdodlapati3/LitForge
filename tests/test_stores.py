"""
Tests for vector stores.
"""

import tempfile
from pathlib import Path

import pytest


class TestBaseVectorStore:
    """Test the base vector store interface."""

    def test_abstract_methods(self):
        """Ensure BaseVectorStore cannot be instantiated."""
        from litforge.stores.base import BaseVectorStore

        with pytest.raises(TypeError):
            BaseVectorStore()


class TestChromaDBStore:
    """Tests for ChromaDB vector store."""

    @pytest.fixture
    def store(self, tmp_path):
        from litforge.stores.chromadb import ChromaDBStore

        return ChromaDBStore(persist_dir=tmp_path, collection_name="test")

    def test_add_and_search(self, store):
        """Test adding and searching documents."""
        store.add(
            id="doc1",
            embedding=[0.1] * 1536,
            metadata={"title": "Test Doc"},
            text="Test content",
        )

        results = store.search([0.1] * 1536, limit=1)

        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["text"] == "Test content"
        assert results[0]["metadata"]["title"] == "Test Doc"

    def test_delete(self, store):
        """Test deleting documents."""
        store.add(
            id="doc1",
            embedding=[0.1] * 1536,
            metadata={"key": "value"},  # ChromaDB requires non-empty metadata
            text="Test",
        )
        assert store.count() == 1

        store.delete("doc1")
        assert store.count() == 0

    def test_clear(self, store):
        """Test clearing all documents."""
        for i in range(5):
            store.add(
                id=f"doc{i}",
                embedding=[0.1] * 1536,
                metadata={"idx": i},  # ChromaDB requires non-empty metadata
                text=f"Test {i}",
            )
        assert store.count() == 5

        store.clear()
        assert store.count() == 0

    def test_batch_add(self, store):
        """Test batch adding documents."""
        store.add_batch(
            ids=["doc1", "doc2", "doc3"],
            embeddings=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],
            metadatas=[{"i": 1}, {"i": 2}, {"i": 3}],
            texts=["Text 1", "Text 2", "Text 3"],
        )

        assert store.count() == 3


class TestQdrantStore:
    """Tests for Qdrant vector store."""

    @pytest.fixture
    def store(self):
        from litforge.stores.qdrant import QdrantStore

        return QdrantStore(in_memory=True, collection_name="test")

    def test_add_and_search(self, store):
        """Test adding and searching documents."""
        store.add(
            id="doc1",
            embedding=[0.1] * 1536,
            metadata={"title": "Test Doc"},
            text="Test content",
        )

        results = store.search([0.1] * 1536, limit=1)

        assert len(results) == 1
        assert results[0]["text"] == "Test content"
        assert results[0]["metadata"]["title"] == "Test Doc"
        assert results[0]["score"] > 0.99  # Should be very high for same vector

    def test_delete(self, store):
        """Test deleting documents."""
        store.add(
            id="doc1",
            embedding=[0.1] * 1536,
            metadata={},
            text="Test",
        )
        assert store.count() == 1

        store.delete("doc1")
        assert store.count() == 0

    def test_clear(self, store):
        """Test clearing all documents."""
        for i in range(5):
            store.add(
                id=f"doc{i}",
                embedding=[0.1] * 1536,
                metadata={},
                text=f"Test {i}",
            )
        assert store.count() == 5

        store.clear()
        assert store.count() == 0


class TestFAISSStore:
    """Tests for FAISS vector store."""

    @pytest.fixture
    def store(self, tmp_path):
        from litforge.stores.faiss import FAISSStore

        return FAISSStore(persist_dir=tmp_path)

    def test_add_and_search(self, store):
        """Test adding and searching documents."""
        store.add(
            id="doc1",
            embedding=[0.1] * 1536,
            metadata={"title": "Test Doc"},
            text="Test content",
        )

        results = store.search([0.1] * 1536, limit=1)

        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["text"] == "Test content"
        assert results[0]["metadata"]["title"] == "Test Doc"
        assert results[0]["score"] > 0.99  # Normalized vectors, should be ~1.0

    def test_count(self, store):
        """Test document count."""
        assert store.count() == 0

        store.add(
            id="doc1",
            embedding=[0.1] * 1536,
            metadata={},
            text="Test",
        )
        assert store.count() == 1

    def test_clear(self, store):
        """Test clearing all documents."""
        for i in range(5):
            store.add(
                id=f"doc{i}",
                embedding=[0.1] * 1536,
                metadata={},
                text=f"Test {i}",
            )
        assert store.count() == 5

        store.clear()
        assert store.count() == 0

    def test_persistence(self, tmp_path):
        """Test that data persists across instances."""
        from litforge.stores.faiss import FAISSStore

        # Create and add
        store1 = FAISSStore(persist_dir=tmp_path)
        store1.add(
            id="doc1",
            embedding=[0.1] * 1536,
            metadata={"key": "value"},
            text="Persistent text",
        )

        # Create new instance, should load data
        store2 = FAISSStore(persist_dir=tmp_path)
        assert store2.count() == 1

        results = store2.search([0.1] * 1536, limit=1)
        assert results[0]["text"] == "Persistent text"
