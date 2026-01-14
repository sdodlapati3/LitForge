"""
Base vector store interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        text: str,
    ) -> None:
        """
        Add a document to the store.
        
        Args:
            id: Unique document ID
            embedding: Vector embedding
            metadata: Document metadata
            text: Document text
        """
        pass
    
    @abstractmethod
    def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            embedding: Query embedding
            limit: Maximum results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of results with id, text, metadata, score
        """
        pass
    
    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a document by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get document count."""
        pass
