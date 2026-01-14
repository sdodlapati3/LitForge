"""
Base embedding provider interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        pass
    
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]
