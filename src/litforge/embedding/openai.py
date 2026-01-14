"""
OpenAI embedding provider.

Uses OpenAI's text-embedding models (cloud-first approach).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from litforge.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding provider.
    
    Uses text-embedding-3-small by default (1536 dimensions, good balance
    of quality and cost).
    """
    
    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client: Any = None
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
    
    @property
    def client(self) -> Any:
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI not installed. Install with: pip install openai"
                )
        return self._client
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.MODEL_DIMENSIONS.get(self.model, 1536)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        if not texts:
            return []
        
        # Clean texts (OpenAI has limits)
        cleaned = [self._clean_text(t) for t in texts]
        
        # Batch if needed (OpenAI recommends < 8191 tokens per text)
        embeddings = []
        batch_size = 100  # OpenAI recommendation
        
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            
            # Extract embeddings in order
            batch_embeddings = [None] * len(batch)
            for item in response.data:
                batch_embeddings[item.index] = item.embedding
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding."""
        # Truncate if too long (rough estimate: 4 chars per token)
        max_chars = 8000 * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # Replace newlines with spaces
        text = text.replace("\n", " ")
        
        return text.strip()
