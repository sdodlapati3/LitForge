"""
Local embedding provider using sentence-transformers.

For offline/local usage without API costs.
"""

from __future__ import annotations

import logging
from typing import Any

from litforge.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


class LocalEmbedder(BaseEmbedder):
    """
    Local embedding provider using sentence-transformers.
    
    Uses all-MiniLM-L6-v2 by default (384 dimensions, fast and good quality).
    Requires: pip install sentence-transformers
    """
    
    # Model dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "multi-qa-mpnet-base-dot-v1": 768,
        "allenai/specter2": 768,
    }
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        """
        Initialize local embedder.
        
        Args:
            model: Model name from HuggingFace
            device: Device to use (cuda, mps, cpu) - auto-detected if None
        """
        self.model_name = model
        self.device = device
        self._model: Any = None
    
    @property
    def model(self) -> Any:
        """Lazy-load sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                logger.info(f"Loaded model {self.model_name} on {self._model.device}")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        # Check known models first
        if self.model_name in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.model_name]
        
        # Fall back to getting from model
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        if not texts:
            return []
        
        # sentence-transformers handles batching internally
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        
        return embeddings.tolist()
