"""
Local Embedding Index for Scientific Papers.

This module provides a local FAISS-based index for fast similarity search
using SPECTER2 embeddings from Semantic Scholar or computed locally.

Features:
- Fast approximate nearest neighbor search (FAISS)
- Persistent storage (save/load index)
- Incremental updates (add new papers)
- Hybrid search with Semantic Scholar API

Storage:
- Index file: ~/.litforge/embeddings/index.faiss (~50GB for full S2AG)
- Metadata: ~/.litforge/embeddings/metadata.json
- Small index: ~/.litforge/embeddings/small_index.faiss (~3GB for arXiv subset)

Dependencies:
- faiss-cpu (or faiss-gpu)
- numpy
- datasets (optional, for HuggingFace datasets)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy import ndarray
else:
    try:
        import numpy as np
        ndarray = np.ndarray
    except ImportError:
        np = None  # type: ignore
        ndarray = Any  # type: ignore

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_INDEX_DIR = Path.home() / ".litforge" / "embeddings"
EMBEDDING_DIM = 768  # SPECTER2 dimension


@dataclass
class IndexedPaper:
    """Minimal paper metadata for index."""
    paper_id: str
    title: str
    year: int | None = None
    doi: str | None = None
    citation_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "doi": self.doi,
            "citation_count": self.citation_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "IndexedPaper":
        return cls(
            paper_id=data["paper_id"],
            title=data["title"],
            year=data.get("year"),
            doi=data.get("doi"),
            citation_count=data.get("citation_count", 0),
        )


@dataclass
class EmbeddingIndex:
    """
    FAISS-based embedding index for fast similarity search.
    
    Uses IVF (Inverted File) index for scalability:
    - ~10M papers: ~10GB index, <100ms query time
    - ~200M papers: ~50GB index, <200ms query time
    """
    
    index_dir: Path = field(default_factory=lambda: DEFAULT_INDEX_DIR)
    index_name: str = "default"
    embedding_dim: int = EMBEDDING_DIM
    
    # Internal state
    _index: Any = None
    _metadata: list[IndexedPaper] = field(default_factory=list)
    _id_to_idx: dict[str, int] = field(default_factory=dict)
    _is_trained: bool = False
    
    def __post_init__(self):
        self.index_dir = Path(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def index_path(self) -> Path:
        return self.index_dir / f"{self.index_name}.faiss"
    
    @property
    def metadata_path(self) -> Path:
        return self.index_dir / f"{self.index_name}_metadata.json"
    
    @property
    def size(self) -> int:
        """Number of papers in the index."""
        return len(self._metadata)
    
    def _ensure_faiss(self):
        """Ensure FAISS is imported."""
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with:\n"
                "  pip install faiss-cpu\n"
                "Or for GPU support:\n"
                "  pip install faiss-gpu"
            )
    
    def create_index(self, nlist: int = 100):
        """
        Create a new empty index.
        
        Args:
            nlist: Number of clusters for IVF index (more = faster but less accurate)
        """
        faiss = self._ensure_faiss()
        
        # Use IVF + Flat for good balance of speed and accuracy
        # For very large indices, consider IVF + PQ
        quantizer = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        self._index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        self._is_trained = False
        self._metadata = []
        self._id_to_idx = {}
        
        logger.info(f"Created new IVF index with {nlist} clusters")
    
    def train(self, embeddings: np.ndarray):
        """
        Train the index on a sample of embeddings.
        
        For IVF index, this learns the cluster centroids.
        Should be called with a representative sample before adding vectors.
        
        Args:
            embeddings: Training embeddings, shape (n_samples, embedding_dim)
        """
        if self._index is None:
            self.create_index()
        
        faiss = self._ensure_faiss()
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        logger.info(f"Training index on {len(embeddings)} vectors...")
        self._index.train(embeddings)
        self._is_trained = True
        logger.info("Index training complete")
    
    def add(
        self,
        papers: list[IndexedPaper],
        embeddings: np.ndarray,
        batch_size: int = 10000,
    ):
        """
        Add papers and their embeddings to the index.
        
        Args:
            papers: List of paper metadata
            embeddings: Embeddings array, shape (n_papers, embedding_dim)
            batch_size: Batch size for adding (memory efficiency)
        """
        if self._index is None:
            self.create_index()
        
        if not self._is_trained:
            # Auto-train on first batch
            logger.info("Auto-training index on first batch...")
            self.train(embeddings[:min(10000, len(embeddings))])
        
        faiss = self._ensure_faiss()
        
        # Normalize for cosine similarity
        embeddings = embeddings.copy()
        faiss.normalize_L2(embeddings)
        
        # Add in batches
        start_idx = len(self._metadata)
        for i in range(0, len(papers), batch_size):
            batch_papers = papers[i:i + batch_size]
            batch_embeds = embeddings[i:i + batch_size]
            
            self._index.add(batch_embeds)
            
            for j, paper in enumerate(batch_papers):
                idx = start_idx + i + j
                self._metadata.append(paper)
                self._id_to_idx[paper.paper_id] = idx
        
        logger.info(f"Added {len(papers)} papers to index. Total: {self.size}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        nprobe: int = 10,
    ) -> list[tuple[IndexedPaper, float]]:
        """
        Search for similar papers.
        
        Args:
            query_embedding: Query embedding, shape (embedding_dim,) or (1, embedding_dim)
            k: Number of results
            nprobe: Number of clusters to search (more = slower but more accurate)
            
        Returns:
            List of (paper, similarity_score) tuples
        """
        if self._index is None or self.size == 0:
            return []
        
        faiss = self._ensure_faiss()
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize
        query_embedding = query_embedding.copy()
        faiss.normalize_L2(query_embedding)
        
        # Set search parameters
        self._index.nprobe = nprobe
        
        # Search
        distances, indices = self._index.search(query_embedding, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self._metadata):
                paper = self._metadata[idx]
                # Convert inner product to similarity score (0-1)
                similarity = float(max(0, min(1, (dist + 1) / 2)))
                results.append((paper, similarity))
        
        return results
    
    def search_multi(
        self,
        query_embeddings: np.ndarray,
        k: int = 20,
        nprobe: int = 10,
    ) -> list[list[tuple[IndexedPaper, float]]]:
        """
        Search for similar papers with multiple queries.
        
        Args:
            query_embeddings: Query embeddings, shape (n_queries, embedding_dim)
            k: Number of results per query
            nprobe: Number of clusters to search
            
        Returns:
            List of result lists, one per query
        """
        if self._index is None or self.size == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        faiss = self._ensure_faiss()
        
        # Normalize
        query_embeddings = query_embeddings.copy()
        faiss.normalize_L2(query_embeddings)
        
        self._index.nprobe = nprobe
        distances, indices = self._index.search(query_embeddings, k)
        
        all_results = []
        for q in range(len(query_embeddings)):
            results = []
            for dist, idx in zip(distances[q], indices[q]):
                if idx >= 0 and idx < len(self._metadata):
                    paper = self._metadata[idx]
                    similarity = float(max(0, min(1, (dist + 1) / 2)))
                    results.append((paper, similarity))
            all_results.append(results)
        
        return all_results
    
    def save(self):
        """Save index and metadata to disk."""
        if self._index is None:
            logger.warning("No index to save")
            return
        
        faiss = self._ensure_faiss()
        
        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))
        logger.info(f"Saved FAISS index to {self.index_path}")
        
        # Save metadata
        metadata_json = {
            "embedding_dim": self.embedding_dim,
            "size": self.size,
            "is_trained": self._is_trained,
            "papers": [p.to_dict() for p in self._metadata],
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata_json, f)
        logger.info(f"Saved metadata to {self.metadata_path}")
    
    def load(self) -> bool:
        """
        Load index and metadata from disk.
        
        Returns:
            True if loaded successfully
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.info(f"No existing index at {self.index_path}")
            return False
        
        faiss = self._ensure_faiss()
        
        try:
            # Load FAISS index
            self._index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index from {self.index_path}")
            
            # Load metadata
            with open(self.metadata_path) as f:
                metadata_json = json.load(f)
            
            self.embedding_dim = metadata_json.get("embedding_dim", EMBEDDING_DIM)
            self._is_trained = metadata_json.get("is_trained", True)
            self._metadata = [
                IndexedPaper.from_dict(p) for p in metadata_json.get("papers", [])
            ]
            self._id_to_idx = {p.paper_id: i for i, p in enumerate(self._metadata)}
            
            logger.info(f"Loaded {self.size} papers from metadata")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        stats = {
            "size": self.size,
            "embedding_dim": self.embedding_dim,
            "is_trained": self._is_trained,
            "index_path": str(self.index_path),
            "index_exists": self.index_path.exists(),
        }
        
        if self.index_path.exists():
            stats["index_size_mb"] = self.index_path.stat().st_size / (1024 * 1024)
        
        if self._metadata:
            years = [p.year for p in self._metadata if p.year]
            if years:
                stats["year_range"] = (min(years), max(years))
        
        return stats


class EmbeddingIndexManager:
    """
    Manager for multiple embedding indices.
    
    Supports:
    - Multiple named indices (e.g., "arxiv", "s2ag", "custom")
    - Automatic index selection based on domain
    - Index downloading from HuggingFace
    """
    
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._indices: dict[str, EmbeddingIndex] = {}
    
    def get_index(self, name: str = "default") -> EmbeddingIndex:
        """Get or create an index by name."""
        if name not in self._indices:
            index = EmbeddingIndex(index_dir=self.index_dir, index_name=name)
            if not index.load():
                index.create_index()
            self._indices[name] = index
        return self._indices[name]
    
    def list_indices(self) -> list[dict]:
        """List all available indices."""
        indices = []
        for path in self.index_dir.glob("*.faiss"):
            name = path.stem
            index = self.get_index(name)
            indices.append({
                "name": name,
                **index.get_stats(),
            })
        return indices
    
    def download_index(
        self,
        source: str = "huggingface",
        dataset: str = "sproos/arxiv-embeddings",
        name: str = "arxiv",
    ) -> EmbeddingIndex:
        """
        Download and build index from a dataset.
        
        Args:
            source: "huggingface" or "s2ag"
            dataset: Dataset identifier
            name: Name for the local index
            
        Returns:
            Built embedding index
        """
        if source == "huggingface":
            return self._download_from_huggingface(dataset, name)
        elif source == "s2ag":
            return self._download_from_s2ag(name)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _download_from_huggingface(self, dataset: str, name: str) -> EmbeddingIndex:
        """Download embeddings from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        logger.info(f"Downloading {dataset} from HuggingFace...")
        
        # Load dataset
        ds = load_dataset(dataset, split="train")
        
        # Create index
        index = EmbeddingIndex(index_dir=self.index_dir, index_name=name)
        index.create_index(nlist=min(100, len(ds) // 100))
        
        # Process in batches
        batch_size = 10000
        papers = []
        embeddings_list = []
        
        for i, item in enumerate(ds):
            # Extract fields (format varies by dataset)
            paper_id = item.get("id", item.get("paper_id", str(i)))
            title = item.get("title", "Unknown")
            embedding = item.get("embedding", item.get("embeddings", []))
            
            if isinstance(embedding, list) and len(embedding) == EMBEDDING_DIM:
                papers.append(IndexedPaper(
                    paper_id=paper_id,
                    title=title,
                    year=item.get("year"),
                    doi=item.get("doi"),
                ))
                embeddings_list.append(embedding)
            
            if len(papers) >= batch_size:
                embeddings_array = np.array(embeddings_list, dtype=np.float32)
                index.add(papers, embeddings_array)
                papers = []
                embeddings_list = []
                logger.info(f"Processed {i + 1} papers...")
        
        # Add remaining
        if papers:
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            index.add(papers, embeddings_array)
        
        # Save
        index.save()
        self._indices[name] = index
        
        logger.info(f"Built index '{name}' with {index.size} papers")
        return index
    
    def _download_from_s2ag(self, name: str) -> EmbeddingIndex:
        """Download embeddings from Semantic Scholar Academic Graph."""
        import requests
        import gzip
        import tempfile
        
        logger.info("Fetching S2AG release info...")
        
        # Get latest release
        releases = requests.get(
            "https://api.semanticscholar.org/datasets/v1/release"
        ).json()
        latest = releases[-1]
        
        # Get embeddings dataset info
        embeddings_info = requests.get(
            f"https://api.semanticscholar.org/datasets/v1/release/{latest}/dataset/embeddings"
        ).json()
        
        file_urls = embeddings_info.get("files", [])
        logger.info(f"Found {len(file_urls)} embedding files in release {latest}")
        
        # Create index
        index = EmbeddingIndex(index_dir=self.index_dir, index_name=name)
        index.create_index(nlist=1000)  # More clusters for large index
        
        # Download and process each file
        for i, url in enumerate(file_urls):
            logger.info(f"Downloading file {i + 1}/{len(file_urls)}...")
            
            response = requests.get(url, stream=True)
            
            with tempfile.NamedTemporaryFile(suffix=".gz") as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp.flush()
                
                # Process gzipped JSONL
                papers = []
                embeddings_list = []
                
                with gzip.open(tmp.name, "rt") as f:
                    for line in f:
                        item = json.loads(line)
                        paper_id = item.get("corpusid", item.get("paperId"))
                        embedding = item.get("vector", item.get("embedding"))
                        
                        if paper_id and embedding:
                            papers.append(IndexedPaper(
                                paper_id=str(paper_id),
                                title=item.get("title", "Unknown"),
                            ))
                            embeddings_list.append(embedding)
                
                if papers:
                    embeddings_array = np.array(embeddings_list, dtype=np.float32)
                    index.add(papers, embeddings_array)
                
                logger.info(f"Processed {len(papers)} papers from file {i + 1}")
        
        # Save
        index.save()
        self._indices[name] = index
        
        logger.info(f"Built S2AG index '{name}' with {index.size} papers")
        return index


# Convenience function
def get_embedding_index(name: str = "default") -> EmbeddingIndex:
    """Get or create an embedding index."""
    manager = EmbeddingIndexManager()
    return manager.get_index(name)
