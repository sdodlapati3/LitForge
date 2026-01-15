# Pre-Built Scientific Paper Embedding Indices

## Research Summary (January 2026)

This document catalogs available pre-built embedding indices and APIs for scientific papers that can be used with LitForge for enhanced search capabilities.

---

## 🏆 Recommended: Semantic Scholar API

**Best for**: Production use, no local storage needed

| Feature | Details |
|---------|---------|
| **Papers** | 214 million |
| **Embeddings** | SPECTER2 (768-dim) available via API |
| **Cost** | Free (1 RPS default, more with API key) |
| **Freshness** | Updated weekly |

### API Usage
```python
# Get paper with embedding
GET https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=embedding

# Search with embedding returned
GET https://api.semanticscholar.org/graph/v1/paper/search?query=...&fields=embedding
```

### Bulk Download
```python
# List available releases
GET https://api.semanticscholar.org/datasets/v1/release

# Download embeddings dataset
GET https://api.semanticscholar.org/datasets/v1/release/latest/dataset/embeddings
```

---

## 🔧 SPECTER2 Model (for custom embedding)

**Best for**: Building your own index

| Feature | Details |
|---------|---------|
| **Source** | AllenAI |
| **Dimensions** | 768 |
| **HuggingFace** | `allenai/specter2_base` + adapters |
| **Tasks** | Retrieval, Classification, Regression, Adhoc Query |

### Usage
```python
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

# For search queries
model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="query", set_active=True)
```

---

## 📦 Pre-Built Datasets on HuggingFace

### 1. sproos/arxiv-embeddings
| Feature | Details |
|---------|---------|
| **Size** | 3.78 GB |
| **Papers** | 296K |
| **License** | Apache-2.0 |
| **URL** | https://huggingface.co/datasets/sproos/arxiv-embeddings |

### 2. CShorten/ArXiv-ML-Abstract-Embeddings
| Feature | Details |
|---------|---------|
| **Size** | ~1 GB |
| **Papers** | 118K (ML-focused) |
| **URL** | https://huggingface.co/datasets/CShorten/ArXiv-ML-Abstract-Embeddings |

### 3. Tychos/arxiv-embeddings (larger)
| Feature | Details |
|---------|---------|
| **Papers** | 1.19M |
| **URL** | https://huggingface.co/datasets/Tychos/arxiv-embeddings |

### 4. allenai/scirepeval (training data)
| Feature | Details |
|---------|---------|
| **Size** | 27.4 GB |
| **Records** | 12.3M |
| **URL** | https://huggingface.co/datasets/allenai/scirepeval |

---

## 📊 Semantic Scholar Bulk Datasets (S2AG)

**Best for**: Building offline index with embeddings

### Available Datasets
| Dataset | Description |
|---------|-------------|
| `papers` | Core paper metadata |
| `abstracts` | Paper abstract text (~100M records) |
| `embeddings` | SPECTER2 embeddings |
| `citations` | Citation graph |
| `authors` | Author metadata |

### Download Process
```python
import requests

# Get latest release
releases = requests.get('https://api.semanticscholar.org/datasets/v1/release').json()
latest = releases[-1]  # e.g., "2026-01-14"

# Get embeddings dataset info
embeddings = requests.get(
    f'https://api.semanticscholar.org/datasets/v1/release/{latest}/dataset/embeddings'
).json()

# Download files (usually 10-30 partitions)
for url in embeddings['files']:
    # Download each partition...
```

---

## 🚀 Implementation Status (LitForge)

### ✅ Phase 1: Live API (Complete)
- Scout search with OpenAlex
- RAG-augmented LLM expansion
- Multi-API search (OpenAlex + Semantic Scholar + arXiv)

### ✅ Phase 2: Semantic Scholar Embeddings API (Complete)
```python
from litforge.clients.semantic_scholar import SemanticScholarClient

client = SemanticScholarClient()

# Get single embedding
embedding = client.get_embedding("204e3073870fae3d05bcbc2f6a8e263d9b72e776")

# Batch embeddings
embeddings = client.get_embeddings_batch(["paper_id_1", "paper_id_2"])

# Search with embeddings included
papers = client.search("neural networks", limit=10, include_embedding=True)
# Access via paper._embedding
```

### ✅ Phase 3: Local FAISS Index (Complete)
```python
from litforge.stores.embedding_index import EmbeddingIndex, EmbeddingIndexManager

# Create new index
index = EmbeddingIndex(name="my_papers", dimension=768)

# Add papers
index.add(paper_ids, embeddings, metadata)

# Search
results = index.search(query_embedding, k=10)

# Save/Load
index.save("~/.litforge/embeddings/my_papers")
index.load("~/.litforge/embeddings/my_papers")

# Download pre-built index
manager = EmbeddingIndexManager()
manager.download_index(source="huggingface", dataset="Tychos/arxiv-embeddings", name="arxiv")
```

### ✅ Phase 4: Hybrid Search (Complete)
```python
from litforge.services.hybrid_search import HybridSearchService, hybrid_search

# Quick usage
papers = hybrid_search("liquid neural networks", k=25, index_name="arxiv")

# Full control
service = HybridSearchService(
    index_name="arxiv",
    use_semantic_scholar=True,
    use_openalex=True,
)
results = service.search("liquid neural networks", k=25, local_k=50, api_limit=30)

# Architecture:
# User Query → Local Index (instant, millions) 
#            → Semantic Scholar API (fresh, metadata)
#            → OpenAlex API (free, comprehensive)
#            → Merge + Deduplicate → Ensemble Score → Final results
```

### Build Index CLI
```bash
# Install dependencies
pip install "litforge[embedding-index]"

# List available datasets
python scripts/build_index.py --list

# Download and build from HuggingFace
python scripts/build_index.py --source huggingface --dataset Tychos/arxiv-embeddings --name arxiv

# Download from Semantic Scholar (larger, ~50GB)
python scripts/build_index.py --source s2ag --name s2_full

# Test the index
python scripts/build_index.py --test --name arxiv --query "transformer attention mechanisms"
```

---

## 📚 References

1. **SPECTER Paper**: https://arxiv.org/pdf/2004.07180.pdf
2. **SPECTER2/SciRepEval**: https://api.semanticscholar.org/CorpusID:254018137
3. **Semantic Scholar API Docs**: https://api.semanticscholar.org/api-docs/
4. **S2AG Datasets API**: https://api.semanticscholar.org/api-docs/datasets

---

*Last updated: January 2026*
