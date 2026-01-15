# 🔥 LitForge

**Forging Knowledge from Literature**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An open-source Python library for unified scientific literature search, retrieval, and knowledge synthesis. Built for researchers, AI agents, and knowledge systems.

## ✨ Features

- **🔍 Unified Search** - Query 250M+ papers across OpenAlex, Semantic Scholar, PubMed, arXiv, and more with a single API
- **🧠 Semantic Search** - LLM-powered query understanding with embedding-based retrieval (like Elicit/Consensus)
- **📊 Citation Networks** - Build and analyze citation graphs, find key papers, discover research clusters
- **🔄 Multi-Provider LLM** - Free-tier cascade through Cerebras → Groq → Google → OpenAI
- **📈 Ensemble Scoring** - Smart ranking with keyword, embedding, citation, and recency signals
- **💬 Chat Interface** - Streamlit-based conversational literature search
- **🤖 AI-Ready** - First-class support for MCP (Claude), CrewAI, LangGraph, and custom agents
- **🔌 Pluggable** - Swap vector stores, LLMs, and data sources without code changes

## 🚀 Quick Start

### Installation

```bash
# Clone and install in development mode
git clone https://github.com/sdodlapati3/LitForge.git
cd LitForge
pip install -e ".[dev]"

# Set up API keys in .env file
cp .env.example .env
# Edit .env with your API keys (Cerebras, Groq, Google are FREE)
```

### Environment Variables

```bash
# Free LLM providers (recommended)
CEREBRAS_API_KEY=csk-...     # 14,400 req/day FREE (best free tier)
GROQ_API_KEY=gsk_...         # 1,000+ req/day FREE (fastest)
GOOGLE_API_KEY=AIza...       # 250 req/day FREE (Gemini)

# Paid fallback
OPENAI_API_KEY=sk-...        # Paid, but most reliable
```

### Basic Usage

```python
from litforge.api import search, lookup, citations

# Search for papers (uses LLM-powered semantic search)
papers = search("liquid foundation models")  # Understands → Liquid Neural Networks

# Look up by DOI
paper = lookup("10.1609/aaai.v35i9.16936")

# Get citing papers
citing = citations("10.1609/aaai.v35i9.16936")
```

### Chat Interface

```bash
# Start the web UI
./scripts/start_ui.sh chat

# Access at http://localhost:8503
```

**Try these commands:**
- `Find papers on CRISPR gene editing`
- `citation network for Liquid Time-constant Networks`
- `Download first 10 as CSV`
- `Export as BibTeX`

### Citation Networks

```python
from litforge.api import search
from litforge.services.citation import CitationService

# Find a seed paper
papers = search("Liquid Time-constant Networks", limit=5)
seed = papers[0]

# Build citation network
citation_service = CitationService()
network = citation_service.build_network(
    seed_papers=[seed],
    depth=2,           # How many citation levels to traverse
    max_papers=100     # Limit total papers
)

# Network stats
print(f"Papers: {network.num_nodes}, Citations: {network.num_edges}")

# Find influential papers (by PageRank)
key_papers = citation_service.find_key_papers(network, metric='pagerank', limit=10)

# Export for visualization
network.export("network.graphml")  # For Gephi, Cytoscape
network.export("network.json")     # For web visualization
```

### Semantic Search (LLM-First)

```python
from litforge.services.semantic_search import semantic_search
from litforge.llm.router import get_llm

# Get LLM (auto-selects from available providers)
llm = get_llm()

# Search with natural language - LLM understands intent
papers, metadata = semantic_search(
    query="papers about liquid foundation models",  # → Liquid Neural Networks
    llm=llm,
    max_results=25,
    use_recommendations=True,  # Use SPECTER2 embeddings
)

# See what LLM understood
print(metadata["understanding"]["explanation"])
# "The user is asking about Liquid Neural Networks, a type of continuous-depth neural network..."
```

### With AI Agents (MCP / Claude)

LitForge includes an MCP server for use with Claude Desktop:

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "litforge": {
      "command": "python",
      "args": ["-m", "litforge.mcp"],
      "env": {
        "CEREBRAS_API_KEY": "csk-...",
        "GROQ_API_KEY": "gsk_..."
      }
    }
  }
}
```

Then in Claude:
```
Search for papers on "transformer attention mechanisms" and summarize the key findings
```

## 📦 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              LitForge                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        Interface Layer                          ││
│  │   Python API  │  MCP Server  │  Chat UI  │  CLI                 ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                  ↓                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        Service Layer                            ││
│  │  Discovery │ Semantic Search │ RAG Search │ Citation │ Scoring  ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                  ↓                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        Client Layer                             ││
│  │  OpenAlex  │  Semantic Scholar  │  PubMed  │  arXiv  │  ...     ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                  ↓                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                         LLM Router                              ││
│  │  Cerebras (70B) → Groq (70B) → Google (Gemini) → OpenAI         ││
│  │  (Free, fastest)   (Free)      (Free)            (Paid fallback)││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

LitForge uses a `.env` file for configuration:

```bash
# .env file in project root

# Free LLM providers (recommended - no cost!)
CEREBRAS_API_KEY=csk-...     # 14,400 req/day FREE - https://cloud.cerebras.ai
GROQ_API_KEY=gsk_...         # 1,000+ req/day FREE - https://console.groq.com  
GOOGLE_API_KEY=AIza...       # 250 req/day FREE - https://aistudio.google.com

# Paid fallback (optional)
OPENAI_API_KEY=sk-...        # Paid - https://platform.openai.com

# Data source API keys (optional, increases rate limits)
SEMANTIC_SCHOLAR_API_KEY=... # Optional - https://www.semanticscholar.org/product/api
```

## 🔌 Key Components

### LLM Router (Free-Tier Cascade)

LitForge automatically routes through free LLM providers:

```python
from litforge.llm.router import get_llm

# Auto-selects best available provider
llm = get_llm()  # Cerebras → Groq → Google → OpenAI

# Use for query understanding, verification, etc.
response = llm.complete("Explain CRISPR in one sentence")
```

### Ensemble Scoring

Smart ranking combining multiple signals:

```python
from litforge.services.scoring import EnsembleScorer

scorer = EnsembleScorer()
scored_papers = scorer.score(
    papers=papers,
    query="transformer attention",
    weights={
        "keyword": 0.3,      # BM25-style matching
        "embedding": 0.3,    # SPECTER2 similarity
        "citation": 0.25,    # Citation count
        "recency": 0.15,     # Publication date
    }
)
```

## 📊 Supported Data Sources

| Source | Papers | Citations | Full-Text | Free |
|--------|--------|-----------|-----------|------|
| **OpenAlex** | 250M+ | ✅ | Abstracts | ✅ |
| **Semantic Scholar** | 214M+ | ✅ | Abstracts | ✅ |
| **PubMed** | 36M+ | ✅ | Abstracts | ✅ |
| **arXiv** | 2.4M+ | ❌ | ✅ PDFs | ✅ |
| **Unpaywall** | 50M+ OA | ❌ | ✅ PDFs | ✅ |
| **PubMed Central** | 9M+ | ✅ | ✅ Full | ✅ |
| **Crossref** | 150M+ | ✅ | Metadata | ✅ |
| **Europe PMC** | 45M+ | ✅ | ✅ Full | ✅ |

## 🛠️ Development

```bash
# Clone the repository
git clone https://github.com/sdodlapati3/LitForge.git
cd LitForge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src tests
mypy src

# Pre-commit hooks
pre-commit install
```

## 📚 Examples

See the [examples/](examples/) directory for complete examples:

- [basic_search.py](examples/basic_search.py) - Simple paper search
- [citation_network.py](examples/citation_network.py) - Build citation graphs
- [knowledge_base.py](examples/knowledge_base.py) - RAG-powered Q&A
- [crewai_agent.py](examples/crewai_agent.py) - CrewAI integration
- [mcp_tools.py](examples/mcp_tools.py) - MCP server for Claude

## 🤝 Integration with Other Projects

LitForge is designed to work seamlessly with:

- **[ChemAgent](https://github.com/sdodlapati3/ChemAgent)** - Pharmaceutical research assistant
- **[OmicsOracle](https://github.com/sdodlapati3/OmicsOracle)** - Genomics analysis platform
- **[BioPipelines](https://github.com/sdodlapati3/BioPipelines)** - Bioinformatics workflows

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

LitForge builds upon excellent open-source projects:

- [pyalex](https://github.com/J535D165/pyalex) - OpenAlex Python client
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - Local embeddings
- [MCP](https://github.com/anthropics/mcp) - Model Context Protocol

---

<p align="center">
  <b>🔥 LitForge</b> - Forging Knowledge from Literature
</p>
