# 🔥 LitForge

**Forging Knowledge from Literature**

[![PyPI version](https://badge.fury.io/py/litforge.svg)](https://badge.fury.io/py/litforge)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An open-source Python library for unified scientific literature search, retrieval, and knowledge synthesis. Built for researchers, AI agents, and knowledge systems.

## ✨ Features

- **🔍 Unified Search** - Query 250M+ papers across OpenAlex, Semantic Scholar, PubMed, arXiv, and more with a single API
- **📥 Smart Retrieval** - Download full-text PDFs from Unpaywall, PMC, arXiv, and other open access sources
- **📊 Citation Networks** - Build and analyze citation graphs, find key papers, discover research clusters
- **🧠 Knowledge Base** - Index papers with semantic embeddings for RAG-powered Q&A
- **🤖 AI-Ready** - First-class support for MCP (Claude), CrewAI, LangGraph, and custom agents
- **🔌 Pluggable** - Swap vector stores, LLMs, and data sources without code changes

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install litforge

# With local embeddings (no API keys needed)
pip install litforge[local]

# Full installation with all features
pip install litforge[all]
```

### Basic Usage

```python
from litforge import Forge

# Initialize LitForge
forge = Forge()

# Search for papers
papers = forge.search("CRISPR gene editing mechanisms", limit=50)

# Download full text (automatically finds open access versions)
for paper in papers[:5]:
    forge.retrieve(paper)

# Build a knowledge base
forge.index(papers)

# Ask questions
answer = forge.ask("What are the main mechanisms of CRISPR-Cas9?")
print(answer.text)
print(f"Sources: {answer.citations}")
```

### Citation Networks

```python
from litforge import Forge

forge = Forge()

# Get a seed paper
paper = forge.lookup(doi="10.1126/science.aad5227")  # Original CRISPR paper

# Build citation network
network = forge.build_network(
    seeds=[paper],
    depth=2,           # How many citation levels to traverse
    max_papers=500     # Limit total papers
)

# Find influential papers
influential = network.most_cited(n=10)

# Find research clusters
clusters = network.find_clusters()
for cluster in clusters:
    print(f"Cluster: {cluster.label}, Papers: {len(cluster.papers)}")

# Export for visualization
network.export("crispr_network.graphml")  # For Gephi, Cytoscape
```

### With AI Agents (MCP / Claude)

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "litforge": {
      "command": "python",
      "args": ["-m", "litforge.mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Then in Claude:
```
Search for papers on "transformer attention mechanisms" and summarize the key findings
```

### With CrewAI

```python
from crewai import Agent, Task, Crew
from litforge.integrations.crewai import LitForgeTools

# Get LitForge tools for CrewAI
tools = LitForgeTools()

researcher = Agent(
    role="Literature Researcher",
    goal="Find and synthesize relevant papers",
    tools=tools.all(),  # search, retrieve, cite, ask
)

task = Task(
    description="Research the current state of protein folding prediction",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## 📦 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              LitForge                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        Interface Layer                          ││
│  │   Python SDK  │  MCP Server  │  REST API  │  CLI                ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                  ↓                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        Service Layer                            ││
│  │  Discovery  │  Retrieval  │  Citation  │  Knowledge  │  QA      ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                  ↓                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        Client Layer                             ││
│  │  OpenAlex  │  Semantic Scholar  │  PubMed  │  arXiv  │  ...     ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                  ↓                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                         Data Layer                              ││
│  │  Vector Store  │  Document Store  │  Graph Store  │  Cache      ││
│  │  (ChromaDB)      (SQLite)          (NetworkX)      (DiskCache)  ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

LitForge can be configured via environment variables or a config file:

```bash
# Environment variables
export LITFORGE_OPENAI_API_KEY="sk-..."
export LITFORGE_DEFAULT_SOURCES="openalex,semantic_scholar"
export LITFORGE_CACHE_DIR="~/.litforge/cache"
export LITFORGE_EMBEDDING_MODEL="text-embedding-3-small"
```

Or via `litforge.yaml`:

```yaml
# litforge.yaml
sources:
  default: [openalex, semantic_scholar, pubmed]
  openalex:
    email: your@email.com  # For polite pool (higher rate limits)

embeddings:
  provider: openai  # or "local" for sentence-transformers
  model: text-embedding-3-small

vector_store:
  provider: chroma  # or "qdrant", "faiss"
  persist_dir: ~/.litforge/vectors

llm:
  provider: openai
  model: gpt-4o-mini

cache:
  enabled: true
  dir: ~/.litforge/cache
  ttl: 86400  # 24 hours
```

## 🔌 Pluggable Components

### Vector Stores

```python
# ChromaDB (default, embedded)
forge = Forge(vector_store="chroma")

# Qdrant (cloud or self-hosted)
forge = Forge(vector_store="qdrant", qdrant_url="http://localhost:6333")

# FAISS (high performance)
forge = Forge(vector_store="faiss")
```

### Embedding Providers

```python
# OpenAI (cloud-first, default)
forge = Forge(embeddings="openai")

# Local (no API key needed)
forge = Forge(embeddings="local")  # Uses sentence-transformers

# Custom
from litforge.embeddings import EmbeddingProvider

class MyEmbeddings(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Your implementation
        pass

forge = Forge(embeddings=MyEmbeddings())
```

### LLM Providers

```python
# OpenAI (default)
forge = Forge(llm="openai")

# Anthropic
forge = Forge(llm="anthropic")

# Local (Ollama)
forge = Forge(llm="ollama", ollama_model="llama3.2")
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
