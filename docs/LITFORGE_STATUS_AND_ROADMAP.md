# LitForge: Status Report and Implementation Roadmap

**Version**: 1.0  
**Date**: January 2025  
**Repository**: [github.com/sdodlapati3/LitForge](https://github.com/sdodlapati3/LitForge)

---

## Executive Summary

LitForge is a scientific literature search library designed to be the **literature backbone for AI agents and researchers**. This document provides a comprehensive assessment of the current implementation status, identifies gaps against the original vision, and presents a prioritized roadmap for completing the library.

**Current State**: LitForge v0.1.0 is partially functional. The **Simple API** and **Web UI** work well for basic literature search, but the **core Forge class** and **advanced features** (PDF retrieval, RAG Q&A, citation networks) are skeletal implementations that need significant work.

### Quick Status

| Component | Status | Functionality |
|-----------|--------|---------------|
| Simple API (`litforge.search()`) | ✅ Complete | Production-ready |
| Web UI (Form + Chat) | ✅ Complete | Fully functional |
| MCP Server | ✅ Complete | Claude/AI agent integration |
| Framework Integrations | ✅ Complete | CrewAI, LangChain, LangGraph |
| Vector Stores | ✅ Complete | ChromaDB, FAISS, Qdrant |
| Core Forge Class | ⚠️ Skeleton | Methods exist, don't work |
| PDF/Full-text Retrieval | ❌ Missing | Not implemented |
| RAG Q&A Pipeline | ❌ Missing | Not connected |
| Citation Network Analysis | ❌ Missing | Stub only |

---

## Part 1: Original Vision vs Current Reality

### 1.1 The Original Vision

From the ChemAgent planning documents, LitForge was envisioned as a **4-pipeline system**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Original LitForge Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Pipeline 1: Discovery         Pipeline 2: Retrieval                 │
│  ───────────────────          ────────────────────                   │
│  • OpenAlex (250M+ papers)    • Unpaywall (55M OA)                   │
│  • Semantic Scholar (214M)    • arXiv (2M+ papers)                   │
│  • CrossRef (140M+ DOIs)      • PubMed Central                       │
│  • PubMed (36M biomedical)    • CORE API (200M)                      │
│                                                                      │
│  Pipeline 3: Citation Network  Pipeline 4: Knowledge Base            │
│  ─────────────────────────    ───────────────────────                │
│  • Build citation graphs      • PDF text extraction                  │
│  • Forward/backward refs      • Section detection                    │
│  • Influence analysis         • Vector embeddings                    │
│  • Co-citation clustering     • Conversational Q&A                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Current Reality

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Current Implementation                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Simple API (WORKING)          Web UI (WORKING)                      │
│  ────────────────────          ────────────────                      │
│  • litforge.search()           • Form-based search                   │
│  • litforge.lookup()           • Chat interface                      │
│  • litforge.citations()        • PDF request handling                │
│  • litforge.references()       • Expandable results                  │
│                                                                      │
│  Forge Class (SKELETON)        Services (STUBS)                      │
│  ──────────────────────        ────────────────                      │
│  • Methods exist               • discovery.py (partial)              │
│  • Don't actually work         • retrieval.py (stub)                 │
│  • Not connected to APIs       • knowledge.py (stub)                 │
│  • No RAG pipeline             • qa.py (stub)                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Detailed Implementation Status

### 2.1 ✅ COMPLETE: Simple API (`src/litforge/api.py`)

**Status**: Production-ready, fully functional

The Simple API provides one-liner access to literature search:

```python
import litforge

# Search - WORKING
papers = litforge.search("machine learning drug discovery", limit=20)

# Lookup by DOI - WORKING
paper = litforge.lookup("10.1038/nature12373")

# Get citations - WORKING
citations = litforge.citations("10.1038/nature12373", limit=50)

# Get references - WORKING
references = litforge.references("10.1038/nature12373")
```

**Features**:
- Direct OpenAlex and CrossRef API calls
- Automatic query cleaning
- Relevance scoring
- DOI resolution
- Year filtering
- Clean `Paper` objects

**Test Coverage**: ✅ Manual testing passed

---

### 2.2 ✅ COMPLETE: Web UI (`src/litforge/ui/`)

**Status**: Fully functional

#### Form UI (`app.py` - Port 8502)
```bash
cd src/litforge/ui && streamlit run app.py --server.port 8502
```

- Search form with filters (year, limit)
- Results displayed as expandable cards
- Paper details (title, authors, abstract, DOI, venue)
- Direct links to papers

#### Chat UI (`chat.py` - Port 8503)
```bash
cd src/litforge/ui && streamlit run chat.py --server.port 8503
```

- Conversational interface
- Intent parsing for search/citations/references
- "Show all" expandable results
- PDF request handling (with explanation)
- Multi-turn conversation support

---

### 2.3 ✅ COMPLETE: MCP Server (`src/litforge/mcp/`)

**Status**: Functional, tested with Claude

Tools exposed via MCP:
- `search_papers` - Search scientific literature
- `get_paper_details` - Get full paper metadata
- `get_citations` - Get papers that cite a work
- `get_references` - Get a paper's reference list

**Usage with Claude Desktop**:
```json
{
  "mcpServers": {
    "litforge": {
      "command": "python",
      "args": ["-m", "litforge.mcp.server"]
    }
  }
}
```

---

### 2.4 ✅ COMPLETE: Framework Integrations

| Framework | File | Status |
|-----------|------|--------|
| CrewAI | `integrations/crewai.py` | ✅ LitForgeSearchTool |
| LangChain | `integrations/langchain.py` | ✅ BaseTool implementations |
| LangGraph | `integrations/langgraph.py` | ✅ Node functions |

---

### 2.5 ✅ COMPLETE: Vector Stores (`src/litforge/stores/`)

| Store | File | Status |
|-------|------|--------|
| Base | `base.py` | ✅ Abstract interface |
| ChromaDB | `chromadb.py` | ✅ Persistent storage |
| FAISS | `faiss.py` | ✅ In-memory |
| Qdrant | `qdrant.py` | ✅ Client/server |

---

### 2.6 ⚠️ SKELETON: Core Forge Class (`src/litforge/core/forge.py`)

**Status**: Class structure exists (~550 lines), but methods are not functional

```python
from litforge import Forge

forge = Forge()

# These methods EXIST but DON'T WORK:
forge.search("query")          # Returns placeholder data
forge.get_paper("doi")         # Calls stub service
forge.get_citations("doi")     # Not connected to API
forge.get_references("doi")    # Not connected to API
forge.download_pdf("doi")      # NOT IMPLEMENTED
forge.index_papers([...])      # NOT IMPLEMENTED
forge.ask("question")          # NOT IMPLEMENTED (RAG)
forge.build_network([...])     # NOT IMPLEMENTED
```

**Current Code Issues**:
1. Services are initialized but contain stub implementations
2. No actual API calls in most methods
3. RAG pipeline not connected
4. Vector stores not utilized

---

### 2.7 ⚠️ PARTIAL: Client Libraries (`src/litforge/clients/`)

| Client | File | Status | Notes |
|--------|------|--------|-------|
| Base | `base.py` | ✅ Abstract class | |
| OpenAlex | `openalex.py` | ⚠️ Partial | Structure exists, needs testing |
| CrossRef | `crossref.py` | ⚠️ Partial | Structure exists, needs testing |
| Semantic Scholar | `semantic_scholar.py` | ❌ Stub | Not implemented |
| PubMed | `pubmed.py` | ❌ Stub | Not implemented |
| arXiv | `arxiv.py` | ❌ Stub | Not implemented |
| Unpaywall | `unpaywall.py` | ❌ Stub | Critical for PDFs |

---

### 2.8 ⚠️ STUB: Services (`src/litforge/services/`)

| Service | File | Status | Purpose |
|---------|------|--------|---------|
| Discovery | `discovery.py` | ⚠️ Partial | Multi-source search |
| Retrieval | `retrieval.py` | ❌ Stub | PDF/fulltext download |
| Citation | `citation.py` | ❌ Stub | Network building |
| Knowledge | `knowledge.py` | ❌ Stub | Indexing/RAG |
| QA | `qa.py` | ❌ Stub | Question answering |

---

### 2.9 ❌ MISSING: Critical Features

1. **PDF/Full-text Retrieval**
   - No Unpaywall integration
   - No arXiv PDF download
   - No PubMed Central access
   - No CORE API integration

2. **RAG Q&A Pipeline**
   - Vector stores exist but not connected
   - No document chunking
   - No embedding pipeline
   - `forge.ask()` not functional

3. **Citation Network Analysis**
   - No graph building
   - No influence metrics
   - No visualization
   - NetworkX not integrated

4. **REST API**
   - FastAPI structure missing
   - No REST endpoints
   - No API documentation

5. **CLI Commands**
   - `litforge search` not working
   - `litforge cite` not implemented
   - No progress indicators

---

## Part 3: Architecture Gap Analysis

### 3.1 Planned vs Implemented Data Flow

**PLANNED** (from ChemAgent docs):
```
User Query
    ↓
Intent Parser → Route to appropriate pipeline
    ↓
┌───────────────────────────────────────────────────────┐
│  Discovery: Search OpenAlex + S2 + CrossRef + PubMed  │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│  Retrieval: Get PDFs via Unpaywall + arXiv + CORE     │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│  Knowledge: Extract text, chunk, embed, index         │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│  Q&A: RAG retrieval + LLM synthesis with citations    │
└───────────────────────────────────────────────────────┘
    ↓
Formatted Answer with Sources
```

**ACTUAL** (current implementation):
```
User Query
    ↓
Simple API → Direct OpenAlex/CrossRef call
    ↓
Paper Results (metadata only)
    ↓
[END - No PDF, No RAG, No synthesis]
```

### 3.2 API Coverage Gap

| API Source | Planned | Implemented | Gap |
|------------|---------|-------------|-----|
| OpenAlex | Full integration | Simple search only | Filters, pagination |
| Semantic Scholar | Full with citations | Not implemented | 100% gap |
| CrossRef | Full integration | Basic lookup | Filters, batch |
| PubMed | Full biomedical | Not implemented | 100% gap |
| Unpaywall | OA URL resolution | Not implemented | 100% gap |
| arXiv | PDF download | Not implemented | 100% gap |
| CORE | Fulltext access | Not implemented | 100% gap |

---

## Part 4: Implementation Roadmap

Based on the ChemAgent planning documents and the current gaps, here's the prioritized roadmap:

### Phase 1: Complete Core Discovery (Week 1-2) ⭐ HIGH PRIORITY

**Goal**: Make `Forge.search()` work properly with multi-source search

**Tasks**:
1. **Fix OpenAlex Client** (`clients/openalex.py`)
   - Complete API implementation
   - Add pagination
   - Add filters (year, type, OA status)
   - Test with rate limiting

2. **Implement Semantic Scholar Client** (`clients/semantic_scholar.py`)
   - API wrapper with auth
   - Paper search
   - Citation/reference retrieval
   - Bulk operations

3. **Connect Discovery Service** (`services/discovery.py`)
   - Multi-source aggregation
   - Result deduplication
   - Relevance ranking
   - Source attribution

**Deliverable**: `forge.search()` returns high-quality results from 2+ sources

---

### Phase 2: PDF/Fulltext Retrieval (Week 3-4) ⭐ HIGH PRIORITY

**Goal**: Enable `forge.get_fulltext()` and `forge.download_pdf()`

**Tasks**:
1. **Implement Unpaywall Client** (`clients/unpaywall.py`)
   - OA URL resolution
   - PDF link extraction
   - License detection
   - Batch lookups

2. **Implement arXiv Client** (`clients/arxiv.py`)
   - Paper search
   - PDF download
   - Metadata extraction
   - Version handling

3. **Connect Retrieval Service** (`services/retrieval.py`)
   - Waterfall PDF strategy
   - Local caching
   - Success tracking
   - Fallback chain

**Deliverable**: `forge.get_fulltext("10.xxx/xxx")` returns PDF or extracted text

---

### Phase 3: RAG Knowledge Base (Week 5-6) ⭐⭐ MEDIUM PRIORITY

**Goal**: Enable `forge.index()` and `forge.ask()`

**Tasks**:
1. **PDF Text Extraction**
   - PyMuPDF integration
   - Section detection
   - Table extraction (optional)
   - Clean text output

2. **Chunking & Embedding Pipeline**
   - Sentence-level chunking
   - Overlap strategies
   - Embedding generation (sentence-transformers or OpenAI)
   - Batch processing

3. **Connect Knowledge Service** (`services/knowledge.py`)
   - Index management
   - Multi-paper indexing
   - Metadata tracking
   - Update/delete support

4. **Connect QA Service** (`services/qa.py`)
   - RAG retrieval
   - Context assembly
   - LLM synthesis
   - Citation extraction

**Deliverable**: 
```python
forge.index_papers(papers)
answer = forge.ask("What are the side effects of compound X?")
```

---

### Phase 4: Citation Networks (Week 7-8) ⭐⭐ MEDIUM PRIORITY

**Goal**: Enable `forge.build_network()` and citation analysis

**Tasks**:
1. **NetworkX Integration**
   - Graph construction
   - Node/edge attributes
   - Serialization

2. **Citation Analysis**
   - Influence metrics
   - Co-citation analysis
   - Bibliographic coupling
   - Cluster detection

3. **Connect Citation Service** (`services/citation.py`)
   - Forward citations
   - Backward references
   - Network building
   - Export formats

**Deliverable**:
```python
network = forge.build_network(seed_papers, depth=2)
influential = network.most_influential(n=10)
```

---

### Phase 5: REST API & CLI (Week 9-10) ⭐⭐⭐ LOWER PRIORITY

**Goal**: Production deployment options

**Tasks**:
1. **FastAPI REST Server**
   - Endpoint definitions
   - Request/response models
   - OpenAPI documentation
   - Authentication (optional)

2. **CLI Enhancement**
   - `litforge search "query"`
   - `litforge download DOI`
   - `litforge ask "question"`
   - Progress indicators

**Deliverable**: 
```bash
litforge search "CRISPR" --limit 20 --format json
litforge serve --port 8000
```

---

## Part 5: Technology Recommendations

Based on the ChemAgent planning documents:

### 5.1 API Sources (Priority Order)

| Source | Priority | Reason |
|--------|----------|--------|
| OpenAlex | ⭐⭐⭐ | 250M papers, FREE, CC0 license |
| Semantic Scholar | ⭐⭐⭐ | 214M papers, citation networks |
| Unpaywall | ⭐⭐⭐ | OA URLs, critical for PDFs |
| CrossRef | ⭐⭐ | DOI metadata, reliable |
| arXiv | ⭐⭐ | Preprints, free PDFs |
| PubMed | ⭐ | Biomedical focus |

### 5.2 Python Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `pyalex` | OpenAlex client | `pip install pyalex` |
| `semanticscholar` | S2 client | `pip install semanticscholar` |
| `pypdf` | PDF extraction | `pip install pypdf` |
| `pymupdf` | Advanced PDF | `pip install pymupdf` |
| `sentence-transformers` | Embeddings | `pip install sentence-transformers` |
| `chromadb` | Vector store | `pip install chromadb` |
| `networkx` | Graph analysis | `pip install networkx` |

### 5.3 LLM Options (from ChemAgent LLM Plan)

| Provider | Use Case | Cost |
|----------|----------|------|
| Groq | Fast inference | FREE |
| Gemini Flash | Fallback | FREE |
| OpenAI | Enterprise | ~$0.002/query |

---

## Part 6: Effort Estimates

### Total Effort: 8-10 weeks

| Phase | Weeks | Effort | Risk |
|-------|-------|--------|------|
| Phase 1: Discovery | 2 | Medium | Low |
| Phase 2: Retrieval | 2 | Medium | Medium |
| Phase 3: RAG | 2 | High | Medium |
| Phase 4: Citation | 2 | Medium | Low |
| Phase 5: REST/CLI | 2 | Low | Low |

### Dependencies

```
Phase 1 (Discovery) 
    └─→ Phase 2 (Retrieval) 
            └─→ Phase 3 (RAG)
                    └─→ Phase 5 (REST/CLI)
    
Phase 1 (Discovery)
    └─→ Phase 4 (Citation)
```

---

## Part 7: Quick Wins (Can Do Now)

If you want immediate progress, these can be done in 1-2 days each:

### Quick Win 1: Fix Forge.search() to Use Simple API
```python
# In forge.py, replace stub with:
def search(self, query: str, limit: int = 10) -> List[Paper]:
    return litforge.search(query, limit=limit)
```

### Quick Win 2: Add PubMed to Simple API
The PubMed Entrez API is straightforward:
```python
from Bio import Entrez
Entrez.email = "your@email.com"
```

### Quick Win 3: Add Unpaywall URL Resolution
Single API call per DOI:
```python
response = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={email}")
oa_url = response.json().get("best_oa_location", {}).get("url_for_pdf")
```

---

## Part 8: Decision Points

Before proceeding, consider these architectural decisions:

### Decision 1: Sync vs Async
- **Current**: Synchronous API
- **Option**: Add async versions (aiohttp)
- **Recommendation**: Keep sync for simplicity, add async later

### Decision 2: LLM Provider
- **Free**: Groq (recommended for prototyping)
- **Paid**: OpenAI (for production)
- **Recommendation**: Support both with router

### Decision 3: Vector Store Default
- **Current**: ChromaDB (good choice)
- **Alternative**: FAISS (faster, no persistence)
- **Recommendation**: Keep ChromaDB as default

### Decision 4: PDF Extraction Library
- **Option 1**: pypdf (simple, lightweight)
- **Option 2**: pymupdf (faster, more features)
- **Recommendation**: Start with pypdf, upgrade if needed

---

## Part 9: Success Metrics

### MVP Success Criteria
- [ ] `forge.search()` returns results from 2+ sources
- [ ] `forge.get_fulltext()` works for 50%+ of OA papers
- [ ] `forge.ask()` answers questions with citations
- [ ] Web UI handles all basic workflows

### Production Success Criteria
- [ ] 95%+ uptime for API
- [ ] <2s latency for search
- [ ] <10s for PDF retrieval
- [ ] <30s for RAG Q&A
- [ ] Citation network visualization

---

## Appendix A: File Status Reference

```
src/litforge/
├── __init__.py                 ✅ Complete
├── api.py                      ✅ Complete (Simple API)
├── cli.py                      ⚠️ Partial (needs commands)
├── config.py                   ✅ Complete
├── clients/
│   ├── __init__.py             ✅ Complete
│   ├── base.py                 ✅ Complete
│   ├── arxiv.py                ❌ Stub
│   ├── crossref.py             ⚠️ Partial
│   ├── openalex.py             ⚠️ Partial
│   ├── pubmed.py               ❌ Stub
│   ├── semantic_scholar.py     ❌ Stub
│   └── unpaywall.py            ❌ Stub
├── core/
│   ├── __init__.py             ✅ Complete
│   └── forge.py                ⚠️ Skeleton
├── embedding/
│   ├── __init__.py             ✅ Complete
│   ├── base.py                 ✅ Complete
│   ├── openai.py               ⚠️ Partial
│   └── sentence_transformers.py ⚠️ Partial
├── integrations/
│   ├── crewai.py               ✅ Complete
│   ├── langchain.py            ✅ Complete
│   └── langgraph.py            ✅ Complete
├── llm/
│   ├── __init__.py             ✅ Complete
│   ├── base.py                 ✅ Complete
│   └── openai.py               ⚠️ Partial
├── mcp/
│   └── server.py               ✅ Complete
├── models/
│   ├── __init__.py             ✅ Complete
│   ├── network.py              ⚠️ Partial
│   ├── publication.py          ✅ Complete
│   └── search.py               ✅ Complete
├── services/
│   ├── __init__.py             ✅ Complete
│   ├── citation.py             ❌ Stub
│   ├── discovery.py            ⚠️ Partial
│   ├── knowledge.py            ❌ Stub
│   ├── qa.py                   ❌ Stub
│   └── retrieval.py            ❌ Stub
├── stores/
│   ├── __init__.py             ✅ Complete
│   ├── base.py                 ✅ Complete
│   ├── chromadb.py             ✅ Complete
│   ├── faiss.py                ✅ Complete
│   └── qdrant.py               ✅ Complete
└── ui/
    ├── app.py                  ✅ Complete
    └── chat.py                 ✅ Complete
```

---

## Appendix B: Related Planning Documents

Found in `/home/sdodl001_odu_edu/ChemAgent/docs/planning/`:

| Document | Size | Relevance |
|----------|------|-----------|
| `COMPREHENSIVE_LITERATURE_AGENT_OPTIONS.md` | 47KB | Architecture options (10 approaches) |
| `LITERATURE_SEARCH_AGENT_ANALYSIS.md` | 18KB | API comparisons, 4-week plan |
| `LLM_INTEGRATION_PLAN.md` | 29KB | LLM provider recommendations |
| `MULTI_DOCUMENT_CHAT_ARCHITECTURE.md` | 70KB | RAG pipeline design |

---

## Conclusion

LitForge has a solid foundation with the Simple API and Web UI working well. The primary gaps are:

1. **PDF/Fulltext retrieval** - Critical for RAG
2. **RAG Q&A pipeline** - The main value proposition
3. **Forge class functionality** - Needs to call actual APIs

**Recommended Next Steps**:
1. Implement Unpaywall client for PDF URLs (1-2 days)
2. Connect Forge.search() to the working Simple API (1 day)
3. Add Semantic Scholar client (3-4 days)
4. Build PDF extraction pipeline (1 week)
5. Connect RAG pipeline (1-2 weeks)

The path to a fully functional LitForge is clear - it requires systematic implementation of the outlined phases while building on the working Simple API foundation.

---

*Document Version: 1.0*  
*Last Updated: January 2025*  
*Author: AI-assisted analysis*
