# LitForge Implementation Master Plan

**Version**: 1.0  
**Created**: January 14, 2025  
**Status**: 🚀 Active Development

---

## Pre-Implementation Audit Summary

### Existing Code Analysis (9,462 lines total)

| Component | Lines | Status | Assessment |
|-----------|-------|--------|------------|
| **api.py** | 385 | ✅ Working | Simple API functional, uses OpenAlex directly |
| **core/forge.py** | 549 | ⚠️ Structure | Good structure, needs to connect to services |
| **clients/openalex.py** | 232 | ✅ Implemented | Functional, needs minor enhancements |
| **clients/semantic_scholar.py** | 268 | ✅ Implemented | Functional, needs testing |
| **clients/pubmed.py** | 312 | ✅ Implemented | Functional, uses E-utilities |
| **clients/arxiv.py** | 222 | ✅ Implemented | Functional, XML parsing |
| **clients/crossref.py** | 204 | ✅ Implemented | Functional |
| **clients/unpaywall.py** | 85 | ✅ Implemented | Functional, needs batch support |
| **services/discovery.py** | 271 | ⚠️ Partial | Structure exists, needs multi-source |
| **services/retrieval.py** | 235 | ⚠️ Partial | PDF download exists, needs text extraction |
| **services/knowledge.py** | 311 | ⚠️ Partial | Indexing exists, needs chunking |
| **services/qa.py** | 283 | ⚠️ Partial | RAG structure, needs LLM connection |
| **services/citation.py** | 366 | ⚠️ Partial | Network building exists, needs NetworkX |
| **stores/chromadb.py** | 195 | ✅ Implemented | Fully functional |
| **stores/faiss.py** | 373 | ✅ Implemented | Functional |
| **stores/qdrant.py** | 268 | ✅ Implemented | Functional |
| **embedding/sentence_transformers.py** | 91 | ✅ Implemented | Local embeddings work |
| **embedding/openai.py** | 111 | ✅ Implemented | OpenAI embeddings work |
| **llm/openai.py** | 89 | ✅ Implemented | Basic OpenAI LLM |
| **ui/app.py** | 557 | ✅ Working | Form UI works |
| **ui/chat.py** | 432 | ✅ Working | Chat UI works |
| **mcp/server.py** | 145 | ✅ Working | MCP integration works |
| **integrations/** | 1097 | ✅ Working | CrewAI, LangChain, LangGraph |

### Key Finding: More Complete Than Expected

The codebase has **substantial implementations** already. The main gaps are:
1. **Missing PDF text extraction** (pypdf/pymupdf not integrated)
2. **Missing LLM providers** (only OpenAI, need Groq/Anthropic/Ollama)
3. **Forge class not wired to services** properly
4. **Missing async support** throughout
5. **Missing advanced features** (reranking, hybrid search, agents)

---

## Implementation Phases - Detailed Breakdown

### Legend
- 🔴 Not Started
- 🟡 In Progress
- 🟢 Complete
- ⏭️ Skipped (already exists)

---

## Phase 1: Core Excellence (Weeks 1-3)

### Goal: Make search, retrieval, and Forge class work reliably

### 1.1 Connect Forge Class to Working Services 🔴

**File**: `src/litforge/core/forge.py`

| Task | Status | Notes |
|------|--------|-------|
| Wire `search()` to discovery service | 🔴 | Currently not calling service properly |
| Wire `lookup()` to clients | 🔴 | Need DOI resolution |
| Wire `get_paper()` to return Publication | 🔴 | |
| Wire `get_citations()` to clients | 🔴 | Need S2/OpenAlex |
| Wire `get_references()` to clients | 🔴 | |
| Add proper error handling | 🔴 | |
| Add logging | 🔴 | |

### 1.2 Enhance Discovery Service 🔴

**File**: `src/litforge/services/discovery.py`

| Task | Status | Notes |
|------|--------|-------|
| Parallel multi-source search | 🔴 | Use asyncio.gather |
| Result deduplication by DOI | 🔴 | Merge same papers from diff sources |
| Source attribution | 🔴 | Track which source found each paper |
| Relevance scoring | 🔴 | Combine scores from sources |
| Rate limit handling | 🔴 | Respect API limits |

### 1.3 Add Async Architecture 🔴

**Files**: Multiple

| Task | Status | Notes |
|------|--------|-------|
| Create `async_api.py` | 🔴 | Async versions of simple API |
| Add async to base client | 🔴 | `clients/base.py` |
| Add async to all clients | 🔴 | OpenAlex, S2, etc |
| Add async to services | 🔴 | Discovery, retrieval |
| Keep sync wrappers | ⏭️ | Already have sync API |

### 1.4 Test & Validate Clients 🔴

| Client | Status | Test Query |
|--------|--------|------------|
| OpenAlex | 🔴 | `forge.search("CRISPR", sources=["openalex"])` |
| Semantic Scholar | 🔴 | `forge.search("transformer", sources=["semantic_scholar"])` |
| PubMed | 🔴 | `forge.search("cancer", sources=["pubmed"])` |
| arXiv | 🔴 | `forge.search("neural network", sources=["arxiv"])` |
| CrossRef | 🔴 | `forge.lookup("10.1038/nature12373")` |
| Unpaywall | 🔴 | `forge.get_open_access("10.1038/nature12373")` |

---

## Phase 2: PDF Processing (Weeks 4-5)

### Goal: Extract text from PDFs reliably

### 2.1 Add PDF Extraction 🔴

**New File**: `src/litforge/processors/pdf.py`

| Task | Status | Notes |
|------|--------|-------|
| Create PDFExtractor class | 🔴 | |
| pypdf extraction (simple) | 🔴 | Fallback method |
| pymupdf extraction (advanced) | 🔴 | Better quality |
| Fallback chain | 🔴 | Try pymupdf → pypdf |
| Handle encrypted PDFs | 🔴 | Skip gracefully |
| Handle scanned PDFs | 🔴 | Detect and warn |

### 2.2 Add Section Detection 🔴

**New File**: `src/litforge/processors/sections.py`

| Task | Status | Notes |
|------|--------|-------|
| Create SectionDetector class | 🔴 | |
| Detect Abstract | 🔴 | Pattern matching |
| Detect Introduction | 🔴 | |
| Detect Methods | 🔴 | |
| Detect Results | 🔴 | |
| Detect Discussion | 🔴 | |
| Detect References | 🔴 | |
| Handle non-standard formats | 🔴 | Best effort |

### 2.3 Add Smart Chunking 🔴

**New File**: `src/litforge/processors/chunking.py`

| Task | Status | Notes |
|------|--------|-------|
| Create TextChunker class | 🔴 | |
| Sentence-based chunking | 🔴 | Respect sentence boundaries |
| Paragraph-based chunking | 🔴 | |
| Section-aware chunking | 🔴 | Don't split sections |
| Configurable overlap | 🔴 | 10-20% overlap |
| Token counting | 🔴 | For LLM context limits |

### 2.4 Connect to Retrieval Service 🔴

**File**: `src/litforge/services/retrieval.py`

| Task | Status | Notes |
|------|--------|-------|
| Integrate PDFExtractor | 🔴 | Call from retrieve() |
| Integrate SectionDetector | 🔴 | Optional parsing |
| Add text caching | 🔴 | Cache extracted text |
| Add `retrieve_text()` method | 🔴 | Return structured text |
| Add batch processing | 🔴 | Multiple PDFs |

---

## Phase 3: RAG Pipeline (Weeks 6-7)

### Goal: Answer questions with citations

### 3.1 Add More LLM Providers 🔴

**New Files**: `src/litforge/llm/`

| Provider | File | Status | Notes |
|----------|------|--------|-------|
| Groq | `groq.py` | 🔴 | Free tier, fast |
| Anthropic | `anthropic.py` | 🔴 | Claude models |
| Ollama | `ollama.py` | 🔴 | Local LLMs |
| Router | `router.py` | 🔴 | Smart routing with fallback |

### 3.2 Add Hybrid Retrieval 🔴

**New File**: `src/litforge/retrieval/hybrid.py`

| Task | Status | Notes |
|------|--------|-------|
| BM25 sparse retrieval | 🔴 | Using rank_bm25 |
| Dense retrieval | ⏭️ | Already have via stores |
| Hybrid combination | 🔴 | RRF or weighted |
| Configurable weights | 🔴 | dense_weight, sparse_weight |

### 3.3 Add Cross-Encoder Reranking 🔴

**New File**: `src/litforge/retrieval/reranker.py`

| Task | Status | Notes |
|------|--------|-------|
| CrossEncoderReranker class | 🔴 | |
| Use ms-marco-MiniLM | 🔴 | Default model |
| Batch reranking | 🔴 | Efficient processing |
| Score normalization | 🔴 | 0-1 range |

### 3.4 Add Evidence Extraction 🔴

**New File**: `src/litforge/processors/evidence.py`

| Task | Status | Notes |
|------|--------|-------|
| EvidenceExtractor class | 🔴 | |
| Extract supporting passages | 🔴 | |
| Score relevance | 🔴 | |
| Add citation context | 🔴 | Before/after text |
| Track provenance | 🔴 | Source, page, section |

### 3.5 Connect QA Service 🔴

**File**: `src/litforge/services/qa.py`

| Task | Status | Notes |
|------|--------|-------|
| Use LLM router | 🔴 | Not just OpenAI |
| Integrate hybrid retrieval | 🔴 | Better context |
| Integrate reranker | 🔴 | Better top-k |
| Add evidence in response | 🔴 | With citations |
| Add confidence scoring | 🔴 | Based on evidence |
| Add streaming support | 🔴 | For long answers |

---

## Phase 4: Research Agent (Weeks 8-9)

### Goal: Autonomous multi-step research

### 4.1 Create Research Agent 🔴

**New File**: `src/litforge/agents/research_agent.py`

| Task | Status | Notes |
|------|--------|-------|
| ResearchAgent class | 🔴 | |
| Plan generation | 🔴 | Break query into steps |
| Step execution | 🔴 | Search → Retrieve → Analyze |
| Iterative refinement | 🔴 | Improve answer |
| Progress tracking | 🔴 | For streaming UI |

### 4.2 Add Research Planner 🔴

**New File**: `src/litforge/agents/planner.py`

| Task | Status | Notes |
|------|--------|-------|
| ResearchPlanner class | 🔴 | |
| Query decomposition | 🔴 | Complex → sub-queries |
| Dependency detection | 🔴 | Order matters |
| Resource estimation | 🔴 | Papers needed |

### 4.3 Add Contradiction Detection 🔴

**New File**: `src/litforge/processors/contradictions.py`

| Task | Status | Notes |
|------|--------|-------|
| ContradictionDetector class | 🔴 | |
| Compare evidence pairs | 🔴 | |
| Classify relationship | 🔴 | Support/contradict/neutral |
| Report conflicts | 🔴 | In research result |

---

## Phase 5: Citation Networks (Weeks 10-11)

### Goal: Advanced citation analysis

### 5.1 Integrate NetworkX 🔴

**File**: `src/litforge/services/citation.py`

| Task | Status | Notes |
|------|--------|-------|
| NetworkX graph building | 🔴 | Replace manual graph |
| PageRank centrality | 🔴 | Proper algorithm |
| Betweenness centrality | 🔴 | |
| Community detection | 🔴 | Louvain algorithm |

### 5.2 Add Network Analysis 🔴

**New File**: `src/litforge/network/metrics.py`

| Task | Status | Notes |
|------|--------|-------|
| Influence metrics | 🔴 | |
| Co-citation analysis | 🔴 | |
| Bibliographic coupling | 🔴 | |
| Temporal analysis | 🔴 | Trends over time |

### 5.3 Add Visualization Export 🔴

**New File**: `src/litforge/network/visualization.py`

| Task | Status | Notes |
|------|--------|-------|
| Export to JSON (D3) | 🔴 | |
| Export to Gephi | 🔴 | .gexf format |
| Generate pyvis HTML | 🔴 | Interactive |

---

## Phase 6: Production Features (Weeks 12-14)

### Goal: Production-ready system

### 6.1 Add Streaming Support 🔴

**New File**: `src/litforge/streaming/handler.py`

| Task | Status | Notes |
|------|--------|-------|
| StreamingHandler class | 🔴 | |
| SSE support | 🔴 | Server-sent events |
| Event types | 🔴 | Search, retrieve, answer |
| Progress updates | 🔴 | For UI |

### 6.2 Add REST API 🔴

**New Files**: `src/litforge/api/`

| Task | Status | Notes |
|------|--------|-------|
| FastAPI app | 🔴 | `api/app.py` |
| Search endpoint | 🔴 | `POST /search` |
| Paper endpoint | 🔴 | `GET /paper/{doi}` |
| Ask endpoint | 🔴 | `POST /ask` |
| Research endpoint | 🔴 | `POST /research` |
| OpenAPI docs | 🔴 | Auto-generated |

### 6.3 Enhance CLI 🔴

**File**: `src/litforge/cli.py`

| Task | Status | Notes |
|------|--------|-------|
| `litforge search` | 🔴 | Working command |
| `litforge ask` | 🔴 | RAG Q&A |
| `litforge download` | 🔴 | Get PDFs |
| `litforge serve` | 🔴 | Start REST API |
| Progress bars | 🔴 | Rich/tqdm |

### 6.4 Add Export Formats 🔴

**New File**: `src/litforge/services/export.py`

| Task | Status | Notes |
|------|--------|-------|
| BibTeX export | 🔴 | |
| RIS export | 🔴 | |
| Markdown report | 🔴 | |
| JSON export | 🔴 | |

### 6.5 Error Handling & Observability 🔴

**New Files**: `src/litforge/core/`

| Task | Status | Notes |
|------|--------|-------|
| Custom exceptions | 🔴 | `errors.py` |
| Structured logging | 🔴 | `observability.py` |
| Metrics collection | 🔴 | Optional |
| Rate limit tracking | 🔴 | Per API |

---

## New Files to Create

```
src/litforge/
├── async_api.py                    # Phase 1
├── agents/
│   ├── __init__.py                 # Phase 4
│   ├── research_agent.py           # Phase 4
│   └── planner.py                  # Phase 4
├── api/
│   ├── __init__.py                 # Phase 6
│   ├── app.py                      # Phase 6
│   └── routes/                     # Phase 6
│       ├── search.py
│       ├── papers.py
│       └── qa.py
├── llm/
│   ├── groq.py                     # Phase 3
│   ├── anthropic.py                # Phase 3
│   ├── ollama.py                   # Phase 3
│   └── router.py                   # Phase 3
├── network/
│   ├── __init__.py                 # Phase 5
│   ├── metrics.py                  # Phase 5
│   └── visualization.py            # Phase 5
├── processors/
│   ├── __init__.py                 # Phase 2
│   ├── pdf.py                      # Phase 2
│   ├── sections.py                 # Phase 2
│   ├── chunking.py                 # Phase 2
│   ├── evidence.py                 # Phase 3
│   └── contradictions.py           # Phase 4
├── retrieval/
│   ├── __init__.py                 # Phase 3
│   ├── hybrid.py                   # Phase 3
│   └── reranker.py                 # Phase 3
├── streaming/
│   ├── __init__.py                 # Phase 6
│   └── handler.py                  # Phase 6
└── core/
    ├── errors.py                   # Phase 6
    └── observability.py            # Phase 6
```

## Files to Modify

```
src/litforge/
├── core/forge.py                   # Phase 1 - Wire to services
├── services/discovery.py           # Phase 1 - Multi-source
├── services/retrieval.py           # Phase 2 - PDF extraction
├── services/knowledge.py           # Phase 3 - Better chunking
├── services/qa.py                  # Phase 3 - Full RAG
├── services/citation.py            # Phase 5 - NetworkX
├── clients/base.py                 # Phase 1 - Async support
├── config.py                       # Phase 1 - New providers
└── cli.py                          # Phase 6 - Working commands
```

---

## Dependencies to Add

```toml
[project.dependencies]
# PDF Processing (Phase 2)
pypdf = ">=4.0"
pymupdf = ">=1.24"

# RAG (Phase 3)
rank-bm25 = ">=0.2"
sentence-transformers = ">=2.7"  # May already exist

# LLM Providers (Phase 3)
groq = ">=0.9"
anthropic = ">=0.34"
ollama = ">=0.3"

# Networks (Phase 5)
networkx = ">=3.3"
pyvis = ">=0.3"

# REST API (Phase 6)
fastapi = ">=0.111"
uvicorn = ">=0.30"

# Utilities
tenacity = ">=8.5"  # Retry logic
structlog = ">=24.4"  # Logging
rich = ">=13.7"  # CLI
```

---

## Testing Checkpoints

### After Phase 1
```python
from litforge import Forge
forge = Forge()

# Multi-source search
papers = forge.search("CRISPR", sources=["openalex", "semantic_scholar"])
assert len(papers) > 0
print(f"Found {len(papers)} papers from {set(p.sources[0] for p in papers)}")

# DOI lookup
paper = forge.lookup("10.1126/science.aax5077")
assert paper is not None
```

### After Phase 2
```python
# PDF retrieval and extraction
paper = forge.lookup("10.1126/science.aax5077")
text = forge.get_fulltext(paper)
assert text is not None
assert len(text) > 1000
print(f"Extracted {len(text)} characters")
```

### After Phase 3
```python
# RAG Q&A
papers = forge.search("CRISPR delivery", limit=10)
forge.index(papers)
answer = forge.ask("What are the main CRISPR delivery mechanisms?")
assert answer.text
assert len(answer.evidence) > 0
print(answer.text)
```

### After Phase 4
```python
# Research agent
result = await forge.research(
    "What are the latest advances in mRNA vaccines?",
    depth="standard"
)
assert result.answer
assert result.confidence > 0.5
print(f"Answer with {len(result.evidence)} sources")
```

### After Phase 5
```python
# Citation network
network = forge.build_network(
    ["10.1126/science.aax5077"],
    depth=2
)
assert len(network.nodes) > 10
key_papers = network.most_influential(5)
print(f"Key papers: {[p.title[:50] for p in key_papers]}")
```

### After Phase 6
```bash
# CLI
litforge search "machine learning drug discovery" --limit 10
litforge ask "What ML methods are used for drug discovery?"
litforge serve --port 8000
```

---

## Progress Tracking

Use this checklist to track progress. Update status as you complete each task.

### Phase 1 Progress
- [ ] 1.1 Connect Forge class
- [ ] 1.2 Enhance Discovery service
- [ ] 1.3 Add async architecture
- [ ] 1.4 Test all clients

### Phase 2 Progress
- [ ] 2.1 PDF extraction
- [ ] 2.2 Section detection
- [ ] 2.3 Smart chunking
- [ ] 2.4 Connect to retrieval

### Phase 3 Progress
- [ ] 3.1 LLM providers (Groq, Anthropic, Ollama)
- [ ] 3.2 Hybrid retrieval
- [ ] 3.3 Cross-encoder reranking
- [ ] 3.4 Evidence extraction
- [ ] 3.5 Connect QA service

### Phase 4 Progress
- [ ] 4.1 Research agent
- [ ] 4.2 Research planner
- [ ] 4.3 Contradiction detection

### Phase 5 Progress
- [ ] 5.1 NetworkX integration
- [ ] 5.2 Network analysis
- [ ] 5.3 Visualization export

### Phase 6 Progress
- [ ] 6.1 Streaming support
- [ ] 6.2 REST API
- [ ] 6.3 CLI enhancement
- [ ] 6.4 Export formats
- [ ] 6.5 Error handling

---

## Notes on Avoiding Duplication

### Already Exists - DO NOT RECREATE:
1. ✅ Vector stores (ChromaDB, FAISS, Qdrant) - fully working
2. ✅ Embedding providers (OpenAI, sentence-transformers) - working
3. ✅ OpenAI LLM provider - working
4. ✅ All clients (OpenAlex, S2, PubMed, arXiv, CrossRef, Unpaywall) - implemented
5. ✅ MCP server - working
6. ✅ Framework integrations - working
7. ✅ Web UI (form + chat) - working
8. ✅ Publication/Author models - complete

### Needs Enhancement - MODIFY EXISTING:
1. ⚠️ Forge class - wire to services
2. ⚠️ Discovery service - add multi-source
3. ⚠️ Retrieval service - add PDF extraction
4. ⚠️ Knowledge service - add better chunking
5. ⚠️ QA service - add LLM router
6. ⚠️ Citation service - add NetworkX
7. ⚠️ CLI - add working commands
8. ⚠️ Config - add new provider options

### Needs Creation - NEW FILES:
1. 🔴 PDF processors (pdf.py, sections.py, chunking.py)
2. 🔴 LLM providers (groq.py, anthropic.py, ollama.py, router.py)
3. 🔴 Retrieval enhancements (hybrid.py, reranker.py)
4. 🔴 Evidence processor
5. 🔴 Research agent
6. 🔴 Network analysis
7. 🔴 REST API
8. 🔴 Streaming

---

*Ready to begin implementation. Update this document as progress is made.*
