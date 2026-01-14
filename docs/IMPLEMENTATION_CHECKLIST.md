# LitForge Implementation Checklist

Quick reference for tracking implementation progress.

## Current Status: v0.1.0 (January 2025)

---

## ✅ COMPLETE (Working Now)

### Simple API
- [x] `litforge.search(query, limit)` - Search OpenAlex/CrossRef
- [x] `litforge.lookup(doi)` - Get paper by DOI
- [x] `litforge.citations(doi)` - Get citing papers
- [x] `litforge.references(doi)` - Get reference list

### Web UI
- [x] Form UI (port 8502) - Search with filters
- [x] Chat UI (port 8503) - Conversational search
- [x] Expandable results
- [x] PDF request handling

### MCP Server
- [x] `search_papers` tool
- [x] `get_paper_details` tool
- [x] `get_citations` tool
- [x] `get_references` tool

### Framework Integrations
- [x] CrewAI tools
- [x] LangChain tools
- [x] LangGraph nodes

### Vector Stores
- [x] ChromaDB adapter
- [x] FAISS adapter
- [x] Qdrant adapter

---

## ⚠️ PARTIAL (Needs Work)

### Clients
- [ ] `openalex.py` - Add pagination, filters
- [ ] `crossref.py` - Add batch operations
- [ ] `semantic_scholar.py` - Implement fully
- [ ] `pubmed.py` - Implement fully
- [ ] `arxiv.py` - Implement fully
- [ ] `unpaywall.py` - Implement fully

### Forge Class
- [ ] Connect `search()` to Simple API
- [ ] Connect `get_paper()` to clients
- [ ] Connect `get_citations()` to clients
- [ ] Connect `get_references()` to clients

### Embedding
- [ ] Test OpenAI embeddings
- [ ] Test sentence-transformers

---

## ❌ NOT IMPLEMENTED

### PDF/Fulltext Retrieval
- [ ] Unpaywall OA URL resolution
- [ ] arXiv PDF download
- [ ] PubMed Central access
- [ ] CORE API integration
- [ ] PDF caching
- [ ] `forge.get_fulltext(doi)`
- [ ] `forge.download_pdf(doi)`

### RAG Q&A Pipeline
- [ ] PDF text extraction (pypdf/pymupdf)
- [ ] Text chunking
- [ ] Embedding pipeline
- [ ] Index management
- [ ] RAG retrieval
- [ ] LLM synthesis
- [ ] `forge.index_papers(papers)`
- [ ] `forge.ask(question)`

### Citation Networks
- [ ] NetworkX integration
- [ ] Graph building from citations
- [ ] Influence metrics
- [ ] Co-citation analysis
- [ ] `forge.build_network(papers, depth)`
- [ ] Visualization export

### REST API
- [ ] FastAPI server
- [ ] `/search` endpoint
- [ ] `/paper/{doi}` endpoint
- [ ] `/citations/{doi}` endpoint
- [ ] `/ask` endpoint
- [ ] OpenAPI documentation

### CLI Commands
- [ ] `litforge search "query"`
- [ ] `litforge download DOI`
- [ ] `litforge ask "question"`
- [ ] `litforge serve`
- [ ] Progress indicators

---

## Quick Wins (1-2 days each)

1. **Connect Forge.search() to Simple API**
   ```python
   def search(self, query, limit=10):
       return litforge.search(query, limit=limit)
   ```

2. **Add Unpaywall URL resolution**
   ```python
   url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
   ```

3. **Add basic PDF download for arXiv**
   ```python
   pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
   ```

---

## Dependencies to Install

```bash
# Already installed
pip install requests chromadb

# Need to add
pip install pyalex              # OpenAlex client
pip install semanticscholar     # Semantic Scholar
pip install pypdf               # PDF extraction
pip install sentence-transformers  # Embeddings
pip install networkx            # Citation graphs
pip install fastapi uvicorn     # REST API
```

---

## Test Commands

```bash
# Test Simple API
python -c "import litforge; print(litforge.search('machine learning'))"

# Test Web UI
cd src/litforge/ui && streamlit run app.py --server.port 8502

# Test Chat UI
cd src/litforge/ui && streamlit run chat.py --server.port 8503

# Test MCP Server
python -m litforge.mcp.server
```

---

## Next Steps Priority

1. ⭐⭐⭐ Implement Unpaywall client (unlocks PDF access)
2. ⭐⭐⭐ Implement Semantic Scholar client (better citations)
3. ⭐⭐ Add PDF text extraction
4. ⭐⭐ Connect RAG pipeline
5. ⭐ Add citation network analysis

---

*Last Updated: January 2025*
