# LitForge: State-of-the-Art Architecture and Enhancement Plan

**Version**: 2.0  
**Date**: January 2025  
**Vision**: The definitive scientific literature backbone for AI agents and researchers

---

## Executive Summary

This document presents a comprehensive re-evaluation of LitForge's architecture against state-of-the-art systems (PaperQA2, GPT-Researcher, Semantic Scholar, Elicit) and proposes enhancements to make LitForge a **truly professional, production-ready** library suitable for:

1. **Standalone use** - Researchers using LitForge directly
2. **Library integration** - Import into any Python project
3. **Agentic systems** - Multi-agent orchestration (CrewAI, AutoGen, LangGraph)
4. **MCP tool** - Claude and AI assistant integration

**Key Insight**: Current LitForge has the right *structure* but lacks the *depth* needed for state-of-the-art performance. This plan addresses that gap.

---

## Part 1: State-of-the-Art Competitive Analysis

### 1.1 Leading Systems Comparison

| Feature | PaperQA2 | GPT-Researcher | Elicit | Semantic Scholar | **LitForge (Goal)** |
|---------|----------|----------------|--------|------------------|---------------------|
| **Search Sources** | S2, CrossRef | Web + Arxiv | Multiple | Own DB | ✅ 6+ sources |
| **PDF Retrieval** | Unpaywall | Web scraping | Limited | Direct links | ✅ Waterfall strategy |
| **Full-text Extraction** | Advanced | Basic | Unknown | N/A | ✅ Section-aware |
| **RAG Q&A** | Superhuman | Report-style | Good | N/A | ✅ Evidence-based |
| **Citation Networks** | Basic | None | Basic | Excellent | ✅ Advanced graphs |
| **Streaming** | ✅ | ✅ | ✅ | N/A | ✅ Real-time |
| **Agentic** | ✅ | ✅ | ❌ | ❌ | ✅ Multi-agent |
| **Evidence Grounding** | ✅ | Partial | ✅ | N/A | ✅ Full provenance |
| **Hallucination Prevention** | ✅ | Partial | ✅ | N/A | ✅ Built-in |
| **Local LLM Support** | ✅ | Partial | ❌ | N/A | ✅ Ollama, vLLM |
| **Cost (per query)** | ~$0.40 | ~$0.30 | $$ | Free | ✅ $0-0.10 |

### 1.2 What Makes PaperQA2 "Superhuman"

PaperQA2 achieved #1 on DeepResearchGym with these techniques:

1. **Evidence-based answering** - Every claim linked to source text
2. **Iterative refinement** - Search → Retrieve → Answer → Verify → Refine
3. **Citation context** - Uses surrounding sentences, not just snippets
4. **Contradiction detection** - Identifies conflicting evidence
5. **Confidence scoring** - Indicates answer certainty
6. **Multi-paper synthesis** - Combines evidence across papers

**LitForge must match or exceed these capabilities.**

---

## Part 2: Current Gaps - Deep Analysis

### 2.1 Architecture Gaps

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRENT vs REQUIRED                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CURRENT ARCHITECTURE                 REQUIRED ARCHITECTURE              │
│  ────────────────────                 ─────────────────────              │
│                                                                          │
│  Simple API (sync only)               Async-first with sync wrapper      │
│  Single source per query              Parallel multi-source fusion       │
│  Basic keyword search                 Hybrid search (dense + sparse)     │
│  No reranking                         Cross-encoder reranking            │
│  Simple Paper model                   Rich Document model with chunks    │
│  No streaming                         Full streaming support             │
│  Basic error handling                 Circuit breaker + retry            │
│  No provenance tracking               Full evidence trail                │
│  Single-turn Q&A only                 Multi-turn with memory             │
│  No agent capabilities                Full agentic reasoning             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Missing Critical Features

| Category | Feature | Priority | Effort |
|----------|---------|----------|--------|
| **Search** | Hybrid search (BM25 + semantic) | ⭐⭐⭐ | Medium |
| **Search** | Query expansion/reformulation | ⭐⭐⭐ | Medium |
| **Search** | Cross-encoder reranking | ⭐⭐⭐ | Medium |
| **Retrieval** | Waterfall PDF strategy | ⭐⭐⭐ | High |
| **Retrieval** | Smart caching with TTL | ⭐⭐ | Low |
| **Extraction** | Section-aware parsing | ⭐⭐⭐ | High |
| **Extraction** | Table extraction | ⭐⭐ | High |
| **Extraction** | Figure caption extraction | ⭐⭐ | Medium |
| **RAG** | Chunking strategies | ⭐⭐⭐ | Medium |
| **RAG** | Evidence extraction | ⭐⭐⭐ | High |
| **RAG** | Citation grounding | ⭐⭐⭐ | High |
| **RAG** | Contradiction detection | ⭐⭐ | High |
| **RAG** | Confidence scoring | ⭐⭐ | Medium |
| **Agent** | Research planning | ⭐⭐⭐ | High |
| **Agent** | Iterative refinement | ⭐⭐⭐ | High |
| **Agent** | Tool orchestration | ⭐⭐⭐ | Medium |
| **Network** | Influence metrics | ⭐⭐ | Medium |
| **Network** | Concept clustering | ⭐⭐ | High |
| **Network** | Trend detection | ⭐⭐ | High |
| **Output** | Streaming responses | ⭐⭐⭐ | Medium |
| **Output** | Export (BibTeX, RIS) | ⭐⭐ | Low |
| **Output** | Visualization | ⭐⭐ | Medium |

---

## Part 3: Enhanced Architecture Design

### 3.1 Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LitForge v2.0 Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         API LAYER                                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │ │
│  │  │ Simple   │  │  Forge   │  │  REST    │  │   MCP    │            │ │
│  │  │   API    │  │  Class   │  │   API    │  │  Server  │            │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      ORCHESTRATION LAYER                            │ │
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────┐  │ │
│  │  │   Research Agent    │  │          Task Planner               │  │ │
│  │  │  ┌───────────────┐  │  │  ┌─────────────────────────────┐   │  │ │
│  │  │  │ Plan → Search │  │  │  │ Decompose → Schedule → Track│   │  │ │
│  │  │  │ → Retrieve →  │  │  │  └─────────────────────────────┘   │  │ │
│  │  │  │ → Analyze →   │  │  │                                     │  │ │
│  │  │  │ → Synthesize  │  │  │  ┌─────────────────────────────┐   │  │ │
│  │  │  └───────────────┘  │  │  │    Conversation Memory      │   │  │ │
│  │  └─────────────────────┘  │  └─────────────────────────────┘   │  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        SERVICE LAYER                                │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │ │
│  │  │ Discovery  │ │ Retrieval  │ │ Knowledge  │ │   Q&A      │       │ │
│  │  │  Service   │ │  Service   │ │  Service   │ │  Service   │       │ │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │ │
│  │  │  Citation  │ │  Concept   │ │  Synthesis │ │   Export   │       │ │
│  │  │  Service   │ │  Service   │ │  Service   │ │  Service   │       │ │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        CORE LAYER                                   │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │ │
│  │  │  Clients   │ │ Processors │ │  Stores    │ │   LLM      │       │ │
│  │  │ (6 APIs)   │ │ (PDF/Text) │ │ (Vector)   │ │  Router    │       │ │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │ │
│  │  │ Embeddings │ │  Rankers   │ │   Cache    │ │  Metrics   │       │ │
│  │  │            │ │            │ │            │ │            │       │ │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow for Research Query

```
User: "What are the latest advances in CRISPR delivery mechanisms?"
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌───────────────┐               ┌───────────────┐
           │ Query Parser  │               │ Intent Router │
           │ - Entity NER  │               │ - Search/Q&A  │
           │ - Query expand│               │ - Compare     │
           └───────────────┘               │ - Summarize   │
                    │                      └───────────────┘
                    ▼                               │
           ┌───────────────┐                       │
           │ Research Plan │◀──────────────────────┘
           │ - Steps       │
           │ - Dependencies│
           └───────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │OpenAlex │ │Semantic │ │ PubMed  │  (Parallel search)
   │ Search  │ │ Scholar │ │ Search  │
   └─────────┘ └─────────┘ └─────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
           ┌───────────────┐
           │   Dedup &     │
           │   Merge       │
           └───────────────┘
                    │
                    ▼
           ┌───────────────┐
           │  Reranker     │  (Cross-encoder)
           │  (top-k)      │
           └───────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Unpaywall│ │  arXiv  │ │  CORE   │  (PDF retrieval)
   │   PDF   │ │   PDF   │ │   PDF   │
   └─────────┘ └─────────┘ └─────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
           ┌───────────────┐
           │ PDF Processor │
           │ - Extract     │
           │ - Section     │
           │ - Chunk       │
           └───────────────┘
                    │
                    ▼
           ┌───────────────┐
           │ Vector Index  │
           │ - Embed       │
           │ - Store       │
           └───────────────┘
                    │
                    ▼
           ┌───────────────┐
           │ RAG Retrieval │
           │ - Hybrid      │
           │ - Rerank      │
           └───────────────┘
                    │
                    ▼
           ┌───────────────┐
           │ LLM Synthesis │
           │ - Answer      │
           │ - Citations   │
           │ - Confidence  │
           └───────────────┘
                    │
                    ▼
           ┌───────────────┐
           │ Evidence      │
           │ Verification  │
           └───────────────┘
                    │
                    ▼
              Final Answer
           with Citations
```

---

## Part 4: New Components to Add

### 4.1 Research Agent (`src/litforge/agents/research_agent.py`)

**Purpose**: Autonomous multi-step research with planning and refinement

```python
class ResearchAgent:
    """
    Autonomous research agent that plans and executes literature research.
    
    Capabilities:
    - Decompose complex queries into sub-tasks
    - Search across multiple sources
    - Retrieve and process PDFs
    - Synthesize findings with citations
    - Iteratively refine answers
    """
    
    async def research(
        self,
        query: str,
        *,
        depth: Literal["quick", "standard", "deep"] = "standard",
        max_papers: int = 20,
        require_fulltext: bool = False,
    ) -> ResearchResult:
        """
        Conduct autonomous research on a topic.
        
        Args:
            query: Research question or topic
            depth: How deep to research
                - quick: Top 5 papers, abstracts only
                - standard: Top 20 papers, full text when available
                - deep: Comprehensive search, all available full text
            max_papers: Maximum papers to analyze
            require_fulltext: Only include papers with full text
            
        Returns:
            ResearchResult with answer, evidence, and sources
        """
```

### 4.2 Evidence Extractor (`src/litforge/processors/evidence.py`)

**Purpose**: Extract evidence passages with precise citations

```python
class EvidenceExtractor:
    """
    Extract and score evidence from documents.
    
    Features:
    - Claim-evidence matching
    - Relevance scoring
    - Contradiction detection
    - Context expansion
    """
    
    def extract_evidence(
        self,
        claim: str,
        documents: list[Document],
        *,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> list[Evidence]:
        """Extract evidence passages supporting or contradicting a claim."""
    
    def detect_contradictions(
        self,
        evidence: list[Evidence],
    ) -> list[Contradiction]:
        """Identify contradicting evidence from different sources."""
```

### 4.3 Concept Extractor (`src/litforge/processors/concepts.py`)

**Purpose**: Extract and organize scientific concepts

```python
class ConceptExtractor:
    """
    Extract scientific concepts and relationships from literature.
    
    Features:
    - Named entity recognition (chemicals, genes, diseases, etc.)
    - Relationship extraction
    - Concept clustering
    - Ontology mapping
    """
    
    def extract_concepts(
        self,
        text: str,
        domain: str = "general",
    ) -> list[Concept]:
        """Extract scientific concepts from text."""
    
    def build_concept_graph(
        self,
        papers: list[Publication],
    ) -> ConceptGraph:
        """Build a concept relationship graph from papers."""
```

### 4.4 Hybrid Retriever (`src/litforge/retrieval/hybrid.py`)

**Purpose**: Combine dense and sparse retrieval for best results

```python
class HybridRetriever:
    """
    Hybrid retrieval combining BM25 + dense embeddings.
    
    Based on research showing hybrid retrieval outperforms
    pure dense or sparse approaches.
    """
    
    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.bm25 = BM25Index()
        self.dense = DenseIndex()
        self.reranker = reranker
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Hybrid search with optional reranking."""
```

### 4.5 Streaming Handler (`src/litforge/streaming/handler.py`)

**Purpose**: Real-time streaming for all operations

```python
class StreamingHandler:
    """
    Handle streaming responses for all LitForge operations.
    
    Supports:
    - SSE (Server-Sent Events)
    - WebSocket
    - AsyncIterator
    """
    
    async def stream_research(
        self,
        query: str,
    ) -> AsyncIterator[ResearchEvent]:
        """
        Stream research progress and results.
        
        Yields:
            ResearchEvent objects:
            - PlanEvent: Research plan created
            - SearchEvent: Papers found
            - RetrievalEvent: PDF downloaded
            - ProcessingEvent: Document processed
            - EvidenceEvent: Evidence extracted
            - AnswerChunk: Partial answer text
            - CompleteEvent: Final result
        """
```

### 4.6 LLM Router (`src/litforge/llm/router.py`)

**Purpose**: Smart routing across LLM providers

```python
class LLMRouter:
    """
    Intelligent LLM routing with fallback and cost optimization.
    
    Supports:
    - Groq (free, fast)
    - OpenAI (quality)
    - Anthropic (reasoning)
    - Ollama (local)
    - vLLM (self-hosted)
    """
    
    def __init__(
        self,
        primary: str = "groq",
        fallback: list[str] = ["openai", "ollama"],
        cost_limit: float | None = None,
    ):
        pass
    
    async def complete(
        self,
        prompt: str,
        *,
        task: Literal["parse", "synthesize", "reason"] = "synthesize",
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """Route to appropriate LLM based on task and availability."""
```

---

## Part 5: Enhanced Models

### 5.1 Document Model (extends Publication)

```python
@dataclass
class Document:
    """
    Rich document model with full-text and chunks.
    
    Extends Publication with:
    - Parsed sections
    - Text chunks for RAG
    - Extracted entities
    - Quality metrics
    """
    
    publication: Publication
    
    # Full text content
    full_text: str
    sections: dict[str, Section]  # intro, methods, results, discussion
    
    # Chunks for RAG
    chunks: list[DocumentChunk]
    chunk_embeddings: np.ndarray | None
    
    # Extracted content
    tables: list[Table]
    figures: list[Figure]
    equations: list[Equation]
    references: list[Reference]
    
    # Quality metrics
    extraction_quality: float  # 0-1 score
    has_ocr_errors: bool
    language: str
```

### 5.2 Evidence Model

```python
@dataclass
class Evidence:
    """
    Evidence passage with provenance.
    
    Every claim must be backed by evidence with:
    - Source document
    - Exact text
    - Location (page, section)
    - Relevance score
    - Support type (supports, contradicts, neutral)
    """
    
    text: str
    source: Publication
    
    # Location
    page: int | None
    section: str | None
    paragraph: int | None
    
    # Scoring
    relevance_score: float  # 0-1
    confidence: float  # 0-1
    support_type: Literal["supports", "contradicts", "neutral"]
    
    # Context
    context_before: str
    context_after: str
    
    def to_citation(self) -> str:
        """Generate inline citation."""
        return f"({self.source.first_author.family_name} et al., {self.source.year})"
```

### 5.3 Research Result Model

```python
@dataclass
class ResearchResult:
    """
    Complete research result with answer and evidence.
    """
    
    # The answer
    answer: str
    summary: str
    
    # Evidence
    evidence: list[Evidence]
    supporting_papers: list[Publication]
    
    # Quality metrics
    confidence: float
    evidence_quality: float
    contradictions: list[Contradiction]
    
    # Provenance
    query: str
    research_plan: ResearchPlan
    search_results: list[SearchResult]
    
    # Metadata
    total_papers_searched: int
    papers_with_fulltext: int
    processing_time: float
    llm_tokens_used: int
    estimated_cost: float
```

---

## Part 6: New Services

### 6.1 Synthesis Service (`src/litforge/services/synthesis.py`)

**Purpose**: Generate literature reviews and summaries

```python
class SynthesisService:
    """
    Synthesize findings across multiple papers.
    
    Capabilities:
    - Literature review generation
    - Comparative analysis
    - Gap identification
    - Trend analysis
    - Contradiction resolution
    """
    
    async def literature_review(
        self,
        topic: str,
        papers: list[Publication],
        *,
        style: Literal["narrative", "systematic", "scoping"] = "narrative",
        max_length: int = 2000,
    ) -> LiteratureReview:
        """Generate a literature review on a topic."""
    
    async def compare_papers(
        self,
        papers: list[Publication],
        aspects: list[str] | None = None,
    ) -> ComparisonResult:
        """Compare findings across papers."""
    
    async def identify_gaps(
        self,
        papers: list[Publication],
    ) -> list[ResearchGap]:
        """Identify research gaps in the literature."""
```

### 6.2 Concept Service (`src/litforge/services/concept.py`)

**Purpose**: Build and query concept networks

```python
class ConceptService:
    """
    Manage scientific concept extraction and organization.
    
    Capabilities:
    - Concept extraction from papers
    - Concept graph building
    - Concept clustering
    - Ontology mapping
    - Trend detection
    """
    
    async def extract_concepts(
        self,
        papers: list[Publication],
    ) -> list[Concept]:
        """Extract all concepts from papers."""
    
    async def build_concept_graph(
        self,
        papers: list[Publication],
    ) -> ConceptGraph:
        """Build concept relationship graph."""
    
    async def detect_trends(
        self,
        papers: list[Publication],
        time_window: str = "5y",
    ) -> list[Trend]:
        """Detect emerging trends in concepts."""
```

### 6.3 Export Service (`src/litforge/services/export.py`)

**Purpose**: Export to various formats

```python
class ExportService:
    """
    Export results to various formats.
    
    Formats:
    - BibTeX
    - RIS
    - EndNote XML
    - CSL-JSON
    - Markdown
    - HTML report
    - PDF report
    """
    
    def to_bibtex(self, papers: list[Publication]) -> str:
        """Export to BibTeX format."""
    
    def to_ris(self, papers: list[Publication]) -> str:
        """Export to RIS format."""
    
    def to_markdown_report(
        self,
        result: ResearchResult,
        *,
        include_evidence: bool = True,
        include_figures: bool = False,
    ) -> str:
        """Generate Markdown research report."""
```

---

## Part 7: Enhanced Features Not Previously Considered

### 7.1 🆕 Query Understanding & Expansion

```python
class QueryProcessor:
    """
    Advanced query understanding and expansion.
    
    Features:
    - Entity recognition (chemicals, genes, diseases)
    - Synonym expansion (aspirin → acetylsalicylic acid)
    - Ontology-aware expansion (CRISPR → CRISPR-Cas9, CRISPR-Cas12a)
    - Query decomposition (complex → sub-queries)
    - Intent classification (search, compare, summarize, explain)
    """
```

### 7.2 🆕 Citation Intent Classification

```python
class CitationAnalyzer:
    """
    Analyze citation intent and context.
    
    Categories:
    - Background: General context
    - Method: Uses methodology from cited work
    - Result: Compares results
    - Support: Supports claims
    - Contrast: Contrasts with claims
    - Extension: Extends cited work
    """
```

### 7.3 🆕 Quality & Reliability Scoring

```python
class QualityScorer:
    """
    Score paper and evidence quality.
    
    Metrics:
    - Venue impact factor
    - Author h-index
    - Citation velocity
    - Retraction status
    - Reproducibility indicators
    - Preprint vs peer-reviewed
    """
```

### 7.4 🆕 Multi-Modal Support

```python
class MultiModalProcessor:
    """
    Process figures, tables, and equations.
    
    Features:
    - Figure caption extraction
    - Table parsing (to structured data)
    - Equation OCR (LaTeX)
    - Chemical structure recognition
    - Graph/chart data extraction
    """
```

### 7.5 🆕 Personalization & Learning

```python
class UserProfile:
    """
    User-specific preferences and history.
    
    Features:
    - Search history
    - Favorite papers
    - Reading list
    - Concept interests
    - Citation style preference
    - Notification settings
    """
```

### 7.6 🆕 Collaboration Features

```python
class CollaborationService:
    """
    Multi-user collaboration features.
    
    Features:
    - Shared collections
    - Annotations
    - Comments
    - Highlights
    - Export sharing
    """
```

### 7.7 🆕 Monitoring & Analytics

```python
class AnalyticsService:
    """
    Usage analytics and monitoring.
    
    Metrics:
    - Query latency
    - Cache hit rate
    - API costs
    - Error rates
    - Popular queries
    - User engagement
    """
```

---

## Part 8: Revised Implementation Roadmap

### Phase 1: Core Excellence (Weeks 1-3) ⭐⭐⭐

**Goal**: Make the core search and retrieval world-class

| Task | File | Days | Priority |
|------|------|------|----------|
| Async-first architecture | `core/*.py` | 2 | ⭐⭐⭐ |
| OpenAlex client complete | `clients/openalex.py` | 1 | ⭐⭐⭐ |
| Semantic Scholar client | `clients/semantic_scholar.py` | 2 | ⭐⭐⭐ |
| Unpaywall client | `clients/unpaywall.py` | 1 | ⭐⭐⭐ |
| arXiv client | `clients/arxiv.py` | 1 | ⭐⭐⭐ |
| Multi-source search | `services/discovery.py` | 2 | ⭐⭐⭐ |
| Result deduplication | `services/discovery.py` | 1 | ⭐⭐⭐ |
| PDF waterfall retrieval | `services/retrieval.py` | 2 | ⭐⭐⭐ |
| Smart caching | `core/cache.py` | 1 | ⭐⭐ |

**Deliverable**: `forge.search()` and `forge.get_fulltext()` work reliably

### Phase 2: Document Processing (Weeks 4-5) ⭐⭐⭐

**Goal**: State-of-the-art document understanding

| Task | File | Days | Priority |
|------|------|------|----------|
| PDF text extraction | `processors/pdf.py` | 2 | ⭐⭐⭐ |
| Section detection | `processors/sections.py` | 2 | ⭐⭐⭐ |
| Smart chunking | `processors/chunking.py` | 2 | ⭐⭐⭐ |
| Table extraction | `processors/tables.py` | 2 | ⭐⭐ |
| Reference parsing | `processors/references.py` | 1 | ⭐⭐ |

**Deliverable**: Full-text papers parsed into structured chunks

### Phase 3: RAG Pipeline (Weeks 6-7) ⭐⭐⭐

**Goal**: Evidence-based Q&A with citations

| Task | File | Days | Priority |
|------|------|------|----------|
| Embedding pipeline | `embedding/pipeline.py` | 2 | ⭐⭐⭐ |
| Hybrid retriever | `retrieval/hybrid.py` | 2 | ⭐⭐⭐ |
| Cross-encoder reranker | `retrieval/reranker.py` | 2 | ⭐⭐⭐ |
| Evidence extractor | `processors/evidence.py` | 2 | ⭐⭐⭐ |
| LLM synthesis | `llm/synthesis.py` | 2 | ⭐⭐⭐ |
| Citation grounding | `processors/citations.py` | 1 | ⭐⭐⭐ |

**Deliverable**: `forge.ask()` returns answers with evidence

### Phase 4: Research Agent (Weeks 8-9) ⭐⭐⭐

**Goal**: Autonomous multi-step research

| Task | File | Days | Priority |
|------|------|------|----------|
| Research planner | `agents/planner.py` | 2 | ⭐⭐⭐ |
| Research agent | `agents/research_agent.py` | 3 | ⭐⭐⭐ |
| Iterative refinement | `agents/refinement.py` | 2 | ⭐⭐⭐ |
| Contradiction detection | `processors/contradictions.py` | 2 | ⭐⭐ |
| Confidence scoring | `processors/confidence.py` | 1 | ⭐⭐ |

**Deliverable**: `forge.research()` conducts autonomous research

### Phase 5: Citation Networks (Weeks 10-11) ⭐⭐

**Goal**: Advanced citation analysis

| Task | File | Days | Priority |
|------|------|------|----------|
| Citation graph building | `network/builder.py` | 2 | ⭐⭐⭐ |
| Influence metrics | `network/metrics.py` | 2 | ⭐⭐ |
| Co-citation clustering | `network/clustering.py` | 2 | ⭐⭐ |
| Trend detection | `network/trends.py` | 2 | ⭐⭐ |
| Visualization export | `network/visualization.py` | 1 | ⭐⭐ |

**Deliverable**: `forge.build_network()` with analysis

### Phase 6: Production Features (Weeks 12-14) ⭐⭐

**Goal**: Production-ready system

| Task | File | Days | Priority |
|------|------|------|----------|
| Streaming support | `streaming/handler.py` | 2 | ⭐⭐⭐ |
| LLM router | `llm/router.py` | 2 | ⭐⭐⭐ |
| REST API | `api/rest.py` | 3 | ⭐⭐ |
| CLI commands | `cli.py` | 2 | ⭐⭐ |
| Export formats | `services/export.py` | 2 | ⭐⭐ |
| Error handling | `core/errors.py` | 1 | ⭐⭐ |
| Logging/metrics | `core/observability.py` | 1 | ⭐⭐ |

**Deliverable**: Production-deployable system

---

## Part 9: New Directory Structure

```
src/litforge/
├── __init__.py                 # Public API
├── api.py                      # Simple API (sync)
├── async_api.py                # NEW: Async API
├── cli.py                      # CLI commands
├── config.py                   # Configuration
│
├── agents/                     # NEW: Agentic components
│   ├── __init__.py
│   ├── base.py                 # Base agent class
│   ├── research_agent.py       # Main research agent
│   ├── planner.py              # Research planner
│   └── refinement.py           # Answer refinement
│
├── api/                        # NEW: REST API
│   ├── __init__.py
│   ├── app.py                  # FastAPI app
│   ├── routes/
│   │   ├── search.py
│   │   ├── papers.py
│   │   ├── qa.py
│   │   └── research.py
│   └── models.py               # API models
│
├── clients/                    # External API clients
│   ├── __init__.py
│   ├── base.py
│   ├── openalex.py             # ENHANCE
│   ├── semantic_scholar.py     # IMPLEMENT
│   ├── crossref.py             # ENHANCE
│   ├── pubmed.py               # IMPLEMENT
│   ├── arxiv.py                # IMPLEMENT
│   ├── unpaywall.py            # IMPLEMENT
│   └── core_api.py             # NEW: CORE API
│
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── forge.py                # Main Forge class
│   ├── cache.py                # NEW: Smart caching
│   ├── errors.py               # NEW: Error types
│   └── observability.py        # NEW: Logging/metrics
│
├── embedding/                  # Embedding providers
│   ├── __init__.py
│   ├── base.py
│   ├── openai.py
│   ├── sentence_transformers.py
│   └── cohere.py               # NEW
│
├── integrations/               # Framework integrations
│   ├── __init__.py
│   ├── crewai.py
│   ├── langchain.py
│   ├── langgraph.py
│   ├── autogen.py              # NEW
│   └── llamaindex.py           # NEW
│
├── llm/                        # LLM providers
│   ├── __init__.py
│   ├── base.py
│   ├── router.py               # NEW: Smart routing
│   ├── openai.py
│   ├── anthropic.py            # NEW
│   ├── groq.py                 # NEW
│   ├── ollama.py               # NEW
│   └── synthesis.py            # NEW: Answer synthesis
│
├── mcp/                        # MCP server
│   ├── __init__.py
│   └── server.py
│
├── models/                     # Data models
│   ├── __init__.py
│   ├── publication.py
│   ├── document.py             # NEW: Rich document
│   ├── evidence.py             # NEW: Evidence model
│   ├── network.py
│   ├── search.py
│   ├── research.py             # NEW: Research result
│   └── concepts.py             # NEW: Concept model
│
├── network/                    # NEW: Citation networks
│   ├── __init__.py
│   ├── builder.py
│   ├── metrics.py
│   ├── clustering.py
│   ├── trends.py
│   └── visualization.py
│
├── processors/                 # NEW: Document processing
│   ├── __init__.py
│   ├── pdf.py                  # PDF extraction
│   ├── sections.py             # Section detection
│   ├── chunking.py             # Smart chunking
│   ├── tables.py               # Table extraction
│   ├── references.py           # Reference parsing
│   ├── evidence.py             # Evidence extraction
│   ├── concepts.py             # Concept extraction
│   ├── citations.py            # Citation grounding
│   ├── contradictions.py       # Contradiction detection
│   └── confidence.py           # Confidence scoring
│
├── retrieval/                  # NEW: Advanced retrieval
│   ├── __init__.py
│   ├── hybrid.py               # Hybrid search
│   ├── reranker.py             # Cross-encoder
│   └── query.py                # Query processing
│
├── services/                   # Business logic
│   ├── __init__.py
│   ├── discovery.py            # ENHANCE
│   ├── retrieval.py            # IMPLEMENT
│   ├── citation.py             # IMPLEMENT
│   ├── knowledge.py            # IMPLEMENT
│   ├── qa.py                   # IMPLEMENT
│   ├── synthesis.py            # NEW
│   ├── concept.py              # NEW
│   └── export.py               # NEW
│
├── stores/                     # Vector stores
│   ├── __init__.py
│   ├── base.py
│   ├── chromadb.py
│   ├── faiss.py
│   ├── qdrant.py
│   └── pgvector.py             # NEW
│
├── streaming/                  # NEW: Streaming support
│   ├── __init__.py
│   ├── handler.py
│   └── events.py
│
└── ui/                         # Web interfaces
    ├── __init__.py
    ├── app.py
    └── chat.py
```

---

## Part 10: Dependencies to Add

```toml
[project]
dependencies = [
    # Existing
    "httpx>=0.27",
    "pydantic>=2.0",
    "chromadb>=0.4",
    
    # API Clients
    "pyalex>=0.13",              # OpenAlex
    "semanticscholar>=0.8",       # Semantic Scholar
    "arxiv>=2.1",                 # arXiv
    "biopython>=1.83",            # PubMed via Entrez
    
    # PDF Processing
    "pypdf>=4.0",                 # Basic PDF
    "pymupdf>=1.24",              # Advanced PDF
    "pdfplumber>=0.11",           # Table extraction
    
    # Embeddings
    "sentence-transformers>=2.7", # Local embeddings
    "openai>=1.40",               # OpenAI embeddings
    
    # RAG
    "rank-bm25>=0.2",             # BM25 sparse retrieval
    "faiss-cpu>=1.8",             # FAISS vector search
    
    # LLM
    "groq>=0.9",                  # Groq (free tier)
    "anthropic>=0.34",            # Claude
    "ollama>=0.3",                # Local LLMs
    
    # Graphs
    "networkx>=3.3",              # Citation networks
    "pyvis>=0.3",                 # Visualization
    
    # API
    "fastapi>=0.111",             # REST API
    "uvicorn>=0.30",              # ASGI server
    
    # Utilities
    "tenacity>=8.5",              # Retry logic
    "cachetools>=5.5",            # Caching
    "structlog>=24.4",            # Structured logging
]
```

---

## Part 11: Success Metrics

### Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Search relevance | >90% precision@10 | Human evaluation |
| PDF retrieval | >70% OA success | Automated testing |
| Evidence accuracy | >95% correct citations | LLM evaluation |
| Answer quality | Match PaperQA2 | DeepResearchGym |
| Hallucination rate | <5% | LLM-as-judge |

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Search latency | <2s | p95 |
| PDF retrieval | <10s | p95 |
| RAG Q&A | <15s | p95 |
| Research (deep) | <120s | p95 |
| Streaming TTFB | <500ms | p95 |

### Cost Metrics

| Operation | Target Cost | Breakdown |
|-----------|-------------|-----------|
| Search | $0.00 | Free APIs |
| PDF retrieval | $0.00 | Free sources |
| Simple Q&A | $0.01 | Groq free tier |
| Deep research | $0.10 | 10 LLM calls |

---

## Conclusion

This enhanced plan transforms LitForge from a basic search library into a **state-of-the-art research agent platform** that can:

1. **Search** across 6+ sources with intelligent fusion
2. **Retrieve** PDFs with 70%+ success rate
3. **Process** documents into structured, searchable knowledge
4. **Answer** questions with evidence-grounded citations
5. **Research** autonomously with multi-step reasoning
6. **Analyze** citation networks and detect trends
7. **Integrate** seamlessly with multi-agent systems

The 14-week roadmap prioritizes:
- **Weeks 1-7**: Core excellence (search, retrieval, RAG)
- **Weeks 8-11**: Advanced features (agents, networks)
- **Weeks 12-14**: Production readiness

**Recommended Immediate Actions**:
1. Implement Semantic Scholar client (best citation data)
2. Implement Unpaywall client (unlocks PDFs)
3. Add async architecture (enables parallel operations)
4. Build hybrid retrieval (better search quality)

---

*Document Version: 2.0*  
*Last Updated: January 2025*
