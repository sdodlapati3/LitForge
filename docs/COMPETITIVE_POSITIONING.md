# LitForge: Competitive Positioning & Unique Value

## Why Build LitForge?

### The Problem with Existing Solutions

| Solution | Problem for Us |
|----------|---------------|
| **PaperQA2** | Opinionated architecture, hard to customize, API costs |
| **GPT-Researcher** | Report-focused, not a library, web-centric |
| **Elicit** | Closed source, SaaS only, no self-hosting |
| **Semantic Scholar API** | Read-only API, no RAG, no agents |
| **LlamaIndex** | Too general, heavy dependencies, learning curve |

### LitForge's Unique Value Proposition

```
┌─────────────────────────────────────────────────────────────────┐
│                  LitForge Unique Advantages                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. LIBRARY-FIRST                                                │
│     • import litforge; litforge.search("query")                  │
│     • No servers, no setup, just import                          │
│     • Designed for embedding in other systems                    │
│                                                                  │
│  2. AGENT-READY                                                  │
│     • MCP server for Claude                                      │
│     • CrewAI, LangChain, LangGraph tools                         │
│     • Autonomous research agent built-in                         │
│                                                                  │
│  3. COST-OPTIMIZED                                               │
│     • Free APIs first (OpenAlex, Unpaywall)                      │
│     • Groq free tier for LLM                                     │
│     • $0.00 - $0.10 per research query                           │
│                                                                  │
│  4. SELF-HOSTABLE                                                │
│     • 100% open source                                           │
│     • Local LLM support (Ollama)                                 │
│     • No vendor lock-in                                          │
│                                                                  │
│  5. MODULAR                                                      │
│     • Use just search, or full RAG                               │
│     • Swap vector stores (ChromaDB, FAISS, Qdrant)               │
│     • Swap LLM providers freely                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Target Use Cases

### 1. AI Agent Integration

```python
# CrewAI Example
from crewai import Agent, Task
from litforge.integrations.crewai import LitForgeSearchTool, LitForgeQATool

research_agent = Agent(
    role="Research Scientist",
    tools=[LitForgeSearchTool(), LitForgeQATool()],
    goal="Find evidence for drug-target interactions"
)
```

### 2. Research Automation

```python
from litforge import Forge

forge = Forge()

# Autonomous deep research
result = await forge.research(
    "What are the mechanisms of action of GLP-1 receptor agonists?",
    depth="deep",
    max_papers=50
)

print(result.answer)  # Synthesized answer
print(result.evidence)  # Supporting evidence with citations
print(result.confidence)  # Confidence score
```

### 3. Knowledge Base Building

```python
# Build a domain-specific knowledge base
papers = forge.search("CRISPR delivery mechanisms", limit=100)
fulltext_papers = await forge.get_fulltext_batch(papers)

kb = forge.create_knowledge_base("crispr-delivery")
await kb.index(fulltext_papers)

# Query the knowledge base
answer = await kb.ask("What delivery methods work best for in vivo CRISPR?")
```

### 4. Citation Network Analysis

```python
# Build citation network from seed papers
network = await forge.build_network(
    seed_dois=["10.1126/science.aax5077"],
    depth=2,
    direction="both"  # forward and backward
)

# Analyze
influential = network.most_influential(n=10)
clusters = network.find_clusters()
trends = network.detect_trends(window="5y")
```

---

## Feature Comparison Matrix

| Feature | LitForge | PaperQA2 | GPT-Researcher | Elicit |
|---------|----------|----------|----------------|--------|
| **Simplicity** | | | | |
| One-liner API | ✅ | ❌ | ❌ | ❌ |
| Zero config start | ✅ | ⚠️ | ⚠️ | ✅ |
| Library (not service) | ✅ | ✅ | ❌ | ❌ |
| **Data Sources** | | | | |
| OpenAlex (250M) | ✅ | ❌ | ❌ | ✅ |
| Semantic Scholar | ✅ | ✅ | ❌ | ✅ |
| PubMed | ✅ | ❌ | ❌ | ✅ |
| arXiv | ✅ | ❌ | ✅ | ✅ |
| CrossRef | ✅ | ✅ | ❌ | ❌ |
| Web search | ❌ | ❌ | ✅ | ❌ |
| **PDF Retrieval** | | | | |
| Unpaywall | ✅ | ✅ | ❌ | ✅ |
| arXiv direct | ✅ | ❌ | ✅ | ✅ |
| CORE API | ✅ | ❌ | ❌ | ❌ |
| PMC | ✅ | ❌ | ❌ | ✅ |
| **Document Processing** | | | | |
| Section detection | ✅ | ✅ | ❌ | ✅ |
| Table extraction | ✅ | ✅ | ❌ | ⚠️ |
| Smart chunking | ✅ | ✅ | ⚠️ | ✅ |
| **RAG & Q&A** | | | | |
| Hybrid retrieval | ✅ | ✅ | ❌ | ❌ |
| Cross-encoder rerank | ✅ | ✅ | ❌ | ❌ |
| Evidence extraction | ✅ | ✅ | ⚠️ | ✅ |
| Citation grounding | ✅ | ✅ | ⚠️ | ✅ |
| Contradiction detect | ✅ | ✅ | ❌ | ❌ |
| **Agent Features** | | | | |
| Autonomous research | ✅ | ✅ | ✅ | ❌ |
| Multi-step planning | ✅ | ✅ | ✅ | ❌ |
| Iterative refinement | ✅ | ✅ | ✅ | ❌ |
| **Citation Networks** | | | | |
| Graph building | ✅ | ❌ | ❌ | ⚠️ |
| Influence metrics | ✅ | ❌ | ❌ | ❌ |
| Trend detection | ✅ | ❌ | ❌ | ❌ |
| **Integration** | | | | |
| MCP server | ✅ | ❌ | ❌ | ❌ |
| CrewAI tools | ✅ | ❌ | ❌ | ❌ |
| LangChain tools | ✅ | ❌ | ✅ | ❌ |
| REST API | ✅ | ❌ | ✅ | ✅ |
| **Cost** | | | | |
| Free tier viable | ✅ | ⚠️ | ❌ | ❌ |
| Local LLM support | ✅ | ✅ | ⚠️ | ❌ |
| Self-hostable | ✅ | ✅ | ✅ | ❌ |

---

## Architecture Principles

### 1. Async-First, Sync-Available

```python
# Async for performance
async with Forge() as forge:
    papers = await forge.search_async("query")

# Sync for simplicity
papers = litforge.search("query")
```

### 2. Progressive Enhancement

```python
# Level 1: Simple search (no deps)
papers = litforge.search("CRISPR")

# Level 2: With full text (needs PDF libs)
papers = forge.search("CRISPR", include_fulltext=True)

# Level 3: With RAG (needs embeddings + LLM)
answer = forge.ask("How does CRISPR work?")

# Level 4: Deep research (full stack)
result = forge.research("CRISPR delivery mechanisms", depth="deep")
```

### 3. Fail Gracefully

```python
# If Semantic Scholar is down, fall back to OpenAlex
# If Unpaywall fails, try CORE
# If OpenAI fails, use Groq
# If all LLMs fail, return raw search results
```

### 4. Explicit Provenance

```python
# Every answer includes its evidence chain
answer = forge.ask("What are the side effects of X?")

for evidence in answer.evidence:
    print(f"[{evidence.source.first_author}, {evidence.source.year}]")
    print(f"  '{evidence.text}'")
    print(f"  Page {evidence.page}, Section: {evidence.section}")
    print(f"  Relevance: {evidence.relevance_score:.2f}")
```

---

## Quick Start Examples

### Minimal Example

```python
import litforge

# That's it - search 250M+ papers
papers = litforge.search("transformer attention mechanisms")
for p in papers[:5]:
    print(f"{p.title} ({p.year}) - {p.citations} citations")
```

### Full Research Example

```python
from litforge import Forge

async def main():
    forge = Forge(
        llm="groq",  # Free!
        embeddings="local",  # sentence-transformers
    )
    
    # Conduct autonomous research
    result = await forge.research(
        query="What are the latest advances in DNA origami for drug delivery?",
        depth="standard",
        max_papers=30,
    )
    
    print("## Answer")
    print(result.answer)
    
    print("\n## Key Findings")
    for i, evidence in enumerate(result.evidence[:5], 1):
        print(f"{i}. {evidence.text[:100]}...")
        print(f"   - {evidence.source.first_author} et al. ({evidence.source.year})")
    
    print(f"\n## Confidence: {result.confidence:.0%}")
    print(f"## Papers analyzed: {result.total_papers_searched}")
    print(f"## Full-text available: {result.papers_with_fulltext}")
    print(f"## Estimated cost: ${result.estimated_cost:.3f}")

asyncio.run(main())
```

### Agent Integration Example

```python
from crewai import Agent, Task, Crew
from litforge.integrations.crewai import (
    LitForgeSearchTool,
    LitForgeQATool,
    LitForgeCitationTool,
)

# Define a literature research agent
literature_agent = Agent(
    role="Scientific Literature Expert",
    goal="Find and synthesize scientific evidence",
    backstory="Expert at navigating scientific literature...",
    tools=[
        LitForgeSearchTool(),
        LitForgeQATool(),
        LitForgeCitationTool(),
    ],
)

# Use in a crew
research_task = Task(
    description="Find evidence for {drug} efficacy in {disease}",
    agent=literature_agent,
    expected_output="Summary with citations"
)

crew = Crew(
    agents=[literature_agent],
    tasks=[research_task],
)

result = crew.kickoff(inputs={
    "drug": "semaglutide",
    "disease": "obesity"
})
```

---

## Differentiation Summary

| Aspect | Our Approach | Why It Matters |
|--------|--------------|----------------|
| **Library-first** | Import and use immediately | No deployment overhead |
| **Free-tier viable** | OpenAlex + Groq = $0 | Accessible to everyone |
| **Agent-native** | MCP, CrewAI, LangChain built-in | Ready for AI workflows |
| **Evidence-grounded** | Every claim has a citation | Trustworthy outputs |
| **Modular** | Swap any component | No vendor lock-in |
| **Self-hostable** | 100% open source | Full control |

---

*LitForge: Forging Knowledge from Literature*
