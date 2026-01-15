"""
LitForge Chat Interface - Conversational literature search.

This provides a chat-based interface where users can:
- Search with natural language
- Ask follow-up questions
- Request downloads, summaries, etc.
"""

import streamlit as st
import json
import os
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent.parent.parent / ".env"
_env_loaded = load_dotenv(_env_path)

from litforge.api import search, lookup, citations, references, Paper

# Note: Ensemble scoring is handled inside rag_search.py and semantic_search.py
try:
    from litforge.services.scoring import EnsembleScorer
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

# Try to import RAG search pipeline
try:
    from litforge.services.rag_search import rag_search_pipeline, scout_search
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

# Try to import semantic search (LLM-first approach)
try:
    from litforge.services.semantic_search import semantic_search
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False

# Try to import LLM for semantic search
try:
    from litforge.llm.router import get_llm
    # Check if ANY LLM provider is available
    HAS_LLM = any([
        os.environ.get("CEREBRAS_API_KEY"),
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("GITHUB_TOKEN"),
        os.environ.get("OPENAI_API_KEY"),
    ])
    _llm_debug = f"HAS_LLM={HAS_LLM}, env_loaded={_env_loaded}, env_path={_env_path}"
except ImportError as e:
    HAS_LLM = False
    _llm_debug = f"Import error: {e}"

# Note: LLM-powered query expansion is now handled by semantic_search.py
# The understand_query() function in semantic_search.py provides LLM-based
# query understanding with proper disambiguation (e.g., "liquid foundation models" → Liquid Neural Networks)


def parse_user_intent(message: str, has_results: bool = False) -> dict:
    """
    Parse user message to understand intent.
    Returns: {"intent": str, "params": dict}
    
    Intents:
    - search: User wants to find papers
    - download: User wants to export papers
    - pdf_request: User wants actual PDF files (needs explanation)
    - show_more: User wants to see more results
    - lookup: User wants to look up a specific DOI
    - citations: User wants to find citing papers
    - help: User needs help
    - clarify: Need more info
    """
    import re
    msg_lower = message.lower().strip()
    
    # Show more results intent
    show_more_keywords = ["show more", "see more", "show all", "see all", "rest of", "remaining", "all papers", "more papers", "view all"]
    if any(kw in msg_lower for kw in show_more_keywords):
        return {"intent": "show_more", "params": {}}
    
    # PDF download intent - needs special handling
    pdf_keywords = ["pdf", "pdfs", "pdf file", "full text", "fulltext", "full-text"]
    if any(kw in msg_lower for kw in pdf_keywords):
        # Check if they're asking for actual PDF files
        num_match = re.search(r'(first|top)?\s*(\d+)', msg_lower)
        num = int(num_match.group(2)) if num_match else 5
        return {"intent": "pdf_request", "params": {"num": num}}
    
    # Download/export intent (metadata exports)
    download_keywords = ["download", "export", "save", "csv", "bibtex", "json", "bib"]
    if any(kw in msg_lower for kw in download_keywords):
        # Extract number if present
        num_match = re.search(r'(first|top)?\s*(\d+)', msg_lower)
        num = int(num_match.group(2)) if num_match else 10
        
        # Determine format
        if "csv" in msg_lower:
            fmt = "csv"
        elif "bibtex" in msg_lower or "bib" in msg_lower:
            fmt = "bibtex"
        elif "json" in msg_lower:
            fmt = "json"
        else:
            fmt = "csv"  # default
        
        return {"intent": "download", "params": {"num": num, "format": fmt}}
    
    # Lookup DOI intent
    doi_patterns = [r'10\.\d{4,}/[^\s]+', r'doi[:\s]+']
    for pattern in doi_patterns:
        if re.search(pattern, msg_lower):
            doi_match = re.search(r'10\.\d{4,}/[^\s]+', message)
            if doi_match:
                return {"intent": "lookup", "params": {"doi": doi_match.group()}}
    
    # Citations intent
    if "citing" in msg_lower or "citations" in msg_lower or "who cited" in msg_lower:
        return {"intent": "citations", "params": {}}
    
    # Help intent
    if msg_lower in ["help", "?", "what can you do", "commands"]:
        return {"intent": "help", "params": {}}
    
    # Summary intent
    if "summarize" in msg_lower or "summary" in msg_lower:
        return {"intent": "summarize", "params": {}}
    
    # Default: treat as search query
    # IMPORTANT: Pass raw query to RAG pipeline - let LLM interpret natural language
    # Only do minimal cleanup (punctuation, whitespace)
    cleaned = re.sub(r"['\"\?!]+", " ", msg_lower)  # Remove quotes/punctuation
    cleaned = " ".join(cleaned.split()).strip()
    
    if cleaned:
        return {"intent": "search", "params": {"query": cleaned}}
    else:
        return {"intent": "clarify", "params": {}}


def main():
    st.set_page_config(
        page_title="LitForge Chat",
        page_icon="🔥",
        layout="wide",
    )
    
    st.title("🔥 LitForge Chat")
    st.markdown("**Conversational Literature Search** - Ask me anything about scientific papers!")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        max_results = st.slider("Max Results", 5, 50, 25)
        year_from = st.number_input("Year From", 1900, 2026, 2018)
        year_to = st.number_input("Year To", 1900, 2026, 2026)
        
        st.markdown("---")
        
        # LLM-powered search toggle
        if HAS_LLM:
            st.markdown("### 🧠 AI-Powered Search")
            use_llm = st.toggle("Enable Smart Query Expansion", value=True, 
                               help="Use AI to understand your query and find better results")
            if "use_llm_search" not in st.session_state:
                st.session_state.use_llm_search = use_llm
            else:
                st.session_state.use_llm_search = use_llm
            
            if use_llm:
                st.caption("✅ AI expands queries like 'liquid foundation models' → 'Liquid Neural Networks'")
            
            # Semantic Search toggle (best practice: LLM-first)
            if HAS_SEMANTIC:
                st.markdown("### 🎯 Semantic Search (Best)")
                use_semantic = st.toggle("LLM-First + Embeddings", value=True,
                                        help="Best practice: LLM understands query, embedding-based retrieval")
                st.session_state.use_semantic_search = use_semantic
                if use_semantic:
                    st.caption("✅ LLM interprets → Find seeds → Embedding recommendations")
            
            # Ensemble scoring info
            if HAS_ENSEMBLE:
                st.markdown("### 📊 Smart Ranking")
                st.caption("Ensemble scoring: keyword + embedding + citations + recency")
                use_verify = st.toggle("🔬 LLM Verify (if uncertain)", value=True,
                                      help="Use LLM to verify results only when ranking is uncertain")
                st.session_state.use_llm_verify = use_verify
                if use_verify:
                    st.caption("✅ LLM verifies when uncertainty > 25%")
            
            # RAG Search toggle (fallback)
            if HAS_RAG:
                st.markdown("### 🔍 RAG Search (Fallback)")
                use_rag = st.toggle("Enable Scout + Multi-API", value=True,
                                   help="Fallback: Scout search for context, then search across multiple APIs")
                st.session_state.use_rag_search = use_rag
                if use_rag:
                    st.caption("✅ Scout → RAG → OpenAlex + S2 + arXiv")
            
            # Show available providers
            providers = []
            if os.environ.get("CEREBRAS_API_KEY"): providers.append("Cerebras")
            if os.environ.get("GROQ_API_KEY"): providers.append("Groq")
            if os.environ.get("GOOGLE_API_KEY"): providers.append("Gemini")
            if os.environ.get("GITHUB_TOKEN"): providers.append("GitHub")
            if os.environ.get("OPENAI_API_KEY"): providers.append("OpenAI")
            
            if providers:
                st.caption(f"🔗 Providers: {', '.join(providers)}")
            
            # Debug info
            with st.expander("🔧 Debug Info"):
                st.code(_llm_debug)
                if st.session_state.get("_scoring_meta"):
                    st.json(st.session_state["_scoring_meta"])
                if st.session_state.get("_semantic_meta"):
                    st.markdown("**Semantic Search:**")
                    st.json(st.session_state["_semantic_meta"])
                if st.session_state.get("_rag_meta"):
                    st.markdown("**RAG Pipeline:**")
                    st.json(st.session_state["_rag_meta"])
        else:
            st.warning("⚠️ Set API keys in .env to enable AI-powered search")
            st.caption(f"Debug: {_llm_debug}")
        
        st.markdown("---")
        st.markdown("### 💬 Example Commands")
        st.markdown("""
        - `Find papers on CRISPR`
        - `Download first 10 as CSV`
        - `Export as BibTeX`
        - `Look up 10.1038/nature14539`
        - `Who cited this paper?`
        """)
    
    # Initialize chat history and results
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "papers" not in st.session_state:
        st.session_state.papers = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "show_all" not in st.session_state:
        st.session_state.show_all = {}  # Track which message indices show all papers
    
    # Display chat history
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "papers" in msg and msg["papers"]:
                papers_list = msg["papers"]
                show_all_key = f"msg_{msg_idx}"
                
                # Determine how many to show
                if st.session_state.show_all.get(show_all_key, False) or len(papers_list) <= 5:
                    display_papers_compact(papers_list)
                else:
                    display_papers_compact(papers_list[:5])
                    if st.button(f"📄 Show all {len(papers_list)} papers", key=f"show_{msg_idx}"):
                        st.session_state.show_all[show_all_key] = True
                        st.rerun()
            if "download" in msg:
                st.download_button(
                    msg["download"]["label"],
                    msg["download"]["data"],
                    msg["download"]["filename"],
                    msg["download"]["mime"],
                    key=f"dl_{msg['download']['filename']}"
                )
    
    # Chat input
    if prompt := st.chat_input("Ask me about scientific literature..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Parse intent
        intent = parse_user_intent(prompt, has_results=bool(st.session_state.papers))
        
        # Process intent
        with st.chat_message("assistant"):
            if intent["intent"] == "search":
                # Pass the FULL prompt to LLM - let it handle query understanding
                # This is the best practice: LLM-first, no regex parsing
                query = prompt  # Use full natural language prompt
                papers = []
                expansion_error = None  # Track expansion errors for fallback messaging
                
                # Priority 1: Semantic Search (LLM-first + embedding retrieval)
                # This is the best practice approach used by Elicit.ai / Consensus.app
                use_semantic = st.session_state.get("use_semantic_search", True) and HAS_SEMANTIC and HAS_LLM
                
                if use_semantic:
                    with st.spinner("🧠 Understanding your query (LLM-first)..."):
                        try:
                            llm = get_llm() if HAS_LLM else None
                            papers, semantic_meta = semantic_search(
                                query=query,  # Full natural language query
                                llm=llm,
                                max_results=max_results,
                                use_recommendations=True,  # Embedding-based retrieval
                                use_keyword_fallback=True,
                            )
                            
                            # Show understanding info
                            understanding = semantic_meta.get("understanding", {})
                            if not understanding.get("fallback"):
                                st.info(f"🎯 **Understood**: {understanding.get('explanation', understanding.get('intent', 'Searching...'))}")
                                
                                topic = understanding.get("topic")
                                if topic:
                                    st.caption(f"📚 Topic: {topic}")
                                
                                terms = understanding.get("search_terms", [])
                                if terms:
                                    st.caption(f"🔎 Search terms: {', '.join(terms[:4])}")
                                
                                seed_count = semantic_meta.get("seed_papers_found", 0)
                                rec_count = semantic_meta.get("recommendations", 0)
                                if seed_count > 0 or rec_count > 0:
                                    st.caption(f"🔗 Found {seed_count} seed papers → {rec_count} recommendations (embedding-based)")
                            
                            # Show scoring info
                            scoring = semantic_meta.get("scoring", {})
                            if scoring.get("llm_verified"):
                                st.success("🔬 **LLM Verified**: Results verified for accuracy")
                            
                            # Store metadata for debug
                            st.session_state["_semantic_meta"] = semantic_meta
                            
                        except Exception as e:
                            import traceback
                            st.warning(f"⚠️ Semantic search error: {e}")
                            st.caption(f"Falling back to RAG pipeline...")
                            papers = []
                
                # Priority 2: RAG pipeline (if semantic search not used or failed)
                use_rag = st.session_state.get("use_rag_search", True) and HAS_RAG and HAS_LLM
                
                if not papers and use_rag:
                    with st.spinner("🔍 Scout search + RAG-enhanced expansion..."):
                        try:
                            llm = get_llm() if HAS_LLM else None
                            papers, rag_meta = rag_search_pipeline(
                                query=query,  # Use extracted query, not full prompt
                                llm=llm,
                                max_results=max_results,
                                year_from=year_from,
                                year_to=year_to,
                                use_scout=True,
                                use_multi_api=True,
                            )
                            
                            # Show RAG info
                            if rag_meta.get("rag_enhanced"):
                                exp = rag_meta.get("expansion", {})
                                provider = exp.get("provider", "AI")
                                provider_emoji = {"cerebras": "⚡", "groq": "🚀", "google": "🔮", "github": "🐙", "openai": "💚"}.get(provider, "🧠")
                                st.info(f"{provider_emoji} **RAG-Enhanced Search**: {exp.get('explanation', 'Expanding with context...')}")
                                
                                scout_count = rag_meta.get("scout_results", 0)
                                if scout_count > 0:
                                    st.caption(f"📡 Scout found {scout_count} related papers for context")
                                
                                queries = exp.get("queries", [])
                                if queries:
                                    st.caption(f"🔎 Queries: {', '.join(queries[:3])}")
                            
                            # Show scoring info
                            scoring = rag_meta.get("scoring", {})
                            if scoring.get("llm_verified"):
                                st.success("🔬 **LLM Verified**: Results verified for accuracy")
                            else:
                                uncertainty = scoring.get("uncertainty", {})
                                avg_unc = uncertainty.get("avg_uncertainty", 0)
                                if avg_unc < 0.15:
                                    st.caption(f"✅ High confidence ranking (uncertainty: {avg_unc:.0%})")
                            
                            # Store metadata for debug
                            st.session_state["_rag_meta"] = rag_meta
                            
                        except Exception as e:
                            st.warning(f"⚠️ RAG pipeline error: {e}, falling back to basic search")
                            papers = []
                
                # Simple fallback to basic search if AI search methods failed
                if not papers:
                    st.caption("ℹ️ Using basic keyword search...")
                    with st.spinner(f"Searching for '{query}'..."):
                        papers = search(query, limit=max_results, year_from=year_from, year_to=year_to)
                
                if papers:
                    st.session_state.papers = papers
                    st.session_state.last_query = query
                    msg_idx = len(st.session_state.messages)  # Index for this new message
                    show_all_key = f"msg_{msg_idx}"
                    
                    response = f"Found **{len(papers)} papers** on '{query}'!"
                    st.markdown(response)
                    
                    # Show all papers - use session state to track expansion
                    if len(papers) <= 5 or st.session_state.show_all.get(show_all_key, False):
                        display_papers_compact(papers)
                    else:
                        display_papers_compact(papers[:5])
                        if st.button(f"📄 Show all {len(papers)} papers", key=f"search_{msg_idx}"):
                            st.session_state.show_all[show_all_key] = True
                            st.rerun()
                    
                    st.markdown("\n💡 *'Download as CSV' or 'Export BibTeX' to save*")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "papers": papers
                    })
                else:
                    response = f"No papers found for '{query}'. Try different keywords?"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            elif intent["intent"] == "download":
                if not st.session_state.papers:
                    response = "No papers to download yet. Search for papers first!"
                    st.markdown(response)
                else:
                    num = min(intent["params"]["num"], len(st.session_state.papers))
                    fmt = intent["params"]["format"]
                    export_papers = st.session_state.papers[:num]
                    
                    if fmt == "csv":
                        data = papers_to_csv(export_papers)
                        mime = "text/csv"
                        ext = "csv"
                    elif fmt == "bibtex":
                        data = papers_to_bibtex(export_papers)
                        mime = "text/plain"
                        ext = "bib"
                    else:
                        data = papers_to_json(export_papers)
                        mime = "application/json"
                        ext = "json"
                    
                    filename = f"litforge_{num}papers_{datetime.now().strftime('%Y%m%d')}.{ext}"
                    response = f"Here are {num} papers as {fmt.upper()}:"
                    st.markdown(response)
                    st.download_button(
                        f"📥 Download {filename}",
                        data,
                        filename,
                        mime,
                        key=f"chat_dl_{datetime.now().timestamp()}"
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "download": {
                            "label": f"📥 Download {filename}",
                            "data": data,
                            "filename": filename,
                            "mime": mime
                        }
                    })
            
            elif intent["intent"] == "pdf_request":
                # User wants actual PDF files - explain we can't do that
                num = intent["params"]["num"]
                if not st.session_state.papers:
                    response = "No papers to download yet. Search for papers first!"
                    st.markdown(response)
                else:
                    response = """⚠️ **I can't download actual PDF files** - most academic papers are behind paywalls (publishers like Elsevier, Springer, etc.).

**But here's what I CAN do:**

1. 🔗 **Direct links** - Each paper title links to its DOI page where you can access it (if you have institutional access)

2. 📥 **Export metadata** - I can give you the titles, authors, DOIs, and abstracts as CSV, BibTeX, or JSON

**Your options:**"""
                    st.markdown(response)
                    
                    # Show DOI links for requested papers
                    export_papers = st.session_state.papers[:min(num, len(st.session_state.papers))]
                    st.markdown(f"\n**🔗 DOI links for first {len(export_papers)} papers:**")
                    for i, p in enumerate(export_papers, 1):
                        if p.doi:
                            st.markdown(f"{i}. [{p.title[:60]}...](https://doi.org/{p.doi})")
                        else:
                            st.markdown(f"{i}. {p.title[:60]}... *(no DOI available)*")
                    
                    st.markdown("\n**📥 Or export metadata:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            "📄 CSV",
                            papers_to_csv(export_papers),
                            f"litforge_{len(export_papers)}papers.csv",
                            "text/csv",
                            key=f"pdf_csv_{datetime.now().timestamp()}"
                        )
                    with col2:
                        st.download_button(
                            "📚 BibTeX", 
                            papers_to_bibtex(export_papers),
                            f"litforge_{len(export_papers)}papers.bib",
                            "text/plain",
                            key=f"pdf_bib_{datetime.now().timestamp()}"
                        )
                    with col3:
                        st.download_button(
                            "🔧 JSON",
                            papers_to_json(export_papers),
                            f"litforge_{len(export_papers)}papers.json",
                            "application/json",
                            key=f"pdf_json_{datetime.now().timestamp()}"
                        )
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            elif intent["intent"] == "show_more":
                # User wants to see more results
                if not st.session_state.papers:
                    response = "No papers to show. Search for papers first!"
                    st.markdown(response)
                else:
                    response = f"**All {len(st.session_state.papers)} papers** from your search:"
                    st.markdown(response)
                    display_papers_compact(st.session_state.papers)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "papers": st.session_state.papers
                    })
            
            elif intent["intent"] == "lookup":
                doi = intent["params"]["doi"]
                with st.spinner(f"Looking up {doi}..."):
                    paper = lookup(doi)
                
                if paper:
                    response = f"Found paper:"
                    st.markdown(response)
                    st.markdown(f"**{paper.title}**")
                    st.markdown(f"*{', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}* ({paper.year})")
                    st.markdown(f"Citations: {paper.citations:,}")
                    if paper.abstract:
                        st.markdown(f"**Abstract:** {paper.abstract[:300]}...")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = f"Paper not found for DOI: {doi}"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            elif intent["intent"] == "help":
                response = """
**I can help you with:**

🔍 **Search**: "Find papers on transformers" or "CRISPR gene editing papers"

📥 **Download**: "Download first 10 as CSV" or "Export as BibTeX"

📄 **Lookup**: "Look up DOI 10.1038/nature14539"

📊 **Citations**: "Who cited this paper?" (after looking up a paper)

Just type naturally and I'll understand!
                """
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            else:
                response = "I'm not sure what you mean. Try:\n- 'Find papers on [topic]'\n- 'Download first 10'\n- 'Help'"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


def display_papers_compact(papers: list[Paper]):
    """Display papers in a compact format for chat."""
    for i, p in enumerate(papers, 1):
        doi_link = f"https://doi.org/{p.doi}" if p.doi else ""
        st.markdown(f"**{i}. [{p.title}]({doi_link})**" if doi_link else f"**{i}. {p.title}**")
        
        # Handle authors - could be Author objects or strings
        if p.authors:
            author_names = []
            for a in p.authors[:2]:
                if hasattr(a, 'name'):
                    author_names.append(a.name)
                else:
                    author_names.append(str(a))
            authors_str = ', '.join(author_names)
            if len(p.authors) > 2:
                authors_str += '...'
        else:
            authors_str = "Unknown"
        
        # Handle citation count - field is 'citation_count' not 'citations'
        citation_count = getattr(p, 'citation_count', 0) or 0
        year = p.year if p.year else "?"
        st.caption(f"*{authors_str}* ({year}) - {citation_count:,} citations")


def papers_to_csv(papers):
    """Convert to CSV."""
    import pandas as pd
    
    def get_author_names(authors):
        if not authors:
            return ""
        names = []
        for a in authors:
            if hasattr(a, 'name'):
                names.append(a.name)
            else:
                names.append(str(a))
        return "; ".join(names)
    
    data = [{
        "title": p.title,
        "authors": get_author_names(p.authors),
        "year": p.year,
        "citations": getattr(p, 'citation_count', 0) or 0,
        "doi": p.doi,
        "venue": p.venue,
    } for p in papers]
    return pd.DataFrame(data).to_csv(index=False)


def papers_to_bibtex(papers):
    """Convert to BibTeX."""
    entries = []
    for i, p in enumerate(papers, 1):
        # Handle Author objects or strings
        author_names = []
        if p.authors:
            for a in p.authors:
                if hasattr(a, 'name'):
                    author_names.append(a.name)
                else:
                    author_names.append(str(a))
        
        first_author = author_names[0].split()[-1] if author_names else "unknown"
        key = f"{first_author.lower()}{p.year or 'nd'}_{i}"
        entries.append(f"""@article{{{key},
  title = {{{p.title}}},
  author = {{{' and '.join(author_names)}}},
  year = {{{p.year or 'n.d.'}}},
  journal = {{{p.venue or ''}}},
  doi = {{{p.doi or ''}}}
}}""")
    return "\n\n".join(entries)


def papers_to_json(papers):
    """Convert to JSON."""
    def get_author_names(authors):
        if not authors:
            return []
        names = []
        for a in authors:
            if hasattr(a, 'name'):
                names.append(a.name)
            else:
                names.append(str(a))
        return names
    
    return json.dumps([{
        "title": p.title,
        "authors": get_author_names(p.authors),
        "year": p.year,
        "citations": getattr(p, 'citation_count', 0) or 0,
        "doi": p.doi,
        "venue": p.venue,
        "abstract": p.abstract,
    } for p in papers], indent=2)


if __name__ == "__main__":
    main()
