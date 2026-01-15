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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from litforge.api import search, lookup, citations, references, Paper

# Try to import LLM for query expansion
try:
    from litforge.llm.router import LLMRouter, get_llm
    # Check if ANY LLM provider is available
    HAS_LLM = any([
        os.environ.get("CEREBRAS_API_KEY"),
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("GITHUB_TOKEN"),
        os.environ.get("OPENAI_API_KEY"),
    ])
except ImportError:
    HAS_LLM = False

# LLM-powered query expansion
def expand_query_with_llm(user_query: str, provider: str | None = None) -> dict:
    """
    Use LLM (free tier cascade) to understand the user's intent and expand their query.
    
    Providers tried in order: Cerebras > Groq > Google > GitHub Models > OpenAI
    
    Returns:
        {
            "search_queries": list of optimized search queries,
            "must_contain": list of terms that must appear in results,
            "exclude_terms": list of terms to exclude,
            "explanation": why these queries were chosen,
            "provider": which LLM was used
        }
    """
    if not HAS_LLM:
        return None
    
    try:
        llm = get_llm(provider)  # Smart router with free-tier cascade
        
        system_prompt = """You are an expert scientific literature search assistant. Given a user's natural language question, 
you must convert it into effective search queries for academic databases like OpenAlex and arXiv.

CRITICAL: Users often use informal or incorrect terminology. Your job is to:
1. Identify what the user is REALLY looking for (interpret their intent)
2. Generate 2-4 precise academic search queries using CORRECT technical terminology
3. Identify key terms that MUST appear in relevant results
4. Identify terms to EXCLUDE (homonyms, unrelated fields with similar names)

IMPORTANT EXAMPLES of correcting user terminology:
- "liquid foundation models" → The user likely means "Liquid Neural Networks" (by Hasani et al.), NOT foundation models. Search: ["Liquid Neural Networks", "Liquid Time-constant Networks", "LTC neural networks", "continuous-time neural networks"]
- "transformers" → Could be ML model OR electrical. If ML context, search: ["transformer neural network", "attention mechanism", "BERT", "GPT"]
- "GAN" → Search: ["generative adversarial network", "GAN deep learning", "image synthesis neural network"]
- "diffusion models" → Search: ["diffusion probabilistic models", "denoising diffusion", "score-based generative models"]

Return ONLY valid JSON in this exact format:
{
    "search_queries": ["query1", "query2", ...],
    "must_contain": ["term1", "term2"],
    "exclude_terms": ["term1", "term2"],
    "explanation": "Brief explanation of what you think the user wants"
}
"""
        
        response = llm.generate(
            prompt=f"User query: {user_query}",
            system_prompt=system_prompt,
            max_tokens=500
        )
        
        # Parse JSON from response
        # Handle potential markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        result = json.loads(response)
        # Add which provider was used
        result["provider"] = llm.provider_name if hasattr(llm, 'provider_name') else "unknown"
        return result
    except Exception as e:
        st.warning(f"LLM query expansion failed: {e}")
        return None


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
    msg_lower = message.lower().strip()
    
    # Show more results intent
    show_more_keywords = ["show more", "see more", "show all", "see all", "rest of", "remaining", "all papers", "more papers", "view all"]
    if any(kw in msg_lower for kw in show_more_keywords):
        return {"intent": "show_more", "params": {}}
    
    # PDF download intent - needs special handling
    pdf_keywords = ["pdf", "pdfs", "pdf file", "full text", "fulltext", "full-text"]
    if any(kw in msg_lower for kw in pdf_keywords):
        # Check if they're asking for actual PDF files
        import re
        num_match = re.search(r'(first|top)?\s*(\d+)', msg_lower)
        num = int(num_match.group(2)) if num_match else 5
        return {"intent": "pdf_request", "params": {"num": num}}
    
    # Download/export intent (metadata exports)
    download_keywords = ["download", "export", "save", "csv", "bibtex", "json", "bib"]
    if any(kw in msg_lower for kw in download_keywords):
        # Extract number if present
        import re
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
    import re
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
    # Clean the query
    noise = ["find", "search", "look for", "papers on", "papers about", 
             "can you", "please", "i want", "i need", "show me", "get me",
             "list of", "articles on", "publications"]
    cleaned = msg_lower
    for n in noise:
        cleaned = cleaned.replace(n, " ")
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
            
            # Show available providers
            providers = []
            if os.environ.get("CEREBRAS_API_KEY"): providers.append("Cerebras")
            if os.environ.get("GROQ_API_KEY"): providers.append("Groq")
            if os.environ.get("GOOGLE_API_KEY"): providers.append("Gemini")
            if os.environ.get("GITHUB_TOKEN"): providers.append("GitHub")
            if os.environ.get("OPENAI_API_KEY"): providers.append("OpenAI")
            
            if providers:
                st.caption(f"🔗 Providers: {', '.join(providers)}")
        else:
            st.warning("⚠️ Set API keys in .env to enable AI-powered search")
        
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
                query = intent["params"]["query"]
                
                # Try LLM-powered query expansion for better results
                expanded = None
                if HAS_LLM and st.session_state.get("use_llm_search", True):
                    with st.spinner("🧠 Understanding your query..."):
                        expanded = expand_query_with_llm(prompt)
                
                all_papers = []
                
                if expanded and expanded.get("search_queries"):
                    # Use LLM-expanded queries
                    provider = expanded.get("provider", "AI")
                    provider_emoji = {"cerebras": "⚡", "groq": "🚀", "google": "🔮", "github": "🐙", "openai": "💚"}.get(provider, "🧠")
                    st.info(f"{provider_emoji} **{provider.title()} Insight**: {expanded.get('explanation', 'Expanding search...')}")
                    
                    search_queries = expanded["search_queries"]
                    must_contain = [t.lower() for t in expanded.get("must_contain", [])]
                    exclude_terms = [t.lower() for t in expanded.get("exclude_terms", [])]
                    
                    with st.spinner(f"Searching with {len(search_queries)} optimized queries..."):
                        for sq in search_queries[:4]:  # Limit to 4 queries
                            try:
                                results = search(sq, limit=max_results // 2, year_from=year_from, year_to=year_to)
                                all_papers.extend(results)
                            except Exception as e:
                                pass
                    
                    # Deduplicate by title
                    seen = set()
                    unique_papers = []
                    for p in all_papers:
                        key = p.title.lower()[:60] if p.title else ""
                        if key and key not in seen:
                            seen.add(key)
                            # Apply must_contain filter
                            text = f"{p.title} {p.abstract or ''}".lower()
                            if must_contain and not any(t in text for t in must_contain):
                                continue
                            # Apply exclude filter  
                            if exclude_terms and any(t in text for t in exclude_terms):
                                continue
                            unique_papers.append(p)
                    
                    papers = unique_papers[:max_results]
                else:
                    # Fallback to simple search
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
        st.caption(f"*{', '.join(p.authors[:2])}{'...' if len(p.authors) > 2 else ''}* ({p.year}) - {p.citations:,} citations")


def papers_to_csv(papers):
    """Convert to CSV."""
    import pandas as pd
    data = [{
        "title": p.title,
        "authors": "; ".join(p.authors),
        "year": p.year,
        "citations": p.citations,
        "doi": p.doi,
        "venue": p.venue,
    } for p in papers]
    return pd.DataFrame(data).to_csv(index=False)


def papers_to_bibtex(papers):
    """Convert to BibTeX."""
    entries = []
    for i, p in enumerate(papers, 1):
        first_author = p.authors[0].split()[-1] if p.authors else "unknown"
        key = f"{first_author.lower()}{p.year or 'nd'}_{i}"
        entries.append(f"""@article{{{key},
  title = {{{p.title}}},
  author = {{{' and '.join(p.authors)}}},
  year = {{{p.year or 'n.d.'}}},
  journal = {{{p.venue or ''}}},
  doi = {{{p.doi or ''}}}
}}""")
    return "\n\n".join(entries)


def papers_to_json(papers):
    """Convert to JSON."""
    return json.dumps([{
        "title": p.title,
        "authors": p.authors,
        "year": p.year,
        "citations": p.citations,
        "doi": p.doi,
        "venue": p.venue,
        "abstract": p.abstract,
    } for p in papers], indent=2)


if __name__ == "__main__":
    main()
